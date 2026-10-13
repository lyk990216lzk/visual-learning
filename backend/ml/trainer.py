"""后台训练任务管理：在独立线程中执行训练，并把进度实时写回数据库。"""

from __future__ import annotations

import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch
from django.db import close_old_connections, connections
from django.utils import timezone

from . import zoo
from .datasets import build_dataloaders, scan_image_folder
from .engine import TrainConfig, choose_cpu_threads, resolve_device, train_model


def _dataset_class_counts(loader, num_classes: int) -> list:
    """统计训练集中每个类别的样本数，兼容 ImageFolder 与分层抽样后的 Subset。"""
    dataset = getattr(loader, "dataset", None)
    targets = None
    for candidate in (dataset, getattr(dataset, "dataset", None)):
        if candidate is None:
            continue
        if hasattr(candidate, "targets"):
            targets = list(candidate.targets)
            break
        if hasattr(candidate, "samples"):
            targets = [label for _, label in candidate.samples]
            break
    if targets is None:
        return [0] * num_classes
    indices = getattr(dataset, "indices", None)
    if indices is not None:
        targets = [targets[index] for index in indices]
    counts = [0] * num_classes
    for label in targets:
        if 0 <= int(label) < num_classes:
            counts[int(label)] += 1
    return counts


def _log_line(run, message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{timestamp}] {message}"
    run.log = (run.log + "\n" + line) if run.log else line
    if len(run.log) > 40000:
        run.log = run.log[-40000:]


class TrainingManager:
    """单例：维护运行中的训练线程，支持启动 / 查询 / 取消。"""

    _instance: Optional["TrainingManager"] = None
    _instance_lock = threading.Lock()

    def __init__(self):
        self._threads: Dict[int, threading.Thread] = {}
        self._stop_flags: Dict[int, threading.Event] = {}
        self._lock = threading.Lock()

    @classmethod
    def instance(cls) -> "TrainingManager":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------ 对外接口
    def start(self, run_id: int) -> bool:
        with self._lock:
            thread = self._threads.get(run_id)
            if thread and thread.is_alive():
                return False
            stop_event = threading.Event()
            worker = threading.Thread(target=self._run, args=(run_id, stop_event),
                                      name=f"training-run-{run_id}", daemon=True)
            self._threads[run_id] = worker
            self._stop_flags[run_id] = stop_event
        worker.start()
        return True

    def cancel(self, run_id: int) -> bool:
        with self._lock:
            stop_event = self._stop_flags.get(run_id)
        if stop_event is None:
            return False
        stop_event.set()
        return True

    def is_running(self, run_id: int) -> bool:
        with self._lock:
            thread = self._threads.get(run_id)
            return bool(thread and thread.is_alive())

    def running_ids(self):
        with self._lock:
            return [run_id for run_id, thread in self._threads.items() if thread.is_alive()]

    # ------------------------------------------------------------------ 训练主流程
    def _run(self, run_id: int, stop_event: threading.Event) -> None:
        from apps.recognition.models import TrainedModel, TrainingRun

        close_old_connections()
        previous_threads = torch.get_num_threads()
        run = TrainingRun.objects.filter(pk=run_id).first()
        if run is None:
            self._cleanup(run_id)
            return

        last_db_write = 0.0
        try:
            run.status = TrainingRun.Status.RUNNING
            run.started_at = timezone.now()
            run.progress = 0
            _log_line(run, f"训练任务启动：模型 {run.architecture}，数据集 {run.dataset.name}")
            run.save(update_fields=["status", "started_at", "progress", "log"])

            dataset_dir = Path(run.dataset.dir_path)
            if not dataset_dir.exists():
                raise FileNotFoundError(f"数据集目录不存在：{dataset_dir}")

            device = resolve_device(run.device)
            _log_line(run, f"使用计算设备：{device}")

            train_loader, val_loader, class_names = build_dataloaders(
                dataset_dir,
                image_size=run.image_size,
                batch_size=run.batch_size,
                val_split=run.val_split,
                num_workers=run.num_workers,
                seed=run.seed,
            )
            run.num_classes = len(class_names)
            run.class_names = class_names
            _log_line(run, f"数据加载完成：类别 {class_names}，训练 batch 数 {len(train_loader)}，"
                           f"验证 batch 数 {len(val_loader)}")
            run.save(update_fields=["num_classes", "class_names", "log"])

            model = zoo.build_model(run.architecture, num_classes=len(class_names))
            params = zoo.count_parameters(model, trainable_only=False)
            _log_line(run, f"网络构建完成：{run.architecture}，参数量 {params / 1e6:.3f}M")
            if device.type == "cpu":
                threads = choose_cpu_threads(model)
                torch.set_num_threads(threads)
                _log_line(run, f"CPU 训练线程数设为 {threads}"
                               f"（原线程数 {previous_threads}，深度卷积网络在 CPU 上多线程反而更慢）")

            cfg = TrainConfig(
                epochs=run.epochs,
                batch_size=run.batch_size,
                lr=run.learning_rate,
                weight_decay=run.weight_decay,
                optimizer=run.optimizer,
                scheduler=run.scheduler,
                image_size=run.image_size,
                num_workers=run.num_workers,
                val_split=run.val_split,
                label_smoothing=run.label_smoothing,
                seed=run.seed,
                device=run.device,
                augment=run.augment,
                max_batches_per_epoch=run.max_batches_per_epoch,
                class_weighted_loss=run.class_weighted_loss,
            )

            state = {"last_write": 0.0}

            def on_batch(payload: Dict) -> None:
                now = time.time()
                if now - state["last_write"] < 1.5:
                    return
                state["last_write"] = now
                TrainingRun.objects.filter(pk=run_id).update(
                    progress=payload["progress"],
                    current_epoch=payload["epoch"],
                    latest_metrics=payload,
                )

            def on_epoch(payload: Dict) -> None:
                TrainingRun.objects.filter(pk=run_id).update(
                    progress=payload["progress"],
                    current_epoch=payload["record"]["epoch"],
                    history=payload["history"],
                    best_accuracy=payload["best_accuracy"],
                    latest_metrics=payload["record"],
                )

            def log(message: str) -> None:
                current = TrainingRun.objects.get(pk=run_id)
                _log_line(current, message)
                current.save(update_fields=["log"])

            result = train_model(
                model, train_loader, val_loader, class_names, cfg,
                on_epoch=on_epoch, on_batch=on_batch,
                should_stop=stop_event.is_set, log=log,
                class_counts=_dataset_class_counts(train_loader, len(class_names))
                if run.class_weighted_loss else None,
            )

            # ---------------------------------------------------- 保存权重与模型记录
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            weights_dir = run.dataset.weights_dir
            weights_dir.mkdir(parents=True, exist_ok=True)
            file_path = weights_dir / f"{run.architecture}_{run.dataset_id}_{timestamp}.pt"
            torch.save({
                "state_dict": result["state_dict"],
                "architecture": run.architecture,
                "classes": class_names,
                "image_size": run.image_size,
                "history": result["history"],
                "metrics": result["report"],
                "config": cfg.to_dict(),
            }, file_path)

            report = result["report"]
            trained = TrainedModel.objects.create(
                name=f"{run.architecture}@{run.dataset.name}"[:80],
                architecture=run.architecture,
                dataset=run.dataset,
                training_run=run,
                classes=class_names,
                num_classes=len(class_names),
                input_size=run.image_size,
                accuracy=report["accuracy"],
                macro_f1=report.get("macro_f1", 0),
                metrics=report,
                file_path=str(file_path),
                file_size=file_path.stat().st_size,
                params_million=round(params / 1e6, 3),
                notes=f"由训练任务 #{run.id} 自动生成",
            )

            # 同结构下如果没有默认模型，或者新模型精度更高，则自动设为默认推理模型
            current_active = TrainedModel.objects.filter(
                architecture=run.architecture, is_active=True).exclude(pk=trained.pk).first()
            if current_active is None or report["accuracy"] >= current_active.accuracy:
                trained.is_active = True
                trained.save(update_fields=["is_active", "updated_at"])
                _log_line(run, f"已自动将 {trained.name} 设为 {run.architecture} 的默认推理模型")

            run.status = TrainingRun.Status.CANCELED if report.get("stopped_early") else TrainingRun.Status.SUCCESS
            run.progress = 100
            run.history = result["history"]
            run.metrics = report
            run.best_accuracy = report["best_accuracy"]
            run.finished_at = timezone.now()
            run.trained_model = trained
            run.duration_seconds = report["duration_seconds"]
            _log_line(run, f"权重已保存：{file_path.name}（{trained.file_size / 1024:.1f} KB），"
                           f"准确率 {report['accuracy']:.4f}")
            run.save()

        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            run = TrainingRun.objects.filter(pk=run_id).first()
            if run is not None:
                run.status = TrainingRun.Status.FAILED
                run.error = f"{type(exc).__name__}: {exc}"
                run.finished_at = timezone.now()
                _log_line(run, f"训练失败：{run.error}")
                run.save()
        finally:
            if torch.get_num_threads() != previous_threads:
                torch.set_num_threads(previous_threads)
            connections.close_all()
            self._cleanup(run_id)

    def _cleanup(self, run_id: int) -> None:
        with self._lock:
            self._threads.pop(run_id, None)
            self._stop_flags.pop(run_id, None)


manager = TrainingManager.instance()
