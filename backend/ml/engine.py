"""训练与评估引擎：负责一个完整的「训练 -> 验证 -> 保存最优权重」流程。"""

from __future__ import annotations

import copy
import os
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn

from .metrics import evaluate_model


DEVICE_ALIASES = {"cuda": "gpu"}
SUPPORTED_DEVICES = ("gpu", "cpu", "mps", "auto")


def normalize_device(preference: str = "gpu") -> str:
    """把界面设备名转换为训练引擎使用的规范名称。"""
    name = (preference or "gpu").strip().lower()
    return DEVICE_ALIASES.get(name, name)


def is_device_available(preference: str) -> bool:
    """检测指定训练设备在当前 Python 进程中是否可用。"""
    name = normalize_device(preference)
    if name == "cpu":
        return True
    if name == "gpu":
        return torch.cuda.is_available()
    if name == "mps":
        mps = getattr(torch.backends, "mps", None)
        return bool(mps and mps.is_available())
    return name == "auto"


def get_device_options() -> List[Dict]:
    """返回训练配置可展示的设备及其实时可用状态。"""
    gpu_available = is_device_available("gpu")
    mps_available = is_device_available("mps")
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else ""
    options = [
        {
            "value": "gpu",
            "label": "GPU",
            "available": gpu_available,
            "detail": gpu_name or "当前环境未检测到可用的 CUDA GPU",
        },
        {
            "value": "cpu",
            "label": "CPU",
            "available": True,
            "detail": f"使用 {torch.get_num_threads()} 个 CPU 线程",
        },
        {
            "value": "mps",
            "label": "MPS",
            "available": mps_available,
            "detail": "Apple Metal 加速可用" if mps_available else "当前环境未检测到可用的 Apple MPS",
        },
    ]
    return sorted(options, key=lambda item: not item["available"])


def resolve_device(preference: str = "auto") -> torch.device:
    """解析训练设备；只有 auto 会在设备不可用时自动降级到 CPU。"""
    preference = normalize_device(preference)
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "gpu":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("当前环境未检测到可用的 GPU（CUDA），无法开始 GPU 训练")
    if preference == "mps":
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("当前环境未检测到可用的 Apple MPS，无法开始 MPS 训练")
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    raise ValueError(f"不支持的训练设备：{preference}")


def choose_cpu_threads(model: nn.Module) -> int:
    """为 CPU 训练挑选合适的线程数。

    经验结论（实测于 Apple Silicon + PyTorch 2.12）：标准卷积在 4 线程附近最优，
    而**深度可分离卷积/分组卷积**在 CPU 上对线程数极其敏感 —— 线程越多反而越慢
    （MobileNetV2 实测：1 线程 1.3s/batch，8 线程 13.8s/batch）。因此这里按分组卷积占比
    自动降线程，保证开箱即用的训练速度。
    """
    override = os.environ.get("IR_TORCH_THREADS")
    if override and override.isdigit():
        return max(1, int(override))

    total_conv, grouped_conv = 0, 0
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            total_conv += 1
            if module.groups > 1:
                grouped_conv += 1
    if total_conv and grouped_conv / total_conv > 0.4:
        return 1
    return max(1, min(4, os.cpu_count() or 4))


@dataclass
class TrainConfig:
    """训练超参数，与前端「训练配置」表单一一对应。"""

    epochs: int = 10
    batch_size: int = 32
    lr: float = 0.01
    weight_decay: float = 1e-4
    momentum: float = 0.9
    optimizer: str = "sgd"          # sgd | adam | adamw
    scheduler: str = "cosine"       # none | step | cosine
    image_size: int = 64
    num_workers: int = 0
    val_split: float = 0.2
    label_smoothing: float = 0.0
    seed: int = 42
    device: str = "auto"
    augment: bool = True
    max_batches_per_epoch: int = 0  # >0 时每个 epoch 只跑前 N 个 batch（快速演示用）
    class_weighted_loss: bool = False

    def to_dict(self) -> Dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def build_optimizer(model: nn.Module, cfg: TrainConfig):
    name = (cfg.optimizer or "sgd").lower()
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    return torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum,
                           weight_decay=cfg.weight_decay, nesterov=cfg.momentum > 0)


def build_scheduler(optimizer, cfg: TrainConfig):
    name = (cfg.scheduler or "none").lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, cfg.epochs))
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, cfg.epochs // 3), gamma=0.1)
    return None


def _forward_loss(model: nn.Module, images, targets, criterion):
    """兼容 GoogLeNet 这类带辅助分类器的网络。"""
    outputs = model(images)
    if isinstance(outputs, (tuple, list)):
        main, *auxes = outputs
        loss = criterion(main, targets)
        for aux in auxes:
            loss = loss + 0.3 * criterion(aux, targets)
        return main, loss
    return outputs, criterion(outputs, targets)


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    class_names: List[str],
    cfg: TrainConfig,
    on_epoch: Optional[Callable[[Dict], None]] = None,
    on_batch: Optional[Callable[[Dict], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    log: Optional[Callable[[str], None]] = None,
    class_counts: Optional[List[int]] = None,
) -> Dict:
    """执行训练并返回完整结果（历史曲线、最优权重、最终评估报告）。"""
    log = log or (lambda message: None)
    torch.manual_seed(cfg.seed)
    device = resolve_device(cfg.device)
    model = model.to(device)

    if cfg.class_weighted_loss and class_counts:
        total = sum(class_counts)
        weights = torch.tensor([total / (len(class_counts) * max(1, c)) for c in class_counts],
                               dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=cfg.label_smoothing)
        log(f"启用类别加权损失：{ [round(w, 2) for w in weights.tolist()] }")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history: List[Dict] = []
    best_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())
    total_epochs = max(1, cfg.epochs)
    stopped = False
    start_time = time.time()

    for epoch in range(1, total_epochs + 1):
        model.train()
        epoch_start = time.time()
        running_loss, seen, correct = 0.0, 0, 0
        batch_index = 0
        num_batches = len(train_loader)
        limit = num_batches if cfg.max_batches_per_epoch <= 0 else min(num_batches, cfg.max_batches_per_epoch)

        for images, targets in train_loader:
            if batch_index >= limit:
                break
            if should_stop and should_stop():
                stopped = True
                break
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits, loss = _forward_loss(model, images, targets, criterion)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, loss = _forward_loss(model, images, targets, criterion)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            batch_size = images.size(0)
            running_loss += float(loss.item()) * batch_size
            seen += batch_size
            correct += int((logits.argmax(dim=1) == targets).sum().item())
            batch_index += 1

            if on_batch:
                epoch_fraction = (epoch - 1 + batch_index / max(1, limit)) / total_epochs
                on_batch({
                    "epoch": epoch,
                    "total_epochs": total_epochs,
                    "batch": batch_index,
                    "total_batches": limit,
                    "progress": round(min(99.0, epoch_fraction * 100), 2),
                    "loss": round(running_loss / max(1, seen), 4),
                    "accuracy": round(correct / max(1, seen), 4),
                    "elapsed": round(time.time() - start_time, 1),
                })

        if stopped:
            log(f"第 {epoch} 轮收到停止指令，提前结束训练")
            break

        train_loss = running_loss / max(1, seen)
        train_acc = correct / max(1, seen)
        val_report = evaluate_model(model, val_loader, device, class_names, criterion)
        val_acc = float(val_report["accuracy"])
        val_loss = float(val_report["loss"])

        if scheduler is not None:
            scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_seconds = time.time() - epoch_start

        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_acc, 4),
            "lr": round(current_lr, 6),
            "seconds": round(epoch_seconds, 2),
        }
        history.append(record)

        improved = val_acc >= best_acc
        if improved:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

        log(f"[Epoch {epoch}/{total_epochs}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} lr={current_lr:.5f} ({epoch_seconds:.1f}s)"
            + ("  *best" if improved else ""))

        if on_epoch:
            on_epoch({"record": record, "history": history, "best_accuracy": round(best_acc, 4),
                      "progress": round(epoch / total_epochs * 100, 2)})

    model.load_state_dict(best_state)
    final_report = evaluate_model(model, val_loader, device, class_names, criterion)
    final_report["best_accuracy"] = round(best_acc, 4)
    final_report["history"] = history
    final_report["epochs_completed"] = len(history)
    final_report["duration_seconds"] = round(time.time() - start_time, 2)
    final_report["device"] = str(device)
    final_report["stopped_early"] = stopped
    log(f"训练结束：最优验证准确率 {best_acc:.4f}，总耗时 {final_report['duration_seconds']}s")
    return {"history": history, "report": final_report, "state_dict": best_state,
            "class_names": class_names, "config": cfg.to_dict()}


def cosine_curve_x(total: int) -> List[int]:
    return list(range(1, max(1, total) + 1))


def warmup_lr(step: int, warmup: int, base_lr: float) -> float:
    if warmup <= 0 or step >= warmup:
        return base_lr
    return base_lr * (step + 1) / warmup


__all__ = ["TrainConfig", "train_model", "resolve_device", "normalize_device",
           "is_device_available", "get_device_options", "build_optimizer",
           "build_scheduler", "cosine_curve_x"]
