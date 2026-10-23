"""目标检测：YOLO 数据准备、训练适配与评估。"""

from __future__ import annotations

import json
import random
import shutil
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .datasets import IMAGE_EXTENSIONS
from .engine import resolve_device
from .task_datasets import DatasetFormatError, scan_dataset

YOLO_CONFIG_CANDIDATES = {
    "yolov3": ("yolov3.yaml", "yolov3u.yaml", "yolov3-spp.yaml"),
    "yolov5": ("yolov5.yaml", "yolov5nu.yaml", "yolov5n.yaml"),
    "yolov8": ("yolov8.yaml", "yolov8n.yaml"),
    "yolov10": ("yolov10n.yaml",),
    "yolov11": ("yolo11.yaml", "yolo11n.yaml", "yolov11n.yaml"),
}


class DetectionBackendError(RuntimeError):
    """YOLO 依赖或模型后端不可用。"""


def _require_ultralytics():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise DetectionBackendError(
            "目标检测需要安装 ultralytics："
            "uv pip install --python .venv/bin/python 'ultralytics>=8.3,<9'"
        ) from exc
    return YOLO


def _image_files(directory: Path) -> List[Path]:
    return sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _collect_pairs(image_root: Path, label_root: Path) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    for image_path in _image_files(image_root):
        relative = image_path.relative_to(image_root).with_suffix(".txt")
        label_path = label_root / relative
        if not label_path.is_file():
            raise DatasetFormatError([f"图片 {image_path.name} 缺少对应标注 {relative.as_posix()}"])
        pairs.append((image_path, label_path))
    if not pairs:
        raise DatasetFormatError([f"{image_root} 中没有图片"])
    return pairs


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        destination.symlink_to(source.resolve())
    except OSError:
        shutil.copy2(source, destination)


def _write_yaml(path: Path, dataset_root: Path, class_names: Sequence[str]) -> None:
    lines = [
        f"path: {json.dumps(str(dataset_root.resolve()), ensure_ascii=False)}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    for index, name in enumerate(class_names):
        lines.append(f"  {index}: {json.dumps(str(name), ensure_ascii=False)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_detection_dataset(dataset_dir: str | Path, work_dir: str | Path,
                              val_split: float = 0.2, seed: int = 42) -> Dict:
    """生成 Ultralytics 可用的数据描述；自动模式使用临时符号链接完成配对划分。"""
    source_root = Path(dataset_dir)
    work_root = Path(work_dir)
    info = scan_dataset(source_root, "detection")
    class_names = list(info["classes"])
    train_pairs = _collect_pairs(source_root / "images" / "train",
                                 source_root / "labels" / "train")

    val_split_name = next(
        (name for name in ("val", "test")
         if (source_root / "images" / name).is_dir()
         and (source_root / "labels" / name).is_dir()),
        None,
    )
    if val_split_name:
        val_pairs = _collect_pairs(source_root / "images" / val_split_name,
                                   source_root / "labels" / val_split_name)
        split_mode = "explicit"
    else:
        if len(train_pairs) < 2:
            raise DatasetFormatError(["自动划分训练集和验证集至少需要 2 张图片"])
        rng = random.Random(seed)
        shuffled = list(train_pairs)
        rng.shuffle(shuffled)
        n_val = int(round(len(shuffled) * float(val_split)))
        n_val = max(1, min(len(shuffled) - 1, n_val))
        val_pairs = shuffled[:n_val]
        train_pairs = shuffled[n_val:]
        split_mode = "auto"

    yolo_root = work_root / "yolo_dataset"
    if yolo_root.exists():
        shutil.rmtree(yolo_root)
    source_split = {
        "train": "train",
        "val": val_split_name or "train",
    }
    for split, pairs in (("train", train_pairs), ("val", val_pairs)):
        image_base = source_root / "images" / source_split[split]
        label_base = source_root / "labels" / source_split[split]
        for image_path, label_path in pairs:
            relative_image = image_path.relative_to(image_base)
            relative_label = label_path.relative_to(label_base)
            _link_or_copy(image_path, yolo_root / "images" / split / relative_image)
            _link_or_copy(label_path, yolo_root / "labels" / split / relative_label)

    data_yaml = work_root / "data.yaml"
    _write_yaml(data_yaml, yolo_root, class_names)
    return {
        "data_yaml": data_yaml,
        "dataset_root": yolo_root,
        "classes": class_names,
        "split_mode": split_mode,
        "train_count": len(train_pairs),
        "val_count": len(val_pairs),
    }


def _load_yolo_model(architecture: str):
    YOLO = _require_ultralytics()
    errors = []
    for config in YOLO_CONFIG_CANDIDATES.get(architecture, ()):
        try:
            return YOLO(config), config
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{config}: {exc}")
    raise DetectionBackendError(
        f"当前 Ultralytics 版本找不到 {architecture} 的可用结构配置；"
        f"尝试过 {errors or ['无候选配置']}"
    )


def _metric_value(metrics: Dict, *keys: str) -> float:
    for key in keys:
        value = metrics.get(key)
        try:
            if value is not None:
                return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def _device_arg(device: str) -> str:
    value = (device or "cpu").lower()
    if value == "auto":
        resolved = resolve_device("auto")
        value = "cuda" if resolved.type == "cuda" else resolved.type
    if value in {"gpu", "cuda"}:
        return "0"
    return value


def train_yolo_model(
    architecture: str,
    dataset_dir: str | Path,
    work_dir: str | Path,
    cfg: Dict,
    on_epoch: Optional[Callable[[Dict], None]] = None,
    on_batch: Optional[Callable[[Dict], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    log: Optional[Callable[[str], None]] = None,
) -> Dict:
    """执行 Ultralytics YOLO 训练并返回最优权重及统一指标。"""
    log = log or (lambda message: None)
    prepared = prepare_detection_dataset(
        dataset_dir, work_dir,
        val_split=float(cfg.get("val_split", 0.2)),
        seed=int(cfg.get("seed", 42)),
    )
    model, config_name = _load_yolo_model(architecture)
    epochs = max(1, int(cfg.get("epochs", 10)))
    history: List[Dict] = []
    start = time.time()
    best_map = 0.0
    log(
        f"YOLO 数据准备完成：{prepared['split_mode']}，训练 {prepared['train_count']}，"
        f"验证 {prepared['val_count']}，配置 {config_name}"
    )

    def batch_callback(trainer) -> None:
        if should_stop and should_stop():
            trainer.stop = True
        if on_batch:
            epoch = int(getattr(trainer, "epoch", 0)) + 1
            total_epochs = int(getattr(trainer, "epochs", epochs))
            batch = int(getattr(trainer, "batch_i", 0)) + 1
            total_batches = int(getattr(trainer, "nb", 0)) or 1
            progress = ((epoch - 1) + batch / max(1, total_batches)) / max(1, total_epochs)
            losses = getattr(trainer, "loss_items", None)
            loss = float(losses.sum()) if losses is not None and hasattr(losses, "sum") else None
            on_batch({
                "epoch": epoch,
                "total_epochs": total_epochs,
                "batch": batch,
                "total_batches": total_batches,
                "progress": round(min(99.0, progress * 100), 2),
                "loss": round(loss, 4) if loss is not None else None,
                "elapsed": round(time.time() - start, 1),
            })

    def epoch_callback(trainer) -> None:
        nonlocal best_map
        metrics = getattr(trainer, "metrics", {}) or {}
        map50 = _metric_value(metrics, "metrics/mAP50(B)", "metrics/mAP50")
        map5095 = _metric_value(metrics, "metrics/mAP50-95(B)", "metrics/mAP50-95")
        precision = _metric_value(metrics, "metrics/precision(B)", "metrics/precision")
        recall = _metric_value(metrics, "metrics/recall(B)", "metrics/recall")
        epoch = int(getattr(trainer, "epoch", 0)) + 1
        best_map = max(best_map, map50)
        record = {
            "epoch": epoch,
            "map50": round(map50, 4),
            "map50_95": round(map5095, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "train_loss": None,
            "val_loss": None,
            "train_accuracy": map50,
            "val_accuracy": map50,
            "seconds": None,
        }
        history.append(record)
        log(f"[Epoch {epoch}/{epochs}] mAP50={map50:.4f} mAP50-95={map5095:.4f}")
        if on_epoch:
            on_epoch({
                "record": record,
                "history": history,
                "best_accuracy": round(best_map, 4),
                "progress": round(epoch / epochs * 100, 2),
            })
        if should_stop and should_stop():
            trainer.stop = True

    model.add_callback("on_train_batch_end", batch_callback)
    model.add_callback("on_fit_epoch_end", epoch_callback)
    optimizer = str(cfg.get("optimizer", "sgd")).lower()
    if optimizer not in {"sgd", "adam", "adamw", "auto"}:
        optimizer = "auto"
    model.train(
        data=str(prepared["data_yaml"]),
        epochs=epochs,
        batch=max(1, int(cfg.get("batch_size", 16))),
        imgsz=max(32, int(cfg.get("image_size", 640))),
        lr0=float(cfg.get("learning_rate", 0.01)),
        weight_decay=float(cfg.get("weight_decay", 0.0005)),
        label_smoothing=float(cfg.get("label_smoothing", 0.0)),
        optimizer=optimizer,
        device=_device_arg(str(cfg.get("device", "cpu"))),
        workers=max(0, int(cfg.get("num_workers", 0))),
        seed=int(cfg.get("seed", 42)),
        deterministic=True,
        augment=bool(cfg.get("augment", True)),
        project=str(Path(work_dir) / "yolo_runs"),
        name="train",
        exist_ok=True,
        plots=False,
        verbose=False,
    )
    trainer = getattr(model, "trainer", None)
    metrics = getattr(trainer, "metrics", {}) or {}
    map50 = _metric_value(metrics, "metrics/mAP50(B)", "metrics/mAP50")
    map5095 = _metric_value(metrics, "metrics/mAP50-95(B)", "metrics/mAP50-95")
    best_path = Path(getattr(trainer, "best", "") or "")
    if not best_path.is_file():
        save_dir = Path(getattr(trainer, "save_dir", Path(work_dir) / "yolo_runs" / "train"))
        best_path = save_dir / "weights" / "best.pt"
    if not best_path.is_file():
        raise DetectionBackendError("YOLO 训练结束但没有生成 best.pt 权重文件")

    model_module = getattr(model, "model", None)
    params = sum(p.numel() for p in model_module.parameters()) if model_module is not None else 0
    report = {
        "accuracy": round(map50, 4),
        "best_accuracy": round(max(best_map, map50), 4),
        "map50": round(map50, 4),
        "map50_95": round(map5095, 4),
        "precision": round(_metric_value(metrics, "metrics/precision(B)"), 4),
        "recall": round(_metric_value(metrics, "metrics/recall(B)"), 4),
        "history": history,
        "epochs_completed": len(history),
        "duration_seconds": round(time.time() - start, 2),
        "split_mode": prepared["split_mode"],
        "train_count": prepared["train_count"],
        "val_count": prepared["val_count"],
        "stopped_early": bool(should_stop and should_stop()),
        "params_million": round(params / 1e6, 3) if params else 0.0,
    }
    return {
        "best_path": best_path,
        "history": history,
        "report": report,
        "classes": prepared["classes"],
        "params": params,
    }


def evaluate_yolo_model(trained_model, dataset_dir: str | Path, work_dir: str | Path,
                        val_split: float = 0.2, seed: int = 42,
                        image_size: Optional[int] = None,
                        device: str = "cpu") -> Dict:
    """用模型关联数据集重新计算 YOLO mAP、precision 和 recall。"""
    YOLO = _require_ultralytics()
    prepared = prepare_detection_dataset(
        dataset_dir, work_dir, val_split=val_split, seed=seed
    )
    try:
        model = YOLO(str(trained_model.resolved_path))
        metrics_obj = model.val(
            data=str(prepared["data_yaml"]),
            split="val",
            imgsz=max(32, int(image_size or trained_model.input_size or 640)),
            device=_device_arg(device),
            plots=False,
            verbose=False,
        )
    except Exception as exc:  # noqa: BLE001
        raise DetectionBackendError(f"YOLO 重新评估失败：{exc}") from exc
    metrics = getattr(metrics_obj, "results_dict", {}) or {}
    map50 = _metric_value(metrics, "metrics/mAP50(B)", "metrics/mAP50")
    map5095 = _metric_value(metrics, "metrics/mAP50-95(B)", "metrics/mAP50-95")
    return {
        "accuracy": round(map50, 4),
        "best_accuracy": round(map50, 4),
        "map50": round(map50, 4),
        "map50_95": round(map5095, 4),
        "precision": round(_metric_value(metrics, "metrics/precision(B)"), 4),
        "recall": round(_metric_value(metrics, "metrics/recall(B)"), 4),
        "class_names": prepared["classes"],
        "split_mode": prepared["split_mode"],
        "train_count": prepared["train_count"],
        "val_count": prepared["val_count"],
    }


__all__ = [
    "DetectionBackendError",
    "YOLO_CONFIG_CANDIDATES",
    "evaluate_yolo_model",
    "prepare_detection_dataset",
    "train_yolo_model",
]
