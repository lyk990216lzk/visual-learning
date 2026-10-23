"""语义分割数据加载、训练循环与指标计算。"""

from __future__ import annotations

import copy
import random
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as torch_f
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .datasets import IMAGE_EXTENSIONS, IMAGENET_MEAN, IMAGENET_STD
from .engine import TrainConfig, build_optimizer, build_scheduler, resolve_device
from .task_datasets import DatasetFormatError, scan_dataset


class SegmentationDataset(Dataset):
    """按相对路径配对图片与单通道类别掩码。"""

    def __init__(self, pairs: Sequence[Tuple[Path, Path]], image_size: int,
                 augment: bool = False):
        self.pairs = list(pairs)
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        image_path, mask_path = self.pairs[index]
        try:
            image = Image.open(image_path).convert("RGB")
            mask_image = Image.open(mask_path)
        except (OSError, UnidentifiedImageError) as exc:
            raise DatasetFormatError([f"无法读取分割样本 {image_path.name}：{exc}"]) from exc

        if mask_image.mode not in {"L", "P"}:
            raise DatasetFormatError([f"掩码 {mask_path.name} 必须是灰度或调色板 PNG"])
        mask_array = np.asarray(mask_image, dtype=np.uint8)
        mask = Image.fromarray(mask_array, mode="L")

        image = TF.resize(image, [self.image_size, self.image_size],
                          interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [self.image_size, self.image_size],
                         interpolation=InterpolationMode.NEAREST)

        if self.augment:
            if random.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if random.random() < 0.3:
                angle = random.uniform(-10.0, 10.0)
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)
            if random.random() < 0.4:
                image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
                image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))
                image = TF.adjust_saturation(image, random.uniform(0.8, 1.2))

        image_tensor = TF.normalize(TF.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD)
        mask_tensor = torch.from_numpy(np.asarray(mask, dtype=np.int64)).long()
        return image_tensor, mask_tensor


def _collect_split(root: Path, split: str) -> List[Tuple[Path, Path]]:
    image_root = root / "images" / split
    mask_root = root / "masks" / split
    if not image_root.is_dir() or not mask_root.is_dir():
        raise DatasetFormatError([f"缺少 images/{split} 或 masks/{split}"])
    masks = {
        path.relative_to(mask_root).with_suffix(""): path
        for path in mask_root.rglob("*.png") if path.is_file()
    }
    pairs: List[Tuple[Path, Path]] = []
    for image_path in sorted(
            item for item in image_root.rglob("*")
            if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS):
        key = image_path.relative_to(image_root).with_suffix("")
        mask_path = masks.get(key)
        if mask_path is None:
            raise DatasetFormatError([f"图片 {key} 缺少同名掩码"])
        pairs.append((image_path, mask_path))
    if not pairs:
        raise DatasetFormatError([f"images/{split} 中没有可用图片"])
    return pairs


def build_segmentation_dataloaders(
    dataset_dir: str | Path,
    image_size: int = 512,
    batch_size: int = 4,
    val_split: float = 0.2,
    num_workers: int = 0,
    seed: int = 42,
    augment: bool = True,
) -> Tuple[DataLoader, DataLoader, List[str], Dict]:
    """读取固定验证集，或从唯一的 train 池按 val_split 成对划分。"""
    root = Path(dataset_dir)
    info = scan_dataset(root, "segmentation")
    class_names = list(info["classes"])
    train_pairs = _collect_split(root, "train")
    split_name = next((name for name in ("val", "test") if (root / "images" / name).is_dir()),
                      None)
    if split_name:
        val_pairs = _collect_split(root, split_name)
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

    train_dataset = SegmentationDataset(train_pairs, image_size=image_size, augment=augment)
    val_dataset = SegmentationDataset(val_pairs, image_size=image_size, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=False)
    return train_loader, val_loader, class_names, {
        "split_mode": split_mode,
        "train_count": len(train_pairs),
        "val_count": len(val_pairs),
    }


def segmentation_logits(output):
    """兼容 torchvision 分割模型返回的 ``{"out": tensor}`` 结构。"""
    if isinstance(output, dict):
        output = output.get("out")
    if isinstance(output, (tuple, list)):
        output = output[0]
    if not torch.is_tensor(output):
        raise TypeError("分割模型输出中没有找到主 logits")
    return output


def _segmentation_stats(confusion: np.ndarray, class_names: Sequence[str]) -> Dict:
    total = int(confusion.sum())
    diagonal = np.diag(confusion)
    pixel_accuracy = float(diagonal.sum() / total) if total else 0.0
    per_class = []
    ious = []
    for index, name in enumerate(class_names):
        tp = float(confusion[index, index])
        fp = float(confusion[:, index].sum() - tp)
        fn = float(confusion[index, :].sum() - tp)
        union = tp + fp + fn
        iou = tp / union if union else 0.0
        ious.append(iou)
        per_class.append({
            "index": index,
            "label": name,
            "iou": round(iou, 4),
            "support": int(confusion[index, :].sum()),
        })
    return {
        "accuracy": round(pixel_accuracy, 4),
        "pixel_accuracy": round(pixel_accuracy, 4),
        "miou": round(float(np.mean(ious)) if ious else 0.0, 4),
        "per_class_iou": per_class,
        "class_names": list(class_names),
        "confusion_matrix": confusion.astype(np.int64).tolist(),
    }


@torch.no_grad()
def evaluate_segmentation(model, loader, device, class_names: Sequence[str],
                          criterion=None) -> Dict:
    model.eval()
    criterion = criterion or nn.CrossEntropyLoss(ignore_index=255)
    num_classes = len(class_names)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    total_loss, total_pixels = 0.0, 0
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = segmentation_logits(model(images))
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = torch_f.interpolate(logits, size=masks.shape[-2:], mode="bilinear",
                                         align_corners=False)
        loss = criterion(logits, masks)
        total_loss += float(loss.item()) * masks.numel()
        total_pixels += masks.numel()
        preds = logits.argmax(dim=1)
        valid = (masks >= 0) & (masks < num_classes)
        indices = masks[valid] * num_classes + preds[valid]
        confusion += torch.bincount(indices, minlength=num_classes * num_classes) \
            .detach().cpu().numpy().reshape(num_classes, num_classes)
    report = _segmentation_stats(confusion, class_names)
    report["loss"] = round(total_loss / max(1, total_pixels), 4)
    return report


def train_segmentation_model(
    model,
    train_loader,
    val_loader,
    class_names: Sequence[str],
    cfg: TrainConfig,
    on_epoch: Optional[Callable[[Dict], None]] = None,
    on_batch: Optional[Callable[[Dict], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    log: Optional[Callable[[str], None]] = None,
) -> Dict:
    log = log or (lambda message: None)
    torch.manual_seed(cfg.seed)
    device = resolve_device(cfg.device)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(
        ignore_index=255,
        label_smoothing=max(0.0, min(float(cfg.label_smoothing), 0.3)),
    )
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history: List[Dict] = []
    best_miou = 0.0
    best_state = copy.deepcopy(model.state_dict())
    stopped = False
    start_time = time.time()

    for epoch in range(1, max(1, cfg.epochs) + 1):
        model.train()
        epoch_start = time.time()
        running_loss, seen_batches = 0.0, 0
        num_batches = len(train_loader)
        limit = num_batches if cfg.max_batches_per_epoch <= 0 else min(
            num_batches, cfg.max_batches_per_epoch)
        for batch_index, (images, masks) in enumerate(train_loader, start=1):
            if batch_index > limit:
                break
            if should_stop and should_stop():
                stopped = True
                break
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = segmentation_logits(model(images))
                    if logits.shape[-2:] != masks.shape[-2:]:
                        logits = torch_f.interpolate(logits, size=masks.shape[-2:],
                                                     mode="bilinear", align_corners=False)
                    loss = criterion(logits, masks)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = segmentation_logits(model(images))
                if logits.shape[-2:] != masks.shape[-2:]:
                    logits = torch_f.interpolate(logits, size=masks.shape[-2:],
                                                 mode="bilinear", align_corners=False)
                loss = criterion(logits, masks)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            running_loss += float(loss.item())
            seen_batches += 1
            if on_batch:
                progress = ((epoch - 1) + batch_index / max(1, limit)) / max(1, cfg.epochs)
                on_batch({
                    "epoch": epoch,
                    "total_epochs": cfg.epochs,
                    "batch": batch_index,
                    "total_batches": limit,
                    "progress": round(min(99.0, progress * 100), 2),
                    "loss": round(running_loss / max(1, seen_batches), 4),
                    "elapsed": round(time.time() - start_time, 1),
                })
        if stopped:
            break

        val_report = evaluate_segmentation(model, val_loader, device, class_names, criterion)
        if scheduler:
            scheduler.step()
        record = {
            "epoch": epoch,
            "train_loss": round(running_loss / max(1, seen_batches), 4),
            "val_loss": val_report["loss"],
            "train_accuracy": None,
            "val_accuracy": val_report["accuracy"],
            "val_miou": val_report["miou"],
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": round(time.time() - epoch_start, 2),
        }
        history.append(record)
        if val_report["miou"] >= best_miou:
            best_miou = val_report["miou"]
            best_state = copy.deepcopy(model.state_dict())
        log(f"[Epoch {epoch}/{cfg.epochs}] loss={record['train_loss']:.4f} "
            f"val_loss={record['val_loss']:.4f} pixel_acc={record['val_accuracy']:.4f} "
            f"mIoU={record['val_miou']:.4f}")
        if on_epoch:
            on_epoch({
                "record": record,
                "history": history,
                "best_accuracy": round(best_miou, 4),
                "progress": round(epoch / max(1, cfg.epochs) * 100, 2),
            })

    model.load_state_dict(best_state)
    report = evaluate_segmentation(model, val_loader, device, class_names, criterion)
    report.update({
        "best_accuracy": round(best_miou, 4),
        "best_miou": round(best_miou, 4),
        "history": history,
        "epochs_completed": len(history),
        "duration_seconds": round(time.time() - start_time, 2),
        "device": str(device),
        "stopped_early": stopped,
    })
    return {
        "state_dict": best_state,
        "history": history,
        "report": report,
        "classes": list(class_names),
        "config": cfg.to_dict(),
    }


__all__ = [
    "SegmentationDataset",
    "build_segmentation_dataloaders",
    "evaluate_segmentation",
    "segmentation_logits",
    "train_segmentation_model",
]
