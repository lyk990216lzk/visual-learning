"""数据集扫描、划分、预处理与「演示数据集」生成。"""

from __future__ import annotations

import math
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(image_size: int, augment: bool = True) -> transforms.Compose:
    """训练用（带数据增强）与验证/推理用（确定性强）两套预处理流水线。"""
    if augment:
        # 小图数据集用较温和的裁剪缩放，避免形状被裁掉
        scale = (0.7, 1.0) if image_size <= 96 else (0.6, 1.0)
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomResizedCrop(image_size, scale=scale, ratio=(0.85, 1.18)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15, fill=0),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.15),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def inference_transform(image_size: int) -> transforms.Compose:
    return build_transforms(image_size, augment=False)


def scan_image_folder(root: str | Path) -> Dict:
    """扫描 ImageFolder 结构目录，返回类别与样本统计。"""
    root = Path(root)
    result = {"root": str(root), "classes": [], "class_counts": {}, "image_count": 0,
              "has_split": False, "train_count": 0, "val_count": 0, "preview": []}
    if not root.exists():
        return result

    subdirs = [d for d in sorted(root.iterdir()) if d.is_dir()]
    names = {d.name.lower() for d in subdirs}
    result["has_split"] = {"train", "val"}.issubset(names) or {"train", "test"}.issubset(names)

    class_dirs: List[Path] = []
    if result["has_split"]:
        split_root = next(d for d in subdirs if d.name.lower() == "train")
        class_dirs = [d for d in sorted(split_root.iterdir()) if d.is_dir()]
        val_root = next((d for d in subdirs if d.name.lower() in {"val", "test"}), None)
        result["train_count"] = _count_images(split_root)
        result["val_count"] = _count_images(val_root) if val_root else 0
    else:
        class_dirs = subdirs

    for d in class_dirs:
        count = _count_images(d)
        result["classes"].append(d.name)
        result["class_counts"][d.name] = count
        if not result["has_split"]:
            result["image_count"] += count
    if result["has_split"]:
        result["image_count"] = result["train_count"] + result["val_count"]

    for d in class_dirs[:12]:
        for item in sorted(d.iterdir()):
            if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS:
                result["preview"].append({"label": d.name, "path": str(item)})
                break
    return result


def _count_images(directory: Optional[Path]) -> int:
    if directory is None or not directory.exists():
        return 0
    return sum(1 for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def build_dataloaders(dataset_dir: str | Path, image_size: int = 64, batch_size: int = 32,
                      val_split: float = 0.2, num_workers: int = 0, seed: int = 42,
                      augment: bool = True) -> Tuple[DataLoader, DataLoader, List[str]]:
    """构建训练 / 验证 DataLoader。

    目录既支持 ``root/train/<class>``、``root/val/<class>`` 的显式划分，
    也支持 ``root/<class>`` 的扁平结构（此时按 ``val_split`` 分层随机划分）。
    """
    root = Path(dataset_dir)
    if not root.exists():
        raise FileNotFoundError(f"数据集目录不存在：{root}")

    subdirs = {d.name.lower(): d for d in root.iterdir() if d.is_dir()}
    train_transform = build_transforms(image_size, augment=augment)
    val_transform = build_transforms(image_size, augment=False)

    if "train" in subdirs and ("val" in subdirs or "test" in subdirs):
        train_ds = ImageFolder(str(subdirs["train"]), transform=train_transform)
        val_dir = subdirs.get("val") or subdirs["test"]
        val_ds = ImageFolder(str(val_dir), transform=val_transform)
        if train_ds.classes != val_ds.classes:
            raise ValueError("train 与 val 目录的类别不一致")
        class_names = train_ds.classes
    else:
        full_train = ImageFolder(str(root), transform=train_transform)
        full_val = ImageFolder(str(root), transform=val_transform)
        class_names = full_train.classes
        targets = [label for _, label in full_train.samples]
        train_idx, val_idx = stratified_split(targets, val_split, seed, len(class_names))
        train_ds = Subset(full_train, train_idx)
        val_ds = Subset(full_val, val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=False, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            pin_memory=False, drop_last=False)
    return train_loader, val_loader, class_names


def stratified_split(targets: Sequence[int], val_split: float, seed: int,
                     num_classes: int) -> Tuple[List[int], List[int]]:
    """按类别分层抽样，保证验证集里每个类别都有样本。"""
    rng = random.Random(seed)
    buckets: Dict[int, List[int]] = {i: [] for i in range(num_classes)}
    for index, label in enumerate(targets):
        buckets[int(label)].append(index)
    train_idx: List[int] = []
    val_idx: List[int] = []
    for label, indices in buckets.items():
        rng.shuffle(indices)
        n_val = max(1, int(round(len(indices) * val_split))) if len(indices) > 1 else 0
        val_idx.extend(indices[:n_val])
        train_idx.extend(indices[n_val:])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


# --------------------------------------------------------------------------------------
# 演示数据集：合成几何图形，离线可用，任何模型都能快速训练到高精度
# --------------------------------------------------------------------------------------
DEMO_CLASSES = ["circle", "square", "triangle", "cross", "star", "ring"]
DEMO_CLASS_LABELS = {
    "circle": "圆形",
    "square": "正方形",
    "triangle": "三角形",
    "cross": "十字",
    "star": "五角星",
    "ring": "圆环",
}


def build_demo_dataset(root: str | Path, per_class: int = 240, image_size: int = 64,
                       classes: Optional[Sequence[str]] = None, seed: int = 2024,
                       val_ratio: float = 0.2) -> Dict:
    """生成一个 6 类几何图形数据集（train/val 已划分），用于离线演示与冒烟测试。

    图像加入了随机颜色、亮度、旋转、缩放、位移与噪声，使任务具备一定难度，
    同时保持类别可判别，便于快速对比不同经典模型的收敛速度与精度。
    """
    from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

    classes = list(classes or DEMO_CLASSES)
    root = Path(root)
    rng = random.Random(seed)
    counts = {"train": 0, "val": 0}

    def draw_shape(draw: "ImageDraw.ImageDraw", name: str, box, fill, outline, width, points):
        x0, y0, x1, y1 = box
        if name == "circle":
            draw.ellipse(box, fill=fill, outline=outline, width=width)
        elif name == "square":
            draw.rectangle(box, fill=fill, outline=outline, width=width)
        elif name == "triangle":
            draw.polygon(points, fill=fill, outline=outline)
        elif name == "cross":
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            arm = (x1 - x0) * rng.uniform(0.22, 0.32)
            thickness = (y1 - y0) * rng.uniform(0.3, 0.42)
            draw.rectangle([cx - thickness / 2, y0, cx + thickness / 2, y1], fill=fill)
            draw.rectangle([x0, cy - thickness / 2, x1, cy + thickness / 2], fill=fill)
        elif name == "star":
            draw.polygon(points, fill=fill, outline=outline)
        elif name == "ring":
            draw.ellipse(box, outline=fill, width=max(3, int((x1 - x0) * 0.16)))

    for split, ratio in (("train", 1 - val_ratio), ("val", val_ratio)):
        for name in classes:
            out_dir = root / split / name
            out_dir.mkdir(parents=True, exist_ok=True)
            n = int(round(per_class * ratio))
            for i in range(n):
                size = image_size
                bg = tuple(rng.randint(15, 235) for _ in range(3))
                img = Image.new("RGB", (size, size), bg)
                draw = ImageDraw.Draw(img)
                margin = rng.uniform(0.14, 0.3) * size
                x0, y0 = margin, margin
                x1, y1 = size - margin, size - margin
                color = tuple(rng.randint(0, 255) for _ in range(3))
                box = [x0, y0, x1, y1]
                cx, cy = size / 2, size / 2
                radius = (x1 - x0) / 2

                if name == "triangle":
                    pts = [(cx, y0), (x1, y1), (x0, y1)]
                elif name == "star":
                    pts = []
                    for k in range(10):
                        angle = -math.pi / 2 + k * math.pi / 5
                        r = radius if k % 2 == 0 else radius * 0.45
                        pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
                else:
                    pts = []
                draw_shape(draw, name, box, color, (0, 0, 0), 2, pts)

                if rng.random() < 0.6:
                    angle = rng.uniform(-25, 25)
                    img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=bg)
                if rng.random() < 0.5:
                    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.75, 1.3))
                if rng.random() < 0.4:
                    img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.4, 1.1)))

                img.save(out_dir / f"{name}_{split}_{i:04d}.png")
                counts[split] += 1

    info = scan_image_folder(root)
    info.update({"name": "演示数据集（几何图形）", "classes_label": [DEMO_CLASS_LABELS.get(c, c) for c in classes]})
    return info
