"""按视觉任务校验并解析数据集目录，供上传与统计刷新接口复用。"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from PIL import Image, UnidentifiedImageError

from .datasets import IMAGE_EXTENSIONS, scan_image_folder

TASK_TYPES = ("classification", "detection", "segmentation")
TASK_LABELS = {
    "classification": "图片识别",
    "detection": "目标检测",
    "segmentation": "语义分割",
}
MAX_ERRORS = 30

FORMAT_HINTS = {
    "classification": (
        "仅提供 <类别>/*.jpg 时训练阶段按 val_split 自动划分；"
        "提供 train/val/<类别>/*.jpg 时使用固定验证集。至少需要 2 个类别和 4 张图片。"
    ),
    "detection": (
        "固定验证集：data.yaml + images/train|val + labels/train|val。"
        "自动验证集：只提供 images/train + labels/train，训练时按 val_split 成对划分；"
        "图片与同名 .txt 标注必须一一对应。"
    ),
    "segmentation": (
        "固定验证集：images/train|val + masks/train|val + classes.json。"
        "自动验证集：只提供 images/train + masks/train + classes.json，"
        "训练时按 val_split 成对划分；图片与同名 .png 掩码必须一一对应且尺寸一致。"
    ),
}


class DatasetFormatError(ValueError):
    """数据集目录不符合指定训练任务要求。"""

    def __init__(self, errors: Sequence[str], hint: Optional[str] = None):
        self.errors = list(dict.fromkeys(str(item) for item in errors if item))
        self.hint = hint or ""
        super().__init__("；".join(self.errors) or "数据集格式不符合要求")


def _ids_text(ids: Iterable[str], limit: int = 8) -> str:
    values = sorted(set(ids))
    shown = "、".join(values[:limit])
    return shown + (" 等" if len(values) > limit else "")


def _trim_errors(errors: List[str]) -> None:
    if len(errors) == MAX_ERRORS:
        errors.append(f"错误较多，仅显示前 {MAX_ERRORS} 项。")


def _image_files(directory: Path) -> List[Path]:
    if not directory.exists():
        return []
    return sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _split_dir(root: Path, names: Sequence[str]) -> Optional[Path]:
    return next((root / name for name in names if (root / name).is_dir()), None)


def _split_entry(root: Path, names: Sequence[str]) -> tuple[Optional[str], Optional[Path]]:
    for name in names:
        candidate = root / name
        if candidate.is_dir():
            return name, candidate
    return None, None


def _scan_classification(root: Path) -> Dict:
    info = scan_image_folder(root)
    errors: List[str] = []
    if not info["classes"]:
        errors.append("没有找到类别子目录")
    elif len(info["classes"]) < 2:
        errors.append("图片识别至少需要 2 个类别")
    empty = [name for name, count in info["class_counts"].items() if count == 0]
    if empty:
        errors.append(f"以下类别没有可用图片：{_ids_text(empty)}")
    if info["image_count"] < 4:
        errors.append(f"图片总数不足，当前 {info['image_count']} 张，至少需要 4 张")

    train_dir = root / "train"
    val_dir = _split_dir(root, ("val", "test"))
    if train_dir.is_dir() and val_dir is not None:
        train_classes = {item.name for item in train_dir.iterdir() if item.is_dir()}
        val_classes = {item.name for item in val_dir.iterdir() if item.is_dir()}
        missing_val = train_classes - val_classes
        extra_val = val_classes - train_classes
        if missing_val:
            errors.append(f"验证集缺少类别目录：{_ids_text(missing_val)}")
        if extra_val:
            errors.append(f"验证集包含训练集不存在的类别：{_ids_text(extra_val)}")
        if info["val_count"] == 0:
            errors.append("验证集没有可用图片")
    elif train_dir.is_dir() and val_dir is None:
        errors.append("检测到 train 目录，但缺少 val 或 test 目录")

    if errors:
        raise DatasetFormatError(errors, FORMAT_HINTS["classification"])
    info.update({
        "task_type": "classification",
        "annotation_format": "image_folder",
        "split_mode": "explicit" if info["has_split"] else "auto",
    })
    return info


def _strip_value(value: str) -> str:
    value = value.split(" #", 1)[0].strip().rstrip(",")
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1]
    return value.strip()


def _parse_inline_names(value: str) -> List[str]:
    value = value.strip()
    if not (value.startswith("[") and value.endswith("]")):
        return []
    body = value[1:-1].strip()
    if not body:
        return []
    return [_strip_value(item) for item in body.split(",") if _strip_value(item)]


def _read_yolo_class_names(root: Path) -> List[str]:
    data_file = next(
        (candidate for candidate in (root / "data.yaml", root / "data.yml") if candidate.is_file()),
        None,
    )
    if data_file is None:
        raise DatasetFormatError(
            ["数据集根目录缺少 data.yaml 或 data.yml"],
            FORMAT_HINTS["detection"],
        )

    lines = data_file.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    inline_names: List[str] = []
    indexed_names: Dict[int, str] = {}
    list_names: List[str] = []
    in_names = False
    names_indent = -1

    for raw in lines:
        content = raw.split("#", 1)[0].rstrip()
        if not content.strip():
            continue
        indent = len(content) - len(content.lstrip())

        if in_names and indent <= names_indent:
            in_names = False

        match = re.match(r"^names\s*:\s*(.*)$", content)
        if match:
            value = _strip_value(match.group(1))
            inline_names = _parse_inline_names(value)
            in_names = not inline_names
            names_indent = indent
            continue

        if not in_names:
            continue

        indexed = re.match(r"^\s*(\d+)\s*:\s*(.+?)\s*$", content)
        if indexed:
            indexed_names[int(indexed.group(1))] = _strip_value(indexed.group(2))
            continue
        listed = re.match(r"^\s*-\s*(.+?)\s*$", content)
        if listed:
            list_names.append(_strip_value(listed.group(1)))

    if inline_names:
        return inline_names
    if indexed_names:
        expected = list(range(len(indexed_names)))
        actual = sorted(indexed_names)
        if actual != expected:
            raise DatasetFormatError(
                [f"data.yaml 的类别编号必须从 0 连续编号，当前为 {actual}"],
                FORMAT_HINTS["detection"],
            )
        return [indexed_names[index] for index in expected]
    if list_names:
        return list_names
    raise DatasetFormatError(
        ["data.yaml 中缺少可解析的 names 类别列表"],
        FORMAT_HINTS["detection"],
    )


def _validate_yolo_label(
    label_file: Path,
    class_count: int,
    errors: List[str],
    instance_counts: Dict[str, int],
    class_names: Sequence[str],
) -> None:
    try:
        lines = label_file.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        errors.append(f"无法读取标注 {label_file.name}：{exc}")
        return

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{label_file.name} 第 {line_number} 行不是 5 列 YOLO 标注")
            _trim_errors(errors)
            if len(errors) >= MAX_ERRORS:
                return
            continue
        try:
            class_id = int(parts[0])
            x, y, width, height = (float(item) for item in parts[1:])
        except ValueError:
            errors.append(f"{label_file.name} 第 {line_number} 行包含非数字字段")
            _trim_errors(errors)
            if len(errors) >= MAX_ERRORS:
                return
            continue
        if not 0 <= class_id < class_count:
            errors.append(
                f"{label_file.name} 第 {line_number} 行类别编号 {class_id} "
                f"超出范围 0~{class_count - 1}"
            )
        elif not all(0 <= value <= 1 for value in (x, y, width, height)):
            errors.append(f"{label_file.name} 第 {line_number} 行坐标必须归一化到 0~1")
        elif width <= 0 or height <= 0:
            errors.append(f"{label_file.name} 第 {line_number} 行宽高必须大于 0")
        elif x - width / 2 < -1e-6 or x + width / 2 > 1 + 1e-6 or \
                y - height / 2 < -1e-6 or y + height / 2 > 1 + 1e-6:
            errors.append(f"{label_file.name} 第 {line_number} 行的边界框超出图片范围")
        elif 0 <= class_id < class_count:
            class_name = class_names[class_id]
            instance_counts[class_name] = instance_counts.get(class_name, 0) + 1
        if len(errors) >= MAX_ERRORS:
            return


def _scan_detection(root: Path) -> Dict:
    errors: List[str] = []
    images_root = root / "images"
    labels_root = root / "labels"
    if not images_root.is_dir():
        errors.append("缺少 images 目录")
    if not labels_root.is_dir():
        errors.append("缺少 labels 目录")
    if errors:
        raise DatasetFormatError(errors, FORMAT_HINTS["detection"])

    class_names = _read_yolo_class_names(root)
    class_count = len(class_names)
    train_images_dir = images_root / "train"
    val_images_name, val_images_dir = _split_entry(images_root, ("val", "test"))
    train_labels_dir = labels_root / "train"
    val_labels_name, val_labels_dir = _split_entry(labels_root, ("val", "test"))
    for label, path in (("images/train", train_images_dir), ("labels/train", train_labels_dir)):
        if path is None or not path.is_dir():
            errors.append(f"缺少 {label} 目录")
    if (val_images_dir is None) != (val_labels_dir is None):
        errors.append("images 与 labels 的验证集目录必须同时存在或同时省略")
    elif val_images_dir is not None and val_images_name != val_labels_name:
        errors.append(
            f"images 使用 {val_images_name}，但 labels 使用 {val_labels_name}，验证集名称必须一致"
        )

    if errors:
        raise DatasetFormatError(errors, FORMAT_HINTS["detection"])

    has_split = val_images_dir is not None and val_labels_dir is not None
    instance_counts: Dict[str, int] = {}
    split_counts: Dict[str, int] = {}
    preview: List[Dict] = []
    splits = [("train", train_images_dir, train_labels_dir)]
    if has_split:
        splits.append(("val", val_images_dir, val_labels_dir))
    for split, image_dir, label_dir in splits:
        images = _image_files(image_dir)
        if not images:
            errors.append(f"{image_dir.relative_to(root)} 中没有图片")
            continue
        split_counts[split] = len(images)
        expected_labels = {
            image.relative_to(image_dir).with_suffix(".txt"): image for image in images
        }
        actual_labels = {
            path.relative_to(label_dir): path
            for path in label_dir.rglob("*.txt") if path.is_file()
        }
        missing = expected_labels.keys() - actual_labels.keys()
        extra = actual_labels.keys() - expected_labels.keys()
        if missing:
            errors.append(
                f"labels/{split} 缺少对应标注：{_ids_text(str(item) for item in missing)}"
            )
        if extra:
            errors.append(
                f"labels/{split} 存在无对应图片的标注：{_ids_text(str(item) for item in extra)}"
            )
        for relative, image in sorted(expected_labels.items()):
            label_file = label_dir / relative
            if label_file.exists():
                _validate_yolo_label(label_file, class_count, errors, instance_counts, class_names)
            if len(preview) < 12 and split == "train":
                preview.append({"label": "train", "path": str(image)})
            if len(errors) >= MAX_ERRORS:
                break

    if errors:
        raise DatasetFormatError(errors, FORMAT_HINTS["detection"])
    train_count = split_counts.get("train", 0)
    val_count = split_counts.get("val", 0)
    return {
        "root": str(root),
        "task_type": "detection",
        "annotation_format": "yolo",
        "classes": class_names,
        "class_counts": {name: instance_counts.get(name, 0) for name in class_names},
        "image_count": train_count + val_count,
        "has_split": has_split,
        "split_mode": "explicit" if has_split else "auto",
        "train_count": train_count,
        "val_count": val_count,
        "preview": preview,
    }


def _segmentation_class_names(root: Path) -> List[str]:
    json_file = root / "classes.json"
    text_file = root / "classes.txt"
    if json_file.is_file():
        try:
            data = json.loads(json_file.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DatasetFormatError(
                [f"classes.json 无法解析：{exc}"],
                FORMAT_HINTS["segmentation"],
            )
        if isinstance(data, list) and all(isinstance(item, str) for item in data):
            names = data
        elif isinstance(data, dict) and isinstance(data.get("classes"), list):
            names = data["classes"]
        elif isinstance(data, dict) and all(isinstance(value, int) for value in data.values()):
            ids = sorted(data.values())
            if ids != list(range(len(ids))):
                raise DatasetFormatError(
                    [f"classes.json 的类别编号必须从 0 连续编号，当前为 {ids}"],
                    FORMAT_HINTS["segmentation"],
                )
            names = [key for key, value in sorted(data.items(), key=lambda item: item[1])]
        else:
            raise DatasetFormatError(
                ["classes.json 必须为类别名数组、classes 数组或 {类别名: 编号} 对象"],
                FORMAT_HINTS["segmentation"],
            )
    elif text_file.is_file():
        names = [
            line.strip() for line in text_file.read_text(
                encoding="utf-8-sig", errors="replace"
            ).splitlines() if line.strip()
        ]
    else:
        raise DatasetFormatError(
            ["数据集根目录缺少 classes.json 或 classes.txt"],
            FORMAT_HINTS["segmentation"],
        )
    if not isinstance(names, list) or not all(isinstance(item, str) and item.strip() for item in names):
        raise DatasetFormatError(
            ["类别列表必须全部是非空字符串"],
            FORMAT_HINTS["segmentation"],
        )
    if len(names) < 2:
        raise DatasetFormatError(
            ["语义分割至少需要背景和 1 个前景类别"],
            FORMAT_HINTS["segmentation"],
        )
    return [item.strip() for item in names]


def _mask_class_counts(mask_file: Path, class_count: int, errors: List[str]) -> Dict[int, int]:
    try:
        with Image.open(mask_file) as image:
            if image.mode not in {"L", "P"}:
                errors.append(f"掩码 {mask_file.name} 必须是单通道灰度或调色板 PNG")
                return {}
            histogram = image.histogram()[:256]
    except (OSError, UnidentifiedImageError):
        errors.append(f"掩码 {mask_file.name} 无法读取")
        return {}
    invalid = sum(count for index, count in enumerate(histogram) if index >= class_count and count)
    if invalid:
        errors.append(
            f"掩码 {mask_file.name} 包含超出类别范围 0~{class_count - 1} 的像素值"
        )
    return {
        index: count for index, count in enumerate(histogram)
        if index < class_count and count
    }


def _scan_segmentation(root: Path) -> Dict:
    errors: List[str] = []
    images_root = root / "images"
    masks_root = root / "masks"
    if not images_root.is_dir():
        errors.append("缺少 images 目录")
    if not masks_root.is_dir():
        errors.append("缺少 masks 目录")
    if errors:
        raise DatasetFormatError(errors, FORMAT_HINTS["segmentation"])

    class_names = _segmentation_class_names(root)
    train_images_dir = images_root / "train"
    val_images_name, val_images_dir = _split_entry(images_root, ("val", "test"))
    train_masks_dir = masks_root / "train"
    val_masks_name, val_masks_dir = _split_entry(masks_root, ("val", "test"))
    for label, path in (("images/train", train_images_dir), ("masks/train", train_masks_dir)):
        if path is None or not path.is_dir():
            errors.append(f"缺少 {label} 目录")
    if (val_images_dir is None) != (val_masks_dir is None):
        errors.append("images 与 masks 的验证集目录必须同时存在或同时省略")
    elif val_images_dir is not None and val_images_name != val_masks_name:
        errors.append(
            f"images 使用 {val_images_name}，但 masks 使用 {val_masks_name}，验证集名称必须一致"
        )
    if errors:
        raise DatasetFormatError(errors, FORMAT_HINTS["segmentation"])

    has_split = val_images_dir is not None and val_masks_dir is not None
    pixel_counts = {name: 0 for name in class_names}
    split_counts: Dict[str, int] = {}
    preview: List[Dict] = []
    splits = [("train", train_images_dir, train_masks_dir)]
    if has_split:
        splits.append(("val", val_images_dir, val_masks_dir))
    for split, image_dir, mask_dir in splits:
        images = _image_files(image_dir)
        if not images:
            errors.append(f"images/{split} 中没有图片")
            continue
        split_counts[split] = len(images)
        masks = {
            path.relative_to(mask_dir).with_suffix(""): path
            for path in mask_dir.rglob("*.png") if path.is_file()
        }
        non_png = [
            path.name for path in mask_dir.rglob("*")
            if path.is_file() and path.suffix.lower() != ".png"
        ]
        if non_png:
            errors.append(f"masks/{split} 只允许 PNG 掩码：{_ids_text(non_png)}")
        image_keys = {
            image.relative_to(image_dir).with_suffix(""): image for image in images
        }
        missing = image_keys.keys() - masks.keys()
        extra = masks.keys() - image_keys.keys()
        if missing:
            errors.append(f"masks/{split} 缺少对应掩码：{_ids_text(str(item) for item in missing)}")
        if extra:
            errors.append(f"masks/{split} 存在无对应图片的掩码：{_ids_text(str(item) for item in extra)}")

        for key, image_file in sorted(image_keys.items()):
            mask_file = masks.get(key)
            if mask_file is None:
                continue
            try:
                with Image.open(image_file) as image:
                    image_size = image.size
            except (OSError, UnidentifiedImageError):
                errors.append(f"图片 {image_file.name} 无法读取")
                continue
            try:
                with Image.open(mask_file) as mask_file_image:
                    if mask_file_image.size != image_size:
                        errors.append(
                            f"{image_file.name} 与 {mask_file.name} 尺寸不一致："
                            f"{image_size[0]}×{image_size[1]} / "
                            f"{mask_file_image.size[0]}×{mask_file_image.size[1]}"
                        )
                histogram = _mask_class_counts(mask_file, len(class_names), errors)
                for index, count in histogram.items():
                    pixel_counts[class_names[index]] += count
            except (OSError, UnidentifiedImageError):
                errors.append(f"掩码 {mask_file.name} 无法读取")
            if len(preview) < 12 and split == "train":
                preview.append({"label": "train", "path": str(image_file)})
            if len(errors) >= MAX_ERRORS:
                break

    if errors:
        raise DatasetFormatError(errors, FORMAT_HINTS["segmentation"])
    train_count = split_counts.get("train", 0)
    val_count = split_counts.get("val", 0)
    return {
        "root": str(root),
        "task_type": "segmentation",
        "annotation_format": "png_masks",
        "classes": class_names,
        "class_counts": pixel_counts,
        "image_count": train_count + val_count,
        "has_split": has_split,
        "split_mode": "explicit" if has_split else "auto",
        "train_count": train_count,
        "val_count": val_count,
        "preview": preview,
    }


def scan_dataset(root: str | Path, task_type: str = "classification") -> Dict:
    """按任务类型校验目录并返回可入库的统计信息。"""
    root = Path(root)
    task_type = (task_type or "classification").lower()
    if task_type not in TASK_TYPES:
        raise DatasetFormatError(
            [f"未知的数据集功能类型：{task_type}"],
            "功能必须为图片识别、目标检测或语义分割。",
        )
    if not root.exists() or not root.is_dir():
        raise DatasetFormatError([f"数据集目录不存在：{root}"], FORMAT_HINTS[task_type])
    if task_type == "detection":
        return _scan_detection(root)
    if task_type == "segmentation":
        return _scan_segmentation(root)
    return _scan_classification(root)


__all__ = [
    "DatasetFormatError",
    "FORMAT_HINTS",
    "TASK_LABELS",
    "TASK_TYPES",
    "scan_dataset",
]
