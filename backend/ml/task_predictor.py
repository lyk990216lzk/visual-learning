"""目标检测和语义分割推理服务。"""

from __future__ import annotations

import base64
import io
import threading
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from . import zoo
from .datasets import inference_transform
from .predictor import PredictorError
from .segmentation import segmentation_logits


class TaskModelCache:
    def __init__(self):
        self._items: Dict[int, Dict] = {}
        self._lock = threading.Lock()

    def get(self, trained_model) -> Dict:
        task = getattr(trained_model, "task_type", "classification")
        if task == "classification":
            from .predictor import model_store

            return model_store.load(trained_model)
        key = trained_model.id
        version = trained_model.updated_at.timestamp()
        with self._lock:
            cached = self._items.get(key)
            if cached and cached["version"] == version:
                return cached
        entry = self._build(trained_model, task)
        with self._lock:
            self._items[key] = entry
        return entry

    def _build(self, trained_model, task: str) -> Dict:
        path = Path(trained_model.resolved_path)
        if not path.exists():
            raise PredictorError(f"权重文件不存在：{path}")
        if task == "detection":
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise PredictorError(
                    "目标检测依赖 ultralytics，请先安装训练依赖"
                ) from exc
            try:
                model = YOLO(str(path))
            except Exception as exc:  # noqa: BLE001
                raise PredictorError(f"YOLO 权重加载失败：{exc}") from exc
            names = getattr(model, "names", {}) or {}
            classes = [
                str(names[index]) if isinstance(names, dict) else str(names[index])
                for index in range(len(names))
            ] if names else list(trained_model.classes or [])
            return {
                "task_type": task,
                "model": model,
                "classes": classes,
                "image_size": int(trained_model.input_size or 640),
                "version": trained_model.updated_at.timestamp(),
            }
        if task != "segmentation":
            raise PredictorError(f"不支持的任务类型：{task}")
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:  # noqa: BLE001
            raise PredictorError(f"分割权重加载失败：{exc}") from exc
        classes = list(checkpoint.get("classes") or trained_model.classes or [])
        architecture = checkpoint.get("architecture") or trained_model.architecture
        model = zoo.build_model(architecture, num_classes=len(classes))
        model.load_state_dict(checkpoint.get("state_dict", checkpoint))
        model.eval()
        return {
            "task_type": task,
            "model": model,
            "classes": classes,
            "image_size": int(checkpoint.get("image_size") or trained_model.input_size or 512),
            "version": trained_model.updated_at.timestamp(),
        }

    def invalidate(self, model_id: Optional[int] = None) -> None:
        with self._lock:
            if model_id is None:
                self._items.clear()
            else:
                self._items.pop(model_id, None)


task_model_cache = TaskModelCache()


def _device_arg(device: str) -> str:
    value = (device or "cpu").lower()
    return "0" if value in {"gpu", "cuda"} else value


@torch.no_grad()
def predict_detection(trained_model, image: Image.Image, confidence: float = 0.25,
                      iou: float = 0.45, max_det: int = 100,
                      device: str = "cpu") -> Dict:
    entry = task_model_cache.get(trained_model)
    start = time.perf_counter()
    try:
        results = entry["model"].predict(
            source=np.asarray(image.convert("RGB")),
            conf=max(0.01, min(float(confidence), 0.99)),
            iou=max(0.01, min(float(iou), 0.99)),
            max_det=max(1, min(int(max_det), 1000)),
            imgsz=entry["image_size"],
            device=_device_arg(device),
            verbose=False,
        )
    except Exception as exc:  # noqa: BLE001
        raise PredictorError(f"目标检测推理失败：{exc}") from exc
    elapsed_ms = (time.perf_counter() - start) * 1000
    detections = []
    if results:
        result = results[0]
        names = getattr(result, "names", {}) or {}
        boxes = getattr(result, "boxes", None)
        if boxes is not None:
            xyxy = boxes.xyxy.detach().cpu().numpy()
            scores = boxes.conf.detach().cpu().numpy()
            labels = boxes.cls.detach().cpu().numpy().astype(int)
            for box, score, class_id in zip(xyxy, scores, labels):
                detections.append({
                    "class_id": int(class_id),
                    "label": str(names.get(int(class_id), class_id)),
                    "confidence": round(float(score), 6),
                    "bbox": [round(float(value), 2) for value in box],
                })
    label = detections[0]["label"] if detections else ""
    top_confidence = max((item["confidence"] for item in detections), default=0.0)
    return {
        "task_type": "detection",
        "model_id": trained_model.id,
        "model_name": trained_model.name,
        "architecture": trained_model.architecture,
        "latency_ms": round(elapsed_ms, 2),
        "predicted_label": label,
        "confidence": top_confidence,
        "detections": detections,
        "object_count": len(detections),
    }


def _mask_png(mask: np.ndarray, classes: int) -> str:
    palette = [
        0, 0, 0, 220, 70, 90, 70, 205, 120, 80, 135, 255, 255, 180, 80,
        175, 120, 255, 50, 210, 210, 255, 90, 150,
    ]
    image = Image.fromarray(mask.astype(np.uint8), mode="P")
    image.putpalette((palette * ((classes * 3 // len(palette)) + 1))[:768])
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


@torch.no_grad()
def predict_segmentation(trained_model, image: Image.Image, device: str = "cpu") -> Dict:
    entry = task_model_cache.get(trained_model)
    transform = inference_transform(entry["image_size"])
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    model = entry["model"].to(device)
    classes = entry["classes"]
    start = time.perf_counter()
    try:
        logits = segmentation_logits(model(tensor))
    except Exception as exc:  # noqa: BLE001
        raise PredictorError(f"语义分割推理失败：{exc}") from exc
    if logits.shape[-2:] != (image.height, image.width):
        logits = F.interpolate(logits, size=(image.height, image.width), mode="bilinear",
                               align_corners=False)
    probabilities = F.softmax(logits, dim=1)[0]
    confidence_map, prediction = probabilities.max(dim=0)
    elapsed_ms = (time.perf_counter() - start) * 1000
    mask = prediction.detach().cpu().numpy().astype(np.uint8)
    flat = mask.reshape(-1)
    counts = np.bincount(flat, minlength=len(classes))
    total = max(1, int(flat.size))
    class_stats = [
        {
            "index": index,
            "label": name,
            "pixel_count": int(counts[index]),
            "percentage": round(float(counts[index]) / total, 6),
        }
        for index, name in enumerate(classes)
    ]
    foreground = [item for item in class_stats if item["index"] != 0] or class_stats
    dominant = max(foreground, key=lambda item: item["pixel_count"]) if foreground else None
    return {
        "task_type": "segmentation",
        "model_id": trained_model.id,
        "model_name": trained_model.name,
        "architecture": trained_model.architecture,
        "latency_ms": round(elapsed_ms, 2),
        "predicted_label": dominant["label"] if dominant else "",
        "confidence": round(float(confidence_map.mean().item()), 6),
        "classes": classes,
        "class_stats": class_stats,
        "mask_base64": _mask_png(mask, len(classes)),
        "width": image.width,
        "height": image.height,
    }


__all__ = [
    "task_model_cache",
    "predict_detection",
    "predict_segmentation",
]
