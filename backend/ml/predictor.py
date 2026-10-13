"""推理服务：权重加载缓存、单图预测与多模型对比预测。"""

from __future__ import annotations

import io
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError

from . import zoo
from .datasets import inference_transform


class PredictorError(Exception):
    """推理失败（权重缺失、图片损坏等），由接口层转换为友好的错误响应。"""


class ModelStore:
    """已训练模型的进程内缓存，避免每次请求都重新读盘、重建网络。"""

    def __init__(self, capacity: int = 8):
        self._cache: Dict[int, Dict] = {}
        self._lock = threading.Lock()
        self._capacity = capacity

    # ---------------- 加载与淘汰 ----------------
    def load(self, trained_model) -> Dict:
        key = trained_model.id
        with self._lock:
            cached = self._cache.get(key)
            if cached and cached["version"] == trained_model.updated_at.timestamp():
                cached["last_used"] = time.time()
                return cached

        entry = self._build(trained_model)
        with self._lock:
            self._cache[key] = entry
            if len(self._cache) > self._capacity:
                oldest = min(self._cache.items(), key=lambda item: item[1]["last_used"])[0]
                self._cache.pop(oldest, None)
        return entry

    def _build(self, trained_model) -> Dict:
        # Django 层的 TrainedModel 提供 resolved_path（把相对路径解析成绝对路径）；
        # ml 包本身不依赖 Django，所以这里用 getattr 兜底，路径为绝对路径时行为不变。
        resolved = getattr(trained_model, "resolved_path", None)
        path = Path(resolved) if resolved else Path(trained_model.file_path)
        if not path.exists():
            raise PredictorError(f"权重文件不存在：{path}")
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:  # noqa: BLE001
            raise PredictorError(f"权重文件加载失败：{exc}") from exc

        classes: List[str] = list(checkpoint.get("classes") or trained_model.classes or [])
        arch = checkpoint.get("architecture") or trained_model.architecture
        image_size = int(checkpoint.get("image_size") or trained_model.input_size or 64)
        if not classes:
            raise PredictorError("权重文件中缺少类别信息，无法推理")

        model = zoo.build_model(arch, num_classes=len(classes))
        state = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state)
        model.eval()
        return {
            "model": model,
            "classes": classes,
            "architecture": arch,
            "image_size": image_size,
            "version": trained_model.updated_at.timestamp(),
            "last_used": time.time(),
            "params_million": round(zoo.count_parameters(model, trainable_only=False) / 1e6, 3),
        }

    def invalidate(self, model_id: Optional[int] = None) -> None:
        with self._lock:
            if model_id is None:
                self._cache.clear()
            else:
                self._cache.pop(model_id, None)

    def loaded_ids(self) -> List[int]:
        with self._lock:
            return sorted(self._cache.keys())


model_store = ModelStore()


def open_image(source) -> Image.Image:
    """把上传的文件/字节流安全地读成 RGB 图片。"""
    try:
        if hasattr(source, "read"):
            data = source.read()
            if not data:
                raise PredictorError("上传的图片为空")
            image = Image.open(io.BytesIO(data))
        elif isinstance(source, (bytes, bytearray)):
            image = Image.open(io.BytesIO(bytes(source)))
        else:
            image = Image.open(source)
        return image.convert("RGB")
    except UnidentifiedImageError as exc:
        raise PredictorError("无法识别的图片格式，请上传 JPG / PNG / BMP 等常见格式") from exc
    except PredictorError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise PredictorError(f"图片读取失败：{exc}") from exc


@torch.no_grad()
def predict_image(trained_model, image: Image.Image, top_k: int = 5) -> Dict:
    """单模型推理，返回 Top-K 概率、预测标签与耗时。"""
    entry = model_store.load(trained_model)
    transform = inference_transform(entry["image_size"])
    tensor = transform(image).unsqueeze(0)

    start = time.perf_counter()
    logits = entry["model"](tensor)
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    probabilities = F.softmax(logits, dim=1)[0]
    elapsed_ms = (time.perf_counter() - start) * 1000

    k = max(1, min(int(top_k or 5), len(entry["classes"])))
    values, indices = torch.topk(probabilities, k)
    results = [
        {"index": int(idx), "label": entry["classes"][int(idx)], "probability": round(float(value), 6)}
        for value, idx in zip(values.tolist(), indices.tolist())
    ]
    return {
        "architecture": entry["architecture"],
        "model_id": trained_model.id,
        "model_name": trained_model.name,
        "image_size": entry["image_size"],
        "latency_ms": round(elapsed_ms, 2),
        "predicted_label": results[0]["label"],
        "confidence": results[0]["probability"],
        "topk": results,
        "all_probabilities": {
            label: round(float(p), 6) for label, p in zip(entry["classes"], probabilities.tolist())
        },
    }


def predict_compare(model_map: Dict[str, object], image: Image.Image, top_k: int = 3) -> Dict:
    """同一张图片跑多个模型，用于对比不同经典模型的预测差异。

    ``model_map``: {架构 key: TrainedModel 实例}
    """
    outputs: List[Dict] = []
    errors: List[Dict] = []
    for arch, trained_model in model_map.items():
        try:
            outputs.append(predict_image(trained_model, image, top_k=top_k))
        except PredictorError as exc:
            errors.append({"architecture": arch, "error": str(exc)})
    if not outputs:
        raise PredictorError("没有可用于对比的模型，请先完成至少一个模型的训练")

    labels = [item["predicted_label"] for item in outputs]
    consensus = max(set(labels), key=labels.count)
    return {
        "results": outputs,
        "errors": errors,
        "consensus_label": consensus,
        "agreement": round(labels.count(consensus) / len(labels), 4),
        "latency_ms": round(sum(item["latency_ms"] for item in outputs), 2),
    }
