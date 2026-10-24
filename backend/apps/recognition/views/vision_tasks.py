"""目标检测和语义分割推理接口。"""

from __future__ import annotations

import uuid

from django.core.files.base import ContentFile
from rest_framework import status
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

from ml.predictor import PredictorError, open_image, predict_image
from ml.task_predictor import predict_detection, predict_segmentation

from ..models import PredictionRecord
from .recognize import _check_size, _get_upload
from .utils import resolve_trained_model


def _resolve_task_model(request, task_type: str):
    trained = resolve_trained_model(request.data.get("architecture"),
                                    request.data.get("trained_model"))
    if trained.task_type != task_type:
        raise ValueError(f"所选模型不是{task_type}模型")
    return trained


def predict_task_image(trained, image, options=None):
    """按训练任务功能分派到分类、检测或分割推理。"""
    options = options or {}
    if trained.task_type == "classification":
        result = predict_image(
            trained,
            image,
            top_k=int(options.get("top_k", 5) or 5),
        )
    elif trained.task_type == "detection":
        result = predict_detection(
            trained,
            image,
            confidence=float(options.get("confidence", 0.25)),
            iou=float(options.get("iou", 0.45)),
            max_det=int(options.get("max_det", 100)),
            device=options.get("device", "cpu"),
        )
    elif trained.task_type == "segmentation":
        result = predict_segmentation(
            trained,
            image,
            device=options.get("device", "cpu"),
        )
    else:
        raise ValueError(f"不支持的识别功能：{trained.task_type}")
    result.setdefault("task_type", trained.task_type)
    return result


def _save_task_record(request, upload, raw, trained, result,
                      source=PredictionRecord.Source.UPLOAD, group_id=None):
    result.setdefault("task_type", trained.task_type)
    stored_result = {key: value for key, value in result.items() if key != "mask_base64"}
    record = PredictionRecord(
        group_id=group_id or uuid.uuid4(),
        user=request.user if request.user.is_authenticated else None,
        original_name=upload.name,
        task_type=result["task_type"],
        source=source,
        architecture=result["architecture"],
        model_name=result["model_name"],
        trained_model=trained,
        predicted_label=result["predicted_label"],
        confidence=result["confidence"],
        topk=result.get("topk") or result.get("detections") or result.get("class_stats") or [],
        result_data=stored_result,
        latency_ms=result["latency_ms"],
    )
    record.image.save(f"{record.group_id.hex[:8]}_{upload.name}", ContentFile(raw), save=False)
    record.save()
    return record


class DetectionPredictView(APIView):
    """POST /api/detect/：YOLO 目标检测。"""

    parser_classes = [MultiPartParser, FormParser, JSONParser]

    def post(self, request):
        upload = _get_upload(request)
        if upload is None:
            return Response({"detail": "请上传图片"}, status=status.HTTP_400_BAD_REQUEST)
        message = _check_size(upload)
        if message:
            return Response({"detail": message}, status=status.HTTP_400_BAD_REQUEST)
        try:
            trained = _resolve_task_model(request, "detection")
            raw = upload.read()
            result = predict_task_image(trained, open_image(raw), request.data)
        except (PredictorError, ValueError) as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        record = _save_task_record(request, upload, raw, trained, result)
        result["record_id"] = record.id
        result["training_run_id"] = trained.training_run_id
        result["recognition_type"] = "single"
        result["image_url"] = request.build_absolute_uri(record.image.url) if record.image else ""
        return Response(result)


class SegmentationPredictView(APIView):
    """POST /api/segment/：语义分割。"""

    parser_classes = [MultiPartParser, FormParser, JSONParser]

    def post(self, request):
        upload = _get_upload(request)
        if upload is None:
            return Response({"detail": "请上传图片"}, status=status.HTTP_400_BAD_REQUEST)
        message = _check_size(upload)
        if message:
            return Response({"detail": message}, status=status.HTTP_400_BAD_REQUEST)
        try:
            trained = _resolve_task_model(request, "segmentation")
            raw = upload.read()
            result = predict_task_image(trained, open_image(raw), request.data)
        except (PredictorError, ValueError) as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        record = _save_task_record(request, upload, raw, trained, result)
        result["record_id"] = record.id
        result["training_run_id"] = trained.training_run_id
        result["recognition_type"] = "single"
        result["image_url"] = request.build_absolute_uri(record.image.url) if record.image else ""
        return Response(result)


__all__ = ["DetectionPredictView", "SegmentationPredictView", "predict_task_image"]
