"""目标检测和语义分割推理接口。"""

from __future__ import annotations

import uuid

from django.core.files.base import ContentFile
from rest_framework import status
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

from ml.predictor import PredictorError, open_image
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


def _save_task_record(request, upload, raw, trained, result):
    stored_result = {key: value for key, value in result.items() if key != "mask_base64"}
    record = PredictionRecord(
        group_id=uuid.uuid4(),
        user=request.user if request.user.is_authenticated else None,
        original_name=upload.name,
        task_type=result["task_type"],
        source=PredictionRecord.Source.UPLOAD,
        architecture=result["architecture"],
        model_name=result["model_name"],
        trained_model=trained,
        predicted_label=result["predicted_label"],
        confidence=result["confidence"],
        topk=result.get("detections") or result.get("class_stats") or [],
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
            result = predict_detection(
                trained,
                open_image(raw),
                confidence=float(request.data.get("confidence", 0.25)),
                iou=float(request.data.get("iou", 0.45)),
                max_det=int(request.data.get("max_det", 100)),
                device=request.data.get("device", "cpu"),
            )
        except (PredictorError, ValueError) as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        record = _save_task_record(request, upload, raw, trained, result)
        result["record_id"] = record.id
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
            result = predict_segmentation(
                trained,
                open_image(raw),
                device=request.data.get("device", "cpu"),
            )
        except (PredictorError, ValueError) as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        record = _save_task_record(request, upload, raw, trained, result)
        result["record_id"] = record.id
        return Response(result)


__all__ = ["DetectionPredictView", "SegmentationPredictView"]
