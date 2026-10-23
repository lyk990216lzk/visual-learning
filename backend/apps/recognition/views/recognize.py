"""图像识别接口：单图识别、多模型对比、批量识别、样例图片生成。"""

from __future__ import annotations

import io
import uuid

from django.conf import settings
from django.core.files.base import ContentFile
from django.http import HttpResponse
from rest_framework import status
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

from ml.predictor import PredictorError, open_image, predict_compare, predict_image

from ..models import PredictionRecord, TrainedModel
from ..serializers import PredictionRecordSerializer
from .utils import parse_architecture_list, resolve_trained_model


def _get_upload(request):
    return request.FILES.get("image") or request.FILES.get("file")


def _check_size(upload) -> str | None:
    if upload is not None and upload.size > settings.MAX_IMAGE_UPLOAD_SIZE:
        return f"图片超过 {settings.MAX_IMAGE_UPLOAD_SIZE // 1024 // 1024}MB 限制"
    return None


def _save_record(*, group_id, upload, image_bytes, source, trained, result, user=None):
    record = PredictionRecord(
        group_id=group_id,
        user=user,
        original_name=getattr(upload, "name", "") or "sample.png",
        task_type="classification",
        source=source,
        architecture=result["architecture"],
        model_name=result.get("model_name", ""),
        trained_model=trained,
        predicted_label=result["predicted_label"],
        confidence=result["confidence"],
        topk=result["topk"],
        latency_ms=result["latency_ms"],
    )
    if image_bytes:
        record.image.save(f"{group_id.hex[:8]}_{getattr(upload, 'name', 'image.png')}",
                          ContentFile(image_bytes), save=False)
    record.save()
    return record


class RecognizeView(APIView):
    """POST /api/recognize/ ：单张图片 + 指定模型 -> Top-K 识别结果。"""

    parser_classes = [MultiPartParser, FormParser, JSONParser]

    def post(self, request):
        upload = _get_upload(request)
        if upload is None:
            return Response({"detail": "请上传图片（字段名 image）"}, status=status.HTTP_400_BAD_REQUEST)
        message = _check_size(upload)
        if message:
            return Response({"detail": message}, status=status.HTTP_400_BAD_REQUEST)

        top_k = int(request.data.get("top_k", 5) or 5)
        try:
            trained = resolve_trained_model(request.data.get("architecture"),
                                            request.data.get("trained_model"))
            raw = upload.read()
            image = open_image(raw)
            result = predict_image(trained, image, top_k=top_k)
        except PredictorError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)

        record = _save_record(group_id=uuid.uuid4(), upload=upload, image_bytes=raw,
                              source=PredictionRecord.Source.UPLOAD, trained=trained, result=result,
                              user=request.user if request.user.is_authenticated else None)
        payload = dict(result)
        payload["record_id"] = record.id
        payload["group_id"] = str(record.group_id)
        payload["image_url"] = request.build_absolute_uri(record.image.url) if record.image else ""
        payload["source"] = record.source
        return Response(payload)


class CompareRecognizeView(APIView):
    """POST /api/recognize/compare/ ：一张图片同时跑多个经典模型，对比预测结果。"""

    parser_classes = [MultiPartParser, FormParser, JSONParser]

    def post(self, request):
        upload = _get_upload(request)
        if upload is None:
            return Response({"detail": "请上传图片（字段名 image）"}, status=status.HTTP_400_BAD_REQUEST)
        message = _check_size(upload)
        if message:
            return Response({"detail": message}, status=status.HTTP_400_BAD_REQUEST)

        architectures = parse_architecture_list(request.data.get("architectures"))
        if not architectures:
            architectures = list(TrainedModel.objects.filter(is_active=True)
                                 .values_list("architecture", flat=True))
        if not architectures:
            return Response({"detail": "请先训练至少一个模型，再进行多模型对比"},
                            status=status.HTTP_400_BAD_REQUEST)

        top_k = int(request.data.get("top_k", 3) or 3)
        model_map = {}
        unavailable = []
        for arch in architectures:
            try:
                model_map[arch] = resolve_trained_model(arch, None)
            except Exception:  # noqa: BLE001 —— 没有训练的模型时跳过，不影响其它模型对比
                unavailable.append(arch)

        if not model_map:
            return Response({"detail": "所选模型都还没有训练好的权重，请先完成训练",
                             "unavailable": unavailable}, status=status.HTTP_400_BAD_REQUEST)

        try:
            raw = upload.read()
            image = open_image(raw)
            result = predict_compare(model_map, image, top_k=top_k)
        except PredictorError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)

        group_id = uuid.uuid4()
        records = []
        for item in result["results"]:
            trained = model_map[item["architecture"]]
            record = _save_record(group_id=group_id, upload=upload, image_bytes=raw,
                                  source=PredictionRecord.Source.COMPARE, trained=trained, result=item,
                                  user=request.user if request.user.is_authenticated else None)
            records.append(record.id)
            item["record_id"] = record.id
            item["model_name"] = trained.name
            from ml.registry import get_architecture

            meta = get_architecture(item["architecture"])
            item["architecture_name"] = meta["name"] if meta else item["architecture"]

        first_image = PredictionRecord.objects.filter(pk=records[0]).first()
        result.update({
            "group_id": str(group_id),
            "record_ids": records,
            "image_url": request.build_absolute_uri(first_image.image.url)
            if first_image and first_image.image else "",
            "unavailable": unavailable,
        })
        return Response(result)


class BatchRecognizeView(APIView):
    """POST /api/recognize/batch/ ：一次上传多张图片，使用同一个模型批量识别。"""

    parser_classes = [MultiPartParser, FormParser, JSONParser]

    def post(self, request):
        uploads = request.FILES.getlist("images") or request.FILES.getlist("files")
        if not uploads:
            single = _get_upload(request)
            uploads = [single] if single else []
        if not uploads:
            return Response({"detail": "请上传至少一张图片（字段名 images）"},
                            status=status.HTTP_400_BAD_REQUEST)
        if len(uploads) > 20:
            return Response({"detail": "单次批量识别最多 20 张图片"}, status=status.HTTP_400_BAD_REQUEST)

        top_k = int(request.data.get("top_k", 3) or 3)
        try:
            trained = resolve_trained_model(request.data.get("architecture"),
                                            request.data.get("trained_model"))
        except Exception as exc:  # noqa: BLE001
            detail = getattr(exc, "detail", str(exc))
            return Response({"detail": detail}, status=status.HTTP_400_BAD_REQUEST)

        group_id = uuid.uuid4()
        results, failed = [], []
        for upload in uploads:
            message = _check_size(upload)
            if message:
                failed.append({"name": upload.name, "error": message})
                continue
            try:
                raw = upload.read()
                image = open_image(raw)
                result = predict_image(trained, image, top_k=top_k)
            except PredictorError as exc:
                failed.append({"name": upload.name, "error": str(exc)})
                continue
            record = _save_record(group_id=group_id, upload=upload, image_bytes=raw,
                                  source=PredictionRecord.Source.BATCH, trained=trained, result=result,
                                  user=request.user if request.user.is_authenticated else None)
            results.append({
                "record_id": record.id,
                "name": upload.name,
                "predicted_label": result["predicted_label"],
                "confidence": result["confidence"],
                "topk": result["topk"],
                "latency_ms": result["latency_ms"],
                "image_url": request.build_absolute_uri(record.image.url) if record.image else "",
            })

        return Response({
            "group_id": str(group_id),
            "model_id": trained.id,
            "model_name": trained.name,
            "total": len(results),
            "results": results,
            "failed": failed,
        })


class SampleImageView(APIView):
    """GET /api/recognize/sample/?label=circle ：生成一张合成样例图，方便没有图片时快速体验。"""

    def get(self, request):
        import random
        import tempfile
        from pathlib import Path

        from PIL import Image

        from ml.datasets import DEMO_CLASSES, build_demo_dataset

        label = request.query_params.get("label") or DEMO_CLASSES[0]
        size = max(32, min(int(request.query_params.get("size", 64)), 224))
        if label not in DEMO_CLASSES:
            return Response({"detail": f"不支持的样例类别：{label}"},
                            status=status.HTTP_400_BAD_REQUEST)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_demo_dataset(root, per_class=5, image_size=size, classes=[label],
                               seed=random.randint(1, 10 ** 6))
            sample = next((root / "val" / label).glob("*.png"), None) or \
                next((root / "train" / label).glob("*.png"), None)
            if sample is None:
                return Response({"detail": "样例生成失败"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            with Image.open(sample) as image:
                buffer = io.BytesIO()
                image.convert("RGB").save(buffer, format="PNG")
        response = HttpResponse(buffer.getvalue(), content_type="image/png")
        response["Content-Disposition"] = f'inline; filename="sample_{label}.png"'
        return response
