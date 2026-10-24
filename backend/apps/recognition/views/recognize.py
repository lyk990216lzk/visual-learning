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

from ml.comparison import compare_detection_outputs, compare_segmentation_outputs
from ml.predictor import PredictorError, open_image, predict_image

from ..models import PredictionRecord, TrainedModel, TrainingRun
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
        payload["training_run_id"] = trained.training_run_id
        payload["recognition_type"] = "single"
        payload["group_id"] = str(record.group_id)
        payload["image_url"] = request.build_absolute_uri(record.image.url) if record.image else ""
        payload["source"] = record.source
        return Response(payload)


class CompareRecognizeView(APIView):
    """POST /api/recognize/compare/：同一数据集的任务模型对比。"""

    parser_classes = [MultiPartParser, FormParser, JSONParser]

    def post(self, request):
        upload = _get_upload(request)
        if upload is None:
            return Response({"detail": "请上传图片（字段名 image）"}, status=status.HTTP_400_BAD_REQUEST)
        message = _check_size(upload)
        if message:
            return Response({"detail": message}, status=status.HTTP_400_BAD_REQUEST)

        raw_ids = request.data.get("trained_models")
        if isinstance(raw_ids, (list, tuple)):
            model_id_values = raw_ids
        else:
            model_id_values = str(raw_ids or "").split(",")
        model_ids = []
        for value in model_id_values:
            value = str(value).strip()
            if not value:
                continue
            if not value.isdigit():
                return Response({"detail": "模型 ID 格式不正确"},
                                status=status.HTTP_400_BAD_REQUEST)
            model_id = int(value)
            if model_id not in model_ids:
                model_ids.append(model_id)

        if model_ids:
            if not 2 <= len(model_ids) <= 6:
                return Response({"detail": "多模型对比需要选择 2~6 个模型"},
                                status=status.HTTP_400_BAD_REQUEST)
            queryset = TrainedModel.objects.select_related("dataset", "training_run").filter(
                pk__in=model_ids
            )
            by_id = {model.id: model for model in queryset}
            models = [by_id[model_id] for model_id in model_ids if model_id in by_id]
            if len(models) != len(model_ids):
                return Response({"detail": "部分所选模型不存在"},
                                status=status.HTTP_400_BAD_REQUEST)
        else:
            # 兼容旧接口：按网络结构取分类模型，但仍要求共享同一数据集。
            architectures = parse_architecture_list(request.data.get("architectures"))
            models = []
            for architecture in architectures:
                try:
                    models.append(resolve_trained_model(architecture, None))
                except Exception:  # noqa: BLE001
                    continue
            if not models:
                return Response({"detail": "请先训练至少两个模型，再进行多模型对比"},
                                status=status.HTTP_400_BAD_REQUEST)

        task_types = {model.task_type for model in models}
        dataset_ids = {model.dataset_id for model in models}
        if len(task_types) != 1:
            return Response({"detail": "参与对比的模型必须使用同一识别功能"},
                            status=status.HTTP_400_BAD_REQUEST)
        if len(dataset_ids) != 1 or None in dataset_ids:
            return Response({"detail": "参与对比的模型必须使用同一数据集"},
                            status=status.HTTP_400_BAD_REQUEST)
        invalid_tasks = [
            model.id
            for model in models
            if model.training_run is None
            or model.training_run.status != TrainingRun.Status.SUCCESS
        ]
        if invalid_tasks:
            return Response({
                "detail": f"以下模型没有关联已完成的训练任务：{invalid_tasks}"
            }, status=status.HTTP_400_BAD_REQUEST)
        task_type = task_types.pop()
        requested_task_type = (request.data.get("task_type") or task_type).strip().lower()
        if requested_task_type != task_type:
            return Response({"detail": "所选功能与模型任务类型不一致"},
                            status=status.HTTP_400_BAD_REQUEST)
        missing_weights = [model.id for model in models if not model.resolved_path.is_file()]
        if missing_weights:
            return Response({"detail": f"模型权重文件不存在：{missing_weights}"},
                            status=status.HTTP_400_BAD_REQUEST)

        def metric_score(model):
            metrics = model.metrics or {}
            if task_type == "detection":
                return float(
                    metrics.get("map50_95")
                    or metrics.get("map5095")
                    or metrics.get("map50")
                    or model.accuracy
                    or 0
                )
            if task_type == "segmentation":
                return float(metrics.get("miou") or metrics.get("best_miou")
                             or model.macro_f1 or model.accuracy or 0)
            return float(metrics.get("macro_f1") or model.macro_f1 or model.accuracy or 0)

        ranked_models = sorted(models, key=lambda model: (metric_score(model), model.id), reverse=True)
        best_model = ranked_models[0]

        from .vision_tasks import _save_task_record, predict_task_image

        try:
            raw = upload.read()
            image = open_image(raw)
        except PredictorError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)

        group_id = uuid.uuid4()
        outputs = []
        errors = []
        records = []
        for trained in models:
            try:
                result = predict_task_image(trained, image, request.data)
                result.setdefault("task_type", trained.task_type)
            except (PredictorError, ValueError) as exc:
                errors.append({
                    "model_id": trained.id,
                    "model_name": trained.name,
                    "error": str(exc),
                })
                continue
            record = _save_task_record(
                request,
                upload,
                raw,
                trained,
                result,
                source=PredictionRecord.Source.COMPARE,
                group_id=group_id,
            )
            records.append(record)
            result.update({
                "record_id": record.id,
                "model_id": trained.id,
                "model_name": trained.name,
                "architecture": trained.architecture,
                "training_run_id": trained.training_run_id,
                "task_id": trained.training_run_id,
                "task_name": trained.training_run.name if trained.training_run else "",
                "model_metric": metric_score(trained),
            })
            outputs.append(result)

        if not outputs:
            return Response({
                "detail": "所有模型推理失败",
                "errors": errors,
            }, status=status.HTTP_400_BAD_REQUEST)

        successful_best = next(
            (item for item in outputs if item["model_id"] == best_model.id),
            max(outputs, key=lambda item: item.get("model_metric", 0)),
        )
        reference_id = successful_best["model_id"]
        response = {
            "group_id": str(group_id),
            "task_type": task_type,
            "dataset_id": models[0].dataset_id,
            "dataset_name": models[0].dataset.name,
            "recognition_type": "compare",
            "selected_model_count": len(models),
            "successful_model_count": len(outputs),
            "best_model_id": best_model.id,
            "best_model_name": best_model.name,
            "best_model_metric": metric_score(best_model),
            "reference_model_id": reference_id,
            "reference_model_name": successful_best["model_name"],
            "reference_fallback": reference_id != best_model.id,
            "results": outputs,
            "errors": errors,
            "latency_ms": round(sum(item.get("latency_ms", 0) for item in outputs), 2),
            "record_ids": [record.id for record in records],
            "image_url": request.build_absolute_uri(records[0].image.url) if records else "",
        }

        try:
            if task_type == "classification":
                labels = [item["predicted_label"] for item in outputs]
                consensus = max(set(labels), key=labels.count)
                response.update({
                    "consensus_label": consensus,
                    "agreement": round(labels.count(consensus) / len(labels), 4),
                })
            elif task_type == "detection":
                response["detection_comparison"] = compare_detection_outputs(
                    outputs,
                    reference_model_id=reference_id,
                    iou_threshold=float(request.data.get("iou_threshold", 0.5)),
                )
            else:
                segmentation_comparison = compare_segmentation_outputs(outputs, reference_id)
                response["segmentation_comparison"] = segmentation_comparison
                for item in response["results"]:
                    item.pop("mask_base64", None)
        except ValueError as exc:
            return Response({"detail": f"对比结果聚合失败：{exc}"},
                            status=status.HTTP_400_BAD_REQUEST)
        return Response(response)


class BatchRecognizeView(APIView):
    """POST /api/recognize/batch/：使用同一训练任务模型批量识别图片。"""

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

        try:
            trained = resolve_trained_model(request.data.get("architecture"),
                                            request.data.get("trained_model"))
        except Exception as exc:  # noqa: BLE001
            detail = getattr(exc, "detail", str(exc))
            return Response({"detail": detail}, status=status.HTTP_400_BAD_REQUEST)

        from .vision_tasks import _save_task_record, predict_task_image

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
                result = predict_task_image(trained, image, request.data)
                result.setdefault("task_type", trained.task_type)
            except (PredictorError, ValueError) as exc:
                failed.append({"name": upload.name, "error": str(exc)})
                continue
            record = _save_task_record(
                request,
                upload,
                raw,
                trained,
                result,
                source=PredictionRecord.Source.BATCH,
                group_id=group_id,
            )
            item = {
                "record_id": record.id,
                "name": upload.name,
                "predicted_label": result["predicted_label"],
                "confidence": result["confidence"],
                "latency_ms": result["latency_ms"],
                "image_url": request.build_absolute_uri(record.image.url) if record.image else "",
                "task_type": result["task_type"],
                "recognition_type": "batch",
            }
            if result["task_type"] == "detection":
                item.update({
                    "detections": result.get("detections", []),
                    "object_count": result.get("object_count", 0),
                })
            elif result["task_type"] == "segmentation":
                item.update({
                    "classes": result.get("classes", []),
                    "class_stats": result.get("class_stats", []),
                    "width": result.get("width"),
                    "height": result.get("height"),
                })
            else:
                item.update({
                    "topk": result.get("topk", []),
                    "all_probabilities": result.get("all_probabilities", {}),
                })
            results.append(item)

        return Response({
            "group_id": str(group_id),
            "model_id": trained.id,
            "model_name": trained.name,
            "task_type": trained.task_type,
            "training_run_id": trained.training_run_id,
            "recognition_type": "batch",
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
