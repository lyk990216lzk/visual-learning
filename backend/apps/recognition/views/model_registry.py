"""已训练模型接口：列表、激活、重新评估、下载、删除。"""

from __future__ import annotations

from django.http import FileResponse
from rest_framework import status
from rest_framework.decorators import action
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.viewsets import ModelViewSet

from ml.datasets import build_dataloaders
from ml.detection import DetectionBackendError, evaluate_yolo_model
from ml.engine import resolve_device
from ml.metrics import evaluate_model
from ml.predictor import PredictorError, model_store
from ml.segmentation import build_segmentation_dataloaders, evaluate_segmentation
from ml.task_predictor import task_model_cache

from ..models import TrainedModel
from ..permissions import IsAdminOrReadOnly
from ..serializers import TrainedModelSerializer


class TrainedModelViewSet(ModelViewSet):
    """模型管理：/api/trained-models/"""

    queryset = TrainedModel.objects.select_related("dataset").all()
    serializer_class = TrainedModelSerializer
    # 激活 / 重新评估 / 删除 = 管理员；浏览与下载 = 所有登录用户
    permission_classes = [IsAuthenticated, IsAdminOrReadOnly]

    def get_queryset(self):
        queryset = super().get_queryset()
        architecture = self.request.query_params.get("architecture")
        if architecture:
            queryset = queryset.filter(architecture=architecture)
        task_type = self.request.query_params.get("task_type")
        if task_type:
            queryset = queryset.filter(task_type=task_type)
        active = self.request.query_params.get("is_active")
        if active in {"1", "true", "True"}:
            queryset = queryset.filter(is_active=True)
        search = self.request.query_params.get("search")
        if search:
            queryset = queryset.filter(name__icontains=search)
        ordering = self.request.query_params.get("ordering")
        if ordering in {"accuracy", "-accuracy", "created_at", "-created_at"}:
            queryset = queryset.order_by(ordering)
        return queryset

    @action(detail=True, methods=["post"])
    def activate(self, request, pk=None):
        """把该模型设为所属网络结构的默认推理模型。"""
        model = self.get_object()
        model.is_active = True
        model.save(update_fields=["is_active", "updated_at"])
        return Response({"detail": f"已将「{model.name}」设为 {model.architecture} 的默认模型",
                         "model": TrainedModelSerializer(model).data})

    @action(detail=True, methods=["post"])
    def evaluate(self, request, pk=None):
        """用训练数据集重新评估该模型的验证指标（准确率、混淆矩阵等）。"""
        trained = self.get_object()
        if trained.dataset is None:
            return Response({"detail": "该模型没有关联数据集，无法重新评估"},
                            status=status.HTTP_400_BAD_REQUEST)
        try:
            if trained.task_type == "detection":
                report = evaluate_yolo_model(
                    trained,
                    trained.dataset.dir_path,
                    trained.dataset.weights_dir / f"_eval_{trained.id}",
                    val_split=float(request.data.get("val_split", 0.2)),
                    seed=int(request.data.get("seed", 42)),
                    image_size=trained.input_size,
                    device=request.data.get("device", "cpu"),
                )
            elif trained.task_type == "segmentation":
                train_loader, val_loader, class_names, _ = build_segmentation_dataloaders(
                    trained.dataset.dir_path,
                    image_size=trained.input_size,
                    batch_size=int(request.data.get("batch_size", 4)),
                    val_split=float(request.data.get("val_split", 0.2)),
                    num_workers=0,
                )
                entry = task_model_cache.get(trained)
                device = resolve_device(request.data.get("device", "auto"))
                model = entry["model"].to(device)
                report = evaluate_segmentation(model, val_loader, device, class_names)
            else:
                entry = model_store.load(trained)
                dataset = trained.dataset
                _, val_loader, class_names = build_dataloaders(
                    dataset.dir_path, image_size=trained.input_size, batch_size=32,
                    val_split=request.data.get("val_split", 0.2), num_workers=0,
                )
                device = resolve_device("auto")
                model = entry["model"].to(device)
                report = evaluate_model(model, val_loader, device, class_names)
        except (PredictorError, DetectionBackendError, ValueError) as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        trained.metrics = report
        trained.accuracy = report.get("accuracy", 0)
        trained.macro_f1 = report.get("macro_f1", report.get("miou", report.get("map50_95", 0)))
        trained.save(update_fields=["metrics", "accuracy", "macro_f1", "updated_at"])
        return Response({"detail": "评估完成", "metrics": report,
                         "model": TrainedModelSerializer(trained).data})

    @action(detail=True, methods=["get"])
    def download(self, request, pk=None):
        """下载权重文件（.pt）。"""
        trained = self.get_object()
        path = trained.resolved_path
        if not path.exists():
            return Response({"detail": "权重文件已丢失"}, status=status.HTTP_404_NOT_FOUND)
        return FileResponse(open(path, "rb"), as_attachment=True, filename=path.name)

    @action(detail=False, methods=["get"])
    def overview(self, request):
        """按网络结构汇总已训练模型，用于模型库页面。"""
        result = {}
        for model in self.get_queryset():
            item = result.setdefault(model.architecture, {"count": 0, "best_accuracy": 0.0,
                                                          "active_model_id": None,
                                                          "latest_model_id": None})
            item["count"] += 1
            item["best_accuracy"] = max(item["best_accuracy"], round(model.accuracy, 4))
            item["latest_model_id"] = model.id
            if model.is_active:
                item["active_model_id"] = model.id
        for item in result.values():
            item["active_model_id"] = item["active_model_id"] or item["latest_model_id"]
        return Response(result)
