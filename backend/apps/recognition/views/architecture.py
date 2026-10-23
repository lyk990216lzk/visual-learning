"""经典模型结构接口：向前端提供模型库元信息与结构统计。"""

from __future__ import annotations

from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.viewsets import ViewSet

from ml.registry import architecture_summary, get_architecture, list_architectures

from ..models import TrainedModel


class ArchitectureViewSet(ViewSet):
    """只读接口：/api/architectures/"""

    def list(self, request):
        """列出全部经典网络结构，并附带当前已训练模型情况。"""
        trained = {}
        for item in TrainedModel.objects.all().only("id", "architecture", "accuracy", "is_active",
                                                    "num_classes", "created_at", "name"):
            trained.setdefault(item.architecture, []).append(item)

        data = []
        for meta in list_architectures():
            models = trained.get(meta["key"], [])
            best = max(models, key=lambda m: m.accuracy) if models else None
            active = next((m for m in models if m.is_active), None)
            data.append({
                **meta,
                "trained_count": len(models),
                "best_accuracy": round(best.accuracy, 4) if best else None,
                "best_model_id": best.id if best else None,
                "active_model_id": (active or best).id if (active or best) else None,
                "active_model_name": (active or best).name if (active or best) else None,
                "num_classes": (active or best).num_classes if (active or best) else None,
                "ready": bool(models),
            })
        return Response({"count": len(data), "results": data})

    def retrieve(self, request, pk=None):
        meta = get_architecture(pk)
        if meta is None:
            return Response({"detail": f"未知的模型：{pk}"}, status=404)
        models = TrainedModel.objects.filter(architecture=meta["key"])
        num_classes = request.query_params.get("num_classes")
        input_size = request.query_params.get("input_size")
        summary = None
        if meta.get("implemented"):
            summary = architecture_summary(meta["key"],
                                           num_classes=int(num_classes) if num_classes else 10,
                                           input_size=int(input_size) if input_size else None)
        return Response({
            **meta,
            "summary": summary,
            "integration_note": None if meta.get("implemented") else
                                "该模型已加入统一模型目录，训练与推理后端尚待接入。",
            "trained_models": list(models.values("id", "name", "accuracy", "num_classes",
                                                 "is_active", "created_at")),
        })

    @action(detail=True, methods=["get"])
    def stat(self, request, pk=None):
        """指定类别数与输入尺寸下的参数量 / 体积 / CPU 推理耗时。"""
        meta = get_architecture(pk)
        if meta is None:
            return Response({"detail": f"未知的模型：{pk}"}, status=404)
        if not meta.get("implemented"):
            return Response(
                {"detail": "该模型尚未接入训练与推理后端，暂时无法统计结构参数。"},
                status=409,
            )
        num_classes = int(request.query_params.get("num_classes", 10))
        input_size = int(request.query_params.get("input_size", meta["default_input_size"]))
        return Response(architecture_summary(meta["key"], num_classes, input_size))
