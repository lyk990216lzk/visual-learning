"""训练任务接口：创建并启动、进度轮询、日志、中止、删除。"""

from __future__ import annotations

from django.db.models import Count
from rest_framework import status
from rest_framework.decorators import action
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.viewsets import ModelViewSet

from ml.trainer import manager

from ..models import TrainingRun
from ..permissions import IsAdminOrReadOnly
from ..serializers import TrainingRunCreateSerializer, TrainingRunSerializer


class TrainingRunViewSet(ModelViewSet):
    """训练任务管理：/api/training-runs/"""

    queryset = TrainingRun.objects.select_related("dataset").all()
    serializer_class = TrainingRunSerializer
    # 发起 / 中止 / 删除训练 = 管理员；查看进度与日志 = 所有登录用户
    permission_classes = [IsAuthenticated, IsAdminOrReadOnly]

    def create(self, request, *args, **kwargs):
        """创建训练任务并立即在后台线程中启动训练。"""
        payload = request.data.copy()
        if not payload.get("name"):
            architecture = payload.get("architecture", "model")
            dataset_id = payload.get("dataset")
            count = TrainingRun.objects.count() + 1
            payload["name"] = f"{architecture}-{dataset_id}-#{count}"

        serializer = TrainingRunCreateSerializer(data=payload)
        serializer.is_valid(raise_exception=True)
        run = serializer.save()
        started = manager.start(run.id)
        run.refresh_from_db()
        data = TrainingRunSerializer(run).data
        data["started"] = started
        return Response(data, status=status.HTTP_201_CREATED)

    def destroy(self, request, *args, **kwargs):
        run = self.get_object()
        if manager.is_running(run.id):
            manager.cancel(run.id)
        if run.trained_model_id:
            run.trained_model.delete()
        run.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)

    @action(detail=True, methods=["post"])
    def cancel(self, request, pk=None):
        """请求中止训练：当前 epoch 结束后安全停止并保存已有效果。"""
        run = self.get_object()
        if run.status not in {TrainingRun.Status.RUNNING, TrainingRun.Status.PENDING}:
            return Response({"detail": "该任务已结束，无需中止"}, status=status.HTTP_400_BAD_REQUEST)
        manager.cancel(run.id)
        return Response({"detail": "已发送中止指令，将在当前 epoch 结束后停止"})

    @action(detail=True, methods=["get"])
    def progress(self, request, pk=None):
        """轻量进度接口，前端按秒轮询：只返回运行态与曲线，不返回大字段。"""
        run = self.get_object()
        running = manager.is_running(run.id)
        if running and run.status != TrainingRun.Status.RUNNING:
            run.refresh_from_db()
        return Response({
            "id": run.id,
            "status": run.status,
            "status_display": run.get_status_display(),
            "running": running,
            "progress": run.progress,
            "current_epoch": run.current_epoch,
            "epochs": run.epochs,
            "best_accuracy": run.best_accuracy,
            "latest_metrics": run.latest_metrics,
            "history": run.history,
            "error": run.error,
            "trained_model_id": run.trained_model_id,
            "duration_seconds": run.duration_seconds,
        })

    @action(detail=True, methods=["get"])
    def log(self, request, pk=None):
        """返回训练日志（支持 ``?tail=200`` 只看最后若干行）。"""
        run = self.get_object()
        tail = int(request.query_params.get("tail", 0))
        lines = (run.log or "").splitlines()
        if tail > 0:
            lines = lines[-tail:]
        return Response({"id": run.id, "status": run.status, "lines": lines,
                         "text": "\n".join(lines)})

    @action(detail=False, methods=["get"])
    def summary(self, request):
        """各状态任务数量统计，用于首页看板。"""
        rows = TrainingRun.objects.values("status").annotate(count=Count("id"))
        return Response({row["status"]: row["count"] for row in rows})
