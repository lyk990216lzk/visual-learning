"""识别记录接口：列表、删除、清空、统计分析。"""

from __future__ import annotations

from datetime import timedelta

from django.db.models import Avg, Count
from django.db.models.functions import TruncDate
from django.utils import timezone
from rest_framework import status
from rest_framework.decorators import action
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.viewsets import ModelViewSet

from ..models import PredictionRecord
from ..permissions import IsAdminOrReadOnly
from ..serializers import PredictionRecordSerializer


class PredictionRecordViewSet(ModelViewSet):
    """识别历史：/api/records/"""

    queryset = PredictionRecord.objects.select_related("trained_model").all()
    serializer_class = PredictionRecordSerializer
    # 删除 / 清空记录 = 管理员；查询与统计 = 所有登录用户
    permission_classes = [IsAuthenticated, IsAdminOrReadOnly]

    def get_queryset(self):
        queryset = super().get_queryset()
        params = self.request.query_params
        if params.get("mine") in {"1", "true", "True"}:
            queryset = queryset.filter(user=self.request.user)
        if params.get("username"):
            queryset = queryset.filter(user__username=params["username"])
        if params.get("architecture"):
            queryset = queryset.filter(architecture=params["architecture"])
        if params.get("task_type"):
            queryset = queryset.filter(task_type=params["task_type"])
        if params.get("label"):
            queryset = queryset.filter(predicted_label=params["label"])
        if params.get("source"):
            queryset = queryset.filter(source=params["source"])
        if params.get("search"):
            queryset = queryset.filter(original_name__icontains=params["search"])
        if params.get("date_from"):
            queryset = queryset.filter(created_at__date__gte=params["date_from"])
        if params.get("date_to"):
            queryset = queryset.filter(created_at__date__lte=params["date_to"])
        ordering = params.get("ordering")
        if ordering in {"created_at", "-created_at", "confidence", "-confidence", "latency_ms",
                        "-latency_ms"}:
            queryset = queryset.order_by(ordering)
        return queryset

    def get_serializer_context(self):
        context = super().get_serializer_context()
        context["request"] = self.request
        return context

    @action(detail=False, methods=["post"])
    def clear(self, request):
        """清空识别记录（默认只清空筛选条件命中的记录）。"""
        queryset = self.filter_queryset(self.get_queryset())
        if str(request.data.get("all", "")).lower() in {"1", "true", "yes"}:
            queryset = PredictionRecord.objects.all()
        count = queryset.count()
        queryset.delete()
        return Response({"detail": f"已删除 {count} 条识别记录", "count": count})

    @action(detail=False, methods=["get"])
    def stats(self, request):
        """识别统计：总量、每日趋势、模型分布、类别分布、置信度与耗时。"""
        queryset = PredictionRecord.objects.all()
        if request.query_params.get("mine") in {"1", "true", "True"}:
            queryset = queryset.filter(user=request.user)
        if request.query_params.get("task_type"):
            queryset = queryset.filter(task_type=request.query_params["task_type"])
        days = int(request.query_params.get("days", 14))
        since = timezone.now() - timedelta(days=days)

        aggregate = queryset.aggregate(avg_conf=Avg("confidence"), avg_latency=Avg("latency_ms"))
        by_day = (queryset.filter(created_at__gte=since)
                  .annotate(day=TruncDate("created_at"))
                  .values("day").annotate(count=Count("id")).order_by("day"))
        by_architecture = (queryset.values("architecture")
                           .annotate(count=Count("id"), avg_confidence=Avg("confidence"))
                           .order_by("-count"))
        by_label = (queryset.exclude(predicted_label="")
                    .values("predicted_label").annotate(count=Count("id")).order_by("-count")[:15])
        by_source = queryset.values("source").annotate(count=Count("id")).order_by("-count")

        return Response({
            "total": queryset.count(),
            "today": queryset.filter(created_at__date=timezone.localdate()).count(),
            "avg_confidence": round(aggregate["avg_conf"] or 0, 4),
            "avg_latency_ms": round(aggregate["avg_latency"] or 0, 2),
            "by_day": [{"date": str(row["day"]), "count": row["count"]} for row in by_day],
            "by_architecture": [
                {"architecture": row["architecture"] or "未知", "count": row["count"],
                 "avg_confidence": round(row["avg_confidence"] or 0, 4)}
                for row in by_architecture
            ],
            "by_label": [{"label": row["predicted_label"], "count": row["count"]} for row in by_label],
            "by_source": [{"source": row["source"], "count": row["count"]} for row in by_source],
        })
