"""识别记录接口：列表、删除、清空、统计分析。"""

from __future__ import annotations

from datetime import timedelta

from django.db.models import Avg, Count, Max
from django.db.models.functions import TruncDate
from django.utils import timezone
from rest_framework import status
from rest_framework.decorators import action
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.viewsets import ModelViewSet

from ..models import PredictionRecord, TrainingRun
from ..permissions import IsAdminOrReadOnly
from ..serializers import PredictionRecordSerializer


class PredictionRecordViewSet(ModelViewSet):
    """识别历史：/api/records/"""

    queryset = PredictionRecord.objects.select_related(
        "user", "trained_model", "trained_model__training_run"
    ).all()
    serializer_class = PredictionRecordSerializer
    # 删除 / 清空记录 = 管理员；查询与统计 = 所有登录用户
    permission_classes = [IsAuthenticated, IsAdminOrReadOnly]

    def _apply_common_filters(self, queryset):
        """列表、统计与导出共用同一套识别历史筛选语义。"""
        params = self.request.query_params
        if params.get("mine") in {"1", "true", "True"}:
            queryset = queryset.filter(user=self.request.user)
        if params.get("username"):
            queryset = queryset.filter(user__username=params["username"])
        if params.get("architecture"):
            queryset = queryset.filter(architecture=params["architecture"])
        if params.get("task_type"):
            queryset = queryset.filter(task_type=params["task_type"])
        if params.get("training_run"):
            try:
                training_run_id = int(params["training_run"])
            except (TypeError, ValueError):
                return queryset.none()
            queryset = queryset.filter(trained_model__training_run_id=training_run_id)
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
        days = params.get("days")
        if days and days != "all":
            try:
                day_count = max(1, int(days))
            except (TypeError, ValueError):
                day_count = 0
            if day_count:
                today = timezone.localdate()
                if day_count == 1:
                    queryset = queryset.filter(created_at__date=today)
                else:
                    queryset = queryset.filter(
                        created_at__date__gte=today - timedelta(days=day_count - 1)
                    )
        return queryset

    def get_queryset(self):
        queryset = self._apply_common_filters(super().get_queryset())
        params = self.request.query_params
        ordering = params.get("ordering")
        if ordering in {"created_at", "-created_at", "confidence", "-confidence", "latency_ms",
                        "-latency_ms"}:
            queryset = queryset.order_by(ordering)
        return queryset

    def get_serializer_context(self):
        context = super().get_serializer_context()
        context["request"] = self.request
        return context

    @staticmethod
    def _record_labels(record):
        """提取一条识别结果中可供查询和展示的类别。"""
        labels = [record.predicted_label] if record.predicted_label else []
        result = record.result_data or {}
        if record.task_type == TrainingRun.TaskType.DETECTION:
            labels.extend(item.get("label", "") for item in result.get("detections", []))
        elif record.task_type == TrainingRun.TaskType.SEGMENTATION:
            labels.extend(
                item.get("label", "")
                for item in result.get("class_stats", [])
                if int(item.get("pixel_count") or 0) > 0
            )
        else:
            labels.extend(item.get("label", "") for item in (record.topk or [])[:3])
        return list(dict.fromkeys(label for label in labels if label))

    def list(self, request, *args, **kwargs):
        """默认返回原始结果；``grouped=1`` 时按一次识别请求聚合。"""
        if request.query_params.get("grouped") not in {"1", "true", "True"}:
            return super().list(request, *args, **kwargs)

        queryset = self.filter_queryset(self.get_queryset())
        group_rows = (
            queryset.order_by()
            .values("group_id")
            .annotate(last_at=Max("created_at"))
            .order_by("-last_at")
        )
        page = self.paginate_queryset(group_rows)
        if page is None:
            page = list(group_rows)
        group_ids = [row["group_id"] for row in page]
        timestamps = {row["group_id"]: row["last_at"] for row in page}

        grouped_records = {group_id: [] for group_id in group_ids}
        rows = queryset.filter(group_id__in=group_ids).order_by("-created_at")
        serializer = self.get_serializer(rows, many=True)
        for record, payload in zip(rows, serializer.data):
            grouped_records.setdefault(record.group_id, []).append((record, payload))

        source_labels = dict(PredictionRecord.Source.choices)
        task_labels = dict(TrainingRun.TaskType.choices)
        groups = []
        for group_id in group_ids:
            items = grouped_records.get(group_id, [])
            if not items:
                continue
            records = [payload for _record, payload in items]
            primary = records[0]
            labels = list(dict.fromkeys(
                label
                for record, _payload in items
                for label in self._record_labels(record)
            ))
            model_names = list(dict.fromkeys(
                payload["model_name"] for payload in records if payload.get("model_name")
            ))
            result_count = len(records)
            is_compare = primary["source"] == PredictionRecord.Source.COMPARE
            images_count = 1 if is_compare else result_count
            groups.append({
                "group_id": str(group_id),
                "created_at": timestamps[group_id].isoformat(),
                "original_name": (
                    primary["original_name"] if images_count == 1
                    else f"{primary['original_name']} 等 {images_count} 张"
                ),
                "image_url": primary.get("image_url", ""),
                "task_type": primary["task_type"],
                "task_type_label": task_labels.get(primary["task_type"], primary["task_type"]),
                "architecture": primary["architecture"],
                "training_run_id": primary.get("training_run_id"),
                "training_run_name": primary.get("training_run_name", ""),
                "source": primary["source"],
                "source_label": source_labels.get(primary["source"], primary["source"]),
                "username": primary.get("username", "—"),
                "labels": labels,
                "model_names": model_names,
                "images_count": images_count,
                "result_count": result_count,
                "confidence": round(
                    sum(float(payload.get("confidence") or 0) for payload in records) / result_count,
                    6,
                ),
                "latency_ms": round(
                    sum(float(payload.get("latency_ms") or 0) for payload in records) / result_count,
                    2,
                ),
                "records": records,
            })

        if self.paginator is not None:
            return self.get_paginated_response(groups)
        return Response(groups)

    @action(detail=False, methods=["get"])
    def options(self, request):
        """返回识别历史可用的识别功能、结构、训练任务与类别。"""
        from ml.registry import get_architecture

        runs = (
            TrainingRun.objects.select_related("trained_model")
            .prefetch_related("models")
            .filter(models__records__isnull=False)
            .distinct()
            .order_by("-created_at")
        )
        tasks = []
        architectures = []
        labels = set()
        for run in runs:
            model = run.trained_model or next(iter(run.models.all()), None)
            classes = list((model.classes if model else None) or run.class_names or [])
            meta = get_architecture(run.architecture) or {}
            tasks.append({
                "id": run.id,
                "name": run.name,
                "task_type": run.task_type,
                "architecture": run.architecture,
                "architecture_name": meta.get("name", run.architecture),
                "classes": classes,
            })
            if run.architecture not in architectures:
                architectures.append(run.architecture)
            labels.update(classes)

        labels.update(
            label
            for label in PredictionRecord.objects.exclude(predicted_label="")
            .values_list("predicted_label", flat=True)
            .distinct()
            if label
        )
        return Response({
            "tasks": tasks,
            "architectures": architectures,
            "labels": sorted(labels),
        })

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
        queryset = self._apply_common_filters(PredictionRecord.objects.all())

        aggregate = queryset.aggregate(avg_conf=Avg("confidence"), avg_latency=Avg("latency_ms"))
        by_day = (queryset.order_by()
                  .annotate(day=TruncDate("created_at"))
                  .values("day").annotate(count=Count("id")).order_by("day"))
        by_architecture = (queryset.order_by().values("architecture")
                           .annotate(count=Count("id"), avg_confidence=Avg("confidence"))
                           .order_by("-count"))
        by_label = (queryset.order_by().exclude(predicted_label="")
                    .values("predicted_label").annotate(count=Count("id")).order_by("-count")[:15])
        by_task_type = (queryset.order_by().values("task_type")
                        .annotate(count=Count("id")).order_by("-count"))
        by_source = (queryset.order_by().values("source")
                     .annotate(count=Count("id")).order_by("-count"))

        return Response({
            "total": queryset.count(),
            "group_count": queryset.values("group_id").distinct().count(),
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
            "by_task_type": [
                {"task_type": row["task_type"], "count": row["count"]}
                for row in by_task_type
            ],
            "by_source": [{"source": row["source"], "count": row["count"]} for row in by_source],
        })
