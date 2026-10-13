"""首页看板、系统信息与健康检查。"""

from __future__ import annotations

import platform
import sys
from datetime import timedelta

import torch
from django.db.models import Avg, Count
from django.db.models.functions import TruncDate
from django.utils import timezone
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView

from ml import zoo
from ml.engine import resolve_device
from ml.predictor import model_store
from ml.registry import list_architectures

from ..models import Dataset, PredictionRecord, TrainedModel, TrainingRun


def _database_info() -> dict:
    """当前实际生效的数据库信息，用于接口与前端展示「到底用的哪个库」。"""
    from django.conf import settings
    from django.db import connection

    conf = settings.DATABASES["default"]
    engine = conf["ENGINE"].rsplit(".", 1)[-1]
    label = {"sqlite3": "SQLite", "mysql": "MySQL", "postgresql": "PostgreSQL"}.get(engine, engine)
    if engine == "sqlite3":
        name = str(conf["NAME"]).split("/")[-1]
        location = "本地文件 backend/" + name
    else:
        name = str(conf["NAME"])
        location = f"{conf.get('USER') or ''}@{conf.get('HOST') or '127.0.0.1'}:{conf.get('PORT') or ''}"

    version = ""
    try:
        sql = "SELECT sqlite_version()" if engine == "sqlite3" else "SELECT VERSION()"
        with connection.cursor() as cursor:
            cursor.execute(sql)
            row = cursor.fetchone()
            version = str(row[0]) if row else ""
    except Exception as exc:  # noqa: BLE001 —— 数据库不可用时不影响其它信息展示
        version = "不可用"

    return {"engine": engine, "label": label, "version": version, "name": name,
            "location": location, "configured_by": "IR_DB_ENGINE" if engine != "sqlite3" else "默认配置"}


def _system_info() -> dict:
    device = resolve_device("auto")
    return {
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.machine()}",
        "torch": torch.__version__,
        "torchvision": getattr(__import__("torchvision"), "__version__", "unknown"),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        "cpu_count": torch.get_num_threads(),
        "architectures": len(list_architectures()),
        "loaded_models": len(model_store.loaded_ids()),
        "database": _database_info(),
    }


class SystemInfoView(APIView):
    """GET /api/system/ ：运行环境与算法能力信息。"""

    def get(self, request):
        info = _system_info()
        info["supported_architectures"] = [
            {"key": item["key"], "name": item["name"], "year": item["year"], "family": item["family"]}
            for item in list_architectures()
        ]
        return Response(info)


class DashboardStatsView(APIView):
    """GET /api/dashboard/ ：首页聚合统计。"""

    def get(self, request):
        days = int(request.query_params.get("days", 14))
        since = timezone.now() - timedelta(days=days)

        records = PredictionRecord.objects.all()
        trained = TrainedModel.objects.select_related("dataset").all()
        best = trained.order_by("-accuracy").first()

        by_day = (records.filter(created_at__gte=since)
                  .annotate(day=TruncDate("created_at"))
                  .values("day").annotate(count=Count("id")).order_by("day"))

        accuracy_series = [
            {
                "id": model.id,
                "name": model.name,
                "architecture": model.architecture,
                "accuracy": round(model.accuracy, 4),
                "created_at": model.created_at.strftime("%Y-%m-%d %H:%M"),
            }
            for model in trained.order_by("created_at")
        ]

        architecture_compare = {}
        for model in trained:
            item = architecture_compare.setdefault(model.architecture, {
                "architecture": model.architecture, "runs": 0, "best_accuracy": 0.0,
                "params_million": model.params_million, "dataset": model.dataset.name if model.dataset else "",
            })
            item["runs"] += 1
            item["best_accuracy"] = max(item["best_accuracy"], round(model.accuracy, 4))

        return Response({
            "counts": {
                "datasets": Dataset.objects.count(),
                "training_runs": TrainingRun.objects.count(),
                "running_runs": TrainingRun.objects.filter(status=TrainingRun.Status.RUNNING).count(),
                "trained_models": trained.count(),
                "records": records.count(),
                "records_today": records.filter(created_at__date=timezone.localdate()).count(),
                "architectures": len(list_architectures()),
                "ready_architectures": trained.values("architecture").distinct().count(),
            },
            "best_model": {
                "id": best.id, "name": best.name, "architecture": best.architecture,
                "accuracy": round(best.accuracy, 4), "classes": best.num_classes,
                "params_million": best.params_million,
            } if best else None,
            "avg_accuracy": round(trained.aggregate(value=Avg("accuracy"))["value"] or 0, 4),
            "avg_confidence": round(records.aggregate(value=Avg("confidence"))["value"] or 0, 4),
            "avg_latency_ms": round(records.aggregate(value=Avg("latency_ms"))["value"] or 0, 2),
            "records_by_day": [{"date": str(row["day"]), "count": row["count"]} for row in by_day],
            "accuracy_series": accuracy_series[-20:],
            "architecture_compare": sorted(architecture_compare.values(),
                                            key=lambda item: item["best_accuracy"], reverse=True),
            "recent_runs": [
                {
                    "id": run.id, "name": run.name, "architecture": run.architecture,
                    "status": run.status, "status_display": run.get_status_display(),
                    "progress": run.progress, "best_accuracy": round(run.best_accuracy, 4),
                    "created_at": run.created_at.strftime("%Y-%m-%d %H:%M"),
                    "dataset": run.dataset.name,
                }
                for run in TrainingRun.objects.select_related("dataset").order_by("-created_at")[:6]
            ],
            "recent_records": [
                {
                    "id": record.id, "name": record.original_name,
                    "architecture": record.architecture, "label": record.predicted_label,
                    "confidence": round(record.confidence, 4),
                    "image_url": request.build_absolute_uri(record.image.url) if record.image else "",
                    "created_at": record.created_at.strftime("%Y-%m-%d %H:%M"),
                }
                for record in records.order_by("-created_at")[:8]
            ],
            "system": _system_info(),
        })


class HealthView(APIView):
    """GET /api/health/ ：服务健康检查。"""

    permission_classes = [AllowAny]
    authentication_classes = []

    def get(self, request):
        return Response({"status": "ok", "time": timezone.now().strftime("%Y-%m-%d %H:%M:%S"),
                         "architectures": len(zoo.BUILDERS)})
