from django.contrib import admin

from .models import Dataset, PredictionRecord, TrainedModel, TrainingRun


@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = ("id", "name", "task_type", "num_classes", "image_count", "is_demo",
                    "created_at")
    list_filter = ("task_type", "is_demo")
    search_fields = ("name",)


@admin.register(TrainingRun)
class TrainingRunAdmin(admin.ModelAdmin):
    list_display = ("id", "name", "task_type", "architecture", "dataset", "status", "progress",
                    "best_accuracy", "created_at")
    list_filter = ("task_type", "status", "architecture", "optimizer")
    search_fields = ("name",)


@admin.register(TrainedModel)
class TrainedModelAdmin(admin.ModelAdmin):
    list_display = ("id", "name", "task_type", "architecture", "accuracy", "macro_f1",
                    "is_active", "created_at")
    list_filter = ("task_type", "architecture", "is_active")
    search_fields = ("name",)


@admin.register(PredictionRecord)
class PredictionRecordAdmin(admin.ModelAdmin):
    list_display = ("id", "original_name", "task_type", "architecture", "predicted_label",
                    "confidence", "created_at")
    list_filter = ("task_type", "architecture", "predicted_label")
    search_fields = ("original_name", "predicted_label")
