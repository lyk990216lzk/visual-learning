from django.contrib import admin

from .models import Dataset, PredictionRecord, TrainedModel, TrainingRun


@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = ("id", "name", "num_classes", "image_count", "is_demo", "created_at")
    search_fields = ("name",)


@admin.register(TrainingRun)
class TrainingRunAdmin(admin.ModelAdmin):
    list_display = ("id", "name", "architecture", "dataset", "status", "progress",
                    "best_accuracy", "created_at")
    list_filter = ("status", "architecture", "optimizer")
    search_fields = ("name",)


@admin.register(TrainedModel)
class TrainedModelAdmin(admin.ModelAdmin):
    list_display = ("id", "name", "architecture", "accuracy", "macro_f1", "is_active", "created_at")
    list_filter = ("architecture", "is_active")
    search_fields = ("name",)


@admin.register(PredictionRecord)
class PredictionRecordAdmin(admin.ModelAdmin):
    list_display = ("id", "original_name", "architecture", "predicted_label", "confidence", "created_at")
    list_filter = ("architecture", "predicted_label")
    search_fields = ("original_name", "predicted_label")
