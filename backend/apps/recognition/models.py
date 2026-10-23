"""领域模型：数据集、训练任务、已训练模型、识别记录。"""

from __future__ import annotations

import uuid
from pathlib import Path

from django.conf import settings
from django.db import models

from .paths import resolve_path, to_stored_path


class Dataset(models.Model):
    """图片识别、YOLO 目标检测或图片-掩码语义分割数据集。"""

    class TaskType(models.TextChoices):
        CLASSIFICATION = "classification", "图片识别"
        DETECTION = "detection", "目标检测"
        SEGMENTATION = "segmentation", "语义分割"

    name = models.CharField("名称", max_length=100, unique=True)
    task_type = models.CharField("功能", max_length=20, choices=TaskType.choices,
                                 default=TaskType.CLASSIFICATION)
    description = models.TextField("说明", blank=True, default="")
    path = models.CharField("目录路径", max_length=500)
    is_demo = models.BooleanField("演示数据集", default=False)
    num_classes = models.IntegerField("类别数", default=0)
    class_names = models.JSONField("类别列表", default=list, blank=True)
    image_count = models.IntegerField("图片总数", default=0)
    train_count = models.IntegerField("训练集图片数", default=0)
    val_count = models.IntegerField("验证集图片数", default=0)
    class_counts = models.JSONField("类别分布", default=dict, blank=True)
    created_at = models.DateTimeField("创建时间", auto_now_add=True)
    updated_at = models.DateTimeField("更新时间", auto_now=True)

    class Meta:
        verbose_name = "数据集"
        verbose_name_plural = "数据集"
        ordering = ["-created_at"]

    def save(self, *args, **kwargs):
        # 入库统一存相对 backend/ 的相对路径，项目改名 / 搬迁后无需改数据
        self.path = to_stored_path(self.path)
        super().save(*args, **kwargs)

    def __str__(self) -> str:
        return self.name

    @property
    def dir_path(self) -> Path:
        """数据集目录的绝对路径（库里存的是相对 backend/ 的路径）。"""
        return resolve_path(self.path)

    @property
    def weights_dir(self) -> Path:
        """该数据集训练产出的权重目录：``data/weights/<数据集名>``。"""
        safe_name = "".join(ch for ch in self.name if ch.isalnum() or ch in "-_") or f"dataset_{self.pk}"
        return Path(settings.WEIGHTS_DIR) / safe_name

    def refresh_stats(self, save: bool = True) -> dict:
        from ml.task_datasets import scan_dataset

        info = scan_dataset(self.dir_path, self.task_type)
        if not info["has_split"] and info["image_count"]:
            # 扁平目录结构在训练时才会按比例分层划分，这里给出 8:2 的预估数量便于展示
            info["train_count"] = int(info["image_count"] * 0.8)
            info["val_count"] = info["image_count"] - info["train_count"]
        self.num_classes = len(info["classes"])
        self.class_names = info["classes"]
        self.class_counts = info["class_counts"]
        self.image_count = info["image_count"]
        self.train_count = info["train_count"]
        self.val_count = info["val_count"]
        if save:
            self.save(update_fields=["num_classes", "class_names", "class_counts", "image_count",
                                     "train_count", "val_count", "updated_at"])
        return info


class TrainingRun(models.Model):
    """一次训练任务：超参数 + 实时进度 + 历史曲线 + 日志。"""

    class TaskType(models.TextChoices):
        CLASSIFICATION = "classification", "图片识别"
        DETECTION = "detection", "目标检测"
        SEGMENTATION = "segmentation", "语义分割"

    class Status(models.TextChoices):
        PENDING = "pending", "排队中"
        RUNNING = "running", "训练中"
        SUCCESS = "success", "已完成"
        FAILED = "failed", "失败"
        CANCELED = "canceled", "已中止"

    name = models.CharField("任务名称", max_length=120)
    task_type = models.CharField("功能", max_length=20, choices=TaskType.choices,
                                 default=TaskType.CLASSIFICATION)
    architecture = models.CharField("网络结构", max_length=40)
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name="training_runs",
                                verbose_name="数据集")
    status = models.CharField("状态", max_length=20, choices=Status.choices, default=Status.PENDING)

    # 超参数
    epochs = models.IntegerField("训练轮数", default=10)
    batch_size = models.IntegerField("批大小", default=32)
    learning_rate = models.FloatField("学习率", default=0.01)
    weight_decay = models.FloatField("权重衰减", default=0.0001)
    optimizer = models.CharField("优化器", max_length=20, default="sgd")
    scheduler = models.CharField("学习率策略", max_length=20, default="cosine")
    image_size = models.IntegerField("输入尺寸", default=64)
    num_workers = models.IntegerField("数据加载线程", default=0)
    val_split = models.FloatField("验证集比例", default=0.2)
    label_smoothing = models.FloatField("标签平滑", default=0.0)
    seed = models.IntegerField("随机种子", default=42)
    device = models.CharField("计算设备", max_length=10, default="auto")
    augment = models.BooleanField("数据增强", default=True)
    max_batches_per_epoch = models.IntegerField("每轮最大 batch 数", default=0)
    class_weighted_loss = models.BooleanField("类别加权损失", default=False)

    # 运行态
    progress = models.FloatField("进度", default=0.0)
    current_epoch = models.IntegerField("当前轮次", default=0)
    num_classes = models.IntegerField("类别数", default=0)
    class_names = models.JSONField("类别列表", default=list, blank=True)
    history = models.JSONField("训练曲线", default=list, blank=True)
    metrics = models.JSONField("最终指标", default=dict, blank=True)
    latest_metrics = models.JSONField("最近指标", default=dict, blank=True)
    best_accuracy = models.FloatField("最优准确率", default=0.0)
    duration_seconds = models.FloatField("耗时(秒)", default=0.0)
    log = models.TextField("训练日志", blank=True, default="")
    error = models.TextField("错误信息", blank=True, default="")

    created_at = models.DateTimeField("创建时间", auto_now_add=True)
    started_at = models.DateTimeField("开始时间", null=True, blank=True)
    finished_at = models.DateTimeField("结束时间", null=True, blank=True)
    trained_model = models.OneToOneField("TrainedModel", on_delete=models.SET_NULL, null=True,
                                         blank=True, related_name="source_run", verbose_name="产出模型")

    class Meta:
        verbose_name = "训练任务"
        verbose_name_plural = "训练任务"
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.name}({self.status})"

    @property
    def is_finished(self) -> bool:
        return self.status in {self.Status.SUCCESS, self.Status.FAILED, self.Status.CANCELED}


class TrainedModel(models.Model):
    """训练产出的可用模型：保存权重路径、类别、评估指标，支持「激活」为默认模型。"""

    name = models.CharField("模型名称", max_length=120)
    architecture = models.CharField("网络结构", max_length=40)
    dataset = models.ForeignKey(Dataset, on_delete=models.SET_NULL, null=True, blank=True,
                                related_name="trained_models", verbose_name="训练数据集")
    training_run = models.ForeignKey(TrainingRun, on_delete=models.SET_NULL, null=True, blank=True,
                                     related_name="models", verbose_name="来源任务")
    classes = models.JSONField("类别列表", default=list, blank=True)
    num_classes = models.IntegerField("类别数", default=0)
    input_size = models.IntegerField("输入尺寸", default=64)
    accuracy = models.FloatField("验证准确率", default=0.0)
    macro_f1 = models.FloatField("宏平均 F1", default=0.0)
    metrics = models.JSONField("完整评估指标", default=dict, blank=True)
    params_million = models.FloatField("参数量(M)", default=0.0)
    file_path = models.CharField("权重路径", max_length=500)
    file_size = models.BigIntegerField("权重体积(字节)", default=0)
    is_active = models.BooleanField("是否激活", default=False)
    notes = models.CharField("备注", max_length=200, blank=True, default="")
    created_at = models.DateTimeField("创建时间", auto_now_add=True)
    updated_at = models.DateTimeField("更新时间", auto_now=True)

    class Meta:
        verbose_name = "已训练模型"
        verbose_name_plural = "已训练模型"
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.name} ({self.accuracy:.3f})"

    def save(self, *args, **kwargs):
        # 与 Dataset 一致：权重路径以相对 backend/ 的形式入库
        self.file_path = to_stored_path(self.file_path)
        super().save(*args, **kwargs)
        # 同一种网络结构只保留一个「激活」模型，作为该结构的默认推理模型
        if self.is_active:
            TrainedModel.objects.filter(architecture=self.architecture, is_active=True) \
                .exclude(pk=self.pk).update(is_active=False)

    @property
    def resolved_path(self) -> Path:
        """权重文件的绝对路径（库里存的是相对 backend/ 的路径）。"""
        return resolve_path(self.file_path)

    def delete(self, *args, **kwargs):
        from ml.predictor import model_store

        model_store.invalidate(self.pk)
        path = self.resolved_path
        result = super().delete(*args, **kwargs)
        try:
            if path.exists() and path.is_file() and path.suffix == ".pt":
                path.unlink()
        except OSError:
            pass
        return result


class PredictionRecord(models.Model):
    """一次识别请求的记录（对比模式下一次请求会写入多条，共享 group_id）。"""

    class Source(models.TextChoices):
        UPLOAD = "upload", "单图上传"
        COMPARE = "compare", "多模型对比"
        BATCH = "batch", "批量识别"
        SAMPLE = "sample", "内置样例"

    group_id = models.UUIDField("请求分组", default=uuid.uuid4, db_index=True)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True,
                             related_name="prediction_records", verbose_name="操作用户")
    image = models.FileField("图片", upload_to="uploads/%Y%m%d/", blank=True, null=True)
    original_name = models.CharField("原始文件名", max_length=255, blank=True, default="")
    source = models.CharField("来源", max_length=20, choices=Source.choices, default=Source.UPLOAD)
    architecture = models.CharField("网络结构", max_length=40, blank=True, default="")
    model_name = models.CharField("模型名", max_length=120, blank=True, default="")
    trained_model = models.ForeignKey(TrainedModel, on_delete=models.SET_NULL, null=True, blank=True,
                                      related_name="records", verbose_name="模型")
    predicted_label = models.CharField("预测类别", max_length=120, blank=True, default="")
    confidence = models.FloatField("置信度", default=0.0)
    topk = models.JSONField("Top-K 结果", default=list, blank=True)
    latency_ms = models.FloatField("推理耗时(ms)", default=0.0)
    created_at = models.DateTimeField("识别时间", auto_now_add=True, db_index=True)

    class Meta:
        verbose_name = "识别记录"
        verbose_name_plural = "识别记录"
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.original_name} -> {self.predicted_label}({self.confidence:.2f})"
