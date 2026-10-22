"""DRF 序列化器：请求校验与响应结构定义。"""

from __future__ import annotations

from pathlib import Path

from django.contrib.auth.models import User
from rest_framework import serializers

from ml.engine import SUPPORTED_DEVICES, is_device_available, normalize_device
from ml.registry import architecture_keys

from .models import Dataset, PredictionRecord, TrainedModel, TrainingRun


class DatasetSerializer(serializers.ModelSerializer):
    class Meta:
        model = Dataset
        fields = ["id", "name", "description", "path", "is_demo", "num_classes", "class_names",
                  "image_count", "train_count", "val_count", "class_counts", "created_at"]
        read_only_fields = ["path", "is_demo", "num_classes", "class_names", "image_count",
                            "train_count", "val_count", "class_counts", "created_at"]


class DatasetBriefSerializer(serializers.ModelSerializer):
    class Meta:
        model = Dataset
        fields = ["id", "name", "num_classes", "image_count"]


class TrainingRunSerializer(serializers.ModelSerializer):
    dataset_name = serializers.CharField(source="dataset.name", read_only=True)
    status_display = serializers.CharField(source="get_status_display", read_only=True)
    architecture_name = serializers.SerializerMethodField()
    trained_model_id = serializers.SerializerMethodField()

    class Meta:
        model = TrainingRun
        fields = [
            "id", "name", "architecture", "architecture_name", "dataset", "dataset_name", "status",
            "status_display", "epochs", "batch_size", "learning_rate", "weight_decay", "optimizer",
            "scheduler", "image_size", "num_workers", "val_split", "label_smoothing", "seed",
            "device", "augment", "max_batches_per_epoch", "class_weighted_loss", "progress",
            "current_epoch", "num_classes", "class_names", "history", "metrics", "latest_metrics",
            "best_accuracy", "duration_seconds", "log", "error", "created_at", "started_at",
            "finished_at", "trained_model_id",
        ]
        read_only_fields = ["status", "progress", "current_epoch", "history", "metrics",
                            "latest_metrics", "best_accuracy", "duration_seconds", "log", "error",
                            "created_at", "started_at", "finished_at"]

    def get_architecture_name(self, obj):
        from ml.registry import get_architecture

        meta = get_architecture(obj.architecture)
        return meta["name"] if meta else obj.architecture

    def get_trained_model_id(self, obj):
        return obj.trained_model_id


class TrainingRunCreateSerializer(serializers.ModelSerializer):
    architecture = serializers.ChoiceField(choices=[(key, key) for key in architecture_keys()])
    dataset = serializers.PrimaryKeyRelatedField(queryset=Dataset.objects.all())

    class Meta:
        model = TrainingRun
        fields = ["name", "architecture", "dataset", "epochs", "batch_size", "learning_rate",
                  "weight_decay", "optimizer", "scheduler", "image_size", "num_workers",
                  "val_split", "label_smoothing", "seed", "device", "augment",
                  "max_batches_per_epoch", "class_weighted_loss"]

    def validate_epochs(self, value):
        if not 1 <= value <= 200:
            raise serializers.ValidationError("训练轮数需在 1~200 之间")
        return value

    def validate_batch_size(self, value):
        if not 1 <= value <= 512:
            raise serializers.ValidationError("批大小需在 1~512 之间")
        return value

    def validate_image_size(self, value):
        if not 16 <= value <= 512:
            raise serializers.ValidationError("输入尺寸需在 16~512 之间")
        return value

    def validate_learning_rate(self, value):
        if not 0 < value <= 1:
            raise serializers.ValidationError("学习率需在 (0, 1] 之间")
        return value

    def validate_val_split(self, value):
        if not 0 <= value < 1:
            raise serializers.ValidationError("验证集比例需在 [0, 1) 之间")
        return value

    def validate_device(self, value):
        device = normalize_device(value)
        if device not in SUPPORTED_DEVICES:
            raise serializers.ValidationError("计算设备仅支持 GPU、CPU 或 MPS")
        if not is_device_available(device):
            label = {"gpu": "GPU", "mps": "MPS"}.get(device, device.upper())
            raise serializers.ValidationError(f"当前环境不支持 {label} 训练，请选择其它设备")
        return device

    def validate(self, attrs):
        dataset = attrs["dataset"]
        if dataset.image_count < 4:
            raise serializers.ValidationError({"dataset": "数据集样本过少（至少 4 张图片）"})
        if len(dataset.class_names or []) < 2:
            raise serializers.ValidationError({"dataset": "数据集至少需要 2 个类别"})
        arch = attrs["architecture"]
        from ml.registry import get_architecture

        meta = get_architecture(arch)
        if meta and attrs["image_size"] < meta["min_input_size"]:
            raise serializers.ValidationError(
                {"image_size": f"{meta['name']} 的输入尺寸建议不小于 {meta['min_input_size']}"})
        if attrs.get("max_batches_per_epoch", 0) and attrs["max_batches_per_epoch"] < 1:
            raise serializers.ValidationError({"max_batches_per_epoch": "该值需为 0 或正整数"})
        return attrs


class TrainedModelSerializer(serializers.ModelSerializer):
    dataset_name = serializers.CharField(source="dataset.name", read_only=True, default="")
    architecture_name = serializers.SerializerMethodField()
    file_name = serializers.SerializerMethodField()
    file_size_kb = serializers.SerializerMethodField()

    class Meta:
        model = TrainedModel
        fields = ["id", "name", "architecture", "architecture_name", "dataset", "dataset_name",
                  "training_run", "classes", "num_classes", "input_size", "accuracy", "macro_f1",
                  "metrics", "params_million", "file_path", "file_size", "file_size_kb", "file_name",
                  "is_active", "notes", "created_at", "updated_at"]
        read_only_fields = ["file_path", "updated_at"]

    def get_architecture_name(self, obj):
        from ml.registry import get_architecture

        meta = get_architecture(obj.architecture)
        return meta["name"] if meta else obj.architecture

    def get_file_name(self, obj):
        return obj.resolved_path.name if obj.file_path else ""

    def get_file_size_kb(self, obj):
        return round((obj.file_size or 0) / 1024, 1)


class PredictionRecordSerializer(serializers.ModelSerializer):
    image_url = serializers.SerializerMethodField()
    username = serializers.SerializerMethodField()

    class Meta:
        model = PredictionRecord
        fields = ["id", "group_id", "user", "username", "image_url", "original_name", "source",
                  "architecture", "model_name", "trained_model", "predicted_label", "confidence",
                  "topk", "latency_ms", "created_at"]

    def get_username(self, obj):
        return obj.user.username if obj.user_id else "—"

    def get_image_url(self, obj):
        request = self.context.get("request")
        if not obj.image:
            return ""
        url = obj.image.url
        return request.build_absolute_uri(url) if request else url


class UserSerializer(serializers.ModelSerializer):
    """用户列表 / 详情（只读展示，含角色与识别次数）。"""

    display_name = serializers.SerializerMethodField()
    role = serializers.SerializerMethodField()
    record_count = serializers.SerializerMethodField()

    class Meta:
        model = User
        fields = ["id", "username", "display_name", "email", "is_staff", "is_active", "role",
                  "record_count", "date_joined", "last_login"]

    def get_display_name(self, obj):
        return obj.get_full_name() or obj.username

    def get_role(self, obj):
        return "admin" if obj.is_staff else "user"

    def get_record_count(self, obj):
        return obj.prediction_records.count()


class UserWriteSerializer(serializers.ModelSerializer):
    """新建 / 修改用户：由管理员操作。"""

    ROLE_CHOICES = [("admin", "管理员"), ("user", "普通用户")]

    password = serializers.CharField(write_only=True, required=False, min_length=6,
                                     allow_blank=True, help_text="留空表示不修改密码")
    role = serializers.ChoiceField(choices=ROLE_CHOICES, required=False, default="user")
    display_name = serializers.CharField(required=False, allow_blank=True, max_length=60)

    class Meta:
        model = User
        fields = ["id", "username", "password", "display_name", "email", "role", "is_active"]
        read_only_fields = ["id"]

    def validate_username(self, value):
        value = (value or "").strip()
        if not value:
            raise serializers.ValidationError("用户名不能为空")
        query = User.objects.filter(username__iexact=value)
        if self.instance:
            query = query.exclude(pk=self.instance.pk)
        if query.exists():
            raise serializers.ValidationError("该用户名已存在")
        return value

    def validate_email(self, value):
        value = (value or "").strip()
        if not value:
            return value
        query = User.objects.filter(email__iexact=value)
        if self.instance:
            query = query.exclude(pk=self.instance.pk)
        if query.exists():
            raise serializers.ValidationError("该邮箱已被其它账号使用")
        return value

    def create(self, validated_data):
        password = validated_data.pop("password", "") or "123456"
        role = validated_data.pop("role", "user")
        display_name = validated_data.pop("display_name", "")
        user = User.objects.create_user(
            username=validated_data["username"],
            password=password,
            email=validated_data.get("email", ""),
            is_staff=(role == "admin"),
            is_active=validated_data.get("is_active", True),
            first_name=display_name or validated_data["username"],
        )
        return user

    def update(self, instance, validated_data):
        password = validated_data.pop("password", "")
        role = validated_data.pop("role", None)
        display_name = validated_data.pop("display_name", None)

        for field, value in validated_data.items():
            setattr(instance, field, value)
        if display_name is not None:
            instance.first_name = display_name or instance.username
        if role is not None:
            instance.is_staff = role == "admin"
        if password:
            instance.set_password(password)
        instance.save()
        return instance

    def get_image_url(self, obj):
        request = self.context.get("request")
        if not obj.image:
            return ""
        url = obj.image.url
        return request.build_absolute_uri(url) if request else url
