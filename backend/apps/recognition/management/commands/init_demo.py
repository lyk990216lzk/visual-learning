"""一键初始化演示环境：生成演示数据集 +（可选）训练几个基线模型。

    python manage.py init_demo            # 只创建数据集
    python manage.py init_demo --train    # 数据集 + 快速训练 lenet5 / alexnet / resnet18
"""

from __future__ import annotations

import time

from django.core.management import call_command
from django.core.management.base import BaseCommand

from apps.recognition.models import Dataset, TrainingRun
from apps.recognition.serializers import TrainingRunCreateSerializer
from ml.trainer import manager

DEFAULT_MODELS = ["lenet5", "alexnet", "resnet18"]


class Command(BaseCommand):
    help = "初始化演示数据（数据集 + 可选的基线模型训练）"

    def add_arguments(self, parser):
        parser.add_argument("--train", action="store_true", help="同时训练几个基线模型")
        parser.add_argument("--epochs", type=int, default=5)
        parser.add_argument("--per-class", type=int, default=180)
        parser.add_argument("--image-size", type=int, default=64)
        parser.add_argument("--architectures", nargs="*", default=DEFAULT_MODELS)

    def handle(self, *args, **options):
        call_command("build_demo_dataset", per_class=options["per_class"], size=options["image_size"])
        if not options["train"]:
            self.stdout.write(self.style.SUCCESS("演示数据集准备完成，可加 --train 继续训练基线模型"))
            return

        dataset = Dataset.objects.filter(is_demo=True).order_by("-created_at").first()
        for architecture in options["architectures"]:
            payload = {
                "name": f"基线-{architecture}",
                "architecture": architecture,
                "dataset": dataset.id,
                "epochs": options["epochs"],
                "batch_size": 32,
                "learning_rate": 0.01,
                "image_size": options["image_size"],
                "optimizer": "sgd",
                "scheduler": "cosine",
                "device": "auto",
            }
            serializer = TrainingRunCreateSerializer(data=payload)
            serializer.is_valid(raise_exception=True)
            run = serializer.save()
            self.stdout.write(f"训练 {architecture} ...")
            manager.start(run.id)
            while True:
                run.refresh_from_db()
                if run.is_finished:
                    break
                time.sleep(1.0)
            self.stdout.write(self.style.SUCCESS(
                f"  {architecture}: 准确率 {run.metrics.get('accuracy', 0):.4f}，状态 {run.status}"
            ))
