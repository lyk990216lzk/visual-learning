"""命令行训练：python manage.py train -a resnet18 -d 演示数据集-几何图形 -e 5"""

from __future__ import annotations

import time

from django.core.management.base import BaseCommand, CommandError

from apps.recognition.models import Dataset, TrainingRun
from apps.recognition.serializers import TrainingRunCreateSerializer
from ml.registry import get_architecture, implemented_architecture_keys
from ml.trainer import manager


class Command(BaseCommand):
    help = "创建并执行一个训练任务（前台阻塞，实时打印进度）"

    def add_arguments(self, parser):
        parser.add_argument("--task", default="classification",
                            choices=["classification", "detection", "segmentation"])
        parser.add_argument("--architecture", "-a", required=True,
                            choices=implemented_architecture_keys())
        parser.add_argument("--dataset", "-d", required=True, help="数据集名称或 id")
        parser.add_argument("--epochs", "-e", type=int, default=5)
        parser.add_argument("--batch-size", type=int, default=32)
        parser.add_argument("--lr", type=float, default=0.01)
        parser.add_argument("--image-size", type=int, default=64)
        parser.add_argument("--optimizer", default="sgd", choices=["sgd", "adam", "adamw"])
        parser.add_argument("--scheduler", default="cosine", choices=["none", "step", "cosine"])
        parser.add_argument("--device", default="auto",
                            choices=["auto", "gpu", "cpu", "cuda", "mps"])
        parser.add_argument("--max-batches", type=int, default=0, help="每轮最多训练的 batch 数（0 表示全部）")
        parser.add_argument("--name", default="", help="任务名称")
        parser.add_argument("--quiet", action="store_true", help="只输出关键结果")

    def handle(self, *args, **options):
        dataset = None
        key = options["dataset"]
        if key.isdigit():
            dataset = Dataset.objects.filter(pk=int(key)).first()
        dataset = dataset or Dataset.objects.filter(name=key).first()
        if dataset is None:
            raise CommandError(f"找不到数据集：{key}")

        payload = {
            "name": options["name"] or f"{options['architecture']}@{dataset.name}",
            "task_type": options["task"],
            "architecture": options["architecture"],
            "dataset": dataset.id,
            "epochs": options["epochs"],
            "batch_size": options["batch_size"],
            "learning_rate": options["lr"],
            "image_size": options["image_size"],
            "optimizer": options["optimizer"],
            "scheduler": options["scheduler"],
            "device": options["device"],
            "max_batches_per_epoch": options["max_batches"],
        }
        serializer = TrainingRunCreateSerializer(data=payload)
        serializer.is_valid(raise_exception=True)
        run = serializer.save()
        meta = get_architecture(run.architecture)
        self.stdout.write(self.style.HTTP_INFO(
            f"开始训练 #{run.id}：{meta['name']} / {dataset.name} / {run.epochs} epochs / 输入 {run.image_size}"
        ))
        manager.start(run.id)

        last_line = None
        while True:
            run.refresh_from_db()
            if run.log:
                lines = run.log.splitlines()
                if lines and lines[-1] != last_line and not options["quiet"]:
                    last_line = lines[-1]
                    self.stdout.write(last_line)
            if run.is_finished:
                break
            time.sleep(1.0)

        if run.status == TrainingRun.Status.SUCCESS:
            self.stdout.write(self.style.SUCCESS(
                f"训练完成：验证准确率 {run.metrics.get('accuracy', 0):.4f}，"
                f"宏平均 F1 {run.metrics.get('macro_f1', 0):.4f}，耗时 {run.duration_seconds:.1f}s，"
                f"模型 ID {run.trained_model_id}"
            ))
        else:
            self.stdout.write(self.style.ERROR(f"训练未成功完成：{run.status} {run.error}"))
