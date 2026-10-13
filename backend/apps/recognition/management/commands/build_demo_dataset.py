"""生成离线演示数据集：python manage.py build_demo_dataset --per-class 240 --size 64"""

from __future__ import annotations

import shutil
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand

from apps.recognition.models import Dataset
from ml.datasets import build_demo_dataset


class Command(BaseCommand):
    help = "生成合成几何图形演示数据集（离线可用），并登记到数据库"

    def add_arguments(self, parser):
        parser.add_argument("--name", default="演示数据集-几何图形", help="数据集名称")
        parser.add_argument("--per-class", type=int, default=240, help="每个类别的图片数量")
        parser.add_argument("--size", type=int, default=64, help="图片边长（像素）")

    def handle(self, *args, **options):
        name = options["name"]
        target = Path(settings.DATASET_DIR) / name.replace(" ", "_")
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        self.stdout.write(f"正在生成数据集 -> {target}")
        info = build_demo_dataset(target, per_class=options["per_class"], image_size=options["size"])

        dataset, created = Dataset.objects.get_or_create(
            name=name,
            defaults={"description": "合成几何图形数据集（圆形/正方形/三角形/十字/五角星/圆环）",
                      "path": str(target), "is_demo": True},
        )
        dataset.path = str(target)
        dataset.description = "合成几何图形数据集（圆形/正方形/三角形/十字/五角星/圆环）"
        dataset.is_demo = True
        dataset.refresh_stats()
        self.stdout.write(self.style.SUCCESS(
            f"{'已创建' if created else '已更新'}数据集「{name}」：{info['image_count']} 张图片，"
            f"{len(info['classes'])} 个类别 -> {info['classes']}"
        ))
