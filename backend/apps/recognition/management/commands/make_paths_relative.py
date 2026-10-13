"""把已入库的绝对路径统一转换成相对 ``backend/`` 的相对路径。

    python manage.py make_paths_relative            # 实际写入
    python manage.py make_paths_relative --dry-run  # 只预览

幂等，可重复执行；SQLite 与 MySQL 通用（走 ORM，与具体数据库无关）。
"""

from __future__ import annotations

from django.core.management.base import BaseCommand

from apps.recognition.models import Dataset, TrainedModel
from apps.recognition.paths import to_stored_path


class Command(BaseCommand):
    help = "把数据集 / 权重路径转成相对 backend/ 的相对路径（幂等，项目改名或搬迁后无需改数据）"

    def add_arguments(self, parser):
        parser.add_argument("--dry-run", action="store_true", help="只显示将要修改的条目，不写库")

    def handle(self, *args, **options):
        dry_run = options["dry_run"]
        changed = 0
        scanned = 0

        for model, field in ((Dataset, "path"), (TrainedModel, "file_path")):
            for obj in model.objects.all().iterator():
                scanned += 1
                old = getattr(obj, field) or ""
                new = to_stored_path(old)
                if new == old:
                    continue
                changed += 1
                self.stdout.write(f"  {model.__name__}#{obj.pk} ({obj}):\n    - {old}\n    + {new}")
                if not dry_run:
                    setattr(obj, field, new)
                    obj.save(update_fields=[field, "updated_at"])

        prefix = "将修改" if dry_run else "已修改"
        self.stdout.write(self.style.SUCCESS(
            f"扫描 {scanned} 条记录，{prefix} {changed} 条路径"
            + ("（--dry-run，未写库）" if dry_run else "")
        ))
