"""数据集接口：列表、压缩包上传、演示数据集生成、样本预览。"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from django.conf import settings
from django.http import FileResponse, Http404
from rest_framework import status
from rest_framework.decorators import action
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.viewsets import ModelViewSet

from ml.datasets import IMAGE_EXTENSIONS, build_demo_dataset
from ml.task_datasets import DatasetFormatError, TASK_LABELS, TASK_TYPES, scan_dataset

from ..models import Dataset, TrainingRun
from ..permissions import IsAdminOrReadOnly
from ..serializers import DatasetSerializer


def _safe_dir_name(name: str) -> str:
    cleaned = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in name.strip())
    return cleaned or "dataset"


class DatasetViewSet(ModelViewSet):
    """数据集管理：/api/datasets/"""

    queryset = Dataset.objects.all()
    serializer_class = DatasetSerializer
    parser_classes = [JSONParser, FormParser, MultiPartParser]
    # 上传 / 删除 / 生成演示数据集 = 管理员；浏览与样本预览 = 所有登录用户
    permission_classes = [IsAuthenticated, IsAdminOrReadOnly]

    def create(self, request, *args, **kwargs):
        """上传 zip 数据集（内部结构为 ``<类别>/*.jpg`` 或 ``train/``、``val/`` 子目录）。"""
        payload = request.data
        name = (payload.get("name") or "").strip()
        task_type = (payload.get("task_type") or Dataset.TaskType.CLASSIFICATION).strip().lower()
        upload = request.FILES.get("file") or request.FILES.get("archive")
        if not name:
            return Response({"detail": "请填写数据集名称"}, status=status.HTTP_400_BAD_REQUEST)
        if task_type not in TASK_TYPES:
            return Response(
                {
                    "detail": "请选择正确的数据集功能",
                    "errors": ["功能必须为图片识别、目标检测或语义分割"],
                },
                status=status.HTTP_400_BAD_REQUEST,
            )
        if Dataset.objects.filter(name=name).exists():
            return Response({"detail": f"数据集「{name}」已存在"}, status=status.HTTP_400_BAD_REQUEST)
        if upload is None:
            return Response({"detail": "请上传 zip 压缩包"}, status=status.HTTP_400_BAD_REQUEST)
        if not upload.name.lower().endswith(".zip"):
            return Response({"detail": "目前仅支持 zip 压缩包"}, status=status.HTTP_400_BAD_REQUEST)
        if upload.size > settings.MAX_DATASET_UPLOAD_SIZE:
            return Response({"detail": "压缩包超过 500MB 限制"}, status=status.HTTP_400_BAD_REQUEST)

        target = Path(settings.DATASET_DIR) / _safe_dir_name(name)
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        temp_zip = Path(settings.UPLOAD_DIR) / f"{_safe_dir_name(name)}.zip"
        target.mkdir(parents=True, exist_ok=True)
        try:
            with open(temp_zip, "wb") as handle:
                for chunk in upload.chunks():
                    handle.write(chunk)
            with zipfile.ZipFile(temp_zip) as archive:
                for member in archive.namelist():
                    member_path = (target / member).resolve()
                    if not str(member_path).startswith(str(target.resolve())):
                        raise ValueError("压缩包内包含非法路径")
                archive.extractall(target)
        except zipfile.BadZipFile:
            shutil.rmtree(target, ignore_errors=True)
            return Response({"detail": "压缩包损坏或不是合法的 zip 文件"},
                            status=status.HTTP_400_BAD_REQUEST)
        except ValueError as exc:
            shutil.rmtree(target, ignore_errors=True)
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        finally:
            temp_zip.unlink(missing_ok=True)

        # 压缩包内只有一层目录时，把它当作数据集根目录
        entries = [item for item in target.iterdir() if not item.name.startswith("__")]
        root = target
        if len(entries) == 1 and entries[0].is_dir():
            root = entries[0]

        try:
            scan_dataset(root, task_type)
        except DatasetFormatError as exc:
            shutil.rmtree(target, ignore_errors=True)
            return Response(
                {
                    "detail": f"数据集格式不符合{TASK_LABELS[task_type]}要求",
                    "task_type": task_type,
                    "errors": exc.errors,
                    "hint": exc.hint,
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        dataset = Dataset.objects.create(
            name=name,
            task_type=task_type,
            description=payload.get("description", ""),
            path=str(root),
            is_demo=False,
        )
        dataset.refresh_stats()
        return Response(DatasetSerializer(dataset).data, status=status.HTTP_201_CREATED)

    def destroy(self, request, *args, **kwargs):
        """删除数据集记录；若目录位于受管的数据集根目录内则一并清理。"""
        dataset = self.get_object()
        if TrainingRun.objects.filter(dataset=dataset, status=TrainingRun.Status.RUNNING).exists():
            return Response({"detail": "该数据集上有正在运行的训练任务，请先中止"},
                            status=status.HTTP_400_BAD_REQUEST)
        path = dataset.dir_path
        dataset.delete()
        try:
            dataset_root = Path(settings.DATASET_DIR).resolve()
            if path.exists() and str(path.resolve()).startswith(str(dataset_root)):
                shutil.rmtree(path, ignore_errors=True)
        except OSError:
            pass
        return Response(status=status.HTTP_204_NO_CONTENT)

    @action(detail=False, methods=["post"], url_path="build-demo")
    def build_demo(self, request):
        """一键生成演示数据集（离线可用的合成几何图形），方便快速体验完整流程。"""
        per_class = int(request.data.get("per_class", 240))
        image_size = int(request.data.get("image_size", 64))
        force = str(request.data.get("force", "true")).lower() in {"1", "true", "yes"}
        name = request.data.get("name") or "演示数据集-几何图形"

        per_class = max(20, min(per_class, 2000))
        image_size = max(32, min(image_size, 224))

        existing = Dataset.objects.filter(name=name).first()
        if existing and not force:
            return Response({"detail": f"数据集「{name}」已存在"}, status=status.HTTP_400_BAD_REQUEST)

        target = Path(settings.DATASET_DIR) / _safe_dir_name(name)
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        info = build_demo_dataset(target, per_class=per_class, image_size=image_size)

        if existing:
            existing.path = str(target)
            existing.task_type = Dataset.TaskType.CLASSIFICATION
            existing.description = "由系统生成的合成几何图形数据集（圆形/正方形/三角形/十字/五角星/圆环）"
            existing.refresh_stats()
            dataset = existing
        else:
            dataset = Dataset.objects.create(
                name=name,
                task_type=Dataset.TaskType.CLASSIFICATION,
                description="由系统生成的合成几何图形数据集（圆形/正方形/三角形/十字/五角星/圆环）",
                path=str(target),
                is_demo=True,
            )
            dataset.refresh_stats()
        return Response(DatasetSerializer(dataset).data, status=status.HTTP_201_CREATED)

    @action(detail=True, methods=["post"])
    def refresh(self, request, pk=None):
        """重新扫描数据集目录，刷新类别与样本数量统计。"""
        dataset = self.get_object()
        try:
            info = dataset.refresh_stats()
        except DatasetFormatError as exc:
            return Response(
                {
                    "detail": "数据集目录已不符合当前功能要求",
                    "errors": exc.errors,
                    "hint": exc.hint,
                },
                status=status.HTTP_400_BAD_REQUEST,
            )
        return Response({"dataset": DatasetSerializer(dataset).data, "info": info})

    @action(detail=True, methods=["get"])
    def samples(self, request, pk=None):
        """返回若干张样本图片的访问地址，供前端预览网格使用。"""
        dataset = self.get_object()
        limit = int(request.query_params.get("limit", 12))
        root = dataset.dir_path
        samples = []
        if dataset.task_type != Dataset.TaskType.CLASSIFICATION:
            for split in ("val", "train"):
                image_dir = root / "images" / split
                if not image_dir.exists():
                    continue
                images = [p for p in sorted(image_dir.rglob("*"))
                          if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
                for image in images[:max(1, limit)]:
                    samples.append({
                        "label": split,
                        "name": image.relative_to(image_dir).as_posix(),
                        "url": f"/api/datasets/{dataset.id}/image/?label={split}&name={image.relative_to(image_dir).as_posix()}",
                    })
                break
            return Response({"count": len(samples), "results": samples[:limit]})

        for label in dataset.class_names or []:
            class_dir = root / "train" / label if (root / "train" / label).exists() else root / label
            if not class_dir.exists():
                continue
            images = [p for p in sorted(class_dir.iterdir())
                      if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS][:max(1, limit // max(1, len(dataset.class_names)))]
            for image in images:
                samples.append({
                    "label": label,
                    "name": image.name,
                    "url": f"/api/datasets/{dataset.id}/image/?label={label}&name={image.name}",
                })
        return Response({"count": len(samples), "results": samples[:limit]})

    @action(detail=True, methods=["get"])
    def image(self, request, pk=None):
        """按类别 + 文件名读取数据集中的原始图片。"""
        dataset = self.get_object()
        label = request.query_params.get("label", "")
        name = request.query_params.get("name", "")
        name_path = Path(name)
        if (not label or not name or name_path.is_absolute() or ".." in name_path.parts
                or "/" in label):
            raise Http404("参数不合法")
        root = dataset.dir_path
        if dataset.task_type != Dataset.TaskType.CLASSIFICATION:
            if label not in {"train", "val", "test"}:
                raise Http404("参数不合法")
            image_root = (root / "images" / label).resolve()
            candidate = (image_root / name_path).resolve()
            if candidate.is_relative_to(image_root) and candidate.exists() and candidate.is_file():
                return FileResponse(open(candidate, "rb"))
            raise Http404("样本图片不存在")
        if "/" in name:
            raise Http404("参数不合法")
        candidates = [root / "train" / label / name, root / label / name,
                      root / "val" / label / name]
        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return FileResponse(open(candidate, "rb"))
        raise Http404("样本图片不存在")
