"""托管前端构建产物（npm run build 之后可只启动 Django 单服务演示）。"""

from __future__ import annotations

from pathlib import Path

from django.conf import settings
from django.http import FileResponse, Http404


def _safe_file(dist: Path, relative: str) -> Path | None:
    candidate = (dist / relative).resolve()
    if dist.resolve() in candidate.parents and candidate.is_file():
        return candidate
    return None


def spa_view(request, path: str = ""):
    """静态资源直接返回文件，其余路径回退到 index.html（Vue Router history 模式）。"""
    dist: Path = settings.FRONTEND_DIST
    if not dist.exists():
        raise Http404("前端尚未构建，请先在 frontend 目录执行 npm run build")

    file = _safe_file(dist, path) if path else None
    if file is not None:
        return FileResponse(open(file, "rb"))

    index = dist / "index.html"
    if not index.exists():
        raise Http404("index.html 不存在，请重新执行 npm run build")
    return FileResponse(open(index, "rb"), content_type="text/html")
