"""路径工具：数据库里统一保存相对 ``backend/`` 的相对路径，读取时再解析成绝对路径。

这样整个项目目录改名、搬迁或换机器都不需要修改已入库的数据。

- :func:`to_stored_path` 用于「写库前」把绝对路径转成相对形式（幂等，已是相对则原样返回）
- :func:`resolve_path` 用于「读出来」把相对路径拼回绝对路径（同样兼容历史遗留的绝对路径）
"""

from __future__ import annotations

import os
from pathlib import Path

from django.conf import settings


def _base_dir() -> Path:
    return Path(settings.BASE_DIR).resolve()


def resolve_path(value) -> Path:
    """把库里存的路径解析成绝对路径（空值原样返回，兼容历史绝对路径）。"""
    raw = str(value or "").strip()
    if not raw:
        return Path(raw)
    path = Path(os.path.expanduser(raw))
    return path if path.is_absolute() else _base_dir() / path


def to_stored_path(value) -> str:
    """把绝对路径转成入库形式：位于 ``backend/`` 之下存相对路径，其余原样保留。"""
    raw = str(value or "").strip()
    if not raw:
        return ""
    path = Path(os.path.expanduser(raw))
    if not path.is_absolute():
        return str(path)
    try:
        return str(path.resolve().relative_to(_base_dir()))
    except ValueError:
        return str(path)
