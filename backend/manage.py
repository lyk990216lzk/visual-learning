#!/usr/bin/env python
"""Django 命令行入口。"""

import os
import sys


def main() -> None:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "无法导入 Django，请先激活虚拟环境并执行 pip install -r requirements.txt"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()

'''
cd /Users/liuyakun/Documents/Codex/2026-09-22/recognition/backend

# SQLite（默认）
uv run python3 manage.py runserver 8000

# MySQL 模式
IR_DB_ENGINE=mysql IR_DB_NAME=image_recognition IR_DB_USER=ir_user IR_DB_PASSWORD=ir_pass_2026 \
  uv run python3 manage.py runserver 8000
  
# 只用 SQLite（零配置，不用管 MySQL）
bash scripts/start-backend.sh

# 用 MySQL
IR_DB_ENGINE=mysql IR_DB_USER=ir_user IR_DB_PASSWORD=ir_pass_2026 bash scripts/start-backend.sh
# ↑ 就是这行比较长 —— 用上面 ①②③ 任一方法都行

# 检查 MySQL 容器在不在
docker compose ps
lsof -nP -iTCP:3306 -sTCP:LISTEN



改进：
    1、添加日志
    2、图像识别模块：添加数据集，生成图片，可以快速推理；对应的数据集有对应的模型，前端返回有哪些种类
    3、模型训练模块：不同的数据集生成不同的模型权重
    4、视觉学习：图像识别，目标检测，语义分割，图像分割，行为检测
    5、端口映射
    6、容器化
    7、脚本
    8、摄像头拍照，参数设置
    9、gitlee
'''