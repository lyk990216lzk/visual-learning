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
    import torch

    print(torch.backends.mps.is_built(), torch.backends.mps.is_available())

'''
cd /Users/liuyakun/Documents/Codex/2026-09-22/recognition/backend
cd /Users/liuyakun/Desktop/高级软件工程/大作业/recognition/backend

# SQLite（默认）
uv run python3 manage.py runserver 8000

# MySQL 模式
IR_DB_ENGINE=mysql IR_DB_NAME=image_recognition IR_DB_USER=ir_user IR_DB_PASSWORD=ir_pass_2026 \
  uv run python3 manage.uv run python3 manage.py runserver 8000py runserver 8000
  
# 只用 SQLite（零配置，不用管 MySQL）
bash scripts/start-backend.sh

# 用 MySQL
IR_DB_ENGINE=mysql IR_DB_USER=ir_user IR_DB_PASSWORD=ir_pass_2026 bash scripts/start-backend.sh
# ↑ 就是这行比较长 —— 用上面 ①②③ 任一方法都行

# 检查 MySQL 容器在不在
docker compose ps
lsof -nP -iTCP:3306 -sTCP:LISTEN

#dokcerfile文件进行启动构建的命令：docker compose up -d --build
    这一条命令会自动完成：
        1. 根据 Dockerfile 构建应用镜像。
        2. 拉取 mysql:8.0，如果本地没有的话。
        3. 创建 recognition_default 网络。
        4. 创建 MySQL、数据集和媒体文件数据卷。
        5. 启动 MySQL 容器。
        6. 等待 MySQL 健康检查通过。
        7. 启动应用容器。
        8. 自动执行 manage.py migrate。
        9. 自动创建演示账号。
        10. 在 8000 端口启动 Django。

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
