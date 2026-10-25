# 单容器运行方案：Django API + 已构建的 Vue 页面。
# CUDA 机器运行时仍需通过 --gpus all 或 Compose 的 gpus 配置透传 GPU。
FROM python:3.10-slim-bookworm

# 使用与项目锁文件对应的 uv 版本。
COPY --from=ghcr.io/astral-sh/uv:0.11.17 /uv /uvx /bin/

# 单容器默认使用 SQLite；连接 MySQL 时可通过环境变量覆盖。
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON=3.10 \
    TZ=Asia/Shanghai \
    PATH=/opt/venv/bin:$PATH \
    DJANGO_SETTINGS_MODULE=config.settings \
    IR_DB_ENGINE=sqlite

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender1 \
        tzdata \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先安装依赖，使业务代码变化时仍可复用依赖层。
COPY pyproject.toml uv.lock /app/
RUN uv sync --frozen --no-dev --no-install-project

# 仅复制运行所需代码，不把 .venv、数据集、媒体文件、Node 依赖和 work 缓存带入镜像。
COPY backend/manage.py /app/backend/manage.py
COPY backend/config /app/backend/config
COPY backend/apps /app/backend/apps
COPY backend/ml /app/backend/ml

# 单端口方案由 Django 托管已经构建好的前端。
COPY frontend/dist /app/frontend/dist

RUN mkdir -p /app/backend/data /app/backend/media \
    && ln -s /app/backend/data/db.sqlite3 /app/backend/db.sqlite3

VOLUME ["/app/backend/data", "/app/backend/media"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health/', timeout=3)"]

WORKDIR /app/backend

# 自动执行数据库迁移和默认账号初始化，然后启动单进程 Django。
CMD ["sh", "-c", "uv run --no-sync python manage.py migrate --noinput && uv run --no-sync python manage.py create_default_user && exec uv run --no-sync python manage.py runserver 0.0.0.0:8000 --noreload"]
