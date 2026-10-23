"""Django 配置：数据库、跨域、上传、深度学习资源目录等。"""

from __future__ import annotations

import os
from pathlib import Path

from django.core.exceptions import ImproperlyConfigured

BASE_DIR = Path(__file__).resolve().parent.parent      # backend/
PROJECT_ROOT = BASE_DIR.parent                         # 仓库根目录

SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "dev-only-secret-key-change-me-in-production")
DEBUG = os.environ.get("DJANGO_DEBUG", "1") == "1"
ALLOWED_HOSTS = ["*"]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "rest_framework",
    "rest_framework.authtoken",
    "corsheaders",
    "apps.recognition",
]

MIDDLEWARE = [
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"
ASGI_APPLICATION = "config.asgi.application"

# ------------------------------------------------------------------ 数据库配置
# 默认使用 SQLite（零配置、便于演示）；设置环境变量 IR_DB_ENGINE=mysql 即可切换到 MySQL，
# 无需改动任何业务代码（Django ORM 屏蔽了数据库差异）：
#
#   export IR_DB_ENGINE=mysql
#   export IR_DB_NAME=image_recognition
#   export IR_DB_USER=root
#   export IR_DB_PASSWORD=你的密码
#   export IR_DB_HOST=127.0.0.1
#   export IR_DB_PORT=3306
#
# 驱动优先使用 mysqlclient，未安装时自动回退到纯 Python 的 PyMySQL（无需编译）。
#DB_ENGINE = os.environ.get("IR_DB_ENGINE", "sqlite").strip().lower()

#使用mysql数据库
DB_ENGINE = os.environ.get("IR_DB_ENGINE", "mysql").strip().lower()

if DB_ENGINE in {"mysql", "mariadb"}:
    try:  # mysqlclient
        import MySQLdb  # noqa: F401
    except ImportError:
        try:  # 纯 Python 驱动，通过 install_as_MySQLdb() 兼容 Django
            import pymysql

            pymysql.install_as_MySQLdb()
        except ImportError as exc:  # pragma: no cover
            raise ImproperlyConfigured(
                "使用 MySQL 需要先安装驱动：pip install mysqlclient（或 pip install pymysql）"
            ) from exc

    #mysql数据库的配置
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.mysql",
            "NAME": os.environ.get("IR_DB_NAME", "image_recognition"),
            #"USER": os.environ.get("IR_DB_USER", "root"),
            #"PASSWORD": os.environ.get("IR_DB_PASSWORD", ""),
            "USER": os.environ.get("IR_DB_USER", "ir_user"),
            "PASSWORD": os.environ.get("IR_DB_PASSWORD", "ir_pass_2026"),
            "HOST": os.environ.get("IR_DB_HOST", "127.0.0.1"),
            "PORT": os.environ.get("IR_DB_PORT", "3306"),
            "CONN_MAX_AGE": 60,
            "OPTIONS": {
                # 中文（校名、类别名）必须使用 utf8mb4，否则会出现乱码或写入失败
                "charset": "utf8mb4",
                "init_command": "SET sql_mode='STRICT_TRANS_TABLES'",
            },
            # 模型里使用了 JSONField，MySQL 需要 5.7.8+（推荐 8.0+）
            "TEST": {"CHARSET": "utf8mb4", "COLLATION": "utf8mb4_unicode_ci"},
        }
    }
else:
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": BASE_DIR / "db.sqlite3",
            "OPTIONS": {"timeout": 30},
        }
    }

AUTH_PASSWORD_VALIDATORS = []

LANGUAGE_CODE = "zh-hans"
TIME_ZONE = "Asia/Shanghai"
USE_I18N = True
USE_TZ = True

STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# ------------------------------------------------------------------ 深度学习资源目录
DATA_DIR = Path(os.environ.get("IR_DATA_DIR", BASE_DIR / "data"))
DATASET_DIR = DATA_DIR / "datasets"       # 数据集根目录
WEIGHTS_DIR = DATA_DIR / "weights"        # 训练产出的权重目录
UPLOAD_DIR = DATA_DIR / "uploads"         # 上传数据集压缩包的临时目录
for _directory in (DATA_DIR, DATASET_DIR, WEIGHTS_DIR, UPLOAD_DIR, MEDIA_ROOT):
    _directory.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ 上传限制
MAX_IMAGE_UPLOAD_SIZE = 10 * 1024 * 1024      # 单张识别图片 10MB
MAX_DATASET_UPLOAD_SIZE = 500 * 1024 * 1024   # 数据集压缩包 500MB
DATA_UPLOAD_MAX_MEMORY_SIZE = MAX_DATASET_UPLOAD_SIZE
FILE_UPLOAD_MAX_MEMORY_SIZE = 20 * 1024 * 1024

# ------------------------------------------------------------------ DRF
REST_FRAMEWORK = {
    "DEFAULT_RENDERER_CLASSES": (
        ["rest_framework.renderers.JSONRenderer", "rest_framework.renderers.BrowsableAPIRenderer"]
        if DEBUG else ["rest_framework.renderers.JSONRenderer"]
    ),
    "DEFAULT_PARSER_CLASSES": (
        "rest_framework.parsers.JSONParser",
        "rest_framework.parsers.FormParser",
        "rest_framework.parsers.MultiPartParser",
    ),
    "DEFAULT_PAGINATION_CLASS": "apps.recognition.pagination.StandardPagination",
    "PAGE_SIZE": 10,
    "DATETIME_FORMAT": "%Y-%m-%d %H:%M:%S",
    "DEFAULT_AUTHENTICATION_CLASSES": (
        "rest_framework.authentication.TokenAuthentication",
        "rest_framework.authentication.SessionAuthentication",
    ),
    # 除登录、健康检查等公开接口外，其余接口都需要登录后访问
    "DEFAULT_PERMISSION_CLASSES": ("rest_framework.permissions.IsAuthenticated",),
}

# ------------------------------------------------------------------ 站点与账号
SITE_NAME = "基于多种深度学习经典模型的视觉学习系统"
UNIVERSITY_NAME = "新疆师范大学"
UNIVERSITY_EN_NAME = "Xinjiang Normal University"
UNIVERSITY_BADGE = "XJNU · 1978"
# 是否在登录页展示演示账号（仅用于本地演示与答辩，部署到公网请设为 False）
SHOW_DEMO_ACCOUNT = os.environ.get("IR_SHOW_DEMO_ACCOUNT", "1") == "1"
DEMO_ACCOUNTS = [
    {"username": "admin", "password": "admin123", "name": "系统管理员", "role": "管理员",
     "note": "数据集 / 训练 / 模型全权限"},
    {"username": "student", "password": "123456", "name": "演示学生", "role": "普通用户",
     "note": "识别图片、查看历史记录"},
]

# ------------------------------------------------------------------ 跨域（开发环境放开，前端 5173 端口）
CORS_ALLOW_ALL_ORIGINS = True
CORS_ALLOW_CREDENTIALS = True

# ------------------------------------------------------------------ 前端构建产物（可选，用于单服务部署）
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {"simple": {"format": "[{levelname}] {asctime} {name}: {message}", "style": "{"}},
    "handlers": {"console": {"class": "logging.StreamHandler", "formatter": "simple"}},
    "root": {"handlers": ["console"], "level": "INFO"},
}
