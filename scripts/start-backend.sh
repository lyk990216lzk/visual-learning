#!/usr/bin/env bash
# 启动 Django 开发服务器（API + 可选的已构建前端）
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/backend"

if [ -d "$ROOT/.venv" ]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi

PORT="${1:-8000}"
DB_KIND="${IR_DB_ENGINE:-sqlite}"
case "$DB_KIND" in
  mysql|mariadb)
    echo "数据库：MySQL（${IR_DB_NAME:-image_recognition} @ ${IR_DB_USER:-root}@${IR_DB_HOST:-127.0.0.1}:${IR_DB_PORT:-3306}）"
    ;;
  *)
    echo "数据库：SQLite（backend/db.sqlite3，默认；要改用 MySQL 请设置 IR_DB_ENGINE=mysql）"
    ;;
esac
echo "API 服务启动中： http://127.0.0.1:${PORT}/api/"
if [ -d "$ROOT/frontend/dist" ]; then
  echo "检测到前端构建产物，可直接访问： http://127.0.0.1:${PORT}/"
fi
python manage.py runserver "${PORT}"
