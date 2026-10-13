#!/usr/bin/env bash
# 把现有 SQLite 数据迁移到 MySQL：
#   1) 从 db.sqlite3 导出用户与业务数据
#   2) 在 MySQL 上建表
#   3) 导入数据并校验
#
# 前置条件：
#   - 已安装并启动 MySQL（macOS: brew install mysql && brew services start mysql）
#   - 已执行 mysql -u root -p < scripts/mysql_setup.sql 创建库与账号
#   - 已安装数据库驱动（项目虚拟环境里已自带纯 Python 的 pymysql）
#
# 用法：
#   bash scripts/migrate_to_mysql.sh
#   IR_DB_USER=root IR_DB_PASSWORD=你的密码 bash scripts/migrate_to_mysql.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ -d .venv ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

export IR_DB_NAME="${IR_DB_NAME:-image_recognition}"
export IR_DB_USER="${IR_DB_USER:-ir_user}"
export IR_DB_PASSWORD="${IR_DB_PASSWORD:-ir_pass_2026}"
export IR_DB_HOST="${IR_DB_HOST:-127.0.0.1}"
export IR_DB_PORT="${IR_DB_PORT:-3306}"

# MySQL 连接参数（下面各步骤显式指定使用哪个后端，避免互相干扰）
MYSQL_ENV=(env IR_DB_ENGINE=mysql IR_DB_NAME="$IR_DB_NAME" IR_DB_USER="$IR_DB_USER"
           IR_DB_PASSWORD="$IR_DB_PASSWORD" IR_DB_HOST="$IR_DB_HOST" IR_DB_PORT="$IR_DB_PORT")
SQLITE_ENV=(env IR_DB_ENGINE=sqlite)

DUMP="$ROOT/work/sqlite_dump.json"
mkdir -p "$ROOT/work"

cd backend

echo "[1/5] 检查 MySQL 连接（${IR_DB_USER}@${IR_DB_HOST}:${IR_DB_PORT}/${IR_DB_NAME}）"
"${MYSQL_ENV[@]}" python manage.py migrate --plan >/dev/null

echo "[2/5] 从 SQLite 导出用户与业务数据 -> work/sqlite_dump.json"
"${SQLITE_ENV[@]}" python manage.py dumpdata auth.user recognition --indent 2 -o "$DUMP"

echo "[3/5] 在 MySQL 上创建数据表"
"${MYSQL_ENV[@]}" python manage.py migrate --noinput | tail -3

echo "[4/5] 导入数据"
"${MYSQL_ENV[@]}" python manage.py loaddata "$DUMP"

echo "[5/5] 校验数据"
"${MYSQL_ENV[@]}" python manage.py shell -c "
from django.contrib.auth.models import User
from apps.recognition.models import Dataset, TrainingRun, TrainedModel, PredictionRecord
print('  用户:', User.objects.count(),
      '| 数据集:', Dataset.objects.count(),
      '| 训练任务:', TrainingRun.objects.count(),
      '| 模型:', TrainedModel.objects.count(),
      '| 识别记录:', PredictionRecord.objects.count())
"

echo
echo "迁移完成。之后用下面的方式启动即为 MySQL 模式："
echo "  source .venv/bin/activate"
echo "  IR_DB_ENGINE=mysql IR_DB_NAME=$IR_DB_NAME IR_DB_USER=$IR_DB_USER IR_DB_PASSWORD=$IR_DB_PASSWORD \\"
echo "    python backend/manage.py runserver 8000"
echo "（SQLite 的 backend/db.sqlite3 会保留，可随时用 IR_DB_ENGINE=sqlite 切回去）"
