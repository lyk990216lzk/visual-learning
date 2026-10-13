#!/usr/bin/env bash
# 一键初始化：创建虚拟环境、安装依赖、初始化数据库、生成演示数据集
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> [1/4] 创建 Python 虚拟环境 .venv"
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> [2/4] 安装后端依赖"
# uv 创建的 venv 默认不含 pip：优先用 uv，回退到 python -m pip（避免 pip 落到系统/conda 环境）
if command -v uv >/dev/null 2>&1; then
  uv pip install -r requirements.txt
else
  python -m pip install --upgrade pip >/dev/null
  python -m pip install -r requirements.txt
fi

echo "==> [3/4] 初始化数据库"
cd backend
python manage.py migrate

echo "==> [4/4] 生成演示数据集（6 类几何图形）"
python manage.py build_demo_dataset --per-class 240 --size 64

cd "$ROOT"
echo
echo "后端初始化完成。"
echo "  启动后端：bash scripts/start-backend.sh"
echo "  启动前端：bash scripts/start-frontend.sh   （首次会执行 npm install）"
