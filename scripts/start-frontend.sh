#!/usr/bin/env bash
# 启动前端开发服务器（Vite，自动代理 /api 与 /media 到后端 8000 端口）
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/frontend"

if [ ! -d node_modules ]; then
  echo "首次运行，安装前端依赖…"
  npm install
fi

echo "前端开发服务器启动中： http://localhost:5173/"
echo "（请确保后端已在 8000 端口运行）"
npm run dev
