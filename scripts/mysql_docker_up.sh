#!/usr/bin/env bash
# 一键启动 Docker 中的 MySQL，并等待其变为 healthy
#
#   bash scripts/mysql_docker_up.sh          # 启动并等待
#   bash scripts/mysql_docker_up.sh --migrate  # 启动后直接迁移 SQLite 数据
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! command -v docker >/dev/null 2>&1; then
  echo "未检测到 docker 命令。macOS 请先安装并启动 Docker Desktop：https://www.docker.com/products/docker-desktop/" >&2
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "Docker 守护进程未运行，正在尝试启动 Docker Desktop…"
  if [ -d /Applications/Docker.app ]; then
    open -a Docker || true
    for _ in $(seq 1 60); do
      if docker info >/dev/null 2>&1; then break; fi
      sleep 3
    done
  fi
fi

if ! docker info >/dev/null 2>&1; then
  echo "仍然无法连接 Docker 守护进程，请手动打开 Docker Desktop 后重试。" >&2
  exit 1
fi

echo "==> 启动 MySQL 容器"
docker compose up -d

echo "==> 等待 MySQL 就绪（首次启动需初始化数据目录，约 30~60 秒）"
for i in $(seq 1 80); do
  status="$(docker inspect --format '{{.State.Health.Status}}' ir-mysql 2>/dev/null || echo unknown)"
  case "$status" in
    healthy)
      echo "MySQL 已就绪（healthy）"
      break
      ;;
    unhealthy)
      echo "MySQL 容器状态异常，请执行 docker compose logs -f mysql 查看日志" >&2
      exit 1
      ;;
    *)
      [ $((i % 5)) -eq 0 ] && echo "  等待中… 当前状态：$status"
      sleep 3
      ;;
  esac
done

if [ "${1:-}" = "--migrate" ]; then
  echo "==> 开始迁移 SQLite 数据到 MySQL"
  bash "$ROOT/scripts/migrate_to_mysql.sh"
else
  echo
  echo "数据库已就绪，接下来可以执行："
  echo "  bash scripts/migrate_to_mysql.sh     # 把 SQLite 现有数据迁移过来"
  echo "  cd backend && IR_DB_ENGINE=mysql IR_DB_USER=ir_user IR_DB_PASSWORD=ir_pass_2026 python manage.py runserver 8000"
fi
