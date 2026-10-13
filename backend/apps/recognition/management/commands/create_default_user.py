"""创建系统默认账号：python manage.py create_default_user

默认创建两个演示账号（可在 settings.DEMO_ACCOUNTS 中调整）：
    admin   / admin123   管理员（is_staff，可登录 Django Admin）
    student / 123456     普通用户
已有账号会被跳过，可用 --reset 重置密码。
"""

from __future__ import annotations

from django.conf import settings
from django.contrib.auth.models import User
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "创建登录页所需的默认账号（管理员与普通用户）"

    def add_arguments(self, parser):
        parser.add_argument("--reset", action="store_true", help="账号已存在时重置其密码")
        parser.add_argument("--username", default="", help="只创建指定的单个账号")
        parser.add_argument("--password", default="", help="配合 --username 使用的密码")
        parser.add_argument("--superuser", action="store_true", help="将指定账号设为超级用户")

    def handle(self, *args, **options):
        if options["username"]:
            accounts = [{
                "username": options["username"],
                "password": options["password"] or "admin123",
                "role": "超级管理员" if options["superuser"] else "普通用户",
                "note": "",
                "is_staff": bool(options["superuser"]),
                "is_superuser": bool(options["superuser"]),
            }]
        else:
            admin = next((item for item in settings.DEMO_ACCOUNTS if item["username"] == "admin"),
                         {"username": "admin", "password": "admin123", "role": "管理员", "note": ""})
            accounts = []
            for item in settings.DEMO_ACCOUNTS:
                accounts.append({**item,
                                 "is_staff": item["username"] == admin["username"],
                                 "is_superuser": item["username"] == admin["username"]})

        for account in accounts:
            username = account["username"]
            user = User.objects.filter(username=username).first()
            if user is None:
                User.objects.create_user(
                    username=username,
                    password=account["password"],
                    is_staff=account.get("is_staff", False),
                    is_superuser=account.get("is_superuser", False),
                    first_name=account.get("name") or account.get("role", ""),
                )
                self.stdout.write(self.style.SUCCESS(
                    f"已创建账号 {username} / {account['password']}（{account.get('role', '')}）"))
                continue

            fields = []
            if account.get("name") and user.first_name != account["name"]:
                user.first_name = account["name"]
                fields.append("first_name")
            if account.get("is_staff") and not user.is_staff:
                user.is_staff = True
                fields.append("is_staff")
            if account.get("is_superuser") and not user.is_superuser:
                user.is_superuser = True
                fields.append("is_superuser")
            if options["reset"]:
                user.set_password(account["password"])
                fields.append("password")
            if fields:
                user.save(update_fields=fields)
                self.stdout.write(self.style.WARNING(
                    f"账号 {username} 已存在，已更新：{', '.join(fields)}"))
            else:
                self.stdout.write(f"账号 {username} 已存在，跳过（如需重置密码请加 --reset）")
