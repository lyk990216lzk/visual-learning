"""认证接口：登录、退出登录、当前用户信息、修改密码、登录页配置。"""

from __future__ import annotations

from django.conf import settings
from django.contrib.auth import authenticate
from django.contrib.auth.models import User
from django.utils import timezone
from rest_framework import status
from rest_framework.authtoken.models import Token
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView

from ..models import PredictionRecord


def serialize_user(user: User) -> dict:
    """统一的用户信息结构，前端登录页与顶栏共用。"""
    return {
        "id": user.id,
        "username": user.username,
        "display_name": user.get_full_name() or user.username,
        "email": user.email,
        "is_staff": user.is_staff,
        "is_superuser": user.is_superuser,
        "role": "管理员" if user.is_staff else "普通用户",
        "date_joined": user.date_joined.strftime("%Y-%m-%d %H:%M:%S") if user.date_joined else "",
        "last_login": user.last_login.strftime("%Y-%m-%d %H:%M:%S") if user.last_login else "",
    }


class LoginView(APIView):
    """POST /api/auth/login/  {username, password} -> {token, user}"""

    permission_classes = [AllowAny]
    authentication_classes = []

    def post(self, request):
        username = str(request.data.get("username") or "").strip()
        password = str(request.data.get("password") or "")
        if not username or not password:
            return Response({"detail": "请输入用户名和密码"}, status=status.HTTP_400_BAD_REQUEST)

        user = authenticate(request, username=username, password=password)
        if user is None:
            # 允许使用邮箱登录
            matched = User.objects.filter(email__iexact=username).first()
            if matched is not None:
                user = authenticate(request, username=matched.username, password=password)
        if user is None:
            return Response({"detail": "用户名或密码错误"}, status=status.HTTP_400_BAD_REQUEST)
        if not user.is_active:
            return Response({"detail": "该账号已被禁用，请联系管理员"}, status=status.HTTP_403_FORBIDDEN)

        token, _ = Token.objects.get_or_create(user=user)
        user.last_login = timezone.now()
        user.save(update_fields=["last_login"])
        return Response({"token": token.key, "user": serialize_user(user)})


class LogoutView(APIView):
    """POST /api/auth/logout/ ：注销当前令牌。"""

    def post(self, request):
        Token.objects.filter(user=request.user).delete()
        return Response({"detail": "已安全退出登录"})


class ProfileView(APIView):
    """GET/PATCH /api/auth/profile/ ：查看或更新当前用户资料。"""

    def get(self, request):
        data = serialize_user(request.user)
        data["record_count"] = PredictionRecord.objects.filter(user=request.user).count()
        return Response(data)

    def patch(self, request):
        user = request.user
        display_name = str(request.data.get("display_name") or "").strip()
        email = str(request.data.get("email") or "").strip()
        if email and User.objects.filter(email__iexact=email).exclude(pk=user.pk).exists():
            return Response({"detail": "该邮箱已被其它账号使用"}, status=status.HTTP_400_BAD_REQUEST)
        if display_name:
            parts = display_name.split(" ", 1)
            user.first_name = parts[0]
            user.last_name = parts[1] if len(parts) > 1 else ""
        if email:
            user.email = email
        user.save()
        return Response({"detail": "资料已更新", "user": serialize_user(user)})


class ChangePasswordView(APIView):
    """POST /api/auth/password/  {old_password, new_password} -> 新令牌（其余会话失效）"""

    def post(self, request):
        old_password = str(request.data.get("old_password") or "")
        new_password = str(request.data.get("new_password") or "")
        if not request.user.check_password(old_password):
            return Response({"detail": "原密码不正确"}, status=status.HTTP_400_BAD_REQUEST)
        if len(new_password) < 6:
            return Response({"detail": "新密码长度至少 6 位"}, status=status.HTTP_400_BAD_REQUEST)
        if new_password == old_password:
            return Response({"detail": "新密码不能与原密码相同"}, status=status.HTTP_400_BAD_REQUEST)

        request.user.set_password(new_password)
        request.user.save()
        # 旧令牌全部失效，并为当前会话签发新令牌
        Token.objects.filter(user=request.user).delete()
        token = Token.objects.create(user=request.user)
        return Response({"detail": "密码修改成功", "token": token.key,
                         "user": serialize_user(request.user)})


class AuthConfigView(APIView):
    """GET /api/auth/config/ ：登录页所需的站点信息（无需鉴权）。"""

    permission_classes = [AllowAny]
    authentication_classes = []

    def get(self, request):
        return Response({
            "site_name": settings.SITE_NAME,
            "university": settings.UNIVERSITY_NAME,
            "university_en": settings.UNIVERSITY_EN_NAME,
            "university_badge": settings.UNIVERSITY_BADGE,
            "allow_register": False,
            "demo_accounts": settings.DEMO_ACCOUNTS if settings.SHOW_DEMO_ACCOUNT else [],
        })
