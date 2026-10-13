"""用户管理接口（仅管理员）：新建、修改、重置密码、启用/停用、删除。"""

from __future__ import annotations

from django.contrib.auth.models import User
from rest_framework import status
from rest_framework.authtoken.models import Token
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.viewsets import ModelViewSet

from ..models import PredictionRecord
from ..permissions import IsAdmin
from ..serializers import UserSerializer, UserWriteSerializer


class UserViewSet(ModelViewSet):
    """用户管理：/api/users/（仅管理员）"""

    queryset = User.objects.all().order_by("id")
    permission_classes = [IsAdmin]

    def get_serializer_class(self):
        if self.action in {"create", "update", "partial_update"}:
            return UserWriteSerializer
        return UserSerializer

    def get_queryset(self):
        queryset = super().get_queryset()
        search = self.request.query_params.get("search")
        if search:
            queryset = queryset.filter(username__icontains=search)
        role = self.request.query_params.get("role")
        if role == "admin":
            queryset = queryset.filter(is_staff=True)
        elif role == "user":
            queryset = queryset.filter(is_staff=False)
        return queryset

    def _admin_count(self) -> int:
        return User.objects.filter(is_staff=True, is_active=True).count()

    def update(self, request, *args, **kwargs):
        user = self.get_object()
        payload = request.data
        if user.pk == request.user.pk:
            if payload.get("role") == "user":
                return Response({"detail": "不能取消自己的管理员权限"}, status=status.HTTP_400_BAD_REQUEST)
            if str(payload.get("is_active", "true")).lower() in {"false", "0"}:
                return Response({"detail": "不能停用当前登录的账号"}, status=status.HTTP_400_BAD_REQUEST)
        if user.is_staff and payload.get("role") == "user" and self._admin_count() <= 1:
            return Response({"detail": "系统至少需要保留一个管理员"}, status=status.HTTP_400_BAD_REQUEST)
        return super().update(request, *args, **kwargs)

    def destroy(self, request, *args, **kwargs):
        user = self.get_object()
        if user.pk == request.user.pk:
            return Response({"detail": "不能删除当前登录的账号"}, status=status.HTTP_400_BAD_REQUEST)
        if user.is_staff and self._admin_count() <= 1:
            return Response({"detail": "系统至少需要保留一个管理员"}, status=status.HTTP_400_BAD_REQUEST)
        if PredictionRecord.objects.filter(user=user).exists():
            return Response({"detail": "该账号下已有识别记录，建议改为「停用」而不是删除"},
                            status=status.HTTP_400_BAD_REQUEST)
        Token.objects.filter(user=user).delete()
        return super().destroy(request, *args, **kwargs)

    @action(detail=True, methods=["post"], url_path="reset-password")
    def reset_password(self, request, pk=None):
        """重置密码：管理员可为用户设置新密码，该用户所有登录状态立即失效。"""
        user = self.get_object()
        new_password = str(request.data.get("new_password") or "").strip()
        if len(new_password) < 6:
            return Response({"detail": "新密码长度至少 6 位"}, status=status.HTTP_400_BAD_REQUEST)
        user.set_password(new_password)
        user.save(update_fields=["password"])
        Token.objects.filter(user=user).delete()
        return Response({"detail": f"已重置「{user.username}」的密码，该账号需重新登录"})

    @action(detail=True, methods=["post"], url_path="toggle-active")
    def toggle_active(self, request, pk=None):
        """启用 / 停用账号。停用后无法登录，已登录的令牌会被清除。"""
        user = self.get_object()
        if user.pk == request.user.pk:
            return Response({"detail": "不能停用当前登录的账号"}, status=status.HTTP_400_BAD_REQUEST)
        if user.is_staff and user.is_active and self._admin_count() <= 1:
            return Response({"detail": "系统至少需要保留一个可用的管理员"},
                            status=status.HTTP_400_BAD_REQUEST)
        user.is_active = not user.is_active
        user.save(update_fields=["is_active"])
        if not user.is_active:
            Token.objects.filter(user=user).delete()
        return Response({"detail": f"已{'启用' if user.is_active else '停用'}账号「{user.username}」",
                         "is_active": user.is_active})
