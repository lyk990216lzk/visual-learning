"""权限控制：区分「管理员」与「普通用户」两级角色。

- 普通用户（is_staff=False）：可以识别图片、做多模型对比、查看模型库与识别历史；
- 管理员（is_staff=True）：在普通用户能力之上，可以管理数据集、发起训练、
  管理已训练模型、清空识别记录以及维护用户账号。

权限点直接复用 Django 用户体系的 is_staff 字段，无需额外建表。
"""

from rest_framework.permissions import SAFE_METHODS, BasePermission


class IsAdminOrReadOnly(BasePermission):
    """读操作对所有登录用户开放；写操作（POST/PUT/PATCH/DELETE）仅管理员可用。"""

    message = "该操作需要管理员权限，请联系系统管理员"

    def has_permission(self, request, view) -> bool:
        if request.method in SAFE_METHODS:
            return True
        user = request.user
        return bool(user and user.is_authenticated and user.is_staff)


class IsAdmin(BasePermission):
    """仅管理员可访问（用于用户管理等纯管理接口）。"""

    message = "该功能仅管理员可用"

    def has_permission(self, request, view) -> bool:
        user = request.user
        return bool(user and user.is_authenticated and user.is_staff)
