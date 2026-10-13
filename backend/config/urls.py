"""全局路由：API + 媒体文件 +（可选）前端单页应用。"""

from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path, re_path

from config.views import spa_view

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/", include("apps.recognition.urls")),
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

# 前端执行 npm run build 后，可由 Django 直接托管 dist 目录（单端口演示部署）
if settings.FRONTEND_DIST.exists():
    urlpatterns += [re_path(r"^(?P<path>.*)$", spa_view)]
