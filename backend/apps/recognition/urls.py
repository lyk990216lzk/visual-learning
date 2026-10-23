"""识别应用路由：统一挂载在 /api/ 下。"""

from django.urls import include, path
from rest_framework.routers import DefaultRouter

from .views import (ArchitectureViewSet, AuthConfigView, BatchRecognizeView, ChangePasswordView,
                    CompareRecognizeView, DashboardStatsView, DatasetViewSet,
                    DetectionPredictView, HealthView, LoginView, LogoutView,
                    PredictionRecordViewSet, ProfileView, RecognizeView, SampleImageView,
                    SegmentationPredictView, SystemInfoView, TrainedModelViewSet,
                    TrainingRunViewSet)
from .views import UserViewSet

router = DefaultRouter()
router.register("architectures", ArchitectureViewSet, basename="architecture")
router.register("datasets", DatasetViewSet, basename="dataset")
router.register("training-runs", TrainingRunViewSet, basename="training-run")
router.register("trained-models", TrainedModelViewSet, basename="trained-model")
router.register("records", PredictionRecordViewSet, basename="record")
router.register("users", UserViewSet, basename="user")

urlpatterns = [
    path("health/", HealthView.as_view(), name="health"),
    path("auth/login/", LoginView.as_view(), name="auth-login"),
    path("auth/logout/", LogoutView.as_view(), name="auth-logout"),
    path("auth/profile/", ProfileView.as_view(), name="auth-profile"),
    path("auth/password/", ChangePasswordView.as_view(), name="auth-password"),
    path("auth/config/", AuthConfigView.as_view(), name="auth-config"),
    path("system/", SystemInfoView.as_view(), name="system-info"),
    path("dashboard/", DashboardStatsView.as_view(), name="dashboard"),
    path("recognize/", RecognizeView.as_view(), name="recognize"),
    path("detect/", DetectionPredictView.as_view(), name="detect"),
    path("segment/", SegmentationPredictView.as_view(), name="segment"),
    path("recognize/compare/", CompareRecognizeView.as_view(), name="recognize-compare"),
    path("recognize/batch/", BatchRecognizeView.as_view(), name="recognize-batch"),
    path("recognize/sample/", SampleImageView.as_view(), name="recognize-sample"),
    path("", include(router.urls)),
]
