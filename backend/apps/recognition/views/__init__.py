"""API 视图集合。"""

from .architecture import ArchitectureViewSet
from .auth import (AuthConfigView, ChangePasswordView, LoginView, LogoutView, ProfileView)
from .dashboard import DashboardStatsView, HealthView, SystemInfoView
from .dataset import DatasetViewSet
from .model_registry import TrainedModelViewSet
from .recognize import BatchRecognizeView, CompareRecognizeView, RecognizeView, SampleImageView
from .record import PredictionRecordViewSet
from .training import TrainingRunViewSet
from .users import UserViewSet
from .vision_tasks import DetectionPredictView, SegmentationPredictView

__all__ = [
    "ArchitectureViewSet",
    "LoginView",
    "LogoutView",
    "ProfileView",
    "ChangePasswordView",
    "AuthConfigView",
    "DatasetViewSet",
    "TrainingRunViewSet",
    "UserViewSet",
    "TrainedModelViewSet",
    "RecognizeView",
    "CompareRecognizeView",
    "BatchRecognizeView",
    "SampleImageView",
    "PredictionRecordViewSet",
    "DashboardStatsView",
    "SystemInfoView",
    "HealthView",
    "DetectionPredictView",
    "SegmentationPredictView",
]
