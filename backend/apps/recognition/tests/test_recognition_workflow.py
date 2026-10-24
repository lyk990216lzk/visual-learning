"""训练任务驱动的识别流程接口测试。"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from PIL import Image
from rest_framework.test import APIClient

from apps.recognition.models import Dataset, PredictionRecord, TrainedModel, TrainingRun
from apps.recognition.views.dataset import sample_dataset_images
from ml.comparison import box_iou, compare_detection_outputs, compare_segmentation_outputs
from ml.task_predictor import _mask_png


class RecognitionWorkflowTest(TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.user = User.objects.create_user("tester", password="test-pass-123")
        self.client = APIClient()
        self.client.force_authenticate(self.user)

    def tearDown(self):
        self.temp.cleanup()

    def _make_classification_dataset(self) -> Dataset:
        dataset_root = self.root / "classification"
        for split in ("train", "val"):
            for label in ("star", "circle"):
                target = dataset_root / split / label
                target.mkdir(parents=True)
                for index in range(4):
                    Image.new("RGB", (32, 32), color=(40 + index, 80, 120)).save(
                        target / f"{index}.png"
                    )
        return Dataset.objects.create(
            name="分类测试集",
            task_type=Dataset.TaskType.CLASSIFICATION,
            path=str(dataset_root),
            num_classes=2,
            class_names=["circle", "star"],
            image_count=16,
            train_count=8,
            val_count=8,
        )

    def _make_task(self, dataset: Dataset, task_type: str, architecture: str,
                   model_file: Path) -> TrainingRun:
        model_file.write_bytes(b"test-weight")
        run = TrainingRun.objects.create(
            name=f"{task_type}-completed",
            task_type=task_type,
            architecture=architecture,
            dataset=dataset,
            status=TrainingRun.Status.SUCCESS,
            best_accuracy=0.88,
        )
        model = TrainedModel.objects.create(
            name=f"{architecture}@{dataset.name}",
            architecture=architecture,
            task_type=task_type,
            dataset=dataset,
            training_run=run,
            classes=list(dataset.class_names or ["background", "object"]),
            num_classes=2,
            input_size=64,
            accuracy=0.88,
            file_path=str(model_file),
            file_size=model_file.stat().st_size,
        )
        run.trained_model = model
        run.save(update_fields=["trained_model"])
        return run

    def test_sample_dataset_images_prefers_validation_split(self):
        dataset = self._make_classification_dataset()
        samples = sample_dataset_images(dataset, limit=6)

        self.assertEqual(len(samples), 6)
        self.assertTrue(all(item["split"] == "val" for item in samples))
        self.assertTrue(all("/image/?" in item["url"] for item in samples))

    def test_recognizable_tasks_only_returns_completed_tasks_with_weight_file(self):
        dataset = self._make_classification_dataset()
        completed = self._make_task(
            dataset,
            TrainingRun.TaskType.CLASSIFICATION,
            "resnet18",
            self.root / "completed.pt",
        )
        TrainingRun.objects.create(
            name="still-running",
            task_type=TrainingRun.TaskType.CLASSIFICATION,
            architecture="lenet5",
            dataset=dataset,
            status=TrainingRun.Status.RUNNING,
        )

        response = self.client.get(
            "/api/training-runs/recognizable/",
            {"task_type": "classification"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["count"], 1)
        item = response.data["results"][0]
        self.assertEqual(item["id"], completed.id)
        self.assertEqual(item["trained_model_id"], completed.trained_model_id)
        self.assertEqual(item["dataset_name"], dataset.name)

    def test_task_samples_returns_six_random_samples(self):
        dataset = self._make_classification_dataset()
        run = self._make_task(
            dataset,
            TrainingRun.TaskType.CLASSIFICATION,
            "resnet18",
            self.root / "sample-model.pt",
        )

        response = self.client.get(f"/api/training-runs/{run.id}/samples/", {"limit": 6})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["task_id"], run.id)
        self.assertEqual(response.data["count"], 6)
        self.assertTrue(all(item["split"] == "val" for item in response.data["results"]))
        image_response = self.client.get(response.data["results"][0]["url"])
        self.assertEqual(image_response.status_code, 200)

    def test_batch_recognition_supports_each_task_type(self):
        dataset = self._make_classification_dataset()
        task_payloads = {
            "classification": {
                "predicted_label": "star",
                "confidence": 0.92,
                "latency_ms": 12,
                "topk": [{"index": 1, "label": "star", "probability": 0.92}],
                "all_probabilities": {"circle": 0.08, "star": 0.92},
            },
            "detection": {
                "predicted_label": "car",
                "confidence": 0.81,
                "latency_ms": 30,
                "detections": [{"label": "car", "confidence": 0.81, "bbox": [1, 2, 10, 12]}],
                "object_count": 1,
            },
            "segmentation": {
                "predicted_label": "road",
                "confidence": 0.76,
                "latency_ms": 35,
                "classes": ["background", "road"],
                "class_stats": [
                    {"index": 0, "label": "background", "pixel_count": 20, "percentage": 0.2},
                    {"index": 1, "label": "road", "pixel_count": 80, "percentage": 0.8},
                ],
                "width": 10,
                "height": 10,
            },
        }
        architectures = {
            "classification": "resnet18",
            "detection": "yolov8",
            "segmentation": "unet",
        }

        with override_settings(MEDIA_ROOT=self.root / "media"):
            for task_type, payload in task_payloads.items():
                with self.subTest(task_type=task_type):
                    run = self._make_task(
                        dataset,
                        task_type,
                        architectures[task_type],
                        self.root / f"{task_type}.pt",
                    )
                    result = {
                        "architecture": architectures[task_type],
                        "model_name": f"{architectures[task_type]}@{dataset.name}",
                        **payload,
                    }
                    if task_type != "classification":
                        result["task_type"] = task_type
                    with patch(
                        "apps.recognition.views.vision_tasks.predict_task_image",
                        return_value=result,
                    ):
                        buffer = io.BytesIO()
                        Image.new("RGB", (10, 10), color=(20, 30, 40)).save(buffer, format="PNG")
                        image = SimpleUploadedFile(
                            "sample.png",
                            buffer.getvalue(),
                            content_type="image/png",
                        )
                        response = self.client.post(
                            "/api/recognize/batch/",
                            {"images": [image], "trained_model": run.trained_model_id},
                            format="multipart",
                        )

                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.data["task_type"], task_type)
                    self.assertEqual(response.data["recognition_type"], "batch")
                    self.assertEqual(response.data["total"], 1)
                    self.assertEqual(response.data["results"][0]["task_type"], task_type)
                    self.assertEqual(response.data["results"][0]["recognition_type"], "batch")
                    self.assertTrue(
                        PredictionRecord.objects.filter(
                            task_type=task_type,
                            source=PredictionRecord.Source.BATCH,
                        ).exists()
                    )

        history = self.client.get("/api/records/", {"page_size": 10})
        self.assertEqual(history.status_code, 200)
        self.assertIn("training_run_id", history.data["results"][0])

    def test_detection_instance_matching_uses_reference_model_boxes(self):
        self.assertAlmostEqual(box_iou([0, 0, 10, 10], [5, 0, 15, 10]), 1 / 3, places=5)
        outputs = [
            {
                "model_id": 1,
                "model_name": "YOLO-reference",
                "detections": [
                    {"class_id": 0, "label": "car", "confidence": 0.95, "bbox": [0, 0, 10, 10]},
                    {"class_id": 0, "label": "car", "confidence": 0.90, "bbox": [20, 0, 30, 10]},
                ],
                "latency_ms": 10,
            },
            {
                "model_id": 2,
                "model_name": "YOLO-compare",
                "detections": [
                    {"class_id": 0, "label": "car", "confidence": 0.88, "bbox": [1, 0, 11, 10]},
                    {"class_id": 1, "label": "truck", "confidence": 0.72, "bbox": [21, 0, 31, 10]},
                ],
                "latency_ms": 12,
            },
            {
                "model_id": 3,
                "model_name": "YOLO-extra",
                "detections": [
                    {"class_id": 0, "label": "car", "confidence": 0.85, "bbox": [0.5, 0.5, 10.5, 10.5]},
                    {"class_id": 2, "label": "person", "confidence": 0.70, "bbox": [40, 0, 45, 10]},
                ],
                "latency_ms": 9,
            },
        ]

        comparison = compare_detection_outputs(
            outputs,
            reference_model_id=1,
            iou_threshold=0.5,
        )

        self.assertEqual(comparison["object_count"], 2)
        self.assertEqual(comparison["class_count"], 1)
        self.assertEqual(comparison["instance_consistency"], 0.75)
        self.assertEqual(comparison["class_conflicts"], 1)
        self.assertEqual(comparison["extra_detections"][0]["label"], "person")
        self.assertEqual(comparison["reference_instances"][0]["matches"]["2"]["status"], "matched")
        self.assertEqual(comparison["reference_instances"][1]["matches"]["2"]["status"], "class_conflict")

    def test_compare_endpoint_accepts_completed_models_from_same_dataset(self):
        dataset = self._make_classification_dataset()
        run_one = self._make_task(
            dataset,
            TrainingRun.TaskType.CLASSIFICATION,
            "resnet18",
            self.root / "compare-one.pt",
        )
        run_two = self._make_task(
            dataset,
            TrainingRun.TaskType.CLASSIFICATION,
            "vgg16",
            self.root / "compare-two.pt",
        )

        def fake_predict(trained, image, options):
            return {
                "architecture": trained.architecture,
                "model_name": trained.name,
                "predicted_label": "star",
                "confidence": 0.9,
                "latency_ms": 12,
                "topk": [{"index": 1, "label": "star", "probability": 0.9}],
                "all_probabilities": {"circle": 0.1, "star": 0.9},
            }

        with override_settings(MEDIA_ROOT=self.root / "compare-media"):
            with patch(
                "apps.recognition.views.vision_tasks.predict_task_image",
                side_effect=fake_predict,
            ):
                buffer = io.BytesIO()
                Image.new("RGB", (10, 10), color=(20, 30, 40)).save(buffer, format="PNG")
                response = self.client.post(
                    "/api/recognize/compare/",
                    {
                        "image": SimpleUploadedFile(
                            "compare.png",
                            buffer.getvalue(),
                            content_type="image/png",
                        ),
                        "trained_models": f"{run_one.trained_model_id},{run_two.trained_model_id}",
                        "task_type": "classification",
                    },
                    format="multipart",
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["successful_model_count"], 2)
        self.assertEqual(response.data["consensus_label"], "star")
        self.assertEqual(response.data["agreement"], 1.0)
        self.assertEqual(
            PredictionRecord.objects.filter(source=PredictionRecord.Source.COMPARE).count(),
            2,
        )

    def test_segmentation_comparison_reports_pixel_agreement(self):
        reference_mask = np.zeros((4, 4), dtype=np.uint8)
        reference_mask[:, 2:] = 1
        candidate_mask = reference_mask.copy()
        candidate_mask[0, 2] = 0
        outputs = [
            {
                "model_id": 1,
                "model_name": "U-Net@dataset",
                "classes": ["background", "road"],
                "class_stats": [
                    {"index": 0, "label": "background", "pixel_count": 8, "percentage": 0.5},
                    {"index": 1, "label": "road", "pixel_count": 8, "percentage": 0.5},
                ],
                "mask_base64": _mask_png(reference_mask, 2),
                "latency_ms": 10,
            },
            {
                "model_id": 2,
                "model_name": "DeepLab@dataset",
                "classes": ["background", "road"],
                "class_stats": [
                    {"index": 0, "label": "background", "pixel_count": 9, "percentage": 0.5625},
                    {"index": 1, "label": "road", "pixel_count": 7, "percentage": 0.4375},
                ],
                "mask_base64": _mask_png(candidate_mask, 2),
                "latency_ms": 12,
            },
        ]

        comparison = compare_segmentation_outputs(outputs, reference_model_id=1)

        self.assertEqual(comparison["mean_pixel_agreement"], 0.9375)
        self.assertEqual(len(comparison["class_iou"]), 2)
        self.assertEqual(comparison["model_results"][0]["pixel_agreement"], 1.0)

