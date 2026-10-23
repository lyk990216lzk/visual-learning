"""任务训练基础设施的逻辑测试；不执行真实模型训练或图片推理。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from ml import zoo
from ml.detection import YOLO_CONFIG_CANDIDATES, prepare_detection_dataset
from ml.segmentation import _segmentation_stats, build_segmentation_dataloaders


class TaskPipelineTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def _make_detection(self, with_val: bool) -> Path:
        root = self.root / ("detection_val" if with_val else "detection_auto")
        for split in ("train", "val") if with_val else ("train",):
            (root / "images" / split).mkdir(parents=True)
            (root / "labels" / split).mkdir(parents=True)
            for index in range(8):
                Image.new("RGB", (32, 32)).save(
                    root / "images" / split / f"{index:02d}.jpg"
                )
                (root / "labels" / split / f"{index:02d}.txt").write_text(
                    "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
                )
        (root / "data.yaml").write_text(
            "names:\n  0: cat\n", encoding="utf-8"
        )
        return root

    def _make_segmentation(self, with_val: bool) -> Path:
        root = self.root / ("segmentation_val" if with_val else "segmentation_auto")
        splits = ("train", "val") if with_val else ("train",)
        for split in splits:
            (root / "images" / split).mkdir(parents=True)
            (root / "masks" / split).mkdir(parents=True)
            for index in range(8):
                Image.new("RGB", (32, 32), color=(40, 80, 120)).save(
                    root / "images" / split / f"{index:02d}.png"
                )
                mask = np.full((32, 32), index % 2, dtype=np.uint8)
                Image.fromarray(mask, mode="L").save(
                    root / "masks" / split / f"{index:02d}.png"
                )
        (root / "classes.json").write_text(
            '["background", "road"]', encoding="utf-8"
        )
        return root

    def test_detection_auto_and_fixed_split_preparation(self):
        auto = prepare_detection_dataset(
            self._make_detection(False), self.root / "auto_work",
            val_split=0.25, seed=42,
        )
        self.assertEqual(auto["split_mode"], "auto")
        self.assertEqual(auto["train_count"], 6)
        self.assertEqual(auto["val_count"], 2)
        self.assertTrue(auto["data_yaml"].is_file())
        self.assertEqual(len(list((auto["dataset_root"] / "images/train").glob("*.jpg"))), 6)
        self.assertEqual(len(list((auto["dataset_root"] / "images/val").glob("*.jpg"))), 2)

        fixed = prepare_detection_dataset(
            self._make_detection(True), self.root / "fixed_work",
            val_split=0.5, seed=42,
        )
        self.assertEqual(fixed["split_mode"], "explicit")
        self.assertEqual(fixed["train_count"], 8)
        self.assertEqual(fixed["val_count"], 8)

    def test_detection_model_catalog(self):
        self.assertEqual(
            set(YOLO_CONFIG_CANDIDATES),
            {"yolov3", "yolov5", "yolov8", "yolov10", "yolov11"},
        )

    def test_segmentation_auto_and_fixed_split_loading(self):
        train_loader, val_loader, classes, info = build_segmentation_dataloaders(
            self._make_segmentation(False), image_size=32, batch_size=2,
            val_split=0.25, seed=42, augment=False,
        )
        self.assertEqual(classes, ["background", "road"])
        self.assertEqual(info, {"split_mode": "auto", "train_count": 6, "val_count": 2})
        images, masks = next(iter(train_loader))
        self.assertEqual(tuple(images.shape[1:]), (3, 32, 32))
        self.assertEqual(tuple(masks.shape[1:]), (32, 32))

        _, val_loader, _, fixed = build_segmentation_dataloaders(
            self._make_segmentation(True), image_size=32, batch_size=2,
            val_split=0.8, seed=42, augment=False,
        )
        self.assertEqual(fixed["split_mode"], "explicit")
        self.assertEqual(fixed["train_count"], 8)
        self.assertEqual(fixed["val_count"], 8)

    def test_segmentation_metrics(self):
        report = _segmentation_stats(
            np.array([[5, 1], [2, 4]], dtype=np.int64),
            ["background", "road"],
        )
        self.assertEqual(report["accuracy"], 0.75)
        self.assertGreater(report["miou"], 0)
        self.assertEqual(len(report["per_class_iou"]), 2)

    def test_segmentation_models_can_be_constructed(self):
        for key in ("unet", "fcn", "deeplabv3"):
            model = zoo.build_model(key, num_classes=2)
            self.assertGreater(sum(item.numel() for item in model.parameters()), 0)


if __name__ == "__main__":
    unittest.main()
