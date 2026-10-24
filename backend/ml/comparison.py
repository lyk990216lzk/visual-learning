"""多模型对比结果聚合：检测实例 IoU 匹配与分割像素一致性。"""

from __future__ import annotations

import base64
import io
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image


def box_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """计算两个 xyxy 检测框的 IoU。"""
    if len(box_a) < 4 or len(box_b) < 4:
        return 0.0
    ax1, ay1, ax2, ay2 = (float(value) for value in box_a[:4])
    bx1, by1, bx2, by2 = (float(value) for value in box_b[:4])
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = inter_w * inter_h
    if intersection <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return round(intersection / union, 6) if union > 0 else 0.0


def _match_model_detections(
    reference: Dict,
    candidate: Dict,
    threshold: float,
) -> Tuple[Dict[int, Dict], set[int]]:
    """在同一模型内与参考目标做一对一匹配，优先同类，再识别类别冲突。"""
    refs = list(reference.get("detections") or [])
    candidates = list(candidate.get("detections") or [])
    used_refs: set[int] = set()
    used_candidates: set[int] = set()
    matches: Dict[int, Dict] = {}

    same_class_pairs = []
    cross_class_pairs = []
    for ref_index, ref in enumerate(refs):
        for detection_index, detection in enumerate(candidates):
            iou = box_iou(ref.get("bbox") or [], detection.get("bbox") or [])
            if iou < threshold:
                continue
            pair = (iou, ref_index, detection_index)
            if ref.get("label") == detection.get("label"):
                same_class_pairs.append(pair)
            else:
                cross_class_pairs.append(pair)

    def assign(pairs, status: str):
        for iou, ref_index, detection_index in sorted(pairs, reverse=True):
            if ref_index in used_refs or detection_index in used_candidates:
                continue
            detection = candidates[detection_index]
            matches[ref_index] = {
                "status": status,
                "iou": iou,
                "detection_index": detection_index,
                "label": detection.get("label", ""),
                "confidence": detection.get("confidence", 0.0),
                "bbox": detection.get("bbox", []),
            }
            used_refs.add(ref_index)
            used_candidates.add(detection_index)

    assign(same_class_pairs, "matched")
    assign(cross_class_pairs, "class_conflict")
    return matches, used_candidates


def compare_detection_outputs(
    outputs: Sequence[Dict],
    reference_model_id: int,
    iou_threshold: float = 0.5,
) -> Dict:
    """以最佳模型实例为参考，将其他模型的同类检测框通过 IoU 聚为实例。"""
    threshold = max(0.1, min(float(iou_threshold), 0.95))
    reference = next(item for item in outputs if item["model_id"] == reference_model_id)
    reference_detections = list(reference.get("detections") or [])
    other_outputs = [item for item in outputs if item["model_id"] != reference_model_id]
    model_by_id = {item["model_id"]: item for item in outputs}

    instances = []
    for index, detection in enumerate(reference_detections):
        instance = {
            "id": f"{detection.get('label', 'object')}-{index + 1}",
            "class_id": detection.get("class_id"),
            "label": detection.get("label", ""),
            "confidence": detection.get("confidence", 0.0),
            "bbox": detection.get("bbox", []),
            "matches": {},
        }
        instances.append(instance)

    extra_detections = []
    model_match_counts = {item["model_id"]: 0 for item in other_outputs}
    for candidate in other_outputs:
        matches, used_candidates = _match_model_detections(reference, candidate, threshold)
        for ref_index, instance in enumerate(instances):
            instance["matches"][str(candidate["model_id"])] = matches.get(
                ref_index,
                {"status": "missed", "iou": 0.0, "label": "", "confidence": 0.0, "bbox": []},
            )
        model_match_counts[candidate["model_id"]] = len(matches)
        for detection_index, detection in enumerate(candidate.get("detections") or []):
            if detection_index not in used_candidates:
                extra_detections.append({
                    "model_id": candidate["model_id"],
                    "model_name": candidate.get("model_name", ""),
                    "label": detection.get("label", ""),
                    "confidence": detection.get("confidence", 0.0),
                    "bbox": detection.get("bbox", []),
                })

    total_possible = len(instances) * max(1, len(other_outputs))
    geometric_matches = sum(
        1
        for instance in instances
        for match in instance["matches"].values()
        if match["status"] in {"matched", "class_conflict"}
    )
    class_conflicts = sum(
        1
        for instance in instances
        for match in instance["matches"].values()
        if match["status"] == "class_conflict"
    )

    classes = defaultdict(lambda: {
        "name": "",
        "count": 0,
        "reference_indices": [],
        "covered_models": {reference_model_id},
        "confidences": [],
    })
    for index, instance in enumerate(instances):
        item = classes[instance["label"]]
        item["name"] = instance["label"]
        item["count"] += 1
        item["reference_indices"].append(index)
        item["confidences"].append(instance["confidence"])
        for model_id, match in instance["matches"].items():
            if match["status"] in {"matched", "class_conflict"}:
                item["covered_models"].add(int(model_id))
                item["confidences"].append(match["confidence"])

    class_summary = [
        {
            "name": item["name"],
            "count": item["count"],
            "covered_models": len(item["covered_models"]),
            "total_models": len(outputs),
            "average_confidence": round(float(np.mean(item["confidences"])), 4)
            if item["confidences"] else 0.0,
        }
        for item in classes.values()
    ]
    for instance in instances:
        instance["match_count"] = sum(
            1 for match in instance["matches"].values()
            if match["status"] in {"matched", "class_conflict"}
        )
        instance["coverage"] = round(instance["match_count"] / max(1, len(other_outputs)), 4)

    return {
        "reference_model_id": reference_model_id,
        "reference_model_name": reference.get("model_name", ""),
        "reference_instances": instances,
        "classes": class_summary,
        "class_count": len(class_summary),
        "object_count": len(instances),
        "matches_against_reference": geometric_matches,
        "possible_matches": total_possible,
        "instance_consistency": round(geometric_matches / max(1, total_possible), 4),
        "class_conflicts": class_conflicts,
        "extra_detections": extra_detections,
        "model_match_counts": model_match_counts,
        "iou_threshold": threshold,
        "model_results": [
            {
                "record_id": item.get("record_id"),
                "model_id": item["model_id"],
                "model_name": item.get("model_name", ""),
                "architecture": item.get("architecture", ""),
                "training_run_id": item.get("training_run_id"),
                "task_id": item.get("task_id"),
                "task_name": item.get("task_name", ""),
                "detections": item.get("detections", []),
                "object_count": item.get("object_count", 0),
                "confidence": item.get("confidence", 0.0),
                "latency_ms": item.get("latency_ms", 0.0),
                "matched_count": (
                    len(instances)
                    if item["model_id"] == reference_model_id
                    else model_match_counts.get(item["model_id"], 0)
                ),
            }
            for item in outputs
        ],
    }


def _decode_mask(value: str) -> np.ndarray:
    if not value:
        raise ValueError("分割结果缺少掩膜")
    raw = base64.b64decode(value)
    with Image.open(io.BytesIO(raw)) as image:
        return np.asarray(image.convert("P"), dtype=np.uint8)


def compare_segmentation_outputs(
    outputs: Sequence[Dict],
    reference_model_id: int,
) -> Dict:
    """比较分割模型相对最佳模型的像素一致率与分类别 IoU。"""
    reference = next(item for item in outputs if item["model_id"] == reference_model_id)
    reference_mask = _decode_mask(reference.get("mask_base64", ""))
    reference_result = {
        "reference_model_id": reference_model_id,
        "reference_model_name": reference.get("model_name", ""),
        "reference_mask_base64": reference.get("mask_base64", ""),
        "classes": reference.get("class_stats", []),
        "model_results": [],
    }
    agreements = []
    class_ious = defaultdict(list)
    class_names = list(reference.get("classes") or [])

    for item in outputs:
        mask = _decode_mask(item.get("mask_base64", ""))
        if mask.shape != reference_mask.shape:
            mask_image = Image.fromarray(mask, mode="P").resize(
                (reference_mask.shape[1], reference_mask.shape[0]),
                resample=Image.Resampling.NEAREST,
            )
            mask = np.asarray(mask_image, dtype=np.uint8)
        pixel_agreement = float(np.mean(mask == reference_mask))
        class_iou = []
        for class_index, class_name in enumerate(class_names):
            reference_pixels = reference_mask == class_index
            candidate_pixels = mask == class_index
            union = np.logical_or(reference_pixels, candidate_pixels).sum()
            intersection = np.logical_and(reference_pixels, candidate_pixels).sum()
            iou = float(intersection / union) if union else 1.0
            item_iou = {"index": class_index, "label": class_name, "iou": round(iou, 4)}
            class_iou.append(item_iou)
            class_ious[class_name].append(iou)
        if item["model_id"] != reference_model_id:
            agreements.append(pixel_agreement)
        reference_result["model_results"].append({
            "record_id": item.get("record_id"),
            "model_id": item["model_id"],
            "model_name": item.get("model_name", ""),
            "architecture": item.get("architecture", ""),
            "training_run_id": item.get("training_run_id"),
            "task_id": item.get("task_id"),
            "task_name": item.get("task_name", ""),
            "class_stats": item.get("class_stats", []),
            "confidence": item.get("confidence", 0.0),
            "latency_ms": item.get("latency_ms", 0.0),
            "pixel_agreement": round(pixel_agreement, 4),
            "class_iou": class_iou,
        })

    reference_result["mean_pixel_agreement"] = round(
        float(np.mean(agreements)) if agreements else 1.0, 4
    )
    reference_result["class_iou"] = [
        {
            "label": name,
            "iou": round(float(np.mean(values)), 4) if values else 0.0,
        }
        for name, values in class_ious.items()
    ]
    return reference_result


__all__ = [
    "box_iou",
    "compare_detection_outputs",
    "compare_segmentation_outputs",
]
