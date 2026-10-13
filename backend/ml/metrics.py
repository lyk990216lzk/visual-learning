"""分类任务的评估指标（纯 Python 实现，不依赖 scikit-learn）。"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int) -> List[List[int]]:
    """返回 num_classes x num_classes 的混淆矩阵，行=真实类别，列=预测类别。"""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= int(t) < num_classes and 0 <= int(p) < num_classes:
            cm[int(t), int(p)] += 1
    return cm.tolist()


def classification_report(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
    confusion: List[List[int]] | None = None,
) -> Dict:
    """计算准确率 / 宏平均精确率 / 召回率 / F1 以及每个类别的明细指标。"""
    num_classes = len(class_names)
    cm = np.array(confusion if confusion is not None else confusion_matrix(y_true, y_pred, num_classes))
    total = int(cm.sum())
    correct = int(np.trace(cm))

    per_class = []
    for i, name in enumerate(class_names):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = int(cm[i, :].sum())
        per_class.append(
            {
                "index": i,
                "label": name,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "support": support,
            }
        )

    macro_precision = float(np.mean([c["precision"] for c in per_class])) if per_class else 0.0
    macro_recall = float(np.mean([c["recall"] for c in per_class])) if per_class else 0.0
    macro_f1 = float(np.mean([c["f1"] for c in per_class])) if per_class else 0.0

    return {
        "accuracy": round(correct / total, 4) if total else 0.0,
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "samples": total,
        "num_classes": num_classes,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "class_names": list(class_names),
    }


@torch.no_grad()
def evaluate_model(model, loader, device, class_names: Sequence[str], criterion=None) -> Dict:
    """在给定数据加载器上评估模型，返回损失与完整指标。"""
    model.eval()
    criterion = criterion or torch.nn.CrossEntropyLoss()
    total_loss, total, correct = 0.0, 0, 0
    all_true: List[int] = []
    all_pred: List[int] = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(images)
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]
        loss = criterion(outputs, targets)
        total_loss += float(loss.item()) * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += int((preds == targets).sum().item())
        total += images.size(0)
        all_true.extend(targets.detach().cpu().tolist())
        all_pred.extend(preds.detach().cpu().tolist())

    report = classification_report(all_true, all_pred, list(class_names), None)
    report["loss"] = round(total_loss / total, 4) if total else 0.0
    return report
