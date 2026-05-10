from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def confusion_matrix(pred: np.ndarray, gt: np.ndarray, num_classes: int, ignore: int = 255) -> np.ndarray:
    valid = gt != ignore
    pred = pred[valid].astype(np.int64)
    gt = gt[valid].astype(np.int64)
    k = (gt >= 0) & (gt < num_classes) & (pred >= 0) & (pred < num_classes)
    return np.bincount(
        num_classes * gt[k] + pred[k], minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)


@dataclass
class SegMetrics:
    miou: float
    pixel_acc: float
    per_class_iou: np.ndarray
    confusion: np.ndarray


def compute_metrics(pred: np.ndarray, gt: np.ndarray, num_classes: int, ignore: int = 255) -> SegMetrics:
    cm = confusion_matrix(pred, gt, num_classes, ignore)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = tp + fp + fn
    iou = np.where(denom > 0, tp / np.maximum(denom, 1), np.nan)
    miou = np.nanmean(iou)
    total = cm.sum()
    pixel_acc = tp.sum() / total if total > 0 else 0.0
    return SegMetrics(miou=float(miou), pixel_acc=float(pixel_acc), per_class_iou=iou, confusion=cm)


def metrics_to_row(m: SegMetrics, class_names: list[str]) -> dict:
    row = {"mIoU": m.miou, "pixel_acc": m.pixel_acc}
    for i, name in enumerate(class_names):
        row[f"IoU_{name}"] = float(m.per_class_iou[i]) if i < len(m.per_class_iou) else float("nan")
    return row
