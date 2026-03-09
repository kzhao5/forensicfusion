from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score


def _flatten_valid(pred: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(pred, dtype=np.float32).reshape(-1)
    gt = np.asarray(gt, dtype=np.float32).reshape(-1)
    # remove NaNs if any
    m = np.isfinite(pred) & np.isfinite(gt)
    pred = pred[m]
    gt = gt[m]
    return pred, gt


def pixel_auc(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_f, gt_f = _flatten_valid(pred, gt)
    # roc_auc_score requires both classes present
    if gt_f.max() == gt_f.min():
        return float("nan")
    return float(roc_auc_score(gt_f, pred_f))


def pixel_f1(pred: np.ndarray, gt: np.ndarray, thr: float = 0.5) -> float:
    pred_f, gt_f = _flatten_valid(pred, gt)
    pb = (pred_f >= thr).astype(np.float32)
    gb = (gt_f >= 0.5).astype(np.float32)
    tp = float(np.sum((pb == 1) & (gb == 1)))
    fp = float(np.sum((pb == 1) & (gb == 0)))
    fn = float(np.sum((pb == 0) & (gb == 1)))
    denom = (2 * tp + fp + fn)
    if denom <= 0:
        return 0.0
    return float(2 * tp / denom)


def pixel_iou(pred: np.ndarray, gt: np.ndarray, thr: float = 0.5) -> float:
    pred_f, gt_f = _flatten_valid(pred, gt)
    pb = (pred_f >= thr).astype(np.float32)
    gb = (gt_f >= 0.5).astype(np.float32)
    inter = float(np.sum((pb == 1) & (gb == 1)))
    union = float(np.sum((pb == 1) | (gb == 1)))
    if union <= 0:
        return 0.0
    return float(inter / union)


def image_score(pred: np.ndarray, mode: str = "max") -> float:
    if mode == "max":
        return float(np.max(pred))
    if mode == "mean":
        return float(np.mean(pred))
    if mode == "p95":
        return float(np.quantile(pred, 0.95))
    raise ValueError(mode)


@dataclass
class MetricBundle:
    auc: float
    miou: float
    f1: float

    @staticmethod
    def from_pred_gt(pred: np.ndarray, gt: np.ndarray, thr: float = 0.5) -> "MetricBundle":
        return MetricBundle(
            auc=pixel_auc(pred, gt),
            miou=pixel_iou(pred, gt, thr=thr),
            f1=pixel_f1(pred, gt, thr=thr),
        )

    def as_dict(self) -> Dict[str, float]:
        return {"AUC": self.auc, "mIoU": self.miou, "F1": self.f1}


def nanmean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    if np.all(~np.isfinite(x)):
        return float("nan")
    return float(np.nanmean(x))
