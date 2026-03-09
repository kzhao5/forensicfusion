from __future__ import annotations

import time
from typing import Callable, Dict, List

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from .data import DatasetIndex, read_mask
from .metrics import MetricBundle, image_score, nanmean


PredictionFn = Callable[[object], np.ndarray]


def evaluate_method_on_split(
    dataset: DatasetIndex,
    split: str,
    method_name: str,
    predict_fn: PredictionFn,
    thr: float = 0.5,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for s in dataset.split(split):
        img = cv2.imread(str(s.image_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]

        t0 = time.perf_counter()
        pred = np.asarray(predict_fn(s), dtype=np.float32)
        runtime_s = time.perf_counter() - t0
        if pred.shape[:2] != (h, w):
            pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
        pred = np.clip(np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)

        row: Dict[str, object] = {
            "dataset": dataset.name,
            "id": s.sample_id,
            "split": s.split,
            "manip_type": s.manip_type,
            "label": int(s.label),
            "method": method_name,
            "image_score": image_score(pred, mode="max"),
            "runtime_s": runtime_s,
        }
        if s.mask_path is not None:
            gt = read_mask(s.mask_path, target_shape=(h, w))
            mb = MetricBundle.from_pred_gt(pred, gt, thr=thr)
            row.update({"AUC": mb.auc, "mIoU": mb.miou, "F1": mb.f1})
        else:
            row.update({"AUC": np.nan, "mIoU": np.nan, "F1": np.nan})
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_localization(df: pd.DataFrame) -> Dict[str, float]:
    return {
        "AUC": nanmean(df["AUC"].to_numpy(np.float32)),
        "mIoU": nanmean(df["mIoU"].to_numpy(np.float32)),
        "F1": nanmean(df["F1"].to_numpy(np.float32)),
        "Runtime": nanmean(df["runtime_s"].to_numpy(np.float32)),
    }


def summarize_detection(df: pd.DataFrame) -> Dict[str, float]:
    y = df["label"].to_numpy(np.int32)
    s = df["image_score"].to_numpy(np.float32)
    acc = accuracy_score(y, (s >= 0.5).astype(np.int32)) if len(np.unique(y)) > 1 else float("nan")
    auc = roc_auc_score(y, s) if len(np.unique(y)) > 1 else float("nan")
    return {"Acc": float(acc), "AUC": float(auc)}
