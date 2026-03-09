from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

from ..cache import OutputCache
from ..data import DatasetIndex, read_mask, Sample
from ..fusion.weights import FusionWeights
from ..metrics import MetricBundle, nanmean


@dataclass
class EvalConfig:
    split: str = "test"
    save_heatmaps: bool = False
    heatmap_dir: Optional[str] = None
    thr: float = 0.5


def _save_heatmap(out_map: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m = (np.clip(out_map, 0, 1) * 255.0).astype(np.uint8)
    cv2.imwrite(str(out_path), m)


def evaluate_method(
    dataset: DatasetIndex,
    method_name: str,
    method_fn: Callable[[Sample], np.ndarray],
    cfg: Optional[EvalConfig] = None,
) -> pd.DataFrame:
    cfg = cfg or EvalConfig()
    rows: List[Dict[str, object]] = []

    samples = dataset.split(cfg.split)
    for s in tqdm(samples, desc=f"Eval {method_name} ({cfg.split})"):
        t0 = time.perf_counter()
        pred = method_fn(s)
        dt = time.perf_counter() - t0

        row: Dict[str, object] = {
            "id": s.sample_id,
            "split": s.split,
            "manip_type": s.manip_type,
            "runtime_s": dt,
        }

        if s.mask_path is not None:
            img = cv2.imread(str(s.image_path), cv2.IMREAD_COLOR)
            if img is not None:
                h, w = img.shape[:2]
                gt = read_mask(s.mask_path, target_shape=(h, w))
                mb = MetricBundle.from_pred_gt(pred, gt, thr=cfg.thr)
                row.update(mb.as_dict())

        rows.append(row)

        if cfg.save_heatmaps:
            out_dir = Path(cfg.heatmap_dir or "heatmaps") / dataset.name / method_name
            _save_heatmap(pred, out_dir / f"{s.sample_id}.png")

    df = pd.DataFrame(rows)
    return df


def summarize(df: pd.DataFrame) -> Dict[str, float]:
    out = {}
    for k in ["AUC", "mIoU", "F1", "runtime_s"]:
        if k in df.columns:
            out[k] = float(np.nanmean(df[k].astype(float).values))
    return out
