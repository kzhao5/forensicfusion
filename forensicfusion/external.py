from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np


@dataclass
class ExternalPredictionStore:
    """Reader for externally produced heatmaps.

    Expected layout:
      root/<dataset>/<method>/<sample_id>.png|jpg|npy
    or
      root/<method>/<dataset>/<sample_id>.*
    """

    root: Path

    def _candidate_paths(self, dataset: str, method: str, sample_id: str) -> List[Path]:
        exts = [".npy", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]
        bases = [
            self.root / dataset / method / sample_id,
            self.root / method / dataset / sample_id,
            self.root / method / sample_id,
            self.root / dataset / sample_id,
        ]
        out: List[Path] = []
        for b in bases:
            for ext in exts:
                out.append(Path(str(b) + ext))
        return out

    def has(self, dataset: str, method: str, sample_id: str) -> bool:
        return any(p.exists() for p in self._candidate_paths(dataset, method, sample_id))

    def load(self, dataset: str, method: str, sample_id: str, target_shape: Optional[tuple[int, int]] = None) -> np.ndarray:
        for p in self._candidate_paths(dataset, method, sample_id):
            if p.exists():
                if p.suffix.lower() == ".npy":
                    arr = np.load(p).astype(np.float32)
                else:
                    arr = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                    if arr is None:
                        raise FileNotFoundError(p)
                    arr = arr.astype(np.float32) / 255.0
                if arr.ndim == 3:
                    arr = arr.mean(axis=2)
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                if target_shape is not None and arr.shape[:2] != target_shape:
                    arr = cv2.resize(arr, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
                # robust normalize in case incoming range is not [0,1]
                lo = float(np.percentile(arr, 1))
                hi = float(np.percentile(arr, 99))
                if hi > lo:
                    arr = np.clip((arr - lo) / (hi - lo + 1e-8), 0.0, 1.0)
                else:
                    arr = np.zeros_like(arr, dtype=np.float32)
                return arr.astype(np.float32)
        raise FileNotFoundError(f"No prediction found for dataset={dataset}, method={method}, sample_id={sample_id}")
