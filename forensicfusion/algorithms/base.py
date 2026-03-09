from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class ModuleMeta:
    """Metadata for a forensic module."""
    module_id: str
    family: str  # e.g., "jpeg", "noise", "cfa", "copy_move"
    desc: str = ""
    default_params: Dict[str, Any] = field(default_factory=dict)


class ForensicModule:
    """Base class for a forensic algorithm module.

    Convention:
      - Input: image file path
      - Output: heatmap float32 in [0,1] with shape (H, W) aligned to the input image.
    """

    meta: ModuleMeta

    def __init__(self, meta: ModuleMeta):
        self.meta = meta

    def run(self, impath: str) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def normalize_map(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """Min-max normalize to [0,1] in a robust way."""
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 3:
            # Convert RGB/BGR residual maps to single channel
            x = x.mean(axis=2)
        vmin = float(np.nanmin(x))
        vmax = float(np.nanmax(x))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < eps:
            return np.zeros_like(x, dtype=np.float32)
        x = (x - vmin) / (vmax - vmin + eps)
        x = np.clip(x, 0.0, 1.0)
        return x.astype(np.float32)

    @staticmethod
    def ensure_2d(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if x.ndim == 2:
            return x
        if x.ndim == 3:
            return x.mean(axis=2)
        raise ValueError(f"Expected 2D/3D array, got shape={x.shape}")
