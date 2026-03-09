from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class ForensicPath:
    """A candidate forensic analysis path: subset of modules + fusion rule."""
    module_ids: Tuple[str, ...]
    fusion: str = "uniform"  # {"uniform","max","learned"}; learned is handled by fusion module
    # Optional explicit weights aligned to module_ids
    weights: Optional[Tuple[float, ...]] = None

    def __post_init__(self):
        if len(self.module_ids) == 0:
            raise ValueError("ForensicPath must contain at least one module")
        if self.weights is not None and len(self.weights) != len(self.module_ids):
            raise ValueError("weights length must match module_ids length")

    def key(self) -> str:
        return "+".join(self.module_ids) + f"|{self.fusion}"


def fuse_maps(
    maps: Dict[str, np.ndarray],
    module_ids: Sequence[str],
    fusion: str = "uniform",
    weights: Optional[Sequence[float]] = None,
    eps: float = 1e-8,
) -> np.ndarray:
    """Fuse module heatmaps into a single heatmap.

    Args:
      maps: dict of already-aligned (H,W) float maps in [0,1]
      module_ids: ordering
      fusion: 'uniform'|'max'|'weighted'
      weights: optional weights for 'weighted'
    """
    valid = [mid for mid in module_ids if mid in maps]
    if len(valid) == 0:
        # fallback: infer shape from any map
        if len(maps) == 0:
            raise ValueError("No maps to fuse")
        shape = next(iter(maps.values())).shape
        return np.zeros(shape, dtype=np.float32)

    stack = np.stack([maps[mid].astype(np.float32) for mid in valid], axis=0)  # (N,H,W)

    if fusion in ("uniform", "avg", "mean"):
        return stack.mean(axis=0).astype(np.float32)

    if fusion == "max":
        return stack.max(axis=0).astype(np.float32)

    if fusion in ("weighted", "learned"):
        if weights is None:
            # default to uniform
            w = np.ones((len(valid),), dtype=np.float32) / max(len(valid), 1)
        else:
            # map weights from module_ids order -> valid order
            if len(weights) != len(module_ids):
                raise ValueError("weights must match module_ids length")
            w_full = np.asarray(weights, dtype=np.float32)
            w = np.asarray([w_full[list(module_ids).index(mid)] for mid in valid], dtype=np.float32)
            # normalize
            if np.sum(w) <= eps:
                w = np.ones_like(w) / len(w)
            else:
                w = w / (np.sum(w) + eps)
        return np.tensordot(w, stack, axes=(0, 0)).astype(np.float32)

    raise ValueError(f"Unknown fusion: {fusion}")
