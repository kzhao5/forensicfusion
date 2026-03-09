from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import cv2


def _entropy(gray: np.ndarray, bins: int = 256, eps: float = 1e-12) -> float:
    hist = np.histogram(gray.astype(np.uint8).reshape(-1), bins=bins, range=(0, 255))[0].astype(np.float32)
    p = hist / (np.sum(hist) + eps)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p + eps)))


def extract_image_features(impath: str) -> np.ndarray:
    """Cheap, deterministic image descriptor for path selection.

    Returns a float32 vector.
    """
    bgr = cv2.imread(impath, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(impath)
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    mean = float(np.mean(gray))
    std = float(np.std(gray))
    ent = _entropy(gray)

    # edge density
    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.mean(edges > 0))

    # saturation (clipped highlights/shadows)
    sat = float(np.mean((gray <= 2) | (gray >= 253)))

    is_jpg = float(impath.lower().endswith((".jpg", ".jpeg")))
    is_png = float(impath.lower().endswith(".png"))

    # Normalize numeric ranges a bit
    feats = np.array([
        np.log1p(h), np.log1p(w),
        mean / 255.0, std / 255.0,
        ent / 8.0,  # entropy max is 8 for 256 bins
        edge_density,
        sat,
        is_jpg, is_png,
    ], dtype=np.float32)
    return feats


def manip_type_onehot(manip_type: str) -> np.ndarray:
    types = ["splicing", "copy_move", "inpainting", "ai_generated", "unknown"]
    manip_type = manip_type.lower()
    v = np.zeros((len(types),), dtype=np.float32)
    if manip_type not in types:
        manip_type = "unknown"
    v[types.index(manip_type)] = 1.0
    return v
