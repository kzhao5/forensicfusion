from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


def save_heatmap(path: str | Path, arr: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(arr, dtype=np.float32)
    if x.ndim == 3:
        x = x.mean(axis=2)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi > lo:
        x = (x - lo) / (hi - lo + 1e-8)
    else:
        x = np.zeros_like(x)
    cv2.imwrite(str(path), (255.0 * x).astype(np.uint8))


def overlay_on_image(image_bgr: np.ndarray, heatmap: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    hm = np.asarray(heatmap, dtype=np.float32)
    if hm.shape[:2] != (h, w):
        hm = cv2.resize(hm, (w, h), interpolation=cv2.INTER_LINEAR)
    hm = np.nan_to_num(hm, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = float(np.min(hm)), float(np.max(hm))
    if hi > lo:
        hm = (hm - lo) / (hi - lo + 1e-8)
    else:
        hm = np.zeros_like(hm)
    hm_u8 = (255.0 * hm).astype(np.uint8)
    color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
    out = cv2.addWeighted(image_bgr, 1.0 - alpha, color, alpha, 0.0)
    return out


def qualitative_grid(
    image_path: str | Path,
    panels: Sequence[Tuple[str, np.ndarray]],
    out_path: str | Path,
    gt_mask: np.ndarray | None = None,
) -> None:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    ncols = len(panels) + 1 + (1 if gt_mask is not None else 0)
    fig = plt.figure(figsize=(2.3 * ncols, 4.6))
    ax = plt.subplot(1, ncols, 1)
    ax.imshow(image_rgb)
    ax.set_title("Input")
    ax.axis("off")

    idx = 2
    if gt_mask is not None:
        ax = plt.subplot(1, ncols, idx)
        ax.imshow(gt_mask, cmap="gray")
        ax.set_title("GT")
        ax.axis("off")
        idx += 1

    for title, heatmap in panels:
        ax = plt.subplot(1, ncols, idx)
        ax.imshow(heatmap, cmap="jet", vmin=0.0, vmax=1.0)
        ax.set_title(title)
        ax.axis("off")
        idx += 1

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
