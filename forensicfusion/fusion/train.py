from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from tqdm import tqdm

from ..cache import OutputCache
from ..data import DatasetIndex, read_mask
from ..path import fuse_maps
from ..supernet import ForensicSupernet
from .weights import FusionWeights


@dataclass
class FusionTrainConfig:
    type_names: Tuple[str, ...] = ("splicing", "copy_move", "inpainting", "ai_generated", "unknown")
    lr: float = 1e-2
    weight_decay: float = 1e-4
    epochs: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_side: int = 384  # downsample for faster weight training
    bce_weight: float = 1.0
    dice_weight: float = 1.0


def _resize_max_side(arr: np.ndarray, max_side: int, interp: int) -> np.ndarray:
    h, w = arr.shape[:2]
    if max(h, w) <= max_side:
        return arr
    scale = max_side / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    return cv2.resize(arr, (nw, nh), interpolation=interp)


def _dice_loss(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = pred.reshape(-1)
    gt = gt.reshape(-1)
    inter = (pred * gt).sum()
    union = pred.sum() + gt.sum()
    dice = (2 * inter + eps) / (union + eps)
    return 1.0 - dice


def train_fusion_weights(
    dataset: DatasetIndex,
    supernet: ForensicSupernet,
    cache: OutputCache,
    out_dir: str | Path,
    cfg: Optional[FusionTrainConfig] = None,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = cfg or FusionTrainConfig()

    module_ids = supernet.module_ids
    type_names = list(cfg.type_names)

    # Learnable logits per type x module
    logits = torch.zeros((len(type_names), len(module_ids)), dtype=torch.float32, requires_grad=True, device=cfg.device)
    opt = torch.optim.Adam([logits], lr=cfg.lr, weight_decay=cfg.weight_decay)

    train_samples = [s for s in dataset.split("train") if s.mask_path is not None]
    if len(train_samples) == 0:
        raise RuntimeError("No train samples with masks found.")

    for epoch in range(1, cfg.epochs + 1):
        losses = []
        rng = np.random.default_rng(seed=epoch)
        order = rng.permutation(len(train_samples))

        for idx in tqdm(order, desc=f"fusion epoch {epoch}/{cfg.epochs}"):
            s = train_samples[int(idx)]
            # load GT mask and image size
            img = cv2.imread(str(s.image_path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            h, w = img.shape[:2]
            gt = read_mask(s.mask_path, target_shape=(h, w))
            # downsample
            gt_ds = _resize_max_side(gt, cfg.max_side, interp=cv2.INTER_NEAREST).astype(np.float32)

            # load module maps (all modules, downsample)
            maps = cache.load_many(dataset.name, s.sample_id, module_ids)
            if len(maps) == 0:
                continue
            maps_ds = {}
            for mid, m in maps.items():
                m2 = _resize_max_side(m, cfg.max_side, interp=cv2.INTER_LINEAR).astype(np.float32)
                maps_ds[mid] = m2

            # Select type row
            mt = s.manip_type.lower()
            if mt not in type_names:
                mt = "unknown" if "unknown" in type_names else type_names[-1]
            t = type_names.index(mt)

            # weights over all modules (only those present)
            present = [mid for mid in module_ids if mid in maps_ds]
            if len(present) == 0:
                continue
            idxs = torch.tensor([module_ids.index(mid) for mid in present], device=cfg.device, dtype=torch.long)
            logits_sub = logits[t, idxs]
            w_sub = torch.softmax(logits_sub, dim=0)  # (P,)

            stack = torch.from_numpy(np.stack([maps_ds[mid] for mid in present], 0)).to(cfg.device)  # (P,H,W)
            fused = torch.tensordot(w_sub, stack, dims=([0], [0]))  # (H,W)
            fused = fused.clamp(0.0, 1.0)

            gt_t = torch.from_numpy(gt_ds).to(cfg.device)
            bce = F.binary_cross_entropy(fused, gt_t)
            dice = _dice_loss(fused, gt_t)

            loss = cfg.bce_weight * bce + cfg.dice_weight * dice

            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))

        mean_loss = float(np.mean(losses)) if losses else float("nan")
        print(f"[fusion epoch {epoch:02d}] loss={mean_loss:.4f}")

    ckpt = out_dir / "fusion_weights.pt"
    fw = FusionWeights(module_ids=module_ids, type_names=type_names, logits=logits.detach().cpu().numpy())
    fw.save(ckpt)
    print(f"Saved fusion weights -> {ckpt}")
    return ckpt
