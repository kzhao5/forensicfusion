from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from ..cache import OutputCache
from ..data import DatasetIndex, read_mask
from ..metrics import pixel_auc
from ..path import ForensicPath, fuse_maps
from ..sampling import SampleConfig, sample_paths_random
from ..supernet import ForensicSupernet
from .features import extract_image_features, manip_type_onehot


@dataclass
class GNNTrainConfig:
    K_paths_per_image: int = 50
    min_len: int = 1
    max_len: int = 4
    fusion_inside_path: str = "uniform"
    seed: int = 0
    max_nodes: int = 8  # max path length after padding


class PathPerformanceDataset(Dataset):
    def __init__(
        self,
        module_ids_padded: np.ndarray,  # (N, max_nodes) int64
        mask: np.ndarray,               # (N, max_nodes) float32
        img_feat: np.ndarray,           # (N, F) float32
        type_oh: np.ndarray,            # (N, T) float32
        y: np.ndarray,                  # (N,) float32 in [0,1]
    ):
        self.module_ids_padded = module_ids_padded.astype(np.int64)
        self.mask = mask.astype(np.float32)
        self.img_feat = img_feat.astype(np.float32)
        self.type_oh = type_oh.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.module_ids_padded[idx]),
            torch.from_numpy(self.mask[idx]),
            torch.from_numpy(self.img_feat[idx]),
            torch.from_numpy(self.type_oh[idx]),
            torch.from_numpy(np.array(self.y[idx], dtype=np.float32)),
        )


def _pad_path(module_idx: List[int], max_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
    n = len(module_idx)
    if n > max_nodes:
        module_idx = module_idx[:max_nodes]
        n = max_nodes
    ids = np.zeros((max_nodes,), dtype=np.int64)
    m = np.zeros((max_nodes,), dtype=np.float32)
    ids[:n] = np.array(module_idx, dtype=np.int64)
    m[:n] = 1.0
    return ids, m


def build_path_performance_dataset(
    dataset: DatasetIndex,
    split: str,
    supernet: ForensicSupernet,
    cache: OutputCache,
    module_vocab: Dict[str, int],
    cfg: GNNTrainConfig,
    dataset_name: Optional[str] = None,
) -> PathPerformanceDataset:
    """Build training data by sampling paths and computing their GT performance (pixel AUC)."""
    if dataset_name is None:
        dataset_name = dataset.name

    samples = dataset.split(split)
    rows_ids: List[np.ndarray] = []
    rows_mask: List[np.ndarray] = []
    rows_imgfeat: List[np.ndarray] = []
    rows_type: List[np.ndarray] = []
    rows_y: List[float] = []

    for si, s in enumerate(samples):
        if s.mask_path is None:
            continue  # skip detection-only
        # read GT mask aligned to image
        img = cv2.imread(str(s.image_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        gt = read_mask(s.mask_path, target_shape=(h, w))

        # image + type features
        img_feat = extract_image_features(str(s.image_path))
        type_feat = manip_type_onehot(s.manip_type)

        # candidate paths
        sample_cfg = SampleConfig(
            K=cfg.K_paths_per_image,
            min_len=cfg.min_len,
            max_len=cfg.max_len,
            seed=cfg.seed + si,
            fusion=cfg.fusion_inside_path,
        )
        paths = sample_paths_random(supernet, s.manip_type, sample_cfg)

        # load module maps from cache (all modules)
        module_maps = cache.load_many(dataset_name, s.sample_id, supernet.module_ids)

        for p in paths:
            fused = fuse_maps(module_maps, p.module_ids, fusion=p.fusion, weights=p.weights)
            y = pixel_auc(fused, gt)
            if not np.isfinite(y):
                continue
            module_idx = [module_vocab[mid] for mid in p.module_ids if mid in module_vocab]
            ids_pad, m = _pad_path(module_idx, cfg.max_nodes)

            rows_ids.append(ids_pad)
            rows_mask.append(m)
            rows_imgfeat.append(img_feat)
            rows_type.append(type_feat)
            rows_y.append(float(y))

    if len(rows_y) == 0:
        raise RuntimeError(f"No training samples built for split={split}. Check masks/cache paths.")

    module_ids_padded = np.stack(rows_ids, axis=0)
    mask = np.stack(rows_mask, axis=0)
    img_feat = np.stack(rows_imgfeat, axis=0)
    type_oh = np.stack(rows_type, axis=0)
    y = np.array(rows_y, dtype=np.float32)

    return PathPerformanceDataset(module_ids_padded, mask, img_feat, type_oh, y)
