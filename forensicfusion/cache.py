from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np


@dataclass
class OutputCache:
    root: Path

    def __post_init__(self):
        self.root.mkdir(parents=True, exist_ok=True)

    def sample_dir(self, dataset_name: str, sample_id: str) -> Path:
        d = self.root / dataset_name / sample_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def heatmap_path(self, dataset_name: str, sample_id: str, module_id: str) -> Path:
        return self.sample_dir(dataset_name, sample_id) / f"{module_id}.npy"

    def has(self, dataset_name: str, sample_id: str, module_id: str) -> bool:
        return self.heatmap_path(dataset_name, sample_id, module_id).exists()

    def save(self, dataset_name: str, sample_id: str, module_id: str, heatmap: np.ndarray) -> None:
        p = self.heatmap_path(dataset_name, sample_id, module_id)
        np.save(p, heatmap.astype(np.float32))

    def load(self, dataset_name: str, sample_id: str, module_id: str) -> np.ndarray:
        p = self.heatmap_path(dataset_name, sample_id, module_id)
        return np.load(p).astype(np.float32)

    def load_many(self, dataset_name: str, sample_id: str, module_ids: Iterable[str]) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for mid in module_ids:
            p = self.heatmap_path(dataset_name, sample_id, mid)
            if p.exists():
                out[mid] = np.load(p).astype(np.float32)
        return out
