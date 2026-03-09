from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from .algorithms.base import ForensicModule


@dataclass
class ForensicSupernet:
    modules: List[ForensicModule]

    def __post_init__(self):
        ids = [m.meta.module_id for m in self.modules]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate module ids in supernet: {ids}")
        self._id2module: Dict[str, ForensicModule] = {m.meta.module_id: m for m in self.modules}

    @property
    def module_ids(self) -> List[str]:
        return [m.meta.module_id for m in self.modules]

    def get(self, module_id: str) -> ForensicModule:
        return self._id2module[module_id]

    def run_modules(self, impath: str, module_ids: Sequence[str], verbose: bool = False) -> Dict[str, np.ndarray]:
        """Run a subset of modules; returns dict[module_id] = heatmap (H,W) in [0,1]."""
        outputs: Dict[str, np.ndarray] = {}
        iterator = tqdm(module_ids, desc="Running modules") if verbose else module_ids
        for mid in iterator:
            mod = self.get(mid)
            try:
                outputs[mid] = mod.run(impath)
            except Exception as e:
                # Fail-soft: a single module error should not crash the pipeline.
                if verbose:
                    print(f"[WARN] module {mid} failed on {impath}: {e}")
                continue
        return outputs
