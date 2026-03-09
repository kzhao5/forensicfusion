from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch


@dataclass
class FusionWeights:
    """Per-manipulation-type fusion weights for module outputs."""
    module_ids: List[str]
    type_names: List[str]
    logits: np.ndarray  # (T, M)

    def weights_for(self, manip_type: str, subset: Sequence[str]) -> np.ndarray:
        manip_type = manip_type.lower()
        if manip_type not in self.type_names:
            manip_type = "unknown" if "unknown" in self.type_names else self.type_names[-1]
        t = self.type_names.index(manip_type)
        mid2i = {m: i for i, m in enumerate(self.module_ids)}
        idx = [mid2i[m] for m in subset if m in mid2i]
        if len(idx) == 0:
            return np.ones((len(subset),), dtype=np.float32) / max(len(subset), 1)
        logits = self.logits[t, idx].astype(np.float32)
        # stable softmax
        logits = logits - float(np.max(logits))
        w = np.exp(logits)
        w = w / (np.sum(w) + 1e-8)
        # expand back to subset order (if subset contains non-existing, assign 0)
        out = []
        j = 0
        for m in subset:
            if m in mid2i:
                out.append(w[j])
                j += 1
            else:
                out.append(0.0)
        out = np.array(out, dtype=np.float32)
        if out.sum() <= 1e-8:
            out = np.ones_like(out) / max(len(out), 1)
        else:
            out = out / (out.sum() + 1e-8)
        return out

    def save(self, path: str | Path) -> None:
        path = Path(path)
        torch.save({"module_ids": self.module_ids, "type_names": self.type_names, "logits": self.logits}, path)

    @staticmethod
    def load(path: str | Path) -> "FusionWeights":
        path = Path(path)
        d = torch.load(path, map_location="cpu", weights_only=False)
        return FusionWeights(module_ids=list(d["module_ids"]), type_names=list(d["type_names"]), logits=np.array(d["logits"], dtype=np.float32))
