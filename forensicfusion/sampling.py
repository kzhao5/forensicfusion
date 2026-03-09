from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .path import ForensicPath
from .supernet import ForensicSupernet


DEFAULT_TYPE2FAMILIES: Dict[str, Tuple[str, ...]] = {
    "splicing": ("jpeg", "noise", "cfa"),
    "inpainting": ("jpeg", "noise", "cfa"),
    "copy_move": ("jpeg", "noise"),
    "ai_generated": ("noise", "cfa", "jpeg"),
    "unknown": ("jpeg", "noise", "cfa", "other"),
}


def _eligible_modules(supernet: ForensicSupernet, manip_type: str) -> List[str]:
    fams = DEFAULT_TYPE2FAMILIES.get(manip_type.lower(), DEFAULT_TYPE2FAMILIES["unknown"])
    mids: List[str] = []
    for m in supernet.modules:
        fam = (m.meta.family or "other").lower()
        if fam in fams:
            mids.append(m.meta.module_id)
    return mids if mids else list(supernet.module_ids)


@dataclass
class SampleConfig:
    K: int = 50
    min_len: int = 1
    max_len: int = 4
    seed: int = 0
    fusion: str = "uniform"


def sample_paths_random(supernet: ForensicSupernet, manip_type: str, cfg: SampleConfig) -> List[ForensicPath]:
    rng = random.Random(cfg.seed)
    eligible = _eligible_modules(supernet, manip_type)
    if len(eligible) == 0:
        return []

    min_len = max(1, min(int(cfg.min_len), len(eligible)))
    max_len = max(min_len, min(int(cfg.max_len), len(eligible)))

    seen = set()
    paths: List[ForensicPath] = []
    max_attempts = max(10 * cfg.K, 100)
    attempts = 0
    while len(paths) < cfg.K and attempts < max_attempts:
        attempts += 1
        L = rng.randint(min_len, max_len)
        mids = tuple(sorted(rng.sample(eligible, L)))
        key = (mids, cfg.fusion)
        if key in seen:
            continue
        seen.add(key)
        paths.append(ForensicPath(module_ids=mids, fusion=cfg.fusion))
    return paths


def sample_path_heuristic(supernet: ForensicSupernet, manip_type: str, fusion: str = "uniform") -> ForensicPath:
    eligible = _eligible_modules(supernet, manip_type)
    preferred_order = [
        "pyifd_ela",
        "pyifd_dct",
        "pyifd_noi1",
        "pyifd_noi4",
        "pyifd_ghost",
        "pyifd_blk",
        "pyifd_adq1",
        "pyifd_adq2",
        "pyifd_adq3",
        "pyifd_nadq",
        "pyifd_cfa1",
        "pyifd_cagi",
        "pyifd_cagi_inv",
        "pyifd_noi2",
        "pyifd_noi5",
    ]
    chosen = [mid for mid in preferred_order if mid in eligible]

    mtype = manip_type.lower()
    if mtype == "copy_move":
        keep = {"pyifd_blk", "pyifd_cagi", "pyifd_cagi_inv", "pyifd_ela", "pyifd_noi4", "pyifd_noi1"}
        chosen = [m for m in chosen if m in keep]
    elif mtype == "ai_generated":
        keep = {"pyifd_noi1", "pyifd_noi2", "pyifd_noi4", "pyifd_noi5", "pyifd_cfa1", "pyifd_ghost"}
        chosen = [m for m in chosen if m in keep]
    else:
        keep = {"pyifd_ela", "pyifd_dct", "pyifd_noi1", "pyifd_noi4", "pyifd_ghost", "pyifd_adq1", "pyifd_adq2", "pyifd_blk"}
        chosen = [m for m in chosen if m in keep]

    if len(chosen) == 0:
        chosen = eligible[: min(4, len(eligible))]
    return ForensicPath(module_ids=tuple(sorted(chosen)), fusion=fusion)
