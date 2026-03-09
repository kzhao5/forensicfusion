from __future__ import annotations

from pathlib import Path
from typing import Dict, Type

from .base import BaselineAdapter
from .catnet import CATNetAdapter
from .mantranet import ManTraNetAdapter
from .mmfusion import MMFusionAdapter
from .omgfuser import OMGFuserAdapter
from .trufor import TruForAdapter


REGISTRY: Dict[str, Type[BaselineAdapter]] = {
    "trufor": TruForAdapter,
    "mmfusion": MMFusionAdapter,
    "cat-net": CATNetAdapter,
    "catnet": CATNetAdapter,
    "mantranet": ManTraNetAdapter,
    "omg-fuser": OMGFuserAdapter,
    "omgfuser": OMGFuserAdapter,
}


def list_adapters() -> list[str]:
    return sorted({cls.spec.name for cls in REGISTRY.values()})


def get_adapter(name: str, project_root: str | Path, repos_root: str | Path | None = None) -> BaselineAdapter:
    key = name.strip().lower()
    if key not in REGISTRY:
        raise KeyError(f"Unknown baseline: {name}. Available: {sorted(REGISTRY)}")
    return REGISTRY[key](project_root=project_root, repos_root=repos_root)
