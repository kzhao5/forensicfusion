from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .cache import OutputCache
from .fusion.weights import FusionWeights
from .path import ForensicPath, fuse_maps
from .predictor.infer import LoadedGNN, load_gnn_checkpoint, score_paths
from .sampling import SampleConfig, sample_path_heuristic, sample_paths_random
from .supernet import ForensicSupernet


@dataclass
class RunConfig:
    K: int = 50
    min_len: int = 1
    max_len: int = 4
    seed: int = 0
    top_k: int = 1          # how many paths to keep after selection
    fuse_across_paths: str = "uniform"  # uniform|score_softmax


def _softmax(x: np.ndarray, tau: float = 1.0) -> np.ndarray:
    x = x.astype(np.float32) / max(tau, 1e-6)
    x = x - float(np.max(x))
    e = np.exp(x)
    return e / (np.sum(e) + 1e-8)


def run_single_path(
    supernet: ForensicSupernet,
    impath: str,
    manip_type: str,
    path: ForensicPath,
    cache: Optional[OutputCache] = None,
    dataset_name: Optional[str] = None,
    sample_id: Optional[str] = None,
    fusion_weights: Optional[FusionWeights] = None,
    verbose: bool = False,
) -> np.ndarray:
    """Run a single path to produce fused heatmap."""
    # load maps from cache if available
    maps: Dict[str, np.ndarray] = {}
    if cache is not None and dataset_name is not None and sample_id is not None:
        maps = cache.load_many(dataset_name, sample_id, path.module_ids)

    missing = [mid for mid in path.module_ids if mid not in maps]
    if len(missing) > 0:
        maps.update(supernet.run_modules(impath, missing, verbose=verbose))

    if fusion_weights is not None and path.fusion in ("learned", "weighted"):
        w = fusion_weights.weights_for(manip_type, path.module_ids)
        return fuse_maps(maps, path.module_ids, fusion="weighted", weights=w)

    return fuse_maps(maps, path.module_ids, fusion=path.fusion, weights=path.weights)


def run_method_random_k(
    supernet: ForensicSupernet,
    impath: str,
    manip_type: str,
    cfg: RunConfig,
    cache: Optional[OutputCache] = None,
    dataset_name: Optional[str] = None,
    sample_id: Optional[str] = None,
    fusion_weights: Optional[FusionWeights] = None,
) -> np.ndarray:
    """Random-K baseline: sample K random paths, fuse their *path outputs* uniformly."""
    scfg = SampleConfig(K=cfg.K, min_len=cfg.min_len, max_len=cfg.max_len, seed=cfg.seed, fusion="uniform")
    paths = sample_paths_random(supernet, manip_type, scfg)
    outs = []
    for p in paths:
        out = run_single_path(supernet, impath, manip_type, p, cache, dataset_name, sample_id, fusion_weights=None)
        outs.append(out)
    if len(outs) == 0:
        raise RuntimeError("No outputs in random_k")
    return np.stack(outs, 0).mean(axis=0).astype(np.float32)


def run_method_uniform_all(
    supernet: ForensicSupernet,
    impath: str,
    manip_type: str,
    cache: Optional[OutputCache] = None,
    dataset_name: Optional[str] = None,
    sample_id: Optional[str] = None,
    fusion_weights: Optional[FusionWeights] = None,
    fusion_mode: str = "uniform",  # uniform|learned
) -> np.ndarray:
    """Uniform-all baseline: run all modules and fuse."""
    path = ForensicPath(module_ids=tuple(supernet.module_ids), fusion="learned" if fusion_mode == "learned" else "uniform")
    return run_single_path(supernet, impath, manip_type, path, cache, dataset_name, sample_id, fusion_weights=fusion_weights)


def run_method_heuristic(
    supernet: ForensicSupernet,
    impath: str,
    manip_type: str,
    cache: Optional[OutputCache] = None,
    dataset_name: Optional[str] = None,
    sample_id: Optional[str] = None,
    fusion_weights: Optional[FusionWeights] = None,
    fusion_mode: str = "uniform",
) -> np.ndarray:
    """Heuristic baseline: a hand-picked path per manipulation type."""
    path = sample_path_heuristic(supernet, manip_type, fusion="learned" if fusion_mode == "learned" else "uniform")
    return run_single_path(supernet, impath, manip_type, path, cache, dataset_name, sample_id, fusion_weights=fusion_weights)


def run_method_gnn(
    supernet: ForensicSupernet,
    impath: str,
    manip_type: str,
    gnn: LoadedGNN,
    cfg: RunConfig,
    cache: Optional[OutputCache] = None,
    dataset_name: Optional[str] = None,
    sample_id: Optional[str] = None,
    fusion_weights: Optional[FusionWeights] = None,
    path_fusion_mode: str = "uniform",  # uniform|learned (within path)
) -> Tuple[np.ndarray, List[Tuple[ForensicPath, float]]]:
    """Our method: sample K candidate paths, score with GNN, select top-k, then fuse across paths."""
    scfg = SampleConfig(K=cfg.K, min_len=cfg.min_len, max_len=cfg.max_len, seed=cfg.seed, fusion="learned" if path_fusion_mode == "learned" else "uniform")
    paths = sample_paths_random(supernet, manip_type, scfg)
    if len(paths) == 0:
        raise RuntimeError("No candidate paths")

    scores = score_paths(gnn, paths, impath, manip_type)  # (K,)
    order = np.argsort(-scores)
    top = [paths[i] for i in order[: cfg.top_k]]
    top_scores = [float(scores[i]) for i in order[: cfg.top_k]]

    outs = []
    for p in top:
        out = run_single_path(supernet, impath, manip_type, p, cache, dataset_name, sample_id, fusion_weights=fusion_weights, verbose=False)
        outs.append(out)

    if len(outs) == 1:
        return outs[0], list(zip(top, top_scores))

    stack = np.stack(outs, 0).astype(np.float32)  # (k,H,W)

    if cfg.fuse_across_paths == "uniform":
        fused = stack.mean(axis=0)
    elif cfg.fuse_across_paths == "score_softmax":
        w = _softmax(np.array(top_scores, dtype=np.float32))
        fused = np.tensordot(w, stack, axes=(0, 0))
    else:
        raise ValueError(cfg.fuse_across_paths)

    return fused.astype(np.float32), list(zip(top, top_scores))
