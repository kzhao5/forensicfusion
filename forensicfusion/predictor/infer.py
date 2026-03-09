from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from ..path import ForensicPath
from ..sampling import SampleConfig
from ..supernet import ForensicSupernet
from .features import extract_image_features, manip_type_onehot
from .gnn import PathPredictorGNN


@dataclass
class LoadedGNN:
    model: PathPredictorGNN
    module_vocab: Dict[str, int]
    max_nodes: int
    device: str


def load_gnn_checkpoint(ckpt_path: str | Path, device: str | None = None) -> LoadedGNN:
    ckpt_path = Path(ckpt_path)
    data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    module_vocab = data["module_vocab"]
    img_feat_dim = int(data["img_feat_dim"])
    type_dim = int(data["type_dim"])
    gnn_cfg = data.get("gnn_cfg", {})
    max_nodes = int(gnn_cfg.get("max_nodes", 8))
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = PathPredictorGNN(
        num_modules=len(module_vocab),
        img_feat_dim=img_feat_dim,
        type_dim=type_dim,
    )
    model.load_state_dict(data["model_state"])
    model.eval().to(device)
    return LoadedGNN(model=model, module_vocab=module_vocab, max_nodes=max_nodes, device=device)


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


@torch.no_grad()
def score_paths(
    gnn: LoadedGNN,
    paths: Sequence[ForensicPath],
    impath: str,
    manip_type: str,
) -> np.ndarray:
    """Return predicted scores y_hat for each path."""
    img_feat = extract_image_features(impath)
    type_oh = manip_type_onehot(manip_type)

    ids_batch = []
    mask_batch = []
    for p in paths:
        module_idx = [gnn.module_vocab[mid] for mid in p.module_ids if mid in gnn.module_vocab]
        ids_pad, m = _pad_path(module_idx, gnn.max_nodes)
        ids_batch.append(ids_pad)
        mask_batch.append(m)

    ids = torch.from_numpy(np.stack(ids_batch, 0)).to(gnn.device)
    m = torch.from_numpy(np.stack(mask_batch, 0)).to(gnn.device)
    img = torch.from_numpy(np.tile(img_feat[None, :], (len(paths), 1))).to(gnn.device)
    typ = torch.from_numpy(np.tile(type_oh[None, :], (len(paths), 1))).to(gnn.device)

    y_hat = gnn.model(ids, m, img, typ)
    return y_hat.detach().cpu().numpy().astype(np.float32)
