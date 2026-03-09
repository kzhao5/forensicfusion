from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..cache import OutputCache
from ..data import DatasetIndex
from ..supernet import ForensicSupernet
from .dataset import GNNTrainConfig, build_path_performance_dataset
from .gnn import PathPredictorGNN


@dataclass
class OptimConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    epochs: int = 15
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def train_gnn_predictor(
    dataset: DatasetIndex,
    supernet: ForensicSupernet,
    cache: OutputCache,
    out_dir: str | Path,
    gnn_cfg: Optional[GNNTrainConfig] = None,
    opt_cfg: Optional[OptimConfig] = None,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gnn_cfg = gnn_cfg or GNNTrainConfig()
    opt_cfg = opt_cfg or OptimConfig()

    # Build module vocabulary
    module_vocab: Dict[str, int] = {mid: i for i, mid in enumerate(supernet.module_ids)}

    # Build datasets
    train_ds = build_path_performance_dataset(dataset, "train", supernet, cache, module_vocab, gnn_cfg)
    val_split = "val" if len(dataset.split("val")) > 0 else "train"
    val_ds = build_path_performance_dataset(dataset, val_split, supernet, cache, module_vocab, gnn_cfg)

    train_loader = DataLoader(train_ds, batch_size=opt_cfg.batch_size, shuffle=True, num_workers=opt_cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=opt_cfg.batch_size, shuffle=False, num_workers=opt_cfg.num_workers)

    # Model
    img_feat_dim = train_ds.img_feat.shape[1]
    type_dim = train_ds.type_oh.shape[1]
    model = PathPredictorGNN(
        num_modules=len(module_vocab),
        img_feat_dim=img_feat_dim,
        type_dim=type_dim,
    ).to(opt_cfg.device)

    opt = torch.optim.Adam(model.parameters(), lr=opt_cfg.lr, weight_decay=opt_cfg.weight_decay)
    loss_fn = torch.nn.MSELoss()

    best_val = float("inf")
    best_path = out_dir / "ckpt.pt"

    for epoch in range(1, opt_cfg.epochs + 1):
        model.train()
        tr_losses = []
        for ids, m, img_feat, type_oh, y in train_loader:
            ids = ids.to(opt_cfg.device)
            m = m.to(opt_cfg.device)
            img_feat = img_feat.to(opt_cfg.device)
            type_oh = type_oh.to(opt_cfg.device)
            y = y.to(opt_cfg.device)

            y_hat = model(ids, m, img_feat, type_oh)
            loss = loss_fn(y_hat, y)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tr_losses.append(float(loss.detach().cpu().item()))

        # val
        model.eval()
        va_losses = []
        with torch.no_grad():
            for ids, m, img_feat, type_oh, y in val_loader:
                ids = ids.to(opt_cfg.device)
                m = m.to(opt_cfg.device)
                img_feat = img_feat.to(opt_cfg.device)
                type_oh = type_oh.to(opt_cfg.device)
                y = y.to(opt_cfg.device)
                y_hat = model(ids, m, img_feat, type_oh)
                loss = loss_fn(y_hat, y)
                va_losses.append(float(loss.detach().cpu().item()))

        tr = float(np.mean(tr_losses)) if tr_losses else float("nan")
        va = float(np.mean(va_losses)) if va_losses else float("nan")
        print(f"[epoch {epoch:02d}] train={tr:.4f} val={va:.4f}")

        if va < best_val:
            best_val = va
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "module_vocab": module_vocab,
                    "img_feat_dim": img_feat_dim,
                    "type_dim": type_dim,
                    "gnn_cfg": gnn_cfg.__dict__,
                },
                best_path,
            )
            print(f"  saved best -> {best_path}")

    return best_path
