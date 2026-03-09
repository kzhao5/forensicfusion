from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphSAGELayer(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.lin_self = nn.Linear(dim_in, dim_out)
        self.lin_neigh = nn.Linear(dim_in, dim_out)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, N, D)"""
        # mean neighbor aggregation on a complete graph (excluding self)
        B, N, D = h.shape
        if N == 1:
            neigh = torch.zeros_like(h)
        else:
            sum_all = h.sum(dim=1, keepdim=True)  # (B,1,D)
            neigh = (sum_all - h) / (N - 1)       # (B,N,D)
        out = self.lin_self(h) + self.lin_neigh(neigh)
        return F.relu(out)


class PathPredictorGNN(nn.Module):
    """Predicts path utility score y_hat in [0,1] (regression) given:
      - a subset of module ids (a 'graph')
      - image features
      - manipulation type one-hot
    """

    def __init__(
        self,
        num_modules: int,
        module_emb_dim: int = 64,
        gnn_hidden: int = 64,
        gnn_layers: int = 3,
        img_feat_dim: int = 9,
        type_dim: int = 5,
        mlp_hidden: int = 128,
    ):
        super().__init__()
        self.emb = nn.Embedding(num_modules, module_emb_dim)

        layers = []
        d_in = module_emb_dim
        for _ in range(gnn_layers):
            layers.append(GraphSAGELayer(d_in, gnn_hidden))
            d_in = gnn_hidden
        self.gnn = nn.ModuleList(layers)

        self.head = nn.Sequential(
            nn.Linear(gnn_hidden + img_feat_dim + type_dim, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.ReLU(),
            nn.Linear(mlp_hidden // 2, 1),
        )

    def forward(self, module_ids_padded: torch.Tensor, mask: torch.Tensor, img_feat: torch.Tensor, type_onehot: torch.Tensor) -> torch.Tensor:
        """
        module_ids_padded: (B, Nmax) int64
        mask: (B, Nmax) float32 {0,1} indicating valid nodes
        img_feat: (B, F)
        type_onehot: (B, T)
        """
        h = self.emb(module_ids_padded)  # (B, Nmax, D)
        # zero out padded nodes
        h = h * mask.unsqueeze(-1)

        # run gnn layers
        for layer in self.gnn:
            h = layer(h) * mask.unsqueeze(-1)

        # mean pool (avoid dividing by 0)
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        g = (h.sum(dim=1) / denom)  # (B, D)

        x = torch.cat([g, img_feat, type_onehot], dim=1)
        y = self.head(x)
        y = torch.sigmoid(y).squeeze(1)
        return y
