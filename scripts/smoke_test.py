import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import shutil

import cv2
import numpy as np
import pandas as pd

from forensicfusion.algorithms.pyifd_wrappers import build_default_pyifd_supernet
from forensicfusion.cache import OutputCache
from forensicfusion.data import DatasetIndex
from forensicfusion.fusion.train import FusionTrainConfig, train_fusion_weights
from forensicfusion.predictor.dataset import GNNTrainConfig
from forensicfusion.predictor.train import OptimConfig, train_gnn_predictor
from forensicfusion.supernet import ForensicSupernet


def build_toy(root: Path):
    if root.exists():
        shutil.rmtree(root)
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "masks").mkdir(parents=True, exist_ok=True)
    rows = []
    for i, split in enumerate(["train", "val", "test", "test"]):
        img = np.full((128, 128, 3), 180, np.uint8)
        mask = np.zeros((128, 128), np.uint8)
        if i < 3:
            cv2.rectangle(img, (30, 30), (80, 90), (50 + 40 * i, 30, 200 - 20 * i), -1)
            cv2.rectangle(mask, (30, 30), (80, 90), 255, -1)
            cv2.imwrite(str(root / "masks" / f"{i:06d}.png"), mask)
            label = 1
            mtype = "splicing"
        else:
            label = 0
            mtype = "unknown"
        cv2.imwrite(str(root / "images" / f"{i:06d}.jpg"), img)
        rows.append({"id": f"{i:06d}", "split": split, "manip_type": mtype, "label": label})
    pd.DataFrame(rows).to_csv(root / "meta.csv", index=False)


def main():
    root = ROOT / "tmp_smoke" / "toy"
    build_toy(root)
    ds = DatasetIndex.from_folder("toy", root)
    supernet = ForensicSupernet(build_default_pyifd_supernet(include_optional=False))
    cache = OutputCache(ROOT / "tmp_smoke" / "cache")

    for s in ds.samples:
        for mid in supernet.module_ids:
            if not cache.has(ds.name, s.sample_id, mid):
                cache.save(ds.name, s.sample_id, mid, supernet.get(mid).run(str(s.image_path)))

    gnn_ckpt = train_gnn_predictor(ds, supernet, cache, ROOT / "tmp_smoke" / "runs_gnn", gnn_cfg=GNNTrainConfig(K_paths_per_image=5), opt_cfg=OptimConfig(epochs=2, batch_size=8))
    fusion_ckpt = train_fusion_weights(ds, supernet, cache, ROOT / "tmp_smoke" / "runs_fusion", cfg=FusionTrainConfig(epochs=2))
    print("Smoke test passed")
    print("GNN:", gnn_ckpt)
    print("Fusion:", fusion_ckpt)


if __name__ == "__main__":
    main()
