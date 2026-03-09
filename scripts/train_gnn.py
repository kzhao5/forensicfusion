import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from pathlib import Path

from forensicfusion.algorithms.pyifd_wrappers import build_default_pyifd_supernet
from forensicfusion.cache import OutputCache
from forensicfusion.data import DatasetIndex
from forensicfusion.predictor.dataset import GNNTrainConfig
from forensicfusion.predictor.train import OptimConfig, train_gnn_predictor
from forensicfusion.supernet import ForensicSupernet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--cache_root", default="cache")
    ap.add_argument("--out_dir", default="runs/gnn")
    ap.add_argument("--include_optional", action="store_true")
    ap.add_argument("--K", type=int, default=50, help="sampled paths per training image")
    ap.add_argument("--min_len", type=int, default=1)
    ap.add_argument("--max_len", type=int, default=4)
    ap.add_argument("--max_nodes", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    args = ap.parse_args()

    ds = DatasetIndex.from_folder(args.dataset, Path(args.data_root) / args.dataset)
    supernet = ForensicSupernet(build_default_pyifd_supernet(include_optional=args.include_optional))
    cache = OutputCache(Path(args.cache_root))

    gnn_cfg = GNNTrainConfig(
        K_paths_per_image=args.K,
        min_len=args.min_len,
        max_len=args.max_len,
        max_nodes=args.max_nodes,
    )
    opt_cfg = OptimConfig(
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )
    ckpt = train_gnn_predictor(ds, supernet, cache, args.out_dir, gnn_cfg=gnn_cfg, opt_cfg=opt_cfg)
    print(f"Saved GNN checkpoint -> {ckpt}")


if __name__ == "__main__":
    main()
