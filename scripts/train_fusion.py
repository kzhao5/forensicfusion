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
from forensicfusion.fusion.train import FusionTrainConfig, train_fusion_weights
from forensicfusion.supernet import ForensicSupernet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--cache_root", default="cache")
    ap.add_argument("--out_dir", default="runs/fusion")
    ap.add_argument("--include_optional", action="store_true")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--max_side", type=int, default=384)
    args = ap.parse_args()

    ds = DatasetIndex.from_folder(args.dataset, Path(args.data_root) / args.dataset)
    supernet = ForensicSupernet(build_default_pyifd_supernet(include_optional=args.include_optional))
    cache = OutputCache(Path(args.cache_root))
    cfg = FusionTrainConfig(epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay, max_side=args.max_side)
    ckpt = train_fusion_weights(ds, supernet, cache, args.out_dir, cfg=cfg)
    print(f"Saved fusion weights -> {ckpt}")


if __name__ == "__main__":
    main()
