import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from pathlib import Path

import pandas as pd

from forensicfusion.data import DatasetIndex
from forensicfusion.evaluate import evaluate_method_on_split, summarize_detection, summarize_localization
from forensicfusion.external import ExternalPredictionStore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--pred_root", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--out_csv", default=None)
    args = ap.parse_args()

    ds = DatasetIndex.from_folder(args.dataset, Path(args.data_root) / args.dataset)
    store = ExternalPredictionStore(Path(args.pred_root))
    df = evaluate_method_on_split(ds, args.split, args.method, lambda s: store.load(ds.name, args.method, s.sample_id), thr=0.5)
    loc = summarize_localization(df)
    det = summarize_detection(df)
    print({"method": args.method, **loc, **{f"Det{k}": v for k, v in det.items()}})

    out_csv = Path(args.out_csv) if args.out_csv else Path(args.pred_root) / f"{args.dataset}_{args.method}_metrics.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")


if __name__ == "__main__":
    main()
