import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from pathlib import Path
import random

import numpy as np
import pandas as pd

from forensicfusion.algorithms.pyifd_wrappers import build_default_pyifd_supernet
from forensicfusion.cache import OutputCache
from forensicfusion.data import DatasetIndex
from forensicfusion.evaluate import evaluate_method_on_split, summarize_localization
from forensicfusion.fusion.weights import FusionWeights
from forensicfusion.latex import make_cvpr_table
from forensicfusion.pipeline import RunConfig, run_method_gnn
from forensicfusion.predictor.infer import load_gnn_checkpoint
from forensicfusion.supernet import ForensicSupernet


TYPES = ["splicing", "copy_move", "inpainting", "ai_generated", "unknown"]


def corrupt_type(mt: str, error_rate: float, rng: random.Random) -> str:
    mt = mt.lower()
    if rng.random() > error_rate:
        return mt
    choices = [x for x in TYPES if x != mt]
    return rng.choice(choices) if choices else mt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--cache_root", default="cache")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--include_optional", action="store_true")
    ap.add_argument("--gnn_ckpt", required=True)
    ap.add_argument("--fusion_ckpt", default=None)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--error_rates", default="0.0,0.2,0.4,0.6")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ds = DatasetIndex.from_folder(args.dataset, Path(args.data_root) / args.dataset)
    cache = OutputCache(Path(args.cache_root))
    supernet = ForensicSupernet(build_default_pyifd_supernet(include_optional=args.include_optional))
    gnn = load_gnn_checkpoint(args.gnn_ckpt)
    fusion_weights = FusionWeights.load(args.fusion_ckpt) if args.fusion_ckpt else None
    error_rates = [float(x) for x in args.error_rates.split(",") if x.strip()]

    rows = []
    all_df = []
    for er in error_rates:
        rng = random.Random(args.seed)
        fn = lambda s, _er=er, _rng=rng: run_method_gnn(
            supernet,
            str(s.image_path),
            corrupt_type(s.manip_type, _er, _rng),
            gnn,
            RunConfig(K=50, min_len=1, max_len=4, seed=args.seed, top_k=5, fuse_across_paths="score_softmax" if fusion_weights else "uniform"),
            cache,
            ds.name,
            s.sample_id,
            fusion_weights=fusion_weights,
            path_fusion_mode="learned" if fusion_weights else "uniform",
        )[0]
        df = evaluate_method_on_split(ds, args.split, f"err_{er:.1f}", fn, thr=0.5)
        all_df.append(df)
        loc = summarize_localization(df)
        rows.append({"Type error": er, "AUC$\\uparrow$": loc["AUC"], "mIoU$\\uparrow$": loc["mIoU"], "F1$\\uparrow$": loc["F1"]})

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(all_df, ignore_index=True).to_csv(results_dir / f"type_robustness_{args.dataset}.csv", index=False)
    tex = make_cvpr_table(
        columns=["Type error", "AUC$\\uparrow$", "mIoU$\\uparrow$", "F1$\\uparrow$"],
        rows=rows,
        caption="Robustness to noisy suspected manipulation type input.",
        label="tab:type_robustness",
        align="cccc",
    )
    tex_path = results_dir / f"type_robustness_{args.dataset}.tex"
    tex_path.write_text(tex, encoding="utf-8")
    print(f"Saved {tex_path}")


if __name__ == "__main__":
    main()
