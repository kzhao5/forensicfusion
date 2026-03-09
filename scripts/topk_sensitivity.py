import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from pathlib import Path

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
    ap.add_argument("--K", type=int, default=50)
    ap.add_argument("--topks", default="1,3,5,7")
    args = ap.parse_args()

    ds = DatasetIndex.from_folder(args.dataset, Path(args.data_root) / args.dataset)
    supernet = ForensicSupernet(build_default_pyifd_supernet(include_optional=args.include_optional))
    cache = OutputCache(Path(args.cache_root))
    gnn = load_gnn_checkpoint(args.gnn_ckpt)
    fusion_weights = FusionWeights.load(args.fusion_ckpt) if args.fusion_ckpt else None
    topks = [int(x) for x in args.topks.split(",") if x.strip()]

    rows, frames = [], []
    for top_k in topks:
        fn = lambda s, _top_k=top_k: run_method_gnn(
            supernet,
            str(s.image_path),
            s.manip_type,
            gnn,
            RunConfig(K=args.K, min_len=1, max_len=4, seed=0, top_k=_top_k, fuse_across_paths="score_softmax" if fusion_weights else "uniform"),
            cache,
            ds.name,
            s.sample_id,
            fusion_weights=fusion_weights,
            path_fusion_mode="learned" if fusion_weights else "uniform",
        )[0]
        df = evaluate_method_on_split(ds, args.split, f"topk_{top_k}", fn, thr=0.5)
        frames.append(df)
        loc = summarize_localization(df)
        rows.append({"top-$k$": top_k, "AUC$\\uparrow$": loc["AUC"], "mIoU$\\uparrow$": loc["mIoU"], "F1$\\uparrow$": loc["F1"]})

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(results_dir / f"topk_sensitivity_{args.dataset}.csv", index=False)
    tex = make_cvpr_table(
        columns=["top-$k$", "AUC$\\uparrow$", "mIoU$\\uparrow$", "F1$\\uparrow$"],
        rows=rows,
        caption="Sensitivity to the number of selected paths fused at inference.",
        label="tab:topk_sensitivity",
        align="cccc",
    )
    tex_path = results_dir / f"topk_sensitivity_{args.dataset}.tex"
    tex_path.write_text(tex, encoding="utf-8")
    print(f"Saved {tex_path}")


if __name__ == "__main__":
    main()
