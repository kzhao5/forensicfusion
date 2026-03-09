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
    ap.add_argument("--train_name", required=True, help="name shown in the table, e.g. CASIA_v1")
    ap.add_argument("--test_datasets", required=True, help="comma-separated datasets to evaluate on")
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--cache_root", default="cache")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--include_optional", action="store_true")
    ap.add_argument("--gnn_ckpt", required=True)
    ap.add_argument("--fusion_ckpt", default=None)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = ap.parse_args()

    supernet = ForensicSupernet(build_default_pyifd_supernet(include_optional=args.include_optional))
    cache = OutputCache(Path(args.cache_root))
    gnn = load_gnn_checkpoint(args.gnn_ckpt)
    fusion_weights = FusionWeights.load(args.fusion_ckpt) if args.fusion_ckpt else None

    rows = []
    frames = []
    for dname in [x.strip() for x in args.test_datasets.split(",") if x.strip()]:
        ds = DatasetIndex.from_folder(dname, Path(args.data_root) / dname)
        fn = lambda s: run_method_gnn(
            supernet,
            str(s.image_path),
            s.manip_type,
            gnn,
            RunConfig(K=50, min_len=1, max_len=4, seed=0, top_k=5, fuse_across_paths="score_softmax" if fusion_weights else "uniform"),
            cache,
            ds.name,
            s.sample_id,
            fusion_weights=fusion_weights,
            path_fusion_mode="learned" if fusion_weights else "uniform",
        )[0]
        df = evaluate_method_on_split(ds, args.split, "ForensicFusion-GNN", fn, thr=0.5)
        frames.append(df)
        loc = summarize_localization(df)
        rows.append({"Train": args.train_name, "Test": dname, "AUC$\\uparrow$": loc["AUC"], "mIoU$\\uparrow$": loc["mIoU"], "F1$\\uparrow$": loc["F1"]})

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(results_dir / f"cross_dataset_{args.train_name}.csv", index=False)
    tex = make_cvpr_table(
        columns=["Train", "Test", "AUC$\\uparrow$", "mIoU$\\uparrow$", "F1$\\uparrow$"],
        rows=rows,
        caption="Cross-dataset generalization of the trained path selector.",
        label="tab:cross_dataset_generalization",
        align="lcccc",
    )
    tex_path = results_dir / f"cross_dataset_{args.train_name}.tex"
    tex_path.write_text(tex, encoding="utf-8")
    print(f"Saved {tex_path}")


if __name__ == "__main__":
    main()
