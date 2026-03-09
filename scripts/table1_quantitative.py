import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from forensicfusion.algorithms.pyifd_wrappers import build_default_pyifd_supernet
from forensicfusion.cache import OutputCache
from forensicfusion.data import DatasetIndex
from forensicfusion.evaluate import evaluate_method_on_split, summarize_detection, summarize_localization
from forensicfusion.external import ExternalPredictionStore
from forensicfusion.fusion.weights import FusionWeights
from forensicfusion.pipeline import RunConfig, run_method_gnn
from forensicfusion.predictor.infer import load_gnn_checkpoint
from forensicfusion.supernet import ForensicSupernet
from forensicfusion.latex import make_cvpr_table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", required=True, help="comma-separated dataset names")
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--cache_root", default="cache")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--include_optional", action="store_true")
    ap.add_argument("--gnn_ckpt", required=True)
    ap.add_argument("--fusion_ckpt", default=None)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--external_pred_root", default=None)
    ap.add_argument("--external_methods", default="", help="comma-separated method names in prediction folder, e.g. TruFor,MMFusion")
    ap.add_argument("--top_k", type=int, default=5)
    args = ap.parse_args()

    dataset_names = [x.strip() for x in args.datasets.split(",") if x.strip()]
    external_methods = [x.strip() for x in args.external_methods.split(",") if x.strip()]

    supernet = ForensicSupernet(build_default_pyifd_supernet(include_optional=args.include_optional))
    cache = OutputCache(Path(args.cache_root))
    gnn = load_gnn_checkpoint(args.gnn_ckpt)
    fusion_weights = FusionWeights.load(args.fusion_ckpt) if args.fusion_ckpt else None
    ext_store = ExternalPredictionStore(Path(args.external_pred_root)) if args.external_pred_root else None

    all_eval = []
    summary_rows = []

    for dname in dataset_names:
        ds = DatasetIndex.from_folder(dname, Path(args.data_root) / dname)

        def mod_fn(mid):
            return lambda s: cache.load(ds.name, s.sample_id, mid) if cache.has(ds.name, s.sample_id, mid) else supernet.get(mid).run(str(s.image_path))

        methods: Dict[str, callable] = {
            "PyIFD-ELA": mod_fn("pyifd_ela"),
            "PyIFD-DCT": mod_fn("pyifd_dct"),
            "PyIFD-NOI1": mod_fn("pyifd_noi1"),
            "ForensicFusion-GNN": lambda s: run_method_gnn(
                supernet,
                str(s.image_path),
                s.manip_type,
                gnn,
                RunConfig(K=50, min_len=1, max_len=4, seed=0, top_k=args.top_k, fuse_across_paths="score_softmax" if fusion_weights else "uniform"),
                cache,
                ds.name,
                s.sample_id,
                fusion_weights=fusion_weights,
                path_fusion_mode="learned" if fusion_weights else "uniform",
            )[0],
        }
        if ext_store is not None:
            for mname in external_methods:
                methods[mname] = lambda s, _m=mname: ext_store.load(ds.name, _m, s.sample_id)

        for method_name, fn in methods.items():
            df = evaluate_method_on_split(ds, args.split, method_name, fn, thr=0.5)
            all_eval.append(df)
            loc = summarize_localization(df)
            det = summarize_detection(df)
            summary_rows.append(
                {
                    "dataset": dname,
                    "method": method_name,
                    "LocF1": loc["F1"],
                    "LocAUC": loc["AUC"],
                    "DetAcc": det["Acc"],
                    "DetAUC": det["AUC"],
                }
            )

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    eval_df = pd.concat(all_eval, ignore_index=True) if all_eval else pd.DataFrame()
    eval_csv = results_dir / "table1_per_image.csv"
    eval_df.to_csv(eval_csv, index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = results_dir / "table1_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    # Wide latex table.
    methods = list(dict.fromkeys(summary_df["method"].tolist()))
    rows = []
    cols = ["Method"]
    for dname in dataset_names:
        cols.extend([f"{dname} F1", f"{dname} AUC"])
    cols.extend(["Avg. Loc.", "Avg. Det."])

    for method in methods:
        sub = summary_df[summary_df["method"] == method]
        row = {"Method": method}
        loc_vals, det_vals = [], []
        for dname in dataset_names:
            one = sub[sub["dataset"] == dname]
            if len(one) == 0:
                row[f"{dname} F1"] = "--"
                row[f"{dname} AUC"] = "--"
            else:
                r = one.iloc[0]
                row[f"{dname} F1"] = float(r["LocF1"])
                row[f"{dname} AUC"] = float(r["LocAUC"])
                loc_vals.append(float(r["LocAUC"]))
                det_vals.append(float(r["DetAUC"]))
        row["Avg. Loc."] = sum(loc_vals) / max(len(loc_vals), 1)
        row["Avg. Det."] = sum(det_vals) / max(len(det_vals), 1)
        rows.append(row)

    tex = make_cvpr_table(
        columns=cols,
        rows=rows,
        caption="Quantitative comparison of localization and detection performance. External methods are read from saved prediction maps when provided.",
        label="tab:quantitative_comparison",
        align="l" + "cc" * len(dataset_names) + "cc",
        size="\\scriptsize",
        tabcolsep_pt=3.0,
        resize_to_linewidth=True,
    )
    tex_path = results_dir / "table1_quantitative.tex"
    tex_path.write_text(tex, encoding="utf-8")

    print(f"Saved {eval_csv}")
    print(f"Saved {summary_csv}")
    print(f"Saved {tex_path}")


if __name__ == "__main__":
    main()
