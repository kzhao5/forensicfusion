import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from forensicfusion.algorithms.pyifd_wrappers import build_default_pyifd_supernet
from forensicfusion.cache import OutputCache
from forensicfusion.data import DatasetIndex, read_mask
from forensicfusion.fusion.weights import FusionWeights
from forensicfusion.latex import make_cvpr_table
from forensicfusion.metrics import MetricBundle, nanmean
from forensicfusion.path import ForensicPath, fuse_maps
from forensicfusion.pipeline import RunConfig, run_method_gnn, run_method_heuristic, run_method_random_k, run_method_uniform_all
from forensicfusion.predictor.infer import load_gnn_checkpoint
from forensicfusion.supernet import ForensicSupernet


def best_single_oracle(supernet: ForensicSupernet, cache: OutputCache, ds: DatasetIndex, s):
    img = cv2.imread(str(s.image_path), cv2.IMREAD_COLOR)
    if img is None or s.mask_path is None:
        raise RuntimeError("Need image + mask for oracle")
    h, w = img.shape[:2]
    gt = read_mask(s.mask_path, target_shape=(h, w))
    maps = cache.load_many(ds.name, s.sample_id, supernet.module_ids)
    best_auc = -1.0
    best_map = None
    for mid, m in maps.items():
        mb = MetricBundle.from_pred_gt(m, gt, thr=0.5)
        if np.isfinite(mb.auc) and mb.auc > best_auc:
            best_auc = mb.auc
            best_map = m
    if best_map is None:
        best_map = supernet.get(supernet.module_ids[0]).run(str(s.image_path))
    return best_map


def summarize_method(name: str, df: pd.DataFrame):
    return {
        "Method": name,
        "K": df["K"].iloc[0] if "K" in df.columns and len(df) else "--",
        "Sel.": df["Sel"].iloc[0] if "Sel" in df.columns and len(df) else "--",
        "Fuse": df["Fuse"].iloc[0] if "Fuse" in df.columns and len(df) else "--",
        "AUC$\\uparrow$": nanmean(df["AUC"].to_numpy(np.float32)),
        "mIoU$\\uparrow$": nanmean(df["mIoU"].to_numpy(np.float32)),
    }


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
    ap.add_argument("--top_k", type=int, default=5)
    args = ap.parse_args()

    ds = DatasetIndex.from_folder(args.dataset, Path(args.data_root) / args.dataset)
    supernet = ForensicSupernet(build_default_pyifd_supernet(include_optional=args.include_optional))
    cache = OutputCache(Path(args.cache_root))
    gnn = load_gnn_checkpoint(args.gnn_ckpt)
    fusion_weights = FusionWeights.load(args.fusion_ckpt) if args.fusion_ckpt else None

    samples = [s for s in ds.split(args.split) if s.mask_path is not None]

    methods = []
    methods.append(("Best single $P^{(i)}$ (oracle)", "--", "oracle", "--", lambda s: best_single_oracle(supernet, cache, ds, s)))
    methods.append(("Uniform-all $P^{(avg)}$", "all", "none", "unif.", lambda s: run_method_uniform_all(supernet, str(s.image_path), s.manip_type, cache, ds.name, s.sample_id, fusion_weights=None, fusion_mode="uniform")))
    methods.append(("Random-K", 10, "rand.", "unif.", lambda s: run_method_random_k(supernet, str(s.image_path), s.manip_type, RunConfig(K=10, min_len=1, max_len=4, seed=0, top_k=10), cache, ds.name, s.sample_id)))
    methods.append(("Random-K", 50, "rand.", "unif.", lambda s: run_method_random_k(supernet, str(s.image_path), s.manip_type, RunConfig(K=50, min_len=1, max_len=4, seed=0, top_k=50), cache, ds.name, s.sample_id)))
    methods.append(("Heuristic rules", 50, "heur.", "unif.", lambda s: run_method_heuristic(supernet, str(s.image_path), s.manip_type, cache, ds.name, s.sample_id, fusion_weights=None, fusion_mode="uniform")))
    methods.append(("GNN-guided (select-1)", 50, "GNN", "--", lambda s: run_method_gnn(supernet, str(s.image_path), s.manip_type, gnn, RunConfig(K=50, min_len=1, max_len=4, seed=0, top_k=1), cache, ds.name, s.sample_id, fusion_weights=None, path_fusion_mode="uniform")[0]))
    methods.append(("GNN top-$k$ + uniform fuse", 50, "GNN", "unif.", lambda s: run_method_gnn(supernet, str(s.image_path), s.manip_type, gnn, RunConfig(K=50, min_len=1, max_len=4, seed=0, top_k=args.top_k, fuse_across_paths="uniform"), cache, ds.name, s.sample_id, fusion_weights=None, path_fusion_mode="uniform")[0]))
    if fusion_weights is not None:
        methods.append(("GNN top-$k$ + learned fuse (ours)", 50, "GNN", "learn.", lambda s: run_method_gnn(supernet, str(s.image_path), s.manip_type, gnn, RunConfig(K=50, min_len=1, max_len=4, seed=0, top_k=args.top_k, fuse_across_paths="score_softmax"), cache, ds.name, s.sample_id, fusion_weights=fusion_weights, path_fusion_mode="learned")[0]))

    all_rows = []
    summary_rows = []
    for method_name, K, sel, fuse, fn in methods:
        per = []
        for s in samples:
            img = cv2.imread(str(s.image_path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            h, w = img.shape[:2]
            gt = read_mask(s.mask_path, target_shape=(h, w))
            pred = np.asarray(fn(s), dtype=np.float32)
            if pred.shape[:2] != (h, w):
                pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
            mb = MetricBundle.from_pred_gt(pred, gt, thr=0.5)
            row = {"method": method_name, "id": s.sample_id, "AUC": mb.auc, "mIoU": mb.miou, "F1": mb.f1, "K": K, "Sel": sel, "Fuse": fuse}
            all_rows.append(row)
            per.append(row)
        if per:
            d = pd.DataFrame(per)
            summary_rows.append(summarize_method(method_name, d))

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows)
    csv_path = results_dir / f"table2_ablation_{args.dataset}.csv"
    df.to_csv(csv_path, index=False)

    tex = make_cvpr_table(
        columns=["Method", "K", "Sel.", "Fuse", "AUC$\\uparrow$", "mIoU$\\uparrow$"],
        rows=summary_rows,
        caption="Ablation on path selection and fusion. Higher is better.",
        label="tab:ablation_selection_fusion",
        align="lccccc",
        size="\\scriptsize",
        tabcolsep_pt=3.5,
    )
    tex_path = results_dir / f"table2_ablation_{args.dataset}.tex"
    tex_path.write_text(tex, encoding="utf-8")
    print(f"Saved {csv_path}")
    print(f"Saved {tex_path}")


if __name__ == "__main__":
    main()
