import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import time
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
from forensicfusion.path import fuse_maps
from forensicfusion.pipeline import RunConfig, run_single_path
from forensicfusion.predictor.infer import load_gnn_checkpoint, score_paths
from forensicfusion.sampling import SampleConfig, sample_paths_random
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
    ap.add_argument("--Ks", default="5,10,20,50,100")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--time_without_cache", action="store_true", help="measure path execution without cached outputs")
    args = ap.parse_args()

    ds = DatasetIndex.from_folder(args.dataset, Path(args.data_root) / args.dataset)
    supernet = ForensicSupernet(build_default_pyifd_supernet(include_optional=args.include_optional))
    cache = OutputCache(Path(args.cache_root))
    gnn = load_gnn_checkpoint(args.gnn_ckpt)
    fusion_weights = FusionWeights.load(args.fusion_ckpt) if args.fusion_ckpt else None

    samples = [s for s in ds.split(args.split) if s.mask_path is not None]
    Ks = [int(x) for x in args.Ks.split(",") if x.strip()]
    summary_rows = []
    all_rows = []

    for K in Ks:
        aucs, mious = [], []
        t_total, t_sel, t_mod, t_fuse = [], [], [], []
        for i, s in enumerate(samples):
            img = cv2.imread(str(s.image_path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            h, w = img.shape[:2]
            gt = read_mask(s.mask_path, target_shape=(h, w))

            t0 = time.perf_counter()
            paths = sample_paths_random(supernet, s.manip_type, SampleConfig(K=K, min_len=1, max_len=4, seed=i, fusion="learned" if fusion_weights else "uniform"))
            ts = time.perf_counter()
            scores = score_paths(gnn, paths, str(s.image_path), s.manip_type)
            te = time.perf_counter()
            order = np.argsort(-scores)
            top_idx = order[: min(args.top_k, len(order))]
            top_paths = [paths[j] for j in top_idx]
            top_scores = scores[top_idx] if len(top_idx) else np.array([], dtype=np.float32)

            outs = []
            tm0 = time.perf_counter()
            for p in top_paths:
                out = run_single_path(
                    supernet,
                    str(s.image_path),
                    s.manip_type,
                    p,
                    cache=None if args.time_without_cache else cache,
                    dataset_name=None if args.time_without_cache else ds.name,
                    sample_id=None if args.time_without_cache else s.sample_id,
                    fusion_weights=fusion_weights,
                )
                outs.append(out)
            tm1 = time.perf_counter()

            tf0 = time.perf_counter()
            if len(outs) == 0:
                pred = np.zeros((h, w), dtype=np.float32)
            elif len(outs) == 1:
                pred = outs[0]
            else:
                stack = np.stack(outs, 0)
                if fusion_weights is not None:
                    w = np.exp(top_scores - np.max(top_scores))
                    w = w / (w.sum() + 1e-8)
                    pred = np.tensordot(w, stack, axes=(0, 0)).astype(np.float32)
                else:
                    pred = stack.mean(axis=0).astype(np.float32)
            tf1 = time.perf_counter()
            total = tf1 - t0

            mb = MetricBundle.from_pred_gt(pred, gt, thr=0.5)
            aucs.append(mb.auc)
            mious.append(mb.miou)
            t_total.append(total)
            t_sel.append(te - ts)
            t_mod.append(tm1 - tm0)
            t_fuse.append(tf1 - tf0)
            all_rows.append({
                "K": K,
                "id": s.sample_id,
                "AUC": mb.auc,
                "mIoU": mb.miou,
                "total_s": total,
                "selection_ms": 1000.0 * (te - ts),
                "modules_ms": 1000.0 * (tm1 - tm0),
                "fuse_ms": 1000.0 * (tf1 - tf0),
            })

        summary_rows.append({
            "K": K,
            "AUC$\\uparrow$": nanmean(np.asarray(aucs, dtype=np.float32)),
            "mIoU$\\uparrow$": nanmean(np.asarray(mious, dtype=np.float32)),
            "Total(s)$\\downarrow$": nanmean(np.asarray(t_total, dtype=np.float32)),
            "Sel.(ms)": 1000.0 * nanmean(np.asarray(t_sel, dtype=np.float32)),
            "Modules(ms)": 1000.0 * nanmean(np.asarray(t_mod, dtype=np.float32)),
            "Fuse(ms)": 1000.0 * nanmean(np.asarray(t_fuse, dtype=np.float32)),
        })

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows)
    csv_path = results_dir / f"table3_efficiency_{args.dataset}.csv"
    df.to_csv(csv_path, index=False)
    tex = make_cvpr_table(
        columns=["K", "AUC$\\uparrow$", "mIoU$\\uparrow$", "Total(s)$\\downarrow$", "Sel.(ms)", "Modules(ms)", "Fuse(ms)"],
        rows=summary_rows,
        caption="Accuracy--efficiency trade-off versus candidate size $K$. Time is per-image average.",
        label="tab:efficiency_tradeoff",
        align="ccccccc",
        size="\\scriptsize",
        tabcolsep_pt=3.0,
        resize_to_linewidth=True,
    )
    tex_path = results_dir / f"table3_efficiency_{args.dataset}.tex"
    tex_path.write_text(tex, encoding="utf-8")
    print(f"Saved {csv_path}")
    print(f"Saved {tex_path}")


if __name__ == "__main__":
    main()
