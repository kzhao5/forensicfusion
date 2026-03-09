import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from pathlib import Path

import cv2

from forensicfusion.algorithms.pyifd_wrappers import build_default_pyifd_supernet
from forensicfusion.fusion.weights import FusionWeights
from forensicfusion.pipeline import RunConfig, run_method_gnn, run_method_heuristic, run_method_uniform_all
from forensicfusion.predictor.infer import load_gnn_checkpoint
from forensicfusion.supernet import ForensicSupernet
from forensicfusion.vis import overlay_on_image, qualitative_grid, save_heatmap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--manip_type", default="unknown")
    ap.add_argument("--out_dir", default="demo_out")
    ap.add_argument("--include_optional", action="store_true")
    ap.add_argument("--gnn_ckpt", default=None)
    ap.add_argument("--fusion_ckpt", default=None)
    ap.add_argument("--K", type=int, default=50)
    ap.add_argument("--top_k", type=int, default=5)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    supernet = ForensicSupernet(build_default_pyifd_supernet(include_optional=args.include_optional))

    base = run_method_uniform_all(supernet, args.image, args.manip_type, cache=None, dataset_name=None, sample_id=None, fusion_weights=None, fusion_mode="uniform")
    heur = run_method_heuristic(supernet, args.image, args.manip_type, cache=None, dataset_name=None, sample_id=None, fusion_weights=None, fusion_mode="uniform")
    save_heatmap(out_dir / "uniform_all.png", base)
    save_heatmap(out_dir / "heuristic.png", heur)

    panels = [("Uniform-all", base), ("Heuristic", heur)]

    if args.gnn_ckpt:
        gnn = load_gnn_checkpoint(args.gnn_ckpt)
        fusion_weights = FusionWeights.load(args.fusion_ckpt) if args.fusion_ckpt else None
        cfg = RunConfig(K=args.K, min_len=1, max_len=4, seed=0, top_k=args.top_k, fuse_across_paths="score_softmax" if fusion_weights else "uniform")
        pred, ranked = run_method_gnn(
            supernet,
            args.image,
            args.manip_type,
            gnn,
            cfg,
            cache=None,
            dataset_name=None,
            sample_id=None,
            fusion_weights=fusion_weights,
            path_fusion_mode="learned" if fusion_weights else "uniform",
        )
        save_heatmap(out_dir / "forensicfusion_gnn.png", pred)
        panels.append(("ForensicFusion", pred))
        (out_dir / "top_paths.txt").write_text(
            "\n".join(f"{score:.4f}\t{' + '.join(path.module_ids)}" for path, score in ranked),
            encoding="utf-8",
        )

    image_bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image_bgr is not None:
        for name, hm in panels:
            ov = overlay_on_image(image_bgr, hm)
            cv2.imwrite(str(out_dir / f"overlay_{name.lower().replace(' ', '_').replace('-', '_')}.png"), ov)
        qualitative_grid(args.image, panels, out_dir / "qualitative_grid.png")

    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
