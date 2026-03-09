import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from pathlib import Path

from forensicfusion.algorithms.pyifd_wrappers import build_default_pyifd_supernet
from forensicfusion.cache import OutputCache
from forensicfusion.data import DatasetIndex, read_mask
from forensicfusion.external import ExternalPredictionStore
from forensicfusion.fusion.weights import FusionWeights
from forensicfusion.pipeline import RunConfig, run_method_gnn, run_method_uniform_all
from forensicfusion.predictor.infer import load_gnn_checkpoint
from forensicfusion.supernet import ForensicSupernet
from forensicfusion.vis import qualitative_grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--cache_root", default="cache")
    ap.add_argument("--out_dir", default="results/qualitative")
    ap.add_argument("--include_optional", action="store_true")
    ap.add_argument("--gnn_ckpt", required=True)
    ap.add_argument("--fusion_ckpt", default=None)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--ids", default="", help="comma-separated sample ids; empty means first 5 samples in split")
    ap.add_argument("--external_pred_root", default=None)
    ap.add_argument("--external_methods", default="")
    args = ap.parse_args()

    ds = DatasetIndex.from_folder(args.dataset, Path(args.data_root) / args.dataset)
    cache = OutputCache(Path(args.cache_root))
    supernet = ForensicSupernet(build_default_pyifd_supernet(include_optional=args.include_optional))
    gnn = load_gnn_checkpoint(args.gnn_ckpt)
    fusion_weights = FusionWeights.load(args.fusion_ckpt) if args.fusion_ckpt else None
    ext_store = ExternalPredictionStore(Path(args.external_pred_root)) if args.external_pred_root else None
    external_methods = [x.strip() for x in args.external_methods.split(",") if x.strip()]

    samples = ds.split(args.split)
    if args.ids.strip():
        wanted = set(x.strip() for x in args.ids.split(",") if x.strip())
        samples = [s for s in samples if s.sample_id in wanted]
    else:
        samples = samples[:5]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for s in samples:
        panels = []
        for mid, title in [("pyifd_ela", "ELA"), ("pyifd_dct", "DCT"), ("pyifd_ghost", "GHOST")]:
            hm = cache.load(ds.name, s.sample_id, mid) if cache.has(ds.name, s.sample_id, mid) else supernet.get(mid).run(str(s.image_path))
            panels.append((title, hm))
        panels.append(("Uniform-all", run_method_uniform_all(supernet, str(s.image_path), s.manip_type, cache, ds.name, s.sample_id, fusion_weights=None, fusion_mode="uniform")))
        panels.append(("Ours", run_method_gnn(supernet, str(s.image_path), s.manip_type, gnn, RunConfig(K=50, min_len=1, max_len=4, seed=0, top_k=5, fuse_across_paths="score_softmax" if fusion_weights else "uniform"), cache, ds.name, s.sample_id, fusion_weights=fusion_weights, path_fusion_mode="learned" if fusion_weights else "uniform")[0]))
        if ext_store is not None:
            for m in external_methods:
                try:
                    panels.append((m, ext_store.load(ds.name, m, s.sample_id)))
                except Exception:
                    pass
        gt = read_mask(s.mask_path) if s.mask_path is not None else None
        qualitative_grid(str(s.image_path), panels, out_dir / f"{s.sample_id}.png", gt_mask=gt)
    print(f"Saved grids to {out_dir}")


if __name__ == "__main__":
    main()
