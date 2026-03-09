import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from pathlib import Path

from tqdm import tqdm

from forensicfusion.algorithms.pyifd_wrappers import build_default_pyifd_supernet
from forensicfusion.cache import OutputCache
from forensicfusion.data import DatasetIndex
from forensicfusion.supernet import ForensicSupernet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Dataset name (folder under data_root)")
    ap.add_argument("--data_root", default="data", help="Root folder containing datasets")
    ap.add_argument("--out_root", default="cache", help="Where to store cached heatmaps")
    ap.add_argument("--split", default="all", choices=["train", "val", "test", "all"])
    ap.add_argument("--include_optional", action="store_true", help="Include heavier / JPEG-specific pyIFD modules")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    ds_root = Path(args.data_root) / args.dataset
    ds = DatasetIndex.from_folder(args.dataset, ds_root)

    mods = build_default_pyifd_supernet(include_optional=args.include_optional)
    supernet = ForensicSupernet(mods)
    cache = OutputCache(Path(args.out_root))

    if args.split == "all":
        samples = ds.samples
    else:
        samples = ds.split(args.split)

    for s in tqdm(samples, desc=f"Precompute {args.dataset} ({args.split})"):
        for mid in supernet.module_ids:
            if (not args.overwrite) and cache.has(ds.name, s.sample_id, mid):
                continue
            try:
                out = supernet.get(mid).run(str(s.image_path))
                cache.save(ds.name, s.sample_id, mid, out)
            except Exception as e:
                if args.verbose:
                    print(f"[WARN] {mid} failed on {s.image_path}: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
