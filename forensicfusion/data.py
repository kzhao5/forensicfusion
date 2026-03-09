from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd


@dataclass
class Sample:
    sample_id: str
    image_path: Path
    mask_path: Optional[Path]
    split: str
    manip_type: str
    label: int  # 1 manipulated, 0 authentic


@dataclass
class DatasetIndex:
    name: str
    root: Path
    samples: List[Sample]

    @staticmethod
    def from_folder(name: str, root: str | Path) -> "DatasetIndex":
        root = Path(root)
        meta_path = root / "meta.csv"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing meta.csv at {meta_path}")
        df = pd.read_csv(meta_path, dtype={"id": str})
        required = {"id", "split", "manip_type"}
        if not required.issubset(set(df.columns)):
            raise ValueError(f"meta.csv must contain columns {sorted(required)}; found {list(df.columns)}")

        images_dir = root / "images"
        masks_dir = root / "masks"
        out: List[Sample] = []
        for _, r in df.iterrows():
            sid = str(r["id"])
            split = str(r["split"]).lower()
            manip_type = str(r["manip_type"]).lower() if not pd.isna(r["manip_type"]) else "unknown"

            img_path = None
            for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]:
                p = images_dir / f"{sid}{ext}"
                if p.exists():
                    img_path = p
                    break
            if img_path is None:
                raise FileNotFoundError(f"Could not find image for id={sid} under {images_dir}")

            mask_path = None
            for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]:
                p = masks_dir / f"{sid}{ext}"
                if p.exists():
                    mask_path = p
                    break

            if "label" in df.columns and not pd.isna(r["label"]):
                label = int(r["label"])
            else:
                label = 1 if mask_path is not None else 0

            out.append(Sample(sample_id=sid, image_path=img_path, mask_path=mask_path, split=split, manip_type=manip_type, label=label))
        return DatasetIndex(name=name, root=root, samples=out)

    def split(self, split: str) -> List[Sample]:
        split = split.lower()
        return [s for s in self.samples if s.split.lower() == split]


def read_mask(mask_path: Path, target_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(mask_path)
    if target_shape is not None and (m.shape[0] != target_shape[0] or m.shape[1] != target_shape[1]):
        m = cv2.resize(m, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    return (m.astype(np.float32) > 127).astype(np.float32)
