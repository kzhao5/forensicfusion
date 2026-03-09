from __future__ import annotations

import shutil
from pathlib import Path

from forensicfusion.data import DatasetIndex

from .base import BaselineAdapter, BaselineSpec
from .utils import conda_run_prefix, run_command, stage_split_images, write_scores_csv


OFFICIAL_RUNNER = r"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--repo_root', required=True)
parser.add_argument('--input_dir', required=True)
parser.add_argument('--output_dir', required=True)
parser.add_argument('--pretrain_index', type=int, default=4)
args = parser.parse_args()
repo_root = Path(args.repo_root).resolve()
sys.path.insert(0, str(repo_root / 'src'))
import modelCore  # type: ignore

model = modelCore.load_pretrain_model_by_index(args.pretrain_index, str(repo_root / 'pretrained_weights'))
out_dir = Path(args.output_dir)
out_dir.mkdir(parents=True, exist_ok=True)
score_rows = []
for img_path in sorted(Path(args.input_dir).iterdir()):
    if not img_path.is_file():
        continue
    rgb = cv2.imread(str(img_path), 1)
    if rgb is None:
        continue
    rgb = rgb[..., ::-1]
    x = np.expand_dims(rgb.astype('float32') / 255.0 * 2.0 - 1.0, axis=0)
    y = model.predict(x)[0, ..., 0].astype('float32')
    y = np.clip(np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
    out = (y * 255.0 + 0.5).astype('uint8')
    cv2.imwrite(str(out_dir / (img_path.stem + '.png')), out)
    score_rows.append((img_path.stem, float(y.max())))
with open(out_dir / '_scores.csv', 'w', encoding='utf-8') as f:
    f.write('sample_id,score\n')
    for sid, score in score_rows:
        f.write(f'{sid},{score}\n')
"""


class ManTraNetAdapter(BaselineAdapter):
    spec = BaselineSpec(
        name="ManTraNet",
        repo_url="https://github.com/ISICV/ManTraNet.git",
        branch="master",
        env_name="ff_mantranet",
        official=True,
    )

    def create_env(self) -> None:
        self.clone_or_update(update=False)
        # Prefer a still-installable TF1/Keras stack while staying close to the official repo.
        run_command(["conda", "create", "-n", self.spec.env_name, "-y", "python=3.7"], check=False)
        run_command(["conda", "run", "-n", self.spec.env_name, "pip", "install", "tensorflow==1.15.5", "keras==2.2.4", "numpy", "opencv-python", "pillow", "matplotlib"], check=False)

    def run_dataset(
        self,
        dataset: DatasetIndex,
        split: str,
        predictions_root: str | Path,
        gpu: str = "0",
        use_current_python: bool = False,
        pretrain_index: int = 4,
        **kwargs,
    ) -> Path:
        self.check_ready()
        stage_dir = self.work_root / dataset.name / split / "images"
        raw_out = self.work_root / dataset.name / split / "raw"
        if stage_dir.exists():
            shutil.rmtree(stage_dir)
        if raw_out.exists():
            shutil.rmtree(raw_out)
        stage_split_images(dataset, split, stage_dir, symlink=False)
        runner_path = self.work_root / "run_mantranet_official.py"
        runner_path.write_text(OFFICIAL_RUNNER, encoding="utf-8")
        py = conda_run_prefix(self.spec.env_name, use_current_python=use_current_python)
        env = {"CUDA_VISIBLE_DEVICES": str(gpu)} if gpu not in {"-1", -1, None} else {"CUDA_VISIBLE_DEVICES": ""}
        run_command(
            py + [
                str(runner_path),
                "--repo_root",
                str(self.repo_dir),
                "--input_dir",
                str(stage_dir),
                "--output_dir",
                str(raw_out),
                "--pretrain_index",
                str(pretrain_index),
            ],
            env=env,
            log_path=self.work_root / dataset.name / split / "mantranet.log",
        )
        pred_root = Path(predictions_root) / dataset.name / self.spec.name
        pred_root.mkdir(parents=True, exist_ok=True)
        for s in dataset.split(split):
            src = raw_out / f"{Path(s.image_path).stem}.png"
            if src.exists():
                shutil.copy2(src, pred_root / f"{s.sample_id}.png")
        score_src = raw_out / "_scores.csv"
        if score_src.exists():
            shutil.copy2(score_src, Path(predictions_root) / dataset.name / f"{self.spec.name}_scores.csv")
        return pred_root
