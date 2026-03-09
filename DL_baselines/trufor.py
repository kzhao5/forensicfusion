from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

import numpy as np

from forensicfusion.data import DatasetIndex

from .base import BaselineAdapter, BaselineSpec
from .utils import (
    conda_run_prefix,
    download_file,
    load_npz_field,
    run_command,
    save_map_uint8,
    stage_split_images,
    unzip_file,
    write_scores_csv,
)


class TruForAdapter(BaselineAdapter):
    spec = BaselineSpec(
        name="TruFor",
        repo_url="https://github.com/grip-unina/TruFor.git",
        branch="main",
        env_name="ff_trufor",
        official=True,
    )

    WEIGHTS_URL = "https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip"
    WEIGHTS_MD5 = "7bee48f3476c75616c3c5721ab256ff8"

    def create_env(self) -> None:
        self.clone_or_update(update=False)
        yaml_path = self.repo_dir / "TruFor_train_test" / "trufor_conda.yaml"
        run_command(["conda", "env", "create", "-n", self.spec.env_name, "-f", str(yaml_path)], check=False)

    def ensure_weights(self, auto_download: bool = False) -> Path:
        weights_path = self.repo_dir / "TruFor_train_test" / "pretrained_models" / "trufor.pth.tar"
        if weights_path.exists():
            return weights_path
        if not auto_download:
            raise FileNotFoundError(
                f"Missing TruFor weights at {weights_path}. Run setup with --download_weights or place them manually."
            )
        zip_path = self.work_root / "TruFor_weights.zip"
        download_file(self.WEIGHTS_URL, zip_path, md5=self.WEIGHTS_MD5)
        unzip_file(zip_path, self.repo_dir / "TruFor_train_test" / "pretrained_models")
        if not weights_path.exists():
            raise FileNotFoundError(f"Downloaded TruFor weights, but could not find {weights_path}")
        return weights_path

    def run_dataset(
        self,
        dataset: DatasetIndex,
        split: str,
        predictions_root: str | Path,
        gpu: str = "0",
        use_current_python: bool = False,
        auto_download_weights: bool = False,
        save_np: bool = False,
        **kwargs,
    ) -> Path:
        self.check_ready()
        weights_path = self.ensure_weights(auto_download=auto_download_weights)
        stage_dir = self.work_root / dataset.name / split / "images"
        raw_out = self.work_root / dataset.name / split / "raw"
        if stage_dir.exists():
            shutil.rmtree(stage_dir)
        if raw_out.exists():
            shutil.rmtree(raw_out)
        mapping = stage_split_images(dataset, split, stage_dir, symlink=True)
        py = conda_run_prefix(self.spec.env_name, use_current_python=use_current_python)
        cmd = py + [
            "test.py",
            "-g",
            str(gpu),
            "-in",
            str(stage_dir),
            "-out",
            str(raw_out),
            "-exp",
            "trufor_ph3",
            "TEST.MODEL_FILE",
            str(weights_path),
        ]
        if save_np:
            cmd.append("--save_np")
        log_path = self.work_root / dataset.name / split / "trufor.log"
        run_command(cmd, cwd=self.repo_dir / "TruFor_train_test", log_path=log_path)

        pred_root = Path(predictions_root) / dataset.name / self.spec.name
        conf_root = Path(predictions_root) / dataset.name / f"{self.spec.name}_conf"
        np_root = Path(predictions_root) / dataset.name / f"{self.spec.name}_nppp"
        score_rows = []
        for s in dataset.split(split):
            npz_path = raw_out / (Path(mapping[s.sample_id]).name + ".npz")
            if not npz_path.exists():
                # test.py may preserve relative names only, try glob by stem.
                cands = list(raw_out.rglob(f"{Path(mapping[s.sample_id]).stem}.npz"))
                if not cands:
                    continue
                npz_path = cands[0]
            pred = load_npz_field(npz_path, "map")
            save_map_uint8(pred, pred_root / f"{s.sample_id}.png")
            with np.load(npz_path) as data:
                score = float(data["score"]) if "score" in data.files else None
                if "conf" in data.files:
                    save_map_uint8(data["conf"].astype(np.float32), conf_root / f"{s.sample_id}.png")
                if save_np and "np++" in data.files:
                    arr = data["np++"].astype(np.float32)
                    lo, hi = float(np.percentile(arr, 1)), float(np.percentile(arr, 99))
                    arr = np.clip((arr - lo) / (hi - lo + 1e-8), 0.0, 1.0) if hi > lo else np.zeros_like(arr)
                    save_map_uint8(arr, np_root / f"{s.sample_id}.png")
            score_rows.append({"sample_id": s.sample_id, "score": score if score is not None else ""})
        write_scores_csv(score_rows, Path(predictions_root) / dataset.name / f"{self.spec.name}_scores.csv")
        return pred_root
