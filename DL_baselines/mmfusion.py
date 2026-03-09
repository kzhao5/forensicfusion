from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from forensicfusion.data import DatasetIndex

from .base import BaselineAdapter, BaselineSpec
from .utils import conda_run_prefix, parse_detection_score, run_command, save_map_uint8, stage_split_images, write_scores_csv


class MMFusionAdapter(BaselineAdapter):
    spec = BaselineSpec(
        name="MMFusion",
        repo_url="https://github.com/IDT-ITI/MMFusion-IML.git",
        branch="main",
        env_name="ff_mmfusion",
        official=True,
    )

    def create_env(self) -> None:
        self.clone_or_update(update=False)
        run_command(["conda", "create", "-n", self.spec.env_name, "-y", "python=3.10"], check=False)
        run_command(["conda", "run", "-n", self.spec.env_name, "pip", "install", "-r", str(self.repo_dir / "requirements.txt")], check=False)

    def _default_ckpt(self) -> Optional[Path]:
        cands = [
            self.repo_dir / "ckpt" / "early_fusion_detection.pth",
            self.repo_dir / "ckpt" / "ec_example" / "best_val_loss.pth",
        ]
        for p in cands:
            if p.exists():
                return p
        return None

    def run_dataset(
        self,
        dataset: DatasetIndex,
        split: str,
        predictions_root: str | Path,
        gpu: str = "0",
        use_current_python: bool = False,
        ckpt: str | Path | None = None,
        exp: str = "experiments/ec_example_phase2.yaml",
        **kwargs,
    ) -> Path:
        self.check_ready()
        ckpt_path = Path(ckpt) if ckpt else self._default_ckpt()
        if ckpt_path is None or not ckpt_path.exists():
            raise FileNotFoundError("Could not find an MMFusion checkpoint. Pass --mmfusion_ckpt explicitly.")
        pred_root = Path(predictions_root) / dataset.name / self.spec.name
        pred_root.mkdir(parents=True, exist_ok=True)
        stage_dir = self.work_root / dataset.name / split / "images"
        if stage_dir.exists():
            shutil.rmtree(stage_dir)
        mapping = stage_split_images(dataset, split, stage_dir, symlink=False)
        py = conda_run_prefix(self.spec.env_name, use_current_python=use_current_python)
        scores = []
        for s in dataset.split(split):
            img_path = stage_dir / Path(mapping[s.sample_id]).name
            cmd = py + [
                "inference.py",
                "-gpu",
                str(gpu),
                "-exp",
                exp,
                "-ckpt",
                str(ckpt_path),
                "-path",
                str(img_path),
            ]
            log_path = self.work_root / dataset.name / split / f"{s.sample_id}.log"
            proc = run_command(cmd, cwd=self.repo_dir, log_path=log_path)
            mask_path = img_path.with_name(img_path.stem + "_mask.png")
            if not mask_path.exists():
                raise FileNotFoundError(f"MMFusion did not produce {mask_path}. Check {log_path}")
            shutil.move(str(mask_path), str(pred_root / f"{s.sample_id}.png"))
            score = parse_detection_score(proc.stdout or "")
            scores.append({"sample_id": s.sample_id, "score": score if score is not None else ""})
        write_scores_csv(scores, Path(predictions_root) / dataset.name / f"{self.spec.name}_scores.csv")
        return pred_root
