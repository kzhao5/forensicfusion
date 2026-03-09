from __future__ import annotations

from pathlib import Path
from typing import Optional

from forensicfusion.data import DatasetIndex

from .base import BaselineAdapter, BaselineSpec
from .utils import conda_run_prefix, run_command


class OMGFuserAdapter(BaselineAdapter):
    spec = BaselineSpec(
        name="OMG-Fuser",
        repo_url="https://github.com/mever-team/omgfuser.git",
        branch="main",
        env_name="ff_omgfuser",
        official=True,
    )

    def create_env(self) -> None:
        self.clone_or_update(update=False)
        run_command(["conda", "create", "-n", self.spec.env_name, "-y", "python=3.10"], check=False)
        # Follow the official README: install torch/torchvision via conda, then pip requirements.
        run_command(["conda", "run", "-n", self.spec.env_name, "conda", "install", "-y", "pytorch==1.13.1", "torchvision==0.14.1", "torchaudio==0.13.1", "pytorch-cuda=11.7", "-c", "pytorch", "-c", "nvidia"], check=False)
        run_command(["conda", "run", "-n", self.spec.env_name, "pip", "install", "-r", str(self.repo_dir / "requirements.txt")], check=False)

    def run_dataset(
        self,
        dataset: DatasetIndex,
        split: str,
        predictions_root: str | Path,
        gpu: str = "0",
        use_current_python: bool = False,
        dataset_csv: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        model_name: str | None = None,
        input_signals: str | None = None,
        signals_channels: str | None = None,
        loss_function: str = "class_aware_localization_detection_bce_dice",
        export_best_samples: bool = False,
        **kwargs,
    ) -> Path:
        self.check_ready()
        if dataset_csv is None or checkpoint_path is None or model_name is None or input_signals is None or signals_channels is None:
            raise ValueError(
                "OMG-Fuser requires a prepared dataset_csv plus checkpoint_path/model_name/input_signals/signals_channels. "
                "Use scripts/prepare_omgfuser_csv.py and then call scripts/run_dl_baselines.py with the omgfuser-specific flags."
            )
        py = conda_run_prefix(self.spec.env_name, use_current_python=use_current_python)
        cmd = py + [
            "-m",
            "omgfuser",
            "test",
            "--experiment_name",
            f"forensicfusion_{dataset.name}_{split}",
            "--gpu_id",
            str(gpu),
            "--checkpoint_path",
            str(checkpoint_path),
            "--dataset_csv",
            str(dataset_csv),
            "--dataset_name",
            dataset.name,
            "--model_name",
            model_name,
            "--input_signals",
            input_signals,
            "--signals_channels",
            signals_channels,
            "--loss_function",
            loss_function,
        ]
        if export_best_samples:
            cmd.append("--export_best_samples")
        log_path = self.work_root / dataset.name / split / "omgfuser.log"
        run_command(cmd, cwd=self.repo_dir, log_path=log_path)
        # The official repo is evaluated through its own run directory. We return that path for downstream parsing.
        return self.repo_dir / "runs" / "tests" / f"forensicfusion_{dataset.name}_{split}"
