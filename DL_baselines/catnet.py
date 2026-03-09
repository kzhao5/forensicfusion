from __future__ import annotations

import re
import shutil
from pathlib import Path

from forensicfusion.data import DatasetIndex

from .base import BaselineAdapter, BaselineSpec
from .utils import conda_run_prefix, run_command, stage_split_images


class CATNetAdapter(BaselineAdapter):
    spec = BaselineSpec(
        name="CAT-Net",
        repo_url="https://github.com/mjkwon2021/CAT-Net.git",
        branch="main",
        env_name="ff_catnet",
        official=True,
    )

    def create_env(self) -> None:
        self.clone_or_update(update=False)
        run_command(["conda", "create", "-n", self.spec.env_name, "-y", "python=3.8"], check=False)
        # The official README targets torch 1.1 / python 3.6; this keeps an isolated env and lets the user adjust if needed.
        run_command(["conda", "run", "-n", self.spec.env_name, "pip", "install", "-r", str(self.repo_dir / "requirements.txt")], check=False)

    def _patch_infer_script(self, mode: str = "full") -> Path:
        src = (self.repo_dir / "tools" / "infer.py").read_text(encoding="utf-8")
        if mode == "full":
            src = re.sub(
                r"args = argparse\.Namespace\(cfg='experiments/CAT_full\.yaml'.*?\)\n\s*# args = argparse\.Namespace\(cfg='experiments/CAT_DCT_only\.yaml'.*?\)\n",
                "args = argparse.Namespace(cfg='experiments/CAT_full.yaml', opts=['TEST.MODEL_FILE', 'output/splicing_dataset/CAT_full/CAT_full_v2.pth.tar', 'TEST.FLIP_TEST', 'False', 'TEST.NUM_SAMPLES', '0'])\n",
                src,
                flags=re.S,
            )
            src = src.replace(
                "test_dataset = splicing_dataset(crop_size=None, grid_crop=True, blocks=('RGB', 'DCTvol', 'qtable'), DCT_channels=1, mode='arbitrary', read_from_jpeg=True)  # full model\n    # test_dataset = splicing_dataset(crop_size=None, grid_crop=True, blocks=('DCTvol', 'qtable'), DCT_channels=1, mode='arbitrary', read_from_jpeg=True)  # DCT stream\n",
                "test_dataset = splicing_dataset(crop_size=None, grid_crop=True, blocks=('RGB', 'DCTvol', 'qtable'), DCT_channels=1, mode='arbitrary', read_from_jpeg=True)\n",
            )
        else:
            src = re.sub(
                r"args = argparse\.Namespace\(cfg='experiments/CAT_full\.yaml'.*?\)\n\s*# args = argparse\.Namespace\(cfg='experiments/CAT_DCT_only\.yaml'.*?\)\n",
                "args = argparse.Namespace(cfg='experiments/CAT_DCT_only.yaml', opts=['TEST.MODEL_FILE', 'output/splicing_dataset/CAT_DCT_only/DCT_only_v2.pth.tar', 'TEST.FLIP_TEST', 'False', 'TEST.NUM_SAMPLES', '0'])\n",
                src,
                flags=re.S,
            )
            src = src.replace(
                "test_dataset = splicing_dataset(crop_size=None, grid_crop=True, blocks=('RGB', 'DCTvol', 'qtable'), DCT_channels=1, mode='arbitrary', read_from_jpeg=True)  # full model\n    # test_dataset = splicing_dataset(crop_size=None, grid_crop=True, blocks=('DCTvol', 'qtable'), DCT_channels=1, mode='arbitrary', read_from_jpeg=True)  # DCT stream\n",
                "test_dataset = splicing_dataset(crop_size=None, grid_crop=True, blocks=('DCTvol', 'qtable'), DCT_channels=1, mode='arbitrary', read_from_jpeg=True)\n",
            )
        src = src.replace(
            "import seaborn as sns; sns.set_theme()\nimport matplotlib.pyplot as plt\n",
            "from PIL import Image\n",
        )
        src = re.sub(
            r"\n\s*# plot\n\s*try:\n.*?except:\n\s*print\(f\"Error occurred while saving output\. \(\{get_next_filename\(index\)\}\)\"\)\n",
            "\n            try:\n                arr = (pred.clip(0,1) * 255.0 + 0.5).astype('uint8')\n                Image.fromarray(arr).save(filepath)\n            except Exception:\n                print(f\"Error occurred while saving output. ({get_next_filename(index)})\")\n",
            src,
            flags=re.S,
        )
        patched = self.work_root / "catnet_infer_patched.py"
        patched.write_text(src, encoding="utf-8")
        return patched

    def run_dataset(
        self,
        dataset: DatasetIndex,
        split: str,
        predictions_root: str | Path,
        gpu: str = "0",
        use_current_python: bool = False,
        mode: str = "full",
        **kwargs,
    ) -> Path:
        self.check_ready()
        patched_script = self._patch_infer_script(mode=mode)
        stage_dir = self.work_root / dataset.name / split / "input"
        if stage_dir.exists():
            shutil.rmtree(stage_dir)
        stage_split_images(dataset, split, stage_dir, symlink=False)
        repo_input = self.repo_dir / "input"
        backup_input = self.work_root / dataset.name / split / "input_backup"
        if repo_input.exists() or repo_input.is_symlink():
            if backup_input.exists() or backup_input.is_symlink():
                if backup_input.is_symlink() or backup_input.is_file():
                    backup_input.unlink()
                else:
                    shutil.rmtree(backup_input)
            shutil.move(str(repo_input), str(backup_input))
        shutil.copytree(stage_dir, repo_input)
        repo_output = self.repo_dir / "output_pred"
        if repo_output.exists():
            shutil.rmtree(repo_output)
        py = conda_run_prefix(self.spec.env_name, use_current_python=use_current_python)
        env = {"CUDA_VISIBLE_DEVICES": str(gpu)} if gpu not in {"-1", -1, None} else {"CUDA_VISIBLE_DEVICES": ""}
        try:
            run_command(py + [str(patched_script)], cwd=self.repo_dir, env=env, log_path=self.work_root / dataset.name / split / "catnet.log")
        finally:
            if repo_input.exists():
                shutil.rmtree(repo_input)
            if backup_input.exists() or backup_input.is_symlink():
                shutil.move(str(backup_input), str(repo_input))
        pred_root = Path(predictions_root) / dataset.name / self.spec.name
        pred_root.mkdir(parents=True, exist_ok=True)
        for s in dataset.split(split):
            src = repo_output / f"{Path(s.image_path).stem}.png"
            if src.exists():
                shutil.copy2(src, pred_root / f"{s.sample_id}.png")
        return pred_root
