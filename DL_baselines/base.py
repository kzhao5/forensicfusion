from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from forensicfusion.data import DatasetIndex


@dataclass
class BaselineSpec:
    name: str
    repo_url: str
    branch: str = "main"
    env_name: Optional[str] = None
    official: bool = True


class BaselineAdapter:
    spec: BaselineSpec

    def __init__(self, project_root: str | Path, repos_root: Optional[str | Path] = None):
        self.project_root = Path(project_root).resolve()
        self.repos_root = (Path(repos_root) if repos_root is not None else self.project_root / "DL_baselines" / "repos").resolve()
        self.repos_root.mkdir(parents=True, exist_ok=True)
        self.repo_dir = self.repos_root / self.spec.name.lower().replace("-", "_").replace(" ", "_")
        self.work_root = self.project_root / "DL_baselines" / "work" / self.spec.name
        self.work_root.mkdir(parents=True, exist_ok=True)

    def clone_or_update(self, update: bool = False) -> Path:
        from .utils import clone_or_update_repo
        return clone_or_update_repo(self.spec.repo_url, self.repo_dir, branch=self.spec.branch, update=update)

    def create_env(self) -> None:
        raise NotImplementedError

    def check_ready(self) -> None:
        if not self.repo_dir.exists():
            raise FileNotFoundError(f"Missing repo for {self.spec.name}: {self.repo_dir}. Run setup first.")

    def run_dataset(
        self,
        dataset: DatasetIndex,
        split: str,
        predictions_root: str | Path,
        gpu: str = "0",
        use_current_python: bool = False,
        **kwargs,
    ) -> Path:
        raise NotImplementedError
