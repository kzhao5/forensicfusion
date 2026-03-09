from __future__ import annotations

import csv
import hashlib
import os
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def which(name: str) -> Optional[str]:
    return shutil.which(name)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def run_command(
    cmd: Sequence[str],
    cwd: str | Path | None = None,
    env: Optional[dict[str, str]] = None,
    log_path: str | Path | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    proc = subprocess.run(
        list(map(str, cmd)),
        cwd=str(cwd) if cwd is not None else None,
        env=merged_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if log_path is not None:
        lp = Path(log_path)
        lp.parent.mkdir(parents=True, exist_ok=True)
        lp.write_text(proc.stdout or "", encoding="utf-8")
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(map(str, cmd))}\n\n{proc.stdout}"
        )
    return proc


def conda_run_prefix(env_name: Optional[str], use_current_python: bool = False) -> List[str]:
    if use_current_python or not env_name:
        return [sys.executable]
    conda = which("conda")
    if conda is None:
        raise RuntimeError("conda was not found in PATH. Either install conda or use --use_current_python.")
    return [conda, "run", "-n", env_name, "python"]


def clone_or_update_repo(repo_url: str, dest: str | Path, branch: str = "main", update: bool = False) -> Path:
    dest = Path(dest)
    git = which("git")
    if git is None:
        raise RuntimeError("git was not found in PATH.")
    if dest.exists() and (dest / ".git").exists():
        if update:
            run_command([git, "fetch", "--all"], cwd=dest)
            run_command([git, "checkout", branch], cwd=dest, check=False)
            run_command([git, "pull", "--ff-only"], cwd=dest, check=False)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    run_command([git, "clone", "--depth", "1", "--branch", branch, repo_url, str(dest)])
    return dest


def md5_file(path: str | Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: str | Path, md5: Optional[str] = None) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    if md5 is not None:
        got = md5_file(dest)
        if got.lower() != md5.lower():
            raise RuntimeError(f"MD5 mismatch for {dest}: expected {md5}, got {got}")
    return dest


def unzip_file(zip_path: str | Path, dest_dir: str | Path) -> Path:
    zip_path = Path(zip_path)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    return dest_dir


def stage_split_images(dataset, split: str, stage_dir: str | Path, symlink: bool = True) -> dict[str, Path]:
    stage_dir = ensure_dir(stage_dir)
    mapping: dict[str, Path] = {}
    for s in dataset.split(split):
        src = Path(s.image_path)
        dst = stage_dir / src.name
        if dst.exists() or dst.is_symlink():
            try:
                dst.unlink()
            except OSError:
                pass
        if symlink:
            try:
                os.symlink(src.resolve(), dst)
            except OSError:
                shutil.copy2(src, dst)
        else:
            shutil.copy2(src, dst)
        mapping[s.sample_id] = dst
    return mapping


def save_map_uint8(pred: np.ndarray, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(pred, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    arr = np.clip(arr, 0.0, 1.0)
    arr_u8 = (arr * 255.0 + 0.5).astype(np.uint8)
    cv2.imwrite(str(out_path), arr_u8)
    return out_path


def load_npz_field(path: str | Path, key: str) -> np.ndarray:
    with np.load(path) as data:
        return data[key]


def parse_detection_score(text: str) -> Optional[float]:
    patterns = [
        r"Detection score:\s*([0-9eE+\-.]+)",
        r"score\s*=\s*([0-9eE+\-.]+)",
        r"score\s*:\s*([0-9eE+\-.]+)",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
    return None


def write_scores_csv(rows: list[dict[str, object]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("sample_id,score\n", encoding="utf-8")
        return path
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def list_prediction_maps(method_root: str | Path) -> list[Path]:
    method_root = Path(method_root)
    out: list[Path] = []
    for p in method_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            out.append(p)
    return out
