from __future__ import annotations

import importlib
import io
from typing import Any, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image
from scipy.signal import fftconvolve, medfilt
from skimage.transform import resize

from .base import ForensicModule, ModuleMeta


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def _imread_bgr(impath: str) -> np.ndarray:
    im = cv2.imread(impath, cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(impath)
    return im


def _imshape(impath: str) -> Tuple[int, int]:
    im = _imread_bgr(impath)
    return int(im.shape[0]), int(im.shape[1])


def _resize_to_image(map2d: np.ndarray, impath: str) -> np.ndarray:
    h, w = _imshape(impath)
    arr = np.asarray(map2d, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    if arr.shape[:2] == (h, w):
        return arr.astype(np.float32)
    return cv2.resize(arr.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)


def _import_pyifd_submodule(name: str):
    try:
        return importlib.import_module(f"pyIFD.{name}")
    except Exception:
        return None


def _soft_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if x.ndim == 3:
        x = x.mean(axis=2)
    lo = float(np.percentile(x, 1.0))
    hi = float(np.percentile(x, 99.0))
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < eps:
        return np.zeros_like(x, dtype=np.float32)
    x = (x - lo) / (hi - lo + eps)
    return np.clip(x, 0.0, 1.0).astype(np.float32)


# -----------------------------------------------------------------------------
# Lightweight local fallbacks (pyIFD-inspired, used when pyIFD is unavailable)
# -----------------------------------------------------------------------------


def _ela_fallback(impath: str, quality: int = 75, multiplier: float = 15.0, flatten: bool = True) -> np.ndarray:
    """In-memory ELA approximation equivalent in spirit to pyIFD.ELA."""
    im = _imread_bgr(impath)
    ok, enc = cv2.imencode(".jpg", im, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("cv2.imencode failed in ELA fallback")
    rec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    diff = np.abs(im.astype(np.float32) - rec.astype(np.float32)) * float(multiplier)
    if flatten:
        diff = diff.mean(axis=2)
    return diff.astype(np.float32)


def _dct_blockiness_fallback(impath: str) -> np.ndarray:
    """Cheap block-artifact proxy used if pyIFD.DCT is unavailable.

    This is not a verbatim reimplementation of pyIFD.DCT. It provides a portable,
    deterministic JPEG/grid inconsistency map with similar intent.
    """
    im = _imread_bgr(impath)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    h8, w8 = (h // 8) * 8, (w // 8) * 8
    gray = gray[:h8, :w8]
    # Local 8x8 DCT energy inconsistency.
    blocks = gray.reshape(h8 // 8, 8, w8 // 8, 8).transpose(0, 2, 1, 3)
    energy = np.zeros((h8 // 8, w8 // 8), dtype=np.float32)
    for i in range(blocks.shape[0]):
        for j in range(blocks.shape[1]):
            b = blocks[i, j] - 128.0
            d = cv2.dct(b)
            energy[i, j] = float(np.mean(np.abs(d[1:, 1:])))
    # Compare each block to its local neighborhood.
    k = np.ones((3, 3), dtype=np.float32) / 9.0
    neigh = cv2.filter2D(energy, -1, k, borderType=cv2.BORDER_REFLECT)
    out = np.abs(energy - neigh)
    out = cv2.resize(out, (w8, h8), interpolation=cv2.INTER_LINEAR)
    full = np.zeros((h, w), dtype=np.float32)
    full[:h8, :w8] = out
    return full


def _noi1_fallback(impath: str, block_size: int = 8) -> np.ndarray:
    im = _imread_bgr(impath)
    ycbcr = cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    y = np.round(ycbcr[:, :, 0])
    try:
        from pywt import dwt2
        _, (_, _, cD) = dwt2(y, "db8")
    except Exception:
        # Portable fallback: high-pass residual map when pywavelets is unavailable.
        blur = cv2.GaussianBlur(y.astype(np.float32), (0, 0), 1.2)
        cD = y.astype(np.float32) - blur
    cD = cD[: (cD.shape[0] // block_size) * block_size, : (cD.shape[1] // block_size) * block_size]
    if cD.size == 0:
        return np.zeros((im.shape[0], im.shape[1]), dtype=np.float32)
    out = np.zeros((cD.shape[0] // block_size, cD.shape[1] // block_size), dtype=np.float32)
    for ii in range(0, cD.shape[0], block_size):
        for jj in range(0, cD.shape[1], block_size):
            blk = cD[ii:ii + block_size, jj:jj + block_size]
            out[ii // block_size, jj // block_size] = np.median(np.abs(blk)) / 0.6745
    return out.astype(np.float32)


def _noi4_fallback(impath: str, nsize: int = 3, multiplier: float = 10.0, flatten: bool = True) -> np.ndarray:
    im = np.asarray(Image.open(impath).convert("RGB"), dtype=np.float32)
    med = np.zeros_like(im)
    for c in range(im.shape[2]):
        med[:, :, c] = medfilt(im[:, :, c], [nsize, nsize])
    out = np.abs(im - med) * float(multiplier)
    if flatten:
        out = out.mean(axis=2)
    return out.astype(np.float32)


def _ghost_fallback(impath: str, quality_min: int = 60, quality_max: int = 95) -> np.ndarray:
    """Portable JPEG-ghost style residual search inspired by pyIFD.GHOST."""
    im = _imread_bgr(impath).astype(np.float32)
    h = np.ones((17, 17, 1), dtype=np.float32) / (17 * 17)
    best_score = None
    best_map = None
    for q in range(int(quality_min), int(quality_max) + 1, 5):
        ok, enc = cv2.imencode(".jpg", im, [int(cv2.IMWRITE_JPEG_QUALITY), int(q)])
        if not ok:
            continue
        rec = cv2.imdecode(enc, cv2.IMREAD_COLOR).astype(np.float32)
        comp = (rec - im) ** 2
        sm = fftconvolve(comp, h, mode="same").mean(axis=2)
        score = float(sm.mean())
        if best_score is None or score < best_score:
            best_score = score
            best_map = sm
    if best_map is None:
        return np.zeros((im.shape[0], im.shape[1]), dtype=np.float32)
    small = resize(best_map.astype(np.float32), (max(1, best_map.shape[0] // 4), max(1, best_map.shape[1] // 4)), preserve_range=True, anti_aliasing=True)
    return small.astype(np.float32)


def _blk_fallback(impath: str) -> np.ndarray:
    im = _imread_bgr(impath)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    out = np.zeros_like(gray)
    for x in range(8, h, 8):
        out[x - 1:x + 1, :] += np.abs(gray[x - 1:x + 1, :] - gray[max(x - 2, 0):x, :].mean(axis=0, keepdims=True))
    for y in range(8, w, 8):
        out[:, y - 1:y + 1] += np.abs(gray[:, y - 1:y + 1] - gray[:, max(y - 2, 0):y].mean(axis=1, keepdims=True))
    return out.astype(np.float32)


# -----------------------------------------------------------------------------
# Module wrappers
# -----------------------------------------------------------------------------


class PyIFD_ELA(ForensicModule):
    def __init__(self, quality: int = 75, multiplier: float = 15.0, flatten: bool = True):
        super().__init__(ModuleMeta(
            module_id="pyifd_ela",
            family="jpeg",
            desc="Error Level Analysis",
            default_params={"quality": quality, "multiplier": multiplier, "flatten": flatten},
        ))
        self.quality = quality
        self.multiplier = multiplier
        self.flatten = flatten

    def run(self, impath: str) -> np.ndarray:
        mod = _import_pyifd_submodule("ELA")
        if mod is not None and hasattr(mod, "ELA"):
            out = mod.ELA(impath, Quality=int(self.quality), Multiplier=float(self.multiplier), Flatten=bool(self.flatten))
        else:
            out = _ela_fallback(impath, self.quality, self.multiplier, self.flatten)
        return _soft_normalize(_resize_to_image(out, impath))


class PyIFD_DCT(ForensicModule):
    def __init__(self, force_png_mode: bool = True):
        super().__init__(ModuleMeta(
            module_id="pyifd_dct",
            family="jpeg",
            desc="DCT/block artifact inconsistency",
            default_params={"force_png_mode": force_png_mode},
        ))
        self.force_png_mode = force_png_mode

    def run(self, impath: str) -> np.ndarray:
        mod = _import_pyifd_submodule("DCT")
        if mod is not None:
            try:
                if self.force_png_mode and hasattr(mod, "GetDCTArtifact"):
                    out = mod.GetDCTArtifact(_imread_bgr(impath), png=True)
                elif hasattr(mod, "DCT"):
                    out = mod.DCT(impath)
                else:
                    out = _dct_blockiness_fallback(impath)
            except Exception:
                out = _dct_blockiness_fallback(impath)
        else:
            out = _dct_blockiness_fallback(impath)
        return _soft_normalize(_resize_to_image(out, impath))


class PyIFD_NOI1(ForensicModule):
    def __init__(self, block_size: int = 8):
        super().__init__(ModuleMeta(
            module_id="pyifd_noi1",
            family="noise",
            desc="Noise variance inconsistency (wavelet)",
            default_params={"block_size": block_size},
        ))
        self.block_size = block_size

    def run(self, impath: str) -> np.ndarray:
        mod = _import_pyifd_submodule("NOI1")
        if mod is not None and hasattr(mod, "GetNoiseMap"):
            out = mod.GetNoiseMap(impath, BlockSize=int(self.block_size))
        else:
            out = _noi1_fallback(impath, self.block_size)
        return _soft_normalize(_resize_to_image(out, impath))


class PyIFD_NOI2(ForensicModule):
    def __init__(self, filter_type: str = "rand", filter_size: int = 4, block_rad: int = 8):
        super().__init__(ModuleMeta(
            module_id="pyifd_noi2",
            family="noise",
            desc="Blind local noise estimation",
            default_params={"filter_type": filter_type, "filter_size": filter_size, "block_rad": block_rad},
        ))
        self.filter_type = filter_type
        self.filter_size = filter_size
        self.block_rad = block_rad

    def run(self, impath: str) -> np.ndarray:
        mod = _import_pyifd_submodule("NOI2")
        if mod is None or not hasattr(mod, "GetNoiseMaps"):
            raise ImportError("pyIFD.NOI2 is required for this optional module")
        out = mod.GetNoiseMaps(impath, filter_type=str(self.filter_type), filter_size=int(self.filter_size), block_rad=int(self.block_rad))
        return _soft_normalize(_resize_to_image(out, impath))


class PyIFD_NOI4(ForensicModule):
    def __init__(self, nsize: int = 3, multiplier: float = 10.0, flatten: bool = True):
        super().__init__(ModuleMeta(
            module_id="pyifd_noi4",
            family="noise",
            desc="Median-filter residuals",
            default_params={"nsize": nsize, "multiplier": multiplier, "flatten": flatten},
        ))
        self.nsize = nsize
        self.multiplier = multiplier
        self.flatten = flatten

    def run(self, impath: str) -> np.ndarray:
        mod = _import_pyifd_submodule("NOI4")
        if mod is not None and hasattr(mod, "MedFiltForensics"):
            out = mod.MedFiltForensics(impath, NSize=int(self.nsize), Multiplier=float(self.multiplier), Flatten=bool(self.flatten))
        else:
            out = _noi4_fallback(impath, self.nsize, self.multiplier, self.flatten)
        return _soft_normalize(_resize_to_image(out, impath))


class PyIFD_NOI5(ForensicModule):
    def __init__(self):
        super().__init__(ModuleMeta(
            module_id="pyifd_noi5",
            family="noise",
            desc="PCA-based noise estimation",
            default_params={},
        ))

    def run(self, impath: str) -> np.ndarray:
        mod = _import_pyifd_submodule("NOI5")
        if mod is None or not hasattr(mod, "PCANoise"):
            raise ImportError("pyIFD.NOI5 is required for this optional module")
        out = mod.PCANoise(impath)[0]
        return _soft_normalize(_resize_to_image(out, impath))


class PyIFD_GHOST(ForensicModule):
    def __init__(self, check_displacements: int = 0):
        super().__init__(ModuleMeta(
            module_id="pyifd_ghost",
            family="jpeg",
            desc="JPEG ghost detector",
            default_params={"check_displacements": check_displacements},
        ))
        self.check_displacements = check_displacements

    def run(self, impath: str) -> np.ndarray:
        mod = _import_pyifd_submodule("GHOST")
        out = None
        if mod is not None and hasattr(mod, "GHOST"):
            try:
                res = mod.GHOST(impath, checkDisplacements=int(self.check_displacements))
                if isinstance(res, (list, tuple)) and len(res) >= 3 and isinstance(res[2], dict) and len(res[2]) > 0:
                    if len(res[1]) > 0:
                        idx = int(np.argmin(np.asarray(res[1], dtype=np.float32)))
                    else:
                        idx = sorted(res[2].keys())[0]
                    out = res[2].get(idx, next(iter(res[2].values())))
            except Exception:
                out = None
        if out is None:
            out = _ghost_fallback(impath)
        return _soft_normalize(_resize_to_image(out, impath))


class PyIFD_BLK(ForensicModule):
    def __init__(self):
        super().__init__(ModuleMeta(
            module_id="pyifd_blk",
            family="jpeg",
            desc="JPEG block grid artifacts",
            default_params={},
        ))

    def run(self, impath: str) -> np.ndarray:
        mod = _import_pyifd_submodule("BLK")
        out = None
        if mod is not None and hasattr(mod, "GetBlockGrid"):
            try:
                res = mod.GetBlockGrid(impath)
                out = res[0] if isinstance(res, (list, tuple)) else res
            except Exception:
                out = None
        if out is None:
            out = _blk_fallback(impath)
        return _soft_normalize(_resize_to_image(out, impath))


class PyIFD_CAGI(ForensicModule):
    def __init__(self, use_inversed: bool = False):
        mid = "pyifd_cagi_inv" if use_inversed else "pyifd_cagi"
        super().__init__(ModuleMeta(
            module_id=mid,
            family="jpeg",
            desc="Content-aware grid inconsistencies",
            default_params={"use_inversed": use_inversed},
        ))
        self.use_inversed = use_inversed

    def run(self, impath: str) -> np.ndarray:
        mod = _import_pyifd_submodule("CAGI")
        if mod is None or not hasattr(mod, "CAGI"):
            raise ImportError("pyIFD.CAGI is required for this optional module")
        out1, out2 = mod.CAGI(impath)
        out = out2 if self.use_inversed else out1
        return _soft_normalize(_resize_to_image(out, impath))


class PyIFD_CFA1(ForensicModule):
    def __init__(self):
        super().__init__(ModuleMeta(
            module_id="pyifd_cfa1",
            family="cfa",
            desc="CFA artifact inconsistencies",
            default_params={},
        ))

    def run(self, impath: str) -> np.ndarray:
        mod = _import_pyifd_submodule("CFA1")
        if mod is None or not hasattr(mod, "CFA1"):
            raise ImportError("pyIFD.CFA1 is required for this optional module")
        out = mod.CFA1(impath)
        return _soft_normalize(_resize_to_image(out, impath))


class PyIFD_ADQ1(ForensicModule):
    def __init__(self):
        super().__init__(ModuleMeta(
            module_id="pyifd_adq1",
            family="jpeg",
            desc="Aligned double JPEG compression",
            default_params={},
        ))

    def run(self, impath: str) -> np.ndarray:
        mod = _import_pyifd_submodule("ADQ1")
        if mod is None or not hasattr(mod, "detectDQ"):
            raise ImportError("pyIFD.ADQ1 is required for this optional module")
        out = mod.detectDQ(impath)
        return _soft_normalize(_resize_to_image(out, impath))


class PyIFD_ADQ2(ForensicModule):
    def __init__(self, ncomp: int = 1, c1: int = 1, c2: int = 15):
        super().__init__(ModuleMeta(
            module_id="pyifd_adq2",
            family="jpeg",
            desc="Aligned double JPEG compression (variant)",
            default_params={"ncomp": ncomp, "c1": c1, "c2": c2},
        ))
        self.ncomp = ncomp
        self.c1 = c1
        self.c2 = c2

    def run(self, impath: str) -> np.ndarray:
        mod = _import_pyifd_submodule("ADQ2")
        if mod is None or not hasattr(mod, "getJmap"):
            raise ImportError("pyIFD.ADQ2 is required for this optional module")
        out = mod.getJmap(impath, ncomp=int(self.ncomp), c1=int(self.c1), c2=int(self.c2))[0]
        return _soft_normalize(_resize_to_image(out, impath))


class PyIFD_ADQ3(ForensicModule):
    def __init__(self):
        super().__init__(ModuleMeta(
            module_id="pyifd_adq3",
            family="jpeg",
            desc="Benford/DQ-based artifacts",
            default_params={},
        ))

    def run(self, impath: str) -> np.ndarray:
        mod = _import_pyifd_submodule("ADQ3")
        if mod is None or not hasattr(mod, "BenfordDQ"):
            raise ImportError("pyIFD.ADQ3 is required for this optional module")
        out = mod.BenfordDQ(impath)
        return _soft_normalize(_resize_to_image(out, impath))


class PyIFD_NADQ(ForensicModule):
    def __init__(self):
        super().__init__(ModuleMeta(
            module_id="pyifd_nadq",
            family="jpeg",
            desc="Non-aligned / aligned double JPEG detector",
            default_params={},
        ))

    def run(self, impath: str) -> np.ndarray:
        mod = _import_pyifd_submodule("NADQ")
        if mod is None or not hasattr(mod, "NADQ"):
            raise ImportError("pyIFD.NADQ is required for this optional module")
        out = mod.NADQ(impath)
        out0 = out[0] if isinstance(out, (list, tuple)) else out
        if np.asarray(out0).ndim == 3:
            out0 = np.asarray(out0).mean(axis=2)
        return _soft_normalize(_resize_to_image(out0, impath))


def build_default_pyifd_supernet(include_optional: bool = True) -> List[ForensicModule]:
    """Default module registry for ForensicFusion.

    The first six modules are fast and have local fallbacks, which makes the
    project runnable even if only a subset of pyIFD dependencies is available.
    """
    mods: List[ForensicModule] = [
        PyIFD_ELA(quality=75, multiplier=15.0, flatten=True),
        PyIFD_DCT(force_png_mode=True),
        PyIFD_NOI1(block_size=8),
        PyIFD_NOI4(nsize=3, multiplier=10.0, flatten=True),
        PyIFD_GHOST(check_displacements=0),
        PyIFD_BLK(),
    ]
    if not include_optional:
        return mods
    mods += [
        PyIFD_NOI2(filter_type="rand", filter_size=4, block_rad=8),
        PyIFD_NOI5(),
        PyIFD_CAGI(use_inversed=False),
        PyIFD_CAGI(use_inversed=True),
        PyIFD_CFA1(),
        PyIFD_ADQ1(),
        PyIFD_ADQ2(ncomp=1, c1=1, c2=15),
        PyIFD_ADQ3(),
        PyIFD_NADQ(),
    ]
    return mods
