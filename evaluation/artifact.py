from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from namt.scenes import MAT, SCENES, truth_mask


@dataclass(frozen=True)
class Artifact:
    path: Path
    scene: str
    recon: np.ndarray
    gx: np.ndarray
    gy: np.ndarray
    target_roi: np.ndarray
    background_roi: np.ndarray
    finite: np.ndarray
    direction: float


def expected_direction(scene: str) -> float:
    background = SCENES[scene][0]
    target = SCENES[scene][1][0]["mat"]
    direction = float(np.sign(MAT[target]["inv_x0"] - MAT[background]["inv_x0"]))
    if direction not in (-1.0, 1.0):
        raise ValueError(f"{scene}: target and background have equal contrast")
    return direction


def load_artifact(path: str | Path, scene: str, roi_half_width: float) -> Artifact:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        missing = sorted({"recon", "gx", "gy", "target_roi", "bg_roi", "sgn"} - set(data.files))
        if missing:
            raise KeyError(f"{path}: missing {missing}")
        recon = np.asarray(data["recon"], dtype=np.float64)
        gx = np.asarray(data["gx"], dtype=np.float64)
        gy = np.asarray(data["gy"], dtype=np.float64)
        stored_target = np.asarray(data["target_roi"], dtype=bool)
        stored_background = np.asarray(data["bg_roi"], dtype=bool)
        direction = float(np.asarray(data["sgn"]).item())
    if recon.ndim != 2 or gx.shape != (recon.shape[1],) or gy.shape != (recon.shape[0],):
        raise ValueError(f"{path}: invalid reconstruction or axes shape")
    if len(gx) < 2 or len(gy) < 2:
        raise ValueError(f"{path}: each axis requires at least two centers")
    if not np.isfinite(gx).all() or not np.isfinite(gy).all():
        raise ValueError(f"{path}: axes must be finite")
    if np.any(np.diff(gx) <= 0.0) or np.any(np.diff(gy) <= 0.0):
        raise ValueError(f"{path}: axes must be strictly increasing")
    if direction != expected_direction(scene):
        raise ValueError(f"{path}: invalid predeclared direction {direction}")
    truth = truth_mask(scene, gx, gy)
    if truth is None:
        raise ValueError(f"{scene}: missing target geometry")
    xx, yy = np.meshgrid(gx, gy)
    roi = (np.abs(xx) <= float(roi_half_width)) & (np.abs(yy) <= float(roi_half_width))
    target_roi = np.asarray(truth & roi, dtype=bool)
    background_roi = np.asarray(~truth & roi, dtype=bool)
    if not np.array_equal(stored_target, target_roi) or not np.array_equal(stored_background, background_roi):
        raise ValueError(f"{path}: stored masks differ from scene geometry")
    finite = np.isfinite(recon)
    if not target_roi.any() or not background_roi.any():
        raise ValueError(f"{path}: empty geometry-only target or background ROI")
    return Artifact(path, scene, recon, gx, gy, target_roi, background_roi, finite, direction)


def masks(artifact: Artifact) -> tuple[np.ndarray, np.ndarray]:
    target = artifact.target_roi & artifact.finite
    background = artifact.background_roi & artifact.finite
    if not target.any() or not background.any():
        raise ValueError(f"{artifact.path}: empty finite target or background samples")
    return target, background


def display_z(artifact: Artifact) -> np.ndarray:
    target, background = masks(artifact)
    scale = float(artifact.recon[background].std(ddof=0))
    resolution = np.finfo(np.float64).eps * max(
        float(np.abs(artifact.recon[background]).max()), np.finfo(np.float64).tiny
    )
    if not np.isfinite(scale) or scale <= resolution:
        raise ValueError(f"{artifact.path}: invalid background spread")
    center = float(artifact.recon[background].mean())
    z = (artifact.recon - center) / scale
    z[~(target | background)] = np.nan
    return z


def pixel_extent(artifact: Artifact) -> tuple[float, float, float, float]:
    gx = artifact.gx
    gy = artifact.gy
    return (
        float(gx[0] - 0.5 * (gx[1] - gx[0])),
        float(gx[-1] + 0.5 * (gx[-1] - gx[-2])),
        float(gy[0] - 0.5 * (gy[1] - gy[0])),
        float(gy[-1] + 0.5 * (gy[-1] - gy[-2])),
    )
