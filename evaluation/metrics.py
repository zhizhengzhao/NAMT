from __future__ import annotations

import numpy as np

from evaluation.artifact import Artifact, masks


def auc(target: np.ndarray, background: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64).ravel()
    background = np.asarray(background, dtype=np.float64).ravel()
    if not len(target) or not len(background):
        raise ValueError("AUC requires nonempty samples")
    values = np.concatenate((target, background))
    if not np.isfinite(values).all():
        raise ValueError("AUC requires finite samples")
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_values)) + 1]
    stops = np.r_[starts[1:], len(sorted_values)]
    for start, stop in zip(starts, stops):
        ranks[order[start:stop]] = 0.5 * (start + stop + 1)
    n_target = len(target)
    score = float((ranks[:n_target].sum() - n_target * (n_target + 1) / 2.0) / (n_target * len(background)))
    if not np.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"invalid AUC {score}")
    return score


def cnr(target: np.ndarray, background: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64).ravel()
    background = np.asarray(background, dtype=np.float64).ravel()
    if not len(target) or not len(background):
        raise ValueError("CNR requires nonempty samples")
    if not np.isfinite(target).all() or not np.isfinite(background).all():
        raise ValueError("CNR requires finite samples")
    scale = float(background.std(ddof=0))
    resolution = np.finfo(np.float64).eps * max(float(np.abs(background).max()), np.finfo(np.float64).tiny)
    if not np.isfinite(scale) or scale <= resolution:
        raise ValueError("CNR requires nonzero background spread")
    value = float((target.mean() - background.mean()) / scale)
    if not np.isfinite(value):
        raise ValueError(f"invalid CNR {value}")
    return value


def score_artifact(artifact: Artifact) -> dict[str, float | int]:
    target, background = masks(artifact)
    score_image = artifact.direction * artifact.recon
    return {
        "auc": auc(score_image[target], score_image[background]),
        "cnr": cnr(score_image[target], score_image[background]),
        "n_target": int(target.sum()),
        "n_bg": int(background.sum()),
        "n_valid": int(target.sum() + background.sum()),
    }
