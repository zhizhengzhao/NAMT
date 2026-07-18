from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from namt import data
from namt.calibration import load_blank_calibration
from namt.config import CONDITIONS, METHODS, SCENES
from namt.methods import get
from namt.scenes import MAT, SCENES as SCENE_MODELS, truth_mask


NAMT_METHODS = ("namt_3p", "namt_4p")


def _masks(gx, gy, scene, roi_half_width):
    truth = truth_mask(scene, gx, gy)
    if truth is None:
        raise ValueError(f"{scene} has no target mask")
    xx, yy = np.meshgrid(gx, gy)
    roi = (
        (np.abs(xx) <= float(roi_half_width))
        & (np.abs(yy) <= float(roi_half_width))
    )
    return truth & roi, ~truth & roi


def _load_namt_calibration(method_name, method, calibration, data_root, assets_root, config, sigma, seed):
    method_config = config["methods"][method_name]
    calibration_path = Path(assets_root) / f"seed_{int(seed)}" / calibration / method_config["calibration"]
    with np.load(Path(data_root) / f"seed_{int(seed)}" / "A_blank.npz", allow_pickle=False) as blank:
        layer_z = np.asarray(blank["layer_z"], dtype=np.float64)
    selected_z = layer_z[:3] if method_name == "namt_3p" else layer_z
    return load_blank_calibration(
        calibration_path,
        method_name=method_config["implementation"],
        layer_z=selected_z,
        dev=method.dev,
        sigma_pos_mm=float(sigma),
    )


def reconstruct(
    method_name,
    scene,
    *,
    config,
    data_root,
    assets_root,
    device="cuda:0",
    condition="150k_1mm",
    seed=42,
):
    if method_name not in METHODS:
        raise ValueError(f"unknown method {method_name!r}")
    if scene not in SCENES:
        raise ValueError(f"unknown scene {scene!r}")
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}")
    seed = int(seed)
    if seed not in config["data"]["seeds"]:
        raise ValueError(f"unknown seed: {seed}")
    condition_config = config["conditions"][condition]
    calibration_name = condition_config["calibration"]
    calibration_config = config["calibrations"][calibration_name]
    reconstruction_config = config["reconstruction"]
    method_config = config["methods"][method_name]
    sigma = float(calibration_config["hit_resolution_mm"])
    cap = int(condition_config["event_count"])
    vox = float(reconstruction_config["voxel_size_mm"])
    implementation = method_config["implementation"]

    method_kwargs = {}
    reconstruct_kwargs = {
        "min_cov": int(reconstruction_config["minimum_coverage"]),
    }
    if method_name in NAMT_METHODS:
        prior_path = Path(assets_root) / "momentum_prior.npz"
        tv_weight = float(config["scenes"][scene]["tv"])
        if tv_weight < 0.0:
            raise ValueError("TV must be nonnegative")
        method_kwargs = {
            "momentum_prior_path": str(prior_path),
            "momentum_quad_order": int(config["namt"]["momentum_quadrature_order"]),
            "tv_xy": tv_weight,
            "tv_z": tv_weight,
        }
        reconstruct_kwargs = {
            "steps": int(config["namt"]["optimizer_steps"]),
            "lr": float(config["namt"]["learning_rate"]),
            "tv": config["namt"]["tv_penalty"],
            **reconstruct_kwargs,
        }
    elif method_name == "mlsd_median":
        reconstruct_kwargs["iters"] = int(method_config["iterations"])

    method = get(implementation, dev=device, **method_kwargs)
    calibration = (
        _load_namt_calibration(
            method_name,
            method,
            calibration_name,
            data_root,
            assets_root,
            config,
            sigma,
            seed,
        )
        if method_name in NAMT_METHODS
        else {}
    )
    hits, layer_z = data.load(
        method.tier,
        scene,
        cap=cap,
        smear=sigma,
        seed=seed,
        data_root=data_root,
    )
    if len(hits) != cap:
        raise RuntimeError(f"{scene}: loaded {len(hits)} events, expected {cap}")
    result = method.reconstruct(
        hits,
        layer_z,
        sigma,
        calibration,
        vox,
        scene=scene,
        **reconstruct_kwargs,
    )
    image = np.asarray(result["img"], dtype=np.float64)
    gx = np.asarray(result["gx"], dtype=np.float64)
    gy = np.asarray(result["gy"], dtype=np.float64)
    target, background = _masks(gx, gy, scene, reconstruction_config["roi_half_width_mm"])
    background_material = SCENE_MODELS[scene][0]
    target_material = SCENE_MODELS[scene][1][0]["mat"]
    direction = float(np.sign(MAT[target_material]["inv_x0"] - MAT[background_material]["inv_x0"]))
    return {
        "recon": image,
        "gx": gx,
        "gy": gy,
        "target_roi": target,
        "bg_roi": background,
        "sgn": np.asarray(direction, dtype=np.float64),
        "method": np.asarray(method_name),
        "scene": np.asarray(scene),
        "condition": np.asarray(condition),
        "seed": np.asarray(seed, dtype=np.int64),
        "tv": np.asarray(np.nan if method_name not in NAMT_METHODS else tv_weight, dtype=np.float64),
    }


def save_reconstruction(path, artifact, *, overwrite=False):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **artifact)
    with np.load(temporary, allow_pickle=False) as check:
        if "recon" not in check.files or not np.isfinite(check["recon"]).any():
            raise RuntimeError(f"invalid reconstruction artifact: {temporary}")
    os.replace(temporary, path)
