from __future__ import annotations

from pathlib import Path

import numpy as np

from ablation.method import NAMT4pMLP
from namt import data
from namt.config import load_config
from namt.pipeline import _masks
from namt.scenes import MAT, SCENES as SCENE_MODELS


SCENES = ("u_pb_concrete", "u_explosive_concrete", "u_void_concrete")
VARIANTS = {
    "full": ("marginalized", "blank_calibrated", "momentum_prior.npz"),
    "gaisser_prior": ("marginalized", "blank_calibrated", "momentum_prior_gaisser.npz"),
    "fixed_momentum": ("fixed_mean_g", "blank_calibrated", "momentum_prior.npz"),
    "analytic_instrument": ("marginalized", "analytic_only", "momentum_prior.npz"),
    "no_instrument": ("marginalized", "none", "momentum_prior.npz"),
}


def load_hits(scene, seed, data_root, cap=0):
    hits, layer_z = data.load("A", scene, cap=cap, smear=0.0, seed=seed, data_root=data_root)
    generator = np.random.default_rng(np.random.SeedSequence((seed, data.STREAMS[scene], 1)))
    sigma = 1.0 + 0.15 * np.clip(hits[:, :, 0:1] / 140.0, -1.0, 1.0)
    return hits + generator.normal(0.0, 1.0, hits.shape) * sigma, layer_z


def reconstruct(variant, scene, seed, *, data_root, assets_root, device):
    if variant not in VARIANTS or scene not in SCENES:
        raise ValueError("unknown ablation experiment")
    config = load_config()
    if seed not in config["data"]["seeds"]:
        raise ValueError("unknown seed")
    momentum, instrument, prior = VARIANTS[variant]
    tv = config["scenes"][scene]["tv"]
    method = NAMT4pMLP(
        dev=device,
        momentum_prior_path=str(Path(assets_root) / prior),
        momentum_quad_order=256,
        tv_xy=tv,
        tv_z=tv,
        momentum_mode=momentum,
        instrument_mode=instrument,
    )
    hits, layer_z = load_hits(scene, seed, data_root, cap=150000)
    if len(hits) != 150000:
        raise ValueError("the ablation requires 150000 accepted events")
    calibration = {}
    if instrument == "blank_calibrated":
        blank, blank_z = load_hits("blank", seed, data_root)
        if len(blank) != 120000 or not np.array_equal(blank_z, layer_z):
            raise ValueError("invalid blank scan")
        calibration = method.calibrate(blank, layer_z, 1.0)
    result = method.reconstruct(
        hits, layer_z, 1.0, calibration, 5.0,
        scene=scene, steps=3000, lr=0.03, tv="l1", min_cov=10,
    )
    gx, gy = result["gx"], result["gy"]
    target, background = _masks(gx, gy, scene, 120.0)
    bg_material, objects = SCENE_MODELS[scene]
    direction = np.sign(MAT[objects[0]["mat"]]["inv_x0"] - MAT[bg_material]["inv_x0"])
    return {
        "recon": result["img"], "gx": gx, "gy": gy,
        "target_roi": target, "bg_roi": background, "sgn": np.asarray(direction),
        "method": np.asarray("namt_4p"), "variant": np.asarray(variant),
        "scene": np.asarray(scene), "seed": np.asarray(seed),
        "condition": np.asarray("150k_s1_g0.15"), "tv": np.asarray(tv),
        "sigma0": np.asarray(1.0), "slope": np.asarray(0.15),
        "momentum_mode": np.asarray(momentum), "instrument_mode": np.asarray(instrument),
        "fixed_g": np.asarray(result["fixed_g"]),
    }
