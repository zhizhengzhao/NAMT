from __future__ import annotations

import os
from pathlib import Path

import numpy as np


DEFAULT_ROOT = Path(os.environ.get("NAMT_DATA_ROOT", "data"))
STREAMS = {
    "blank": 0,
    "u_pb_soil": 1,
    "u_pb_concrete": 2,
    "u_explosive_soil": 3,
    "u_explosive_concrete": 4,
    "u_void_soil": 5,
    "u_void_concrete": 6,
}


def load(tier, scene, cap=0, smear=0.0, seed=42, data_root=None):
    if scene not in STREAMS:
        raise ValueError(f"unknown data scene: {scene}")
    root = DEFAULT_ROOT if data_root is None else Path(data_root)
    path = root / f"seed_{int(seed)}" / f"{tier}_{scene}.npz"
    with np.load(path, allow_pickle=False) as artifact:
        required = {"hits", "nhit", "layer_z"}
        missing = sorted(required - set(artifact.files))
        if missing:
            raise KeyError(f"{path}: missing {missing}")
        layer_z = np.asarray(artifact["layer_z"], dtype=np.float64)
        hits = np.asarray(artifact["hits"], dtype=np.float64)
        nhit = np.asarray(artifact["nhit"])
    if hits.ndim != 3 or hits.shape[1:] != (len(layer_z), 2):
        raise ValueError(f"{path}: invalid hit shape {hits.shape}")
    if nhit.shape != (len(hits),):
        raise ValueError(f"{path}: invalid nhit shape {nhit.shape}")
    hits = hits[nhit == len(layer_z)]
    cap = int(cap)
    if cap < 0:
        raise ValueError("cap must be nonnegative")
    if cap:
        if cap > len(hits):
            raise ValueError(f"{path}: requested {cap} events from {len(hits)}")
        hits = hits[:cap]
    smear = float(smear)
    if smear < 0:
        raise ValueError("smear must be nonnegative")
    if smear:
        generator = np.random.default_rng(np.random.SeedSequence((seed, STREAMS[scene], 1)))
        hits = hits + generator.normal(0.0, smear, hits.shape)
    return hits, layer_z
