from __future__ import annotations

import json
from pathlib import Path


METHODS = (
    "poca",
    "asr_q75",
    "mlsd_median",
    "namt_3p",
    "namt_4p",
)
CALIBRATIONS = ("1mm", "2mm", "3mm")
CONDITIONS = ("150k_1mm", "100k_1mm", "50k_1mm", "150k_2mm", "150k_3mm")
SCENES = (
    "u_pb_soil",
    "u_pb_concrete",
    "u_explosive_soil",
    "u_explosive_concrete",
    "u_void_soil",
    "u_void_concrete",
)


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    if tuple(config.get("methods", ())) != METHODS:
        raise ValueError(f"method order must be {METHODS}")
    if tuple(config.get("scenes", ())) != SCENES:
        raise ValueError(f"scene order must be {SCENES}")
    if tuple(config.get("conditions", ())) != CONDITIONS:
        raise ValueError(f"condition order must be {CONDITIONS}")
    if tuple(config.get("calibrations", ())) != CALIBRATIONS:
        raise ValueError(f"calibration order must be {CALIBRATIONS}")
    seeds = tuple(config.get("data", {}).get("seeds", ()))
    if len(seeds) != 5 or len(set(seeds)) != 5 or not all(isinstance(seed, int) for seed in seeds):
        raise ValueError("data seeds must contain five unique integers")
    for calibration in CALIBRATIONS:
        if config["calibrations"][calibration]["hit_resolution_mm"] <= 0:
            raise ValueError(f"{calibration}: hit_resolution_mm must be positive")
    for condition in CONDITIONS:
        if config["conditions"][condition]["event_count"] <= 0:
            raise ValueError(f"{condition}: event_count must be positive")
        if config["conditions"][condition]["calibration"] not in CALIBRATIONS:
            raise ValueError(f"{condition}: unknown calibration")
    if config["reconstruction"]["voxel_size_mm"] <= 0:
        raise ValueError("voxel_size_mm must be positive")
    minimum_coverage = config["reconstruction"]["minimum_coverage"]
    if not isinstance(minimum_coverage, int) or minimum_coverage < 1:
        raise ValueError("minimum_coverage must be a positive integer")
    for scene in SCENES:
        tv = float(config["scenes"][scene]["tv"])
        if tv < 0:
            raise ValueError(f"{scene}: TV must be nonnegative")
    return config
