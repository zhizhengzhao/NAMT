from __future__ import annotations

import argparse
from pathlib import Path

from namt import data
from namt.calibration import save_blank_calibration
from namt.config import CALIBRATIONS, load_config
from namt.methods import get


ROOT = Path(__file__).resolve().parent
NAMT_METHODS = ("namt_3p", "namt_4p")


def selection(value, allowed):
    if value == "all":
        return allowed
    selected = tuple(item for item in value.split(",") if item)
    if not selected or any(item not in allowed for item in selected):
        raise ValueError(f"invalid selection: {value}")
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default="all")
    parser.add_argument("--calibrations", default="all")
    parser.add_argument("--seeds", default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "paper.json")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--assets-root", type=Path, default=ROOT / "assets")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    methods = selection(args.methods, NAMT_METHODS)
    calibrations = selection(args.calibrations, CALIBRATIONS)
    allowed_seeds = tuple(str(seed) for seed in config["data"]["seeds"])
    seeds = tuple(int(seed) for seed in selection(args.seeds, allowed_seeds))
    for seed in seeds:
        for calibration_name in calibrations:
            sigma = float(config["calibrations"][calibration_name]["hit_resolution_mm"])
            hits, layer_z = data.load("A", "blank", smear=sigma, seed=seed, data_root=args.data_root)
            for name in methods:
                method_config = config["methods"][name]
                output = args.assets_root / f"seed_{seed}" / calibration_name / method_config["calibration"]
                if output.exists() and not args.force:
                    raise FileExistsError(output)
                method = get(method_config["implementation"], dev=args.device)
                calibration = method.calibrate(hits, layer_z, sigma)
                selected_z = layer_z[:3] if name == "namt_3p" else layer_z
                save_blank_calibration(
                    output,
                    method_name=method_config["implementation"],
                    layer_z=selected_z,
                    sigma_pos_mm=sigma,
                    calibration=calibration,
                )
                print(f"saved seed={seed} {calibration_name} {name} -> {output}", flush=True)


if __name__ == "__main__":
    main()
