from __future__ import annotations

import argparse
from pathlib import Path

from namt import data
from namt.calibration import save_blank_calibration
from namt.config import CALIBRATIONS, SEEDS, load_config
from namt.methods import get


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "calibrations")
    args = parser.parse_args()
    config = load_config()
    for seed in SEEDS:
        for name in CALIBRATIONS:
            sigma = config["calibrations"][name]["hit_resolution_mm"]
            hits, layer_z = data.load("A", "blank", smear=sigma, seed=seed, data_root=ROOT / "data")
            for method_name in ("namt_3p", "namt_4p"):
                spec = config["methods"][method_name]
                output = args.output_root / f"seed_{seed}" / name / spec["calibration"]
                if output.exists():
                    print(f"skip {output}", flush=True)
                    continue
                method = get(spec["implementation"], dev=args.device)
                calibration = method.calibrate(hits, layer_z, sigma)
                selected_z = layer_z[:3] if method_name == "namt_3p" else layer_z
                save_blank_calibration(
                    output, method_name=spec["implementation"], layer_z=selected_z,
                    sigma_pos_mm=sigma, calibration=calibration,
                )
                print(f"saved {output}", flush=True)


if __name__ == "__main__":
    main()
