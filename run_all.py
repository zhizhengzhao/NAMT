from __future__ import annotations

import argparse
from pathlib import Path

from namt.config import CONDITIONS, METHODS, SCENES, load_config
from namt.pipeline import reconstruct, save_reconstruction


ROOT = Path(__file__).resolve().parent


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
    parser.add_argument("--scenes", default="all")
    parser.add_argument("--conditions", default="all")
    parser.add_argument("--seeds", default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "paper.json")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--assets-root", type=Path, default=ROOT / "assets")
    parser.add_argument("--output-root", type=Path, default=ROOT / "reconstructions")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    methods = selection(args.methods, METHODS)
    scenes = selection(args.scenes, SCENES)
    conditions = selection(args.conditions, CONDITIONS)
    allowed_seeds = tuple(str(seed) for seed in config["data"]["seeds"])
    seeds = tuple(int(seed) for seed in selection(args.seeds, allowed_seeds))
    for condition in conditions:
        for seed in seeds:
            for method in methods:
                for scene in scenes:
                    output = args.output_root / condition / f"seed_{seed}" / method / f"{scene}.npz"
                    if output.exists() and not args.force:
                        print(f"skip {output}", flush=True)
                        continue
                    artifact = reconstruct(
                        method,
                        scene,
                        config=config,
                        data_root=args.data_root,
                        assets_root=args.assets_root,
                        device=args.device,
                        condition=condition,
                        seed=seed,
                    )
                    save_reconstruction(output, artifact, overwrite=args.force)
                    print(f"saved {output}", flush=True)


if __name__ == "__main__":
    main()
