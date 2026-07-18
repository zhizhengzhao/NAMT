from __future__ import annotations

import argparse
from pathlib import Path

from namt.config import CONDITIONS, METHODS, SCENES, load_config
from namt.pipeline import reconstruct, save_reconstruction


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--scene", choices=SCENES, required=True)
    parser.add_argument("--condition", choices=CONDITIONS, default="150k_1mm")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "paper.json")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--assets-root", type=Path, default=ROOT / "assets")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.seed not in config["data"]["seeds"]:
        raise ValueError(f"unknown seed: {args.seed}")
    default_output = ROOT / "reconstructions" / args.condition / f"seed_{args.seed}" / args.method / f"{args.scene}.npz"
    output = args.output or default_output
    if output.exists() and not args.force:
        raise FileExistsError(f"{output} exists; pass --force to replace it")
    artifact = reconstruct(
        args.method,
        args.scene,
        config=config,
        data_root=args.data_root,
        assets_root=args.assets_root,
        device=args.device,
        condition=args.condition,
        seed=args.seed,
    )
    save_reconstruction(output, artifact, overwrite=args.force)
    print(f"saved {args.method}/{args.scene} -> {output}")


if __name__ == "__main__":
    main()
