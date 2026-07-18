from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from evaluation.artifact import display_z, load_artifact, pixel_extent
from evaluation.evaluate import ROOT, reconstruction_matrix, validate_matrix
from evaluation.metrics import score_artifact
from namt.config import CONDITIONS, METHODS, SCENES, load_config


CLIM = 4.0


def panel(ax, artifact, title: str) -> None:
    values = score_artifact(artifact)
    image = display_z(artifact)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#e6e6e6")
    ax.imshow(
        image,
        origin="lower",
        extent=pixel_extent(artifact),
        cmap=cmap,
        vmin=-CLIM,
        vmax=CLIM,
        interpolation="nearest",
    )
    ax.set_title(f"{title}\nAUC {values['auc']:.3f}  CNR {values['cnr']:.2f}", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def colorbar(fig, axes) -> None:
    scalar = ScalarMappable(norm=Normalize(-CLIM, CLIM), cmap="RdBu_r")
    bar = fig.colorbar(scalar, ax=axes, shrink=0.82, pad=0.01)
    bar.set_label("signed background z-score")


def galleries(root: Path, out: Path, seeds: tuple[int, ...], roi: float) -> None:
    matrix = reconstruction_matrix(root, seeds)
    validate_matrix(root, matrix)
    out.mkdir(parents=True, exist_ok=True)
    for condition in CONDITIONS:
        for scene in SCENES:
            fig, axes = plt.subplots(
                len(seeds), len(METHODS), figsize=(12, 11), squeeze=False, constrained_layout=True
            )
            for row, seed in enumerate(seeds):
                artifacts = [load_artifact(matrix[(condition, seed, method, scene)], scene, roi) for method in METHODS]
                for col, (method, artifact) in enumerate(zip(METHODS, artifacts)):
                    panel(axes[row, col], artifact, method)
                    if col == 0:
                        axes[row, col].set_ylabel(f"seed {seed}")
            colorbar(fig, axes)
            fig.suptitle(f"{condition}  {scene}")
            fig.savefig(out / f"{condition}_{scene}.png", dpi=180)
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "paper.json")
    args = parser.parse_args()
    config = load_config(args.config)
    seeds = tuple(int(seed) for seed in config["data"]["seeds"])
    roi = float(config["reconstruction"]["roi_half_width_mm"])
    galleries(args.input_root, args.output_root, seeds, roi)


if __name__ == "__main__":
    main()
