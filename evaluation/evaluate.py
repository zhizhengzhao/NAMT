from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from evaluation.artifact import load_artifact
from evaluation.metrics import score_artifact
from namt.config import CONDITIONS, METHODS, SCENES, load_config


ROOT = Path(__file__).resolve().parents[1]
RUN_FIELDS = (
    "condition",
    "seed",
    "method",
    "scene",
    "auc",
    "cnr",
    "n_target",
    "n_bg",
    "n_valid",
    "file",
)


def reconstruction_matrix(root: Path, seeds: tuple[int, ...]) -> dict[tuple[str, int, str, str], Path]:
    return {
        (condition, seed, method, scene): root / condition / f"seed_{seed}" / method / f"{scene}.npz"
        for condition in CONDITIONS
        for seed in seeds
        for method in METHODS
        for scene in SCENES
    }


def validate_matrix(root: Path, matrix: dict) -> None:
    expected = {path.resolve() for path in matrix.values()}
    observed = {path.resolve() for path in root.rglob("*.npz")}
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra or len(observed) != len(matrix):
        raise RuntimeError(
            f"artifact matrix differs: expected={len(matrix)}, files={len(observed)}, "
            f"missing={len(missing)}, extra={len(extra)}, "
            f"first_missing={missing[:3]}, first_extra={extra[:3]}"
        )


def evaluate(matrix: dict, root: Path, roi: float) -> list[dict]:
    rows = []
    for condition in CONDITIONS:
        seeds = sorted({key[1] for key in matrix})
        for seed in seeds:
            for method in METHODS:
                for scene in SCENES:
                    path = matrix[(condition, seed, method, scene)]
                    artifact = load_artifact(path, scene, roi)
                    rows.append(
                        {
                            "condition": condition,
                            "seed": seed,
                            "method": method,
                            "scene": scene,
                            **score_artifact(artifact),
                            "file": str(path.relative_to(root)),
                        }
                    )
    return rows


def write_csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["condition"], row["method"], row["scene"])].append(row)
    summary = []
    for key in sorted(grouped):
        values = grouped[key]
        auc = np.asarray([row["auc"] for row in values], dtype=np.float64)
        cnr = np.asarray([row["cnr"] for row in values], dtype=np.float64)
        summary.append(
            {
                "condition": key[0],
                "method": key[1],
                "scene": key[2],
                "n": len(values),
                "auc_mean": float(auc.mean()),
                "auc_std": float(auc.std(ddof=1)),
                "cnr_mean": float(cnr.mean()),
                "cnr_std": float(cnr.std(ddof=1)),
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "paper.json")
    parser.add_argument("--runs-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    seeds = tuple(int(seed) for seed in config["data"]["seeds"])
    roi = float(config["reconstruction"]["roi_half_width_mm"])
    matrix = reconstruction_matrix(args.input_root, seeds)
    validate_matrix(args.input_root, matrix)
    rows = evaluate(matrix, args.input_root, roi)
    write_csv(args.runs_output, rows, RUN_FIELDS)
    summary_fields = (
        "condition",
        "method",
        "scene",
        "n",
        "auc_mean",
        "auc_std",
        "cnr_mean",
        "cnr_std",
    )
    write_csv(args.summary_output, summarize(rows), summary_fields)
    print(f"saved {len(rows)} runs to {args.runs_output}")


if __name__ == "__main__":
    main()
