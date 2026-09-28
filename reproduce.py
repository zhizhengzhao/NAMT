from __future__ import annotations

import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ablation.run import SCENES as ABLATION_SCENES, VARIANTS, reconstruct as reconstruct_ablation
from evaluation.report import build_reports
from namt.config import CONDITIONS, METHODS, SCENES, SEEDS, load_config
from namt.pipeline import reconstruct, save_reconstruction


ROOT = Path(__file__).resolve().parent


def experiments(suite):
    jobs = []
    if suite in ("all", "benchmark"):
        jobs.extend(
            ("benchmark", condition, seed, method, scene)
            for condition in CONDITIONS for seed in SEEDS for method in METHODS for scene in SCENES
        )
    if suite in ("all", "ablation"):
        jobs.extend(
            ("ablation", "150k_s1_g0.15", seed, variant, scene)
            for seed in SEEDS for variant in VARIANTS for scene in ABLATION_SCENES
        )
    return jobs


def artifact_path(root, job):
    suite, condition, seed, method, scene = job
    directory = root / suite
    if suite == "benchmark":
        directory /= condition
    return directory / f"seed_{seed}" / method / f"{scene}.npz"


def run_queue(jobs, device, output_root, force):
    config = load_config()
    for job in jobs:
        path = artifact_path(output_root, job)
        suite, condition, seed, method, scene = job
        if suite == "benchmark":
            artifact = reconstruct(
                method, scene, config=config, data_root=ROOT / "data",
                assets_root=ROOT / "assets", device=device, condition=condition, seed=seed,
            )
        else:
            artifact = reconstruct_ablation(
                method, scene, seed, data_root=ROOT / "data",
                assets_root=ROOT / "assets", device=device,
            )
        save_reconstruction(path, artifact, overwrite=force)
        print(f"saved {path} on {device}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", action="append", dest="devices")
    parser.add_argument("--suite", choices=("all", "benchmark", "ablation"), default="all")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.input_root is not None and not args.evaluate_only:
        parser.error("--input-root is only used with --evaluate-only")
    devices = tuple(args.devices or ("cuda:0",))
    if len(set(devices)) != len(devices):
        parser.error("devices must be unique")
    jobs = experiments(args.suite)
    input_root = args.input_root if args.input_root is not None else args.output_root
    if not args.evaluate_only:
        pending = [job for job in jobs if args.force or not artifact_path(args.output_root, job).exists()]
        print(f"{len(pending)} runs pending; {len(jobs) - len(pending)} existing artifacts", flush=True)
        if len(devices) == 1:
            run_queue(pending, devices[0], args.output_root, args.force)
        elif pending:
            queues = [pending[index::len(devices)] for index in range(len(devices))]
            with ProcessPoolExecutor(
                max_workers=min(len(devices), len(pending)),
                mp_context=multiprocessing.get_context("spawn"),
            ) as executor:
                futures = [
                    executor.submit(run_queue, queue, device, args.output_root, args.force)
                    for queue, device in zip(queues, devices) if queue
                ]
                for future in futures:
                    future.result()
    missing = [artifact_path(input_root, job) for job in jobs if not artifact_path(input_root, job).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} artifacts missing, including {missing[0]}")
    build_reports(input_root, args.output_root / "reports", args.suite)


if __name__ == "__main__":
    main()
