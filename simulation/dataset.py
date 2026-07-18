from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import uproot


BRANCHES = (
    "Edeps.Id",
    "Edeps.X",
    "Edeps.Y",
    "Edeps.trackID",
    "Event.Pid",
)


def scalars(value):
    if isinstance(value, np.ndarray):
        for item in value.flat:
            yield from scalars(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from scalars(item)
    elif hasattr(value, "__iter__") and not isinstance(value, (str, bytes, np.generic)):
        for item in value:
            yield from scalars(item)
    else:
        yield value


def scalar(value, cast):
    values = list(scalars(value))
    if not values:
        raise ValueError("empty scalar field")
    return cast(values[0])


def layer_z(root):
    value = root["params"]["Params/Params.LayerZ"].array(library="np")[0]
    return np.sort(np.asarray(list(scalars(value)), dtype=np.float64))[::-1]


def nbeam(root):
    value = root["params"]["Params/Params.NEvent"].array(library="np")[0]
    return scalar(value, int)


def parse_layer_z(value):
    return np.asarray([float(item) for item in value.split(",")], dtype=np.float64)


def job_id(path):
    match = re.search(r"job_(\d+)\.root$", str(path))
    if match is None:
        raise ValueError(f"invalid ROOT filename: {path}")
    return int(match.group(1))


def root_files(pattern, start, count):
    files = sorted(glob.glob(pattern), key=job_id)
    if not files:
        raise FileNotFoundError(f"no ROOT files matched {pattern!r}")
    ids = [job_id(path) for path in files]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate ROOT job index")
    if count is not None:
        expected = list(range(start, start + count))
        if ids != expected:
            raise ValueError("ROOT job range is incomplete")
    return files


def validate_root(path, expected_nbeam, expected_layer_z):
    with uproot.open(path) as root:
        if "tree" not in root or "params" not in root:
            raise ValueError(f"{path}: missing tree or params")
        root["tree"].arrays(BRANCHES, entry_stop=1, library="np")
        observed_nbeam = nbeam(root)
        observed_z = layer_z(root)
        if observed_nbeam != expected_nbeam:
            raise ValueError(f"{path}: primary count differs")
        if not np.allclose(observed_z, expected_layer_z, atol=1e-6, rtol=0.0):
            raise ValueError(f"{path}: detector geometry differs")
        if root["tree"].num_entries > observed_nbeam:
            raise ValueError(f"{path}: event count exceeds primary count")
    return observed_z


def read_muons(path, expected_nbeam, expected_layer_z):
    validate_root(path, expected_nbeam, expected_layer_z)
    with uproot.open(path) as root:
        arrays = root["tree"].arrays(BRANCHES, library="np")
    records = []
    for event in range(len(arrays["Event.Pid"])):
        if abs(scalar(arrays["Event.Pid"][event], int)) != 13:
            continue
        ids = np.asarray(arrays["Edeps.Id"][event], dtype=np.int64)
        tracks = np.asarray(arrays["Edeps.trackID"][event], dtype=np.int64)
        xs = np.asarray(arrays["Edeps.X"][event], dtype=np.float64)
        ys = np.asarray(arrays["Edeps.Y"][event], dtype=np.float64)
        hits = np.empty((len(expected_layer_z), 2), dtype=np.float32)
        for layer in range(len(expected_layer_z)):
            index = np.flatnonzero((ids == layer) & (tracks == 1))
            if not len(index):
                break
            hits[layer] = (xs[index[0]], ys[index[0]])
        else:
            if np.isfinite(hits).all():
                records.append(hits)
    return records


def command_validate(args):
    expected_z = parse_layer_z(args.expected_layer_z)
    observed_z = validate_root(args.file, args.expected_nbeam, expected_z)
    print(f"{args.file} {args.expected_nbeam} {observed_z.tolist()}")


def command_convert(args):
    expected_z = parse_layer_z(args.expected_layer_z)
    records = []
    for path in root_files(args.glob, args.expected_job_start, args.expected_job_count):
        records.extend(read_muons(path, args.expected_nbeam, expected_z))
    if len(records) < args.target_count:
        raise RuntimeError(f"found {len(records)} muons, require {args.target_count}")
    indices = np.random.default_rng(args.selection_seed).permutation(len(records))[: args.target_count]
    hits = np.stack([records[index] for index in indices])
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        hits=hits,
        nhit=np.full(args.target_count, len(expected_z), dtype=np.int8),
        layer_z=expected_z.astype(np.float32),
    )
    print(f"saved {args.target_count} muons -> {output}")


def parser():
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-root")
    validate.add_argument("--file", required=True)
    validate.add_argument("--expected-nbeam", type=int, required=True)
    validate.add_argument("--expected-layer-z", default="415,215,-215,-415")
    validate.set_defaults(func=command_validate)
    convert = commands.add_parser("convert")
    convert.add_argument("--glob", required=True)
    convert.add_argument("--out", required=True)
    convert.add_argument("--target-count", type=int, default=200000)
    convert.add_argument("--selection-seed", type=int, required=True)
    convert.add_argument("--expected-nbeam", type=int, required=True)
    convert.add_argument("--expected-job-start", type=int, default=0)
    convert.add_argument("--expected-job-count", type=int)
    convert.add_argument("--expected-layer-z", default="415,215,-215,-415")
    convert.set_defaults(func=command_convert)
    return root


def main():
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
