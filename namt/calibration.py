from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from namt.instrument_response import RESPONSE_MODEL_VERSION, StructuredInstrumentResponse


BLANK_CALIBRATION_FORMAT = "namt.structured-instrument-blank.v2"
METHOD_LAYERS = {"namt_3p_structured": 3, "namt_4p_structured": 4}


def _scalar(artifact, key):
    if key not in artifact.files:
        raise ValueError(f"blank calibration is missing {key!r}")
    value = np.asarray(artifact[key])
    if value.size != 1:
        raise ValueError(f"blank calibration field {key!r} is not scalar")
    return value.item()


def save_blank_calibration(path, *, method_name, layer_z, sigma_pos_mm, calibration):
    response = calibration.get("instrument_response")
    if not isinstance(response, StructuredInstrumentResponse):
        raise TypeError("calibration is missing its structured response")
    layer_z = np.asarray(layer_z, dtype=np.float64)
    if method_name not in METHOD_LAYERS or len(layer_z) != METHOD_LAYERS[method_name]:
        raise ValueError("method and calibration geometry differ")
    if not np.array_equal(response.layer_z, layer_z):
        raise ValueError("response and calibration geometry differ")
    if response.sigma_hit_mm != float(sigma_pos_mm):
        raise ValueError("response and calibration resolution differ")
    state = response.state_dict()
    arrays = {
        "format_version": np.asarray(BLANK_CALIBRATION_FORMAT),
        "response_model_version": np.asarray(RESPONSE_MODEL_VERSION),
        "method": np.asarray(method_name),
        "layer_z": layer_z,
        "sigma_pos_mm": np.asarray(sigma_pos_mm, dtype=np.float64),
        "hidden": np.asarray(response.hidden, dtype=np.int64),
        "state_keys": np.asarray(list(state)),
    }
    arrays.update({f"state__{key}": value.detach().cpu().numpy() for key, value in state.items()})
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, destination)


def load_blank_calibration(path, *, method_name, layer_z, dev, sigma_pos_mm):
    layer_z = np.asarray(layer_z, dtype=np.float64)
    if method_name not in METHOD_LAYERS or len(layer_z) != METHOD_LAYERS[method_name]:
        raise ValueError("method and reconstruction geometry differ")
    with np.load(path, allow_pickle=False) as artifact:
        exact = {
            "format_version": BLANK_CALIBRATION_FORMAT,
            "response_model_version": RESPONSE_MODEL_VERSION,
            "method": method_name,
        }
        for key, expected in exact.items():
            if str(_scalar(artifact, key)) != str(expected):
                raise ValueError(f"blank calibration {key} mismatch")
        observed_z = np.asarray(artifact["layer_z"], dtype=np.float64)
        if not np.array_equal(observed_z, layer_z):
            raise ValueError("blank calibration geometry differs")
        if float(_scalar(artifact, "sigma_pos_mm")) != float(sigma_pos_mm):
            raise ValueError("blank calibration resolution differs")
        response = StructuredInstrumentResponse(
            observed_z,
            sigma_pos_mm,
            hidden=int(_scalar(artifact, "hidden")),
            init_seed=0,
            dev=dev,
        )
        state = {}
        for key in np.asarray(artifact["state_keys"]).astype(str).tolist():
            value = np.asarray(artifact[f"state__{key}"])
            if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
                raise ValueError(f"invalid blank calibration state: {key}")
            state[key] = torch.as_tensor(value, dtype=torch.float64, device=dev)
        response.load_state_dict(state, strict=True)
        for parameter in response.parameters():
            parameter.requires_grad_(False)
        return {
            "instrument_response": response,
            "response_model_version": RESPONSE_MODEL_VERSION,
            "sigma_pos_mm": float(sigma_pos_mm),
        }
