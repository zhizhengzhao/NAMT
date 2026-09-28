import numpy as np


RPC_FULL_MM = 30.0


def instrument_from_layers(layer_z):
    layer_z = np.asarray(layer_z, dtype=np.float64)
    if layer_z.ndim != 1 or len(layer_z) not in (3, 4):
        raise ValueError("NAMT requires three or four detector layers")
    if not np.isfinite(layer_z).all() or not np.all(np.diff(layer_z) < 0):
        raise ValueError("layer_z must be finite and strictly descending")
    half_rpc = RPC_FULL_MM / 2.0
    return {
        "layer_z": layer_z,
        "n": len(layer_z),
        "sample": (layer_z[2] + half_rpc, layer_z[1] - half_rpc),
    }
