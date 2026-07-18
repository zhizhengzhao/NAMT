import numpy as np


MATERIALS = ("pb", "explosive", "void")
BACKGROUNDS = ("soil", "concrete")

SCENES = {}
for background in BACKGROUNDS:
    for material in MATERIALS:
        SCENES[f"u_{material}_{background}"] = (
            background,
            [{"mat": material, "kind": "ushape", "x": 0.0, "y": 0.0}],
        )

MAT = {
    "soil": {"inv_x0": 1.0 / 159.6},
    "concrete": {"inv_x0": 1.0 / 115.53},
    "pb": {"inv_x0": 1.0 / 5.6125},
    "explosive": {"inv_x0": 1.0 / 200.73},
    "void": {"inv_x0": 0.0},
}


def _mask_of(obj, gx, gy):
    xx, yy = np.meshgrid(gx, gy)
    x = xx - obj["x"]
    y = yy - obj["y"]
    if obj["kind"] != "ushape":
        raise ValueError(f"unsupported target shape {obj['kind']!r}")
    outer = (np.abs(x) <= 75.0) & (np.abs(y) <= 75.0)
    notch = (np.abs(x) <= 30.0) & (y >= -30.0)
    return outer & ~notch


def object_masks(scene, gx, gy):
    return [(obj["mat"], _mask_of(obj, gx, gy)) for obj in SCENES[scene][1]]


def truth_mask(scene, gx, gy):
    objects = SCENES[scene][1]
    if not objects:
        return None
    mask = np.zeros((len(gy), len(gx)), dtype=bool)
    for _, object_mask in object_masks(scene, gx, gy):
        mask |= object_mask
    return mask
