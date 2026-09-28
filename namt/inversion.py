import numpy as np
import torch


def coverage(xy, nq, half=200.0, vox=16.0):
    nx = int(2 * half / vox)
    points = xy.reshape(-1, 2)
    gx = torch.floor((points[:, 0] + half) / vox).long()
    gy = torch.floor((points[:, 1] + half) / vox).long()
    inside = (gx >= 0) & (gx < nx) & (gy >= 0) & (gy < nx)
    counts = torch.zeros(nx, nx, dtype=torch.float64, device=points.device)
    if inside.any():
        counts.index_put_(
            (gy[inside], gx[inside]),
            torch.ones_like(points[inside, 0]),
            accumulate=True,
        )
    return (counts / nq).cpu().numpy()


def grid_axes(half=200.0, vox=16.0):
    nx = int(2 * half / vox)
    return -half + vox * (np.arange(nx) + 0.5)
