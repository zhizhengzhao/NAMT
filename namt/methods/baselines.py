import numpy as np
import torch

from namt.inversion import grid_axes
from namt.methods.base import Method


HALF = 200.0


def _grid_size(vox):
    vox = float(vox)
    n = int(round(2.0 * HALF / vox)) if vox > 0 else 0
    if n < 1 or not np.isclose(n * vox, 2.0 * HALF):
        raise ValueError("voxel size must divide the reconstruction width")
    return n


def _track_parameters(h, lz):
    h = np.asarray(h, dtype=np.float64)
    lz = np.asarray(lz, dtype=np.float64)
    if h.ndim != 3 or h.shape[1:] != (4, 2) or lz.shape != (4,):
        raise ValueError("baseline methods require four detector layers")
    if not np.isfinite(h).all() or not np.isfinite(lz).all():
        raise ValueError("detector data must be finite")
    if lz[1] == lz[0] or lz[3] == lz[2]:
        raise ValueError("detector layer spacing must be nonzero")
    t_up = (h[:, 1] - h[:, 0]) / (lz[1] - lz[0])
    t_dn = (h[:, 3] - h[:, 2]) / (lz[3] - lz[2])
    p_up = np.column_stack((h[:, 1], np.full(len(h), lz[1])))
    p_dn = np.column_stack((h[:, 2], np.full(len(h), lz[2])))
    d_up = np.column_stack((t_up, np.ones(len(h))))
    d_dn = np.column_stack((t_dn, np.ones(len(h))))
    return t_up, t_dn, p_up, p_dn, d_up, d_dn


def _poca_points(p_up, p_dn, d_up, d_dn):
    w = p_up - p_dn
    a = np.einsum("ni,ni->n", d_up, d_up)
    b = np.einsum("ni,ni->n", d_up, d_dn)
    c = np.einsum("ni,ni->n", d_dn, d_dn)
    d = np.einsum("ni,ni->n", d_up, w)
    e = np.einsum("ni,ni->n", d_dn, w)
    denominator = a * c - b * b
    valid = np.abs(denominator) > 1e-12
    s = np.zeros(len(w))
    t = np.zeros(len(w))
    s[valid] = (b[valid] * e[valid] - c[valid] * d[valid]) / denominator[valid]
    t[valid] = (a[valid] * e[valid] - b[valid] * d[valid]) / denominator[valid]
    c_up = p_up + s[:, None] * d_up
    c_dn = p_dn + t[:, None] * d_dn
    points = 0.5 * (c_up + c_dn)
    return points, valid


def _segment_voxels(start, stop, n, vox):
    delta = stop - start
    distance = float(np.linalg.norm(delta))
    if distance <= 1e-12:
        return np.empty(0, np.int64), np.empty(0, np.float64)
    values = [np.array((0.0, 1.0))]
    boundaries = np.linspace(-HALF, HALF, n + 1)
    for axis in range(3):
        if abs(delta[axis]) > 1e-12:
            t = (boundaries - start[axis]) / delta[axis]
            values.append(t[(t > 0.0) & (t < 1.0)])
    t = np.unique(np.concatenate(values))
    mids = start[None, :] + 0.5 * (t[:-1] + t[1:])[:, None] * delta[None, :]
    ijk = np.floor((mids + HALF) / vox).astype(np.int64)
    inside = np.all((ijk >= 0) & (ijk < n), axis=1)
    ijk = ijk[inside]
    lengths = np.diff(t)[inside] * distance
    ids = ijk[:, 2] * n * n + ijk[:, 1] * n + ijk[:, 0]
    if len(ids) < 2:
        return ids, lengths
    starts = np.r_[True, ids[1:] != ids[:-1]]
    groups = np.cumsum(starts) - 1
    return ids[starts], np.bincount(groups, weights=lengths)


def _ray_voxels(entry, poca, exit, n, vox):
    ids_a, lengths_a = _segment_voxels(entry, poca, n, vox)
    ids_b, lengths_b = _segment_voxels(poca, exit, n, vox)
    ids = np.concatenate((ids_a, ids_b))
    lengths = np.concatenate((lengths_a, lengths_b))
    if len(ids_a) and len(ids_b) and ids[len(ids_a) - 1] == ids[len(ids_a)]:
        lengths[len(ids_a) - 1] += lengths[len(ids_a)]
        ids = np.delete(ids, len(ids_a))
        lengths = np.delete(lengths, len(ids_a))
    return ids, lengths


def _geometry(h, lz, vox):
    n = _grid_size(vox)
    t_up, t_dn, p_up, p_dn, d_up, d_dn = _track_parameters(h, lz)
    poca, poca_valid = _poca_points(p_up, p_dn, d_up, d_dn)
    entry = np.column_stack((h[:, 1] + t_up * (HALF - lz[1]), np.full(len(h), HALF)))
    exit = np.column_stack((h[:, 2] + t_dn * (-HALF - lz[2]), np.full(len(h), -HALF)))
    chord_valid = np.all(np.isfinite(entry), axis=1) & np.all(np.isfinite(exit), axis=1)
    chord_valid &= np.all(np.abs(entry[:, :2]) < HALF, axis=1)
    chord_valid &= np.all(np.abs(exit[:, :2]) < HALF, axis=1)
    poca_valid &= np.all(np.abs(poca) < HALF, axis=1)
    return n, t_up, t_dn, poca, entry, exit, chord_valid, poca_valid


def _project(volume):
    finite = np.isfinite(volume)
    count = finite.sum(axis=0)
    total = np.where(finite, volume, 0.0).sum(axis=0)
    return np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)


def poca_img(h, lz, vox, min_cov):
    n, t_up, t_dn, poca, entry, exit, chord_valid, poca_valid = _geometry(h, lz, vox)
    dtheta = np.arctan(t_dn) - np.arctan(t_up)
    path_length = np.linalg.norm(exit - entry, axis=1)
    signal = np.divide(
        0.5 * np.einsum("ni,ni->n", dtheta, dtheta),
        path_length,
        out=np.zeros(len(h), dtype=np.float64),
        where=path_length > 1e-12,
    )
    numerator = np.zeros(n**3, dtype=np.float64)
    coverage = np.zeros(n**3, dtype=np.int64)
    for i in np.flatnonzero(chord_valid):
        ids, _ = _segment_voxels(entry[i], exit[i], n, vox)
        if len(ids):
            ids = np.unique(ids)
            np.add.at(coverage, ids, 1)
        if not poca_valid[i]:
            continue
        ijk = np.floor((poca[i] + HALF) / vox).astype(np.int64)
        voxel = ijk[2] * n * n + ijk[1] * n + ijk[0]
        if not np.any(ids == voxel):
            coverage[voxel] += 1
        numerator[voxel] += signal[i]
    volume = np.divide(
        numerator,
        coverage,
        out=np.full(n**3, np.nan),
        where=coverage >= min_cov,
    ).reshape(n, n, n)
    return _project(volume), n


def _order_statistic(ids, values, size, quantile, min_count):
    image = np.full(size, np.nan)
    if not len(ids):
        return image
    order = np.argsort(ids, kind="stable")
    ids = ids[order]
    values = values[order]
    unique, starts, counts = np.unique(ids, return_index=True, return_counts=True)
    for voxel, start, count in zip(unique, starts, counts):
        if count < min_count:
            continue
        k = max(int(np.floor(quantile * count)) - 1, 0)
        image[voxel] = np.partition(values[start : start + count], k)[k]
    return image


def asr_img(h, lz, vox, min_cov):
    n = _grid_size(vox)
    t_up, t_dn, _, _, _, _ = _track_parameters(h, lz)
    dtheta = np.abs(np.arctan(t_dn) - np.arctan(t_up))
    sec_up = np.sqrt(1.0 + np.einsum("ni,ni->n", t_up, t_up))
    sec_dn = np.sqrt(1.0 + np.einsum("ni,ni->n", t_dn, t_dn))
    reach = int(np.ceil(0.5 * (sec_up.max() + sec_dn.max()))) + 1
    axis = np.arange(-reach, reach + 1, dtype=np.int64)
    offset_x, offset_y = np.meshgrid(axis, axis, indexing="xy")
    offset_x = offset_x.ravel()
    offset_y = offset_y.ravel()
    z_centers = -HALF + vox * (np.arange(n) + 0.5)
    projection_sum = np.zeros((n, n), dtype=np.float64)
    projection_count = np.zeros((n, n), dtype=np.int64)
    event_batch = 20000
    for z in z_centers:
        ids_parts = []
        event_parts = []
        for start in range(0, len(h), event_batch):
            stop = min(start + event_batch, len(h))
            q_up = h[start:stop, 1] + t_up[start:stop] * (z - lz[1])
            q_dn = h[start:stop, 2] + t_dn[start:stop] * (z - lz[2])
            middle = 0.5 * (q_up + q_dn)
            base_x = np.floor((middle[:, 0] + HALF) / vox).astype(np.int64)
            base_y = np.floor((middle[:, 1] + HALF) / vox).astype(np.int64)
            ix = base_x[:, None] + offset_x[None, :]
            iy = base_y[:, None] + offset_y[None, :]
            x = -HALF + vox * (ix + 0.5)
            y = -HALF + vox * (iy + 0.5)
            dx_up = x - q_up[:, 0, None]
            dy_up = y - q_up[:, 1, None]
            dx_dn = x - q_dn[:, 0, None]
            dy_dn = y - q_dn[:, 1, None]
            dot_up = dx_up * t_up[start:stop, 0, None] + dy_up * t_up[start:stop, 1, None]
            dot_dn = dx_dn * t_dn[start:stop, 0, None] + dy_dn * t_dn[start:stop, 1, None]
            distance_up = dx_up * dx_up + dy_up * dy_up - dot_up * dot_up / sec_up[start:stop, None] ** 2
            distance_dn = dx_dn * dx_dn + dy_dn * dy_dn - dot_dn * dot_dn / sec_dn[start:stop, None] ** 2
            selected = (
                (ix >= 0)
                & (ix < n)
                & (iy >= 0)
                & (iy < n)
                & (distance_up < vox * vox)
                & (distance_dn < vox * vox)
            )
            row, column = np.nonzero(selected)
            if len(row):
                ids_parts.append(iy[row, column] * n + ix[row, column])
                event_parts.append(start + row)
        if not ids_parts:
            continue
        ids = np.concatenate(ids_parts)
        events = np.concatenate(event_parts)
        ids = np.concatenate((ids, ids))
        values = np.concatenate((dtheta[events, 0], dtheta[events, 1]))
        layer = _order_statistic(ids, values, n * n, 0.75, 2 * min_cov)
        layer = layer.reshape(n, n)
        finite = np.isfinite(layer)
        projection_sum[finite] += layer[finite]
        projection_count[finite] += 1
    image = np.divide(
        projection_sum,
        projection_count,
        out=np.full((n, n), np.nan),
        where=projection_count > 0,
    )
    return image, n


def _mlsd_measurements(h, lz, t_up, t_dn, sigma_pos):
    theta_up = np.arctan(t_up)
    theta_dn = np.arctan(t_dn)
    dtheta = theta_dn - theta_up
    entry_projection = h[:, 1] + t_up * (-HALF - lz[1])
    exit_position = h[:, 2] + t_dn * (-HALF - lz[2])
    raw_displacement = exit_position - entry_projection
    sec = np.sqrt(1.0 + np.einsum("ni,ni->n", t_up, t_up))
    factor = np.cos(theta_up) * sec[:, None] * np.cos(theta_dn) / np.cos(dtheta)
    displacement = raw_displacement * factor
    du = lz[1] - lz[0]
    dd = lz[3] - lz[2]
    au = (-HALF - lz[1]) / du
    ad = (-HALF - lz[2]) / dd
    displacement_coefficients = np.array((au, -(1.0 + au), 1.0 - ad, ad))
    covariance = []
    for coordinate in range(2):
        inv_up = 1.0 / (1.0 + t_up[:, coordinate] ** 2)
        inv_dn = 1.0 / (1.0 + t_dn[:, coordinate] ** 2)
        angle_coefficients = np.column_stack(
            (
                inv_up / du,
                -inv_up / du,
                -inv_dn / dd,
                inv_dn / dd,
            )
        )
        displacement_coefficients_i = factor[:, coordinate, None] * displacement_coefficients[None, :]
        e00 = sigma_pos**2 * np.einsum("ni,ni->n", angle_coefficients, angle_coefficients)
        e01 = sigma_pos**2 * np.einsum(
            "ni,ni->n", angle_coefficients, displacement_coefficients_i
        )
        e11 = sigma_pos**2 * np.einsum(
            "ni,ni->n", displacement_coefficients_i, displacement_coefficients_i
        )
        covariance.append(np.column_stack((e00, e01, e11)))
    return dtheta, displacement, covariance


def mlsd_median_img(h, lz, vox, sigma_pos, iters, dev, min_cov):
    n, t_up, t_dn, poca, entry, exit, chord_valid, poca_valid = _geometry(h, lz, vox)
    selected = np.flatnonzero(chord_valid)
    ray_parts = []
    voxel_parts = []
    support_parts = []
    length_parts = []
    tail_parts = []
    kept = []
    for ray, event in enumerate(selected):
        if poca_valid[event]:
            ids, lengths = _ray_voxels(entry[event], poca[event], exit[event], n, vox)
        else:
            ids, lengths = _segment_voxels(entry[event], exit[event], n, vox)
        if not len(ids):
            continue
        ray_parts.append(np.full(len(ids), len(kept), np.int64))
        voxel_parts.append(ids)
        support_parts.append(np.unique(ids))
        length_parts.append(lengths)
        tail_parts.append(lengths.sum() - np.cumsum(lengths))
        kept.append(event)
    if not kept:
        return np.full((n, n), np.nan), n
    kept = np.asarray(kept, dtype=np.int64)
    rays = np.concatenate(ray_parts)
    voxels = np.concatenate(voxel_parts)
    lengths = np.concatenate(length_parts)
    tails = np.concatenate(tail_parts)
    w00 = lengths
    w01 = lengths * lengths / 2.0 + lengths * tails
    w11 = lengths**3 / 3.0 + lengths * lengths * tails + lengths * tails * tails
    dtheta, displacement, covariance = _mlsd_measurements(
        h[kept], lz, t_up[kept], t_dn[kept], sigma_pos
    )
    device = torch.device(dev if str(dev).startswith("cuda") and torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    ray_t = torch.as_tensor(rays, dtype=torch.long, device=device)
    voxel_t = torch.as_tensor(voxels, dtype=torch.long, device=device)
    w00_t = torch.as_tensor(w00, dtype=dtype, device=device)
    w01_t = torch.as_tensor(w01, dtype=dtype, device=device)
    w11_t = torch.as_tensor(w11, dtype=dtype, device=device)
    dtheta_t = torch.as_tensor(dtheta, dtype=dtype, device=device)
    displacement_t = torch.as_tensor(displacement, dtype=dtype, device=device)
    covariance_t = [torch.as_tensor(value, dtype=dtype, device=device) for value in covariance]
    n_rays = len(kept)
    n_voxels = n**3
    support = np.bincount(np.concatenate(support_parts), minlength=n_voxels)
    voxel_order = torch.argsort(voxel_t)
    sorted_voxels = voxel_t[voxel_order]
    visited, group_counts = torch.unique_consecutive(sorted_voxels, return_counts=True)
    group_starts = torch.cumsum(group_counts, dim=0) - group_counts
    median_rows = torch.repeat_interleave(
        torch.arange(len(visited), device=device), group_counts
    )
    median_columns = torch.arange(len(voxel_t), device=device) - torch.repeat_interleave(
        group_starts, group_counts
    )
    max_group_count = int(group_counts.max().item())
    lam = torch.full((n_voxels,), 8.225829961565633e-11, dtype=dtype, device=device)
    for _ in range(int(iters)):
        contribution = lam[voxel_t]
        s00 = torch.zeros(n_rays, dtype=dtype, device=device).index_add_(0, ray_t, contribution * w00_t)
        s01 = torch.zeros(n_rays, dtype=dtype, device=device).index_add_(0, ray_t, contribution * w01_t)
        s11 = torch.zeros(n_rays, dtype=dtype, device=device).index_add_(0, ray_t, contribution * w11_t)
        correction = 0.0
        for coordinate in range(2):
            e00, e01, e11 = covariance_t[coordinate].unbind(1)
            a = s00 + e00
            b = s01 + e01
            c = s11 + e11
            determinant = torch.clamp(a * c - b * b, min=1e-24)
            inv00 = c / determinant
            inv01 = -b / determinant
            inv11 = a / determinant
            angle = dtheta_t[:, coordinate]
            shift = displacement_t[:, coordinate]
            v0 = inv00 * angle + inv01 * shift
            v1 = inv01 * angle + inv11 * shift
            pair_quad = v0[ray_t] ** 2 * w00_t + 2.0 * v0[ray_t] * v1[ray_t] * w01_t + v1[ray_t] ** 2 * w11_t
            pair_trace = inv00[ray_t] * w00_t + 2.0 * inv01[ray_t] * w01_t + inv11[ray_t] * w11_t
            correction = correction + 0.5 * (pair_quad - pair_trace)
        hidden_expectation = 2.0 * contribution + contribution * contribution * correction
        grouped = torch.full(
            (len(visited), max_group_count),
            torch.inf,
            dtype=dtype,
            device=device,
        )
        grouped[median_rows, median_columns] = hidden_expectation[voxel_order]
        ordered = torch.sort(grouped, dim=1).values
        lower = torch.div(group_counts - 1, 2, rounding_mode="floor")
        upper = torch.div(group_counts, 2, rounding_mode="floor")
        rows = torch.arange(len(visited), device=device)
        medians = 0.5 * (ordered[rows, lower] + ordered[rows, upper])
        lam = lam.clone()
        lam[visited] = torch.clamp(0.5 * medians, min=1e-12)
    volume = lam.detach().cpu().numpy().reshape(n, n, n)
    volume[support.reshape(n, n, n) < min_cov] = np.nan
    return _project(volume), n


class _Family(Method):
    tier = "A"

    def __init__(self, dev="cpu"):
        self.dev = dev

    def calibrate(self, blank_hits, layer_z, sigma_pos):
        return {}


class PoCA(_Family):
    name = "poca"

    def reconstruct(self, hits, layer_z, sigma_pos, cal, vox, min_cov=10, **kw):
        image, _ = poca_img(hits, layer_z, vox, min_cov)
        axis = grid_axes(HALF, vox)
        return {"img": image, "gx": axis, "gy": axis}


class ASRQ75(_Family):
    name = "asr_q75"

    def reconstruct(self, hits, layer_z, sigma_pos, cal, vox, min_cov=10, **kw):
        image, _ = asr_img(hits, layer_z, vox, min_cov)
        axis = grid_axes(HALF, vox)
        return {"img": image, "gx": axis, "gy": axis}


class MLSDMedian(_Family):
    name = "mlsd_median"

    def reconstruct(self, hits, layer_z, sigma_pos, cal, vox, iters=100, min_cov=10, **kw):
        image, _ = mlsd_median_img(
            hits,
            layer_z,
            vox,
            sigma_pos=sigma_pos,
            iters=iters,
            dev=self.dev,
            min_cov=min_cov,
        )
        axis = grid_axes(HALF, vox)
        return {"img": image, "gx": axis, "gy": axis}
