from __future__ import annotations

from pathlib import Path

import numpy as np

from namt.momentum import MOMENTUM_FORMAT_VERSION

E_S_MEV = 13.6
MUON_MASS_GEV = 0.105658
P_MIN_GEV = 0.2
P_MAX_GEV = 120.0
TABLE_SIZE = 2**16 + 1
GRID_SIZE = 200_001


def gaisser_guan_vertical(energy_gev):
    e = energy_gev
    return 0.14 * (e * (1.0 + 3.64 / e)) ** -2.7 * (
        1.0 / (1.0 + 1.1 * e / 115.0) + 0.054 / (1.0 + 1.1 * e / 850.0)
    )


def momentum_density(p_gev):
    energy = np.hypot(p_gev, MUON_MASS_GEV)
    return gaisser_guan_vertical(energy) * p_gev / energy


def g_of_p(p_gev):
    p_mev = 1000.0 * p_gev
    beta = p_mev / np.hypot(p_mev, 1000.0 * MUON_MASS_GEV)
    return (E_S_MEV / (beta * p_mev)) ** 2


def build():
    log_p = np.linspace(np.log(P_MIN_GEV), np.log(P_MAX_GEV), GRID_SIZE)
    p = np.exp(log_p)

    density = momentum_density(p) * p
    cdf = np.concatenate(([0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(log_p))))
    cdf /= cdf[-1]
    s = np.linspace(0.0, 1.0, TABLE_SIZE)

    log_p_quantile = np.interp(1.0 - s, cdf, log_p)
    log_g = np.log(g_of_p(np.exp(log_p_quantile)))
    if np.any(np.diff(log_g) < 0.0):
        raise AssertionError("log g quantile must be non-decreasing")
    return s, log_g


if __name__ == "__main__":
    u, log_g = build()
    out = Path(__file__).resolve().parents[1] / "assets" / "momentum_prior_gaisser.npz"
    np.savez(
        out,
        quantile_u=u,
        quantile_log_g=log_g,
        format_version=np.asarray(MOMENTUM_FORMAT_VERSION),
    )
    print(f"wrote {out}: log g in [{log_g[0]:.4f}, {log_g[-1]:.4f}]")
