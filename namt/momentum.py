from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


MOMENTUM_FORMAT_VERSION = "namt.log-g-quantile.v1"


@dataclass(frozen=True)
class ContinuousMomentumPrior:
    quantile_u: np.ndarray
    quantile_log_g: np.ndarray
    format_version: str

    @classmethod
    def load(cls, path):
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            required = {
                "quantile_u",
                "quantile_log_g",
                "format_version",
            }
            missing = sorted(required - set(data.files))
            if missing:
                raise KeyError(f"{path}: missing {missing}")
            u = np.asarray(data["quantile_u"], dtype=np.float64)
            log_g = np.asarray(data["quantile_log_g"], dtype=np.float64)
            format_version = str(np.asarray(data["format_version"]).item())
        if u.ndim != 1 or log_g.shape != u.shape or len(u) < 4097:
            raise ValueError(f"invalid continuous momentum prior shape in {path}")
        if not np.isclose(u[0], 0.0) or not np.isclose(u[-1], 1.0):
            raise ValueError("momentum prior must cover probabilities [0, 1]")
        if np.any(np.diff(u) <= 0.0) or np.any(np.diff(log_g) < 0.0):
            raise ValueError("momentum prior quantiles must be monotone")
        if not np.isfinite(log_g).all():
            raise ValueError("momentum prior contains invalid values")
        if format_version != MOMENTUM_FORMAT_VERSION:
            raise ValueError(f"unsupported momentum prior {format_version!r}")
        return cls(u, log_g, format_version)

    def quantile(self, probability):
        probability = np.asarray(probability, dtype=np.float64)
        if np.any((probability < 0.0) | (probability > 1.0)):
            raise ValueError("probabilities must lie in [0, 1]")
        return np.exp(np.interp(probability, self.quantile_u, self.quantile_log_g))

    def quadrature(self, order=256):
        order = int(order)
        if order < 8:
            raise ValueError("momentum quadrature order must be at least 8")
        points, weights = np.polynomial.legendre.leggauss(order)
        probability = 0.5 * (points + 1.0)
        weights = 0.5 * weights
        if not np.isclose(weights.sum(), 1.0, rtol=0.0, atol=2e-15):
            raise AssertionError("quadrature weights do not sum to one")
        return self.quantile(probability), weights
