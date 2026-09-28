from __future__ import annotations

import numpy as np
import torch

from namt.instrument_response import StructuredInstrumentResponse
from namt.momentum import (
    MOMENTUM_FORMAT_VERSION,
    ContinuousMomentumPrior,
)
from namt.physics import instrument_from_layers


DEV = "cuda:0" if torch.cuda.is_available() else "cpu"


class ScatterMix:
    def __init__(
        self,
        layer_z,
        sigma_pos,
        lambda0=1.0 / 100.0,
        vox=6.0,
        dev=DEV,
        *,
        response_hidden=16,
        response_steps=3000,
        response_lr=1e-3,
        response_event_batch=4096,
        response_seed=0,
        response_eval_every=100,
        response_patience_evals=10,
        momentum_prior_path=None,
        momentum_quad_order=256,
    ):
        self.lambda0 = float(lambda0)
        if not np.isfinite(self.lambda0) or self.lambda0 <= 0.0:
            raise ValueError("lambda0 must be finite and positive")
        self.dev = str(dev)
        self.ins = instrument_from_layers(layer_z)
        self.lz = np.asarray(layer_z, dtype=np.float64)
        self.k = len(self.lz)
        if self.k not in (3, 4):
            raise ValueError("NAMT requires exactly three or four RPC layers")
        self.n_down = self.k - 2
        bot, top = self.ins["sample"]
        ratio = (top - bot) / vox
        n_interval = max(round(ratio), 8)
        nq = n_interval + 1 if np.isclose(ratio, n_interval, rtol=0.0, atol=1e-9) else n_interval
        zq = np.linspace(bot, top, nq)
        wq = np.full(nq, (top - bot) / (nq - 1))
        wq[[0, -1]] *= 0.5
        self.wq = torch.as_tensor(wq, dtype=torch.float64, device=self.dev)
        self.zq = zq
        down_z = self.lz[2:]
        self.lever2 = torch.as_tensor(
            (down_z[:, None] - zq[None, :]) ** 2,
            dtype=torch.float64,
            device=self.dev,
        )
        self.sigma_pos = float(sigma_pos)

        self.momentum_prior_kind = None
        self.momentum_prior_format_version = None
        self.momentum_quad_order = None
        self.g_a = None
        self.log_w_a = None
        if momentum_prior_path is not None:
            prior = ContinuousMomentumPrior.load(momentum_prior_path)
            if prior.format_version != MOMENTUM_FORMAT_VERSION:
                raise ValueError(f"unsupported momentum prior {prior.format_version!r}")
            g_nodes, quad_weights = prior.quadrature(momentum_quad_order)
            self.momentum_prior_kind = "continuous_log_g_quantile_gauss_legendre"
            self.momentum_prior_format_version = prior.format_version
            self.momentum_quad_order = len(g_nodes)
            self.g_a = torch.as_tensor(g_nodes, dtype=torch.float64, device=self.dev)
            self.log_w_a = torch.log(
                torch.as_tensor(quad_weights, dtype=torch.float64, device=self.dev)
            )
        self.response_config = {
            "hidden": int(response_hidden),
            "max_steps": int(response_steps),
            "learning_rate": float(response_lr),
            "event_batch": int(response_event_batch),
            "training_seed": int(response_seed),
            "eval_every": int(response_eval_every),
            "patience_evals": int(response_patience_evals),
        }
        self.instrument_response = None

    def prepare(self, hits, max_n=None, seed=1):
        h_np = np.asarray(hits, dtype=np.float64)
        if h_np.ndim != 3 or h_np.shape[1:] != (self.k, 2):
            raise ValueError(f"hits must have shape N x {self.k} x 2")
        if max_n and max_n < h_np.shape[0]:
            rng = np.random.default_rng(seed)
            h_np = h_np[rng.choice(h_np.shape[0], max_n, replace=False)]
        h = torch.as_tensor(h_np, dtype=torch.float64, device=self.dev)
        z1, z2 = self.lz[:2]
        x1, y1 = h[:, 0, 0], h[:, 0, 1]
        x2, y2 = h[:, 1, 0], h[:, 1, 1]
        tx = (x2 - x1) / (z2 - z1)
        ty = (y2 - y1) / (z2 - z1)
        x0 = x1 - tx * z1
        y0 = y1 - ty * z1
        features = torch.stack((x0, y0, tx, ty), dim=1)
        down_z = torch.as_tensor(self.lz[2:], dtype=torch.float64, device=self.dev)
        pred_x = x1[:, None] + tx[:, None] * (down_z[None, :] - z1)
        pred_y = y1[:, None] + ty[:, None] * (down_z[None, :] - z1)
        residuals = torch.stack((h[:, 2:, 0] - pred_x, h[:, 2:, 1] - pred_y), dim=2)
        sec = torch.sqrt(1.0 + tx.square() + ty.square())
        zq_t = torch.as_tensor(self.zq, dtype=torch.float64, device=self.dev)
        xy = torch.stack(
            (
                x1[:, None] + tx[:, None] * (zq_t[None, :] - z1),
                y1[:, None] + ty[:, None] * (zq_t[None, :] - z1),
            ),
            dim=2,
        )
        prepared = {
            "features": features,
            "residuals": residuals,
            "sec": sec,
            "xy": xy,
            "n": h.shape[0],
        }
        if self.instrument_response is not None:
            mean, variance, _ = self.instrument_response.evaluate_all_layers(features)
            prepared["instrument_mean"] = mean
            prepared["instrument_variance"] = variance
        return prepared

    def _Q(self, prep, lam=None):
        if lam is None:
            lam = self.lambda0 * torch.ones(
                prep["n"], self.wq.shape[0], dtype=torch.float64, device=self.dev
            )
        return torch.einsum("jq,nq->nj", self.lever2, lam * self.wq[None, :]) * prep["sec"][
            :, None
        ].pow(3)

    def nll_mix(self, prep, lam=None):
        if self.g_a is None or self.log_w_a is None:
            raise RuntimeError("momentum prior has not been loaded")
        if "instrument_mean" not in prep or "instrument_variance" not in prep:
            raise RuntimeError("instrument response has not been calibrated")
        q_field = self._Q(prep, lam)
        residual = prep["residuals"]
        mean = prep["instrument_mean"]
        instrument_variance = prep["instrument_variance"]
        expected_shape = (prep["n"], self.n_down, 2)
        if (
            residual.shape != expected_shape
            or mean.shape != expected_shape
            or instrument_variance.shape != expected_shape
        ):
            raise RuntimeError("residual and instrument-response geometries differ")
        if q_field.shape != (prep["n"], self.n_down):
            raise RuntimeError("target-scatter geometry differs from downstream residuals")
        variance = torch.clamp(
            instrument_variance[..., None]
            + q_field[:, :, None, None] * self.g_a[None, None, None, :],
            min=1e-300,
        )
        centered = residual[..., None] - mean[..., None]
        log_coordinate = -0.5 * (torch.log(2.0 * np.pi * variance) + centered.square() / variance)
        log_g = log_coordinate.sum(dim=(1, 2)) + self.log_w_a[None, :]
        return -torch.logsumexp(log_g, dim=1).sum()

    def calibrate(self, blank_hits, max_n=250000, seed=1, verbose=False):
        prepared = self.prepare(blank_hits, max_n=max_n, seed=seed)
        self.instrument_response = StructuredInstrumentResponse(
            self.lz,
            self.sigma_pos,
            hidden=self.response_config["hidden"],
            init_seed=self.response_config["training_seed"],
            dev=self.dev,
        ).fit(
            prepared["features"],
            prepared["residuals"],
            steps=self.response_config["max_steps"],
            lr=self.response_config["learning_rate"],
            event_batch=self.response_config["event_batch"],
            seed=self.response_config["training_seed"],
            eval_every=self.response_config["eval_every"],
            patience_evals=self.response_config["patience_evals"],
            verbose=verbose,
        )
        return self.instrument_response
