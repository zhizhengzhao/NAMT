from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from namt.contrast import ScatterMix
from namt.field import HALF
from namt.instrument_response import RESPONSE_MODEL_VERSION, StructuredInstrumentResponse
from namt.inversion import coverage, grid_axes
from namt.methods.base import Method


def elu1(u):
    return 1.0 + F.elu(u)


LAMBDA0 = 1.0 / 100.0


class NAMT4pMLP(Method):
    name = "namt_4p_structured"
    tier = "A"
    RESPONSE_CONFIG = {
        "response_hidden": 16,
        "response_steps": 3000,
        "response_lr": 1e-3,
        "response_event_batch": 4096,
        "response_seed": 0,
        "response_eval_every": 100,
        "response_patience_evals": 10,
    }
    TV_XY = 18.0
    TV_Z = 18.0

    def __init__(
        self,
        dev="cuda:0",
        momentum_prior_path=None,
        momentum_quad_order=256,
        tv_xy=TV_XY,
        tv_z=TV_Z,
        response_steps=None,
    ):
        self.dev = str(dev)
        self.momentum_prior_path = momentum_prior_path
        self.momentum_quad_order = int(momentum_quad_order)
        self.tv_xy = float(tv_xy)
        self.tv_z = float(tv_z)
        self.response_config = dict(self.RESPONSE_CONFIG)
        if response_steps is not None:
            self.response_config["response_steps"] = int(response_steps)

    def scatter_kwargs(self):
        return {
            "momentum_prior_path": self.momentum_prior_path,
            "momentum_quad_order": self.momentum_quad_order,
            **self.response_config,
        }

    def calibrate(self, blank_hits, layer_z, sigma_pos):
        torch.manual_seed(0)
        kwargs = self.scatter_kwargs()
        kwargs["momentum_prior_path"] = None
        model = ScatterMix(
            layer_z,
            sigma_pos=sigma_pos,
            lambda0=LAMBDA0,
            dev=self.dev,
            **kwargs,
        )
        response = model.calibrate(blank_hits, max_n=250000)
        return {
            "instrument_response": response,
            "response_model_version": RESPONSE_MODEL_VERSION,
            "sigma_pos_mm": float(sigma_pos),
        }

    def reconstruct(
        self,
        hits,
        layer_z,
        sigma_pos,
        cal,
        vox,
        scene=None,
        steps=3000,
        lr=0.03,
        tv="l1",
        delta=0.1,
        min_cov=10,
        **kw,
    ):
        model = ScatterMix(
            layer_z,
            sigma_pos=sigma_pos,
            lambda0=LAMBDA0,
            vox=vox,
            dev=self.dev,
            **self.scatter_kwargs(),
        )
        expected = {
            "response_model_version": RESPONSE_MODEL_VERSION,
            "sigma_pos_mm": float(sigma_pos),
        }
        for key, value in expected.items():
            if cal.get(key) != value:
                raise RuntimeError(f"blank response and reconstruction {key} differ")
        model.instrument_response = cal.get("instrument_response")
        if model.instrument_response is None:
            raise RuntimeError("blank calibration is missing its structured response")
        response = model.instrument_response
        if (
            not isinstance(response, StructuredInstrumentResponse)
            or response.n_down != model.n_down
            or response.sigma_hit_mm != model.sigma_pos
            or not np.array_equal(response.layer_z, model.lz)
        ):
            raise RuntimeError("blank response geometry differs from reconstruction geometry")
        prepared = model.prepare(hits)
        n = prepared["n"]
        chord = prepared["xy"]
        nq = chord.shape[1]
        nx = int(2 * HALF / vox)
        gxc = (chord[:, :, 0] + HALF) / vox - 0.5
        gyc = (chord[:, :, 1] + HALF) / vox - 0.5
        ix0 = torch.floor(gxc).long().clamp(0, nx - 2)
        iy0 = torch.floor(gyc).long().clamp(0, nx - 2)
        fx = (gxc - ix0).clamp(0, 1)
        fy = (gyc - iy0).clamp(0, 1)
        inside = ((chord[:, :, 0].abs() <= HALF) & (chord[:, :, 1].abs() <= HALF)).to(torch.float64)
        qidx = torch.arange(nq, device=self.dev).view(1, nq).expand(n, nq)

        def flat(iy, ix):
            return (qidx * nx + iy) * nx + ix

        idx4 = torch.stack(
            (flat(iy0, ix0), flat(iy0, ix0 + 1), flat(iy0 + 1, ix0), flat(iy0 + 1, ix0 + 1)),
            dim=-1,
        )
        w4 = torch.stack(((1 - fy) * (1 - fx), (1 - fy) * fx, fy * (1 - fx), fy * fx), dim=-1)

        def tv_penalty(value):
            if tv == "l1":
                return value.abs().mean()
            absolute = value.abs()
            return torch.where(
                absolute < delta,
                0.5 * value.square() / delta,
                absolute - 0.5 * delta,
            ).mean()

        u = torch.zeros(nq, nx, nx, dtype=torch.float64, device=self.dev, requires_grad=True)
        optimizer = torch.optim.Adam([u], lr=lr)
        for iteration in range(int(steps)):
            optimizer.zero_grad()
            lambda_field = model.lambda0 * elu1(u)
            lambda_iq = (lambda_field.reshape(-1)[idx4] * w4).sum(dim=-1) * inside
            nll = model.nll_mix(prepared, lam=lambda_iq) / n
            loss = nll + self.tv_xy * (
                tv_penalty(lambda_field[:, :, 1:] - lambda_field[:, :, :-1])
                + tv_penalty(lambda_field[:, 1:, :] - lambda_field[:, :-1, :])
            )
            if nq > 1:
                loss = loss + self.tv_z * tv_penalty(lambda_field[1:] - lambda_field[:-1])
            loss.backward()
            torch.nn.utils.clip_grad_norm_([u], 1.0)
            optimizer.step()
        with torch.no_grad():
            lambda_field = model.lambda0 * elu1(u)
            image = torch.einsum("q,qyx->yx", model.wq, lambda_field).cpu().numpy()
        cov = coverage(prepared["xy"], nq, HALF, vox)
        axis = grid_axes(HALF, vox)
        return {
            "img": np.where(cov >= min_cov, image, np.nan),
            "gx": axis,
            "gy": axis,
        }


class NAMT3pMLP(NAMT4pMLP):
    name = "namt_3p_structured"
    tier = "A"

    def calibrate(self, blank_hits, layer_z, sigma_pos):
        return super().calibrate(blank_hits[:, :3], layer_z[:3], sigma_pos)

    def reconstruct(self, hits, layer_z, sigma_pos, cal, vox, scene=None, **kw):
        return super().reconstruct(hits[:, :3], layer_z[:3], sigma_pos, cal, vox, scene=scene, **kw)
