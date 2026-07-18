from __future__ import annotations

import copy
import math

import numpy as np
import torch


LOG_VARIANCE_BOUND = math.log(1.5)
RESPONSE_MODEL_VERSION = "namt.structured-instrument-response.v1"


def split_indices(event_count, split_seed):
    generator = torch.Generator(device="cpu").manual_seed(int(split_seed))
    return torch.randperm(int(event_count), generator=generator)


def analytic_residual_variance(layer_z, sigma_hit_mm):
    z = np.asarray(layer_z, dtype=np.float64)
    if z.ndim != 1 or len(z) not in (3, 4):
        raise ValueError("structured response requires exactly three or four layers")
    if not np.isfinite(z).all() or abs(z[1] - z[0]) < 1e-12:
        raise ValueError("layer_z must be finite with distinct upstream layers")
    sigma = float(sigma_hit_mm)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma_hit_mm must be finite and positive")
    alpha = (z[2:] - z[0]) / (z[1] - z[0])
    per_layer = sigma * sigma * (1.0 + (1.0 - alpha) ** 2 + alpha**2)
    return np.repeat(per_layer[:, None], 2, axis=1)


class StructuredInstrumentResponse(torch.nn.Module):
    def __init__(self, layer_z, sigma_hit_mm, *, hidden=16, init_seed=0, dev="cpu"):
        super().__init__()
        self.layer_z = np.asarray(layer_z, dtype=np.float64)
        self.n_down = len(self.layer_z) - 2
        self.sigma_hit_mm = float(sigma_hit_mm)
        self.hidden = int(hidden)
        self.init_seed = int(init_seed)
        self.dev = str(dev)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.init_seed)
            self.trunk = torch.nn.Sequential(
                torch.nn.Linear(5, self.hidden),
                torch.nn.SiLU(),
                torch.nn.Linear(self.hidden, self.hidden),
                torch.nn.SiLU(),
            )
            self.head = torch.nn.Linear(self.hidden, 4)
        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.zeros_(self.head.bias)
        self.register_buffer("feature_mean", torch.zeros(4, dtype=torch.float64))
        self.register_buffer("feature_std", torch.ones(4, dtype=torch.float64))
        self.register_buffer(
            "analytic_variance",
            torch.as_tensor(
                analytic_residual_variance(self.layer_z, self.sigma_hit_mm),
                dtype=torch.float64,
            ),
        )
        self.to(dtype=torch.float64, device=dev)

    def _raw(self, features, layer_index):
        normalized = (features - self.feature_mean) / self.feature_std
        layer = layer_index.to(dtype=torch.float64).reshape(-1, 1)
        return self.head(self.trunk(torch.cat([normalized, layer], dim=-1)))

    def params(self, features, layer_index):
        if torch.any(layer_index < 0) or torch.any(layer_index >= self.n_down):
            raise ValueError("layer_index is outside the calibrated downstream layers")
        raw = self._raw(features, layer_index)
        mean = self.sigma_hit_mm * torch.tanh(raw[:, :2])
        log_scale = LOG_VARIANCE_BOUND * torch.tanh(raw[:, 2:])
        base = self.analytic_variance[layer_index]
        variance = base * torch.exp(log_scale)
        return mean, variance, log_scale

    def event_nll(self, features, residuals):
        if features.ndim != 2 or features.shape[1] != 4:
            raise ValueError("features must have shape N x 4")
        if residuals.ndim != 3 or residuals.shape[1:] != (self.n_down, 2):
            raise ValueError(f"residuals must have shape N x {self.n_down} x 2")
        n = len(features)
        layers = torch.arange(self.n_down, device=self.dev)[None, :].expand(n, -1)
        repeated = features[:, None, :].expand(-1, self.n_down, -1)
        mean, variance, _ = self.params(repeated.reshape(-1, 4), layers.reshape(-1))
        mean = mean.reshape(n, self.n_down, 2)
        variance = variance.reshape(n, self.n_down, 2)
        return 0.5 * (torch.log(2.0 * np.pi * variance) + (residuals - mean) ** 2 / variance).sum(
            dim=(1, 2)
        )

    def _zero_head(self):
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.bias.zero_()

    def fit(
        self,
        features,
        residuals,
        *,
        steps=3000,
        lr=1e-3,
        event_batch=4096,
        seed=0,
        eval_every=100,
        patience_evals=10,
        verbose=False,
    ):
        features = torch.as_tensor(features, dtype=torch.float64, device=self.dev)
        residuals = torch.as_tensor(residuals, dtype=torch.float64, device=self.dev)
        n = len(features)
        if n < 100:
            raise ValueError("at least 100 blank events are required")
        if residuals.shape != (n, self.n_down, 2):
            raise ValueError("blank residual shape differs from instrument geometry")
        permutation = split_indices(n, seed + 1)
        n_train = int(0.8 * n)
        n_validation = int(0.1 * n)
        train_index = permutation[:n_train].to(self.dev)
        validation_index = permutation[n_train : n_train + n_validation].to(self.dev)
        test_index = permutation[n_train + n_validation :].to(self.dev)
        with torch.no_grad():
            train_features = features[train_index]
            self.feature_mean.copy_(train_features.mean(dim=0))
            self.feature_std.copy_(train_features.std(dim=0, unbiased=False).clamp(min=1e-6))

        def nll_for(index, chunk=4096):
            values = []
            for start in range(0, len(index), chunk):
                selected = index[start : start + chunk]
                values.append(self.event_nll(features[selected], residuals[selected]))
            return torch.cat(values)

        with torch.no_grad():
            baseline_validation = float(nll_for(validation_index).mean().item())
        best_validation = baseline_validation
        best_step = -1
        best_state = copy.deepcopy(self.state_dict())
        optimizer = torch.optim.Adam(self.parameters(), lr=float(lr))
        train_generator = torch.Generator(device="cpu").manual_seed(seed)
        stale_evals = 0
        for step in range(int(steps)):
            sampled = torch.randint(
                0,
                len(train_index),
                (min(int(event_batch), len(train_index)),),
                generator=train_generator,
            ).to(self.dev)
            selected = train_index[sampled]
            loss = self.event_nll(features[selected], residuals[selected]).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % eval_every == 0 or step == steps - 1:
                with torch.no_grad():
                    validation = float(nll_for(validation_index).mean().item())
                if validation < best_validation:
                    best_validation = validation
                    best_step = step
                    best_state = copy.deepcopy(self.state_dict())
                    stale_evals = 0
                else:
                    stale_evals += 1
                if stale_evals >= int(patience_evals):
                    break
            if verbose and (step % 500 == 0 or step == steps - 1):
                print(f"    instrument step {step}: train_nll={loss.item():.6g}")
        self.load_state_dict(best_state, strict=True)
        with torch.no_grad():
            learned_test = nll_for(test_index)
            self._zero_head()
            analytic_test = nll_for(test_index)
            self.load_state_dict(best_state, strict=True)
            paired_improvement = analytic_test - learned_test
            improvement_mean = float(paired_improvement.mean().item())
            improvement_se = float(
                paired_improvement.std(unbiased=True).item() / math.sqrt(len(paired_improvement))
            )
            gate_passed = bool(improvement_mean > 2.0 * improvement_se and best_step >= 0)
            if not gate_passed:
                self._zero_head()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        return self

    @torch.no_grad()
    def evaluate_all_layers(self, features):
        features = torch.as_tensor(features, dtype=torch.float64, device=self.dev)
        n = len(features)
        layers = torch.arange(self.n_down, device=self.dev)[None, :].expand(n, -1)
        repeated = features[:, None, :].expand(-1, self.n_down, -1)
        mean, variance, log_scale = self.params(repeated.reshape(-1, 4), layers.reshape(-1))
        shape = (n, self.n_down, 2)
        return mean.reshape(shape), variance.reshape(shape), log_scale.reshape(shape)
