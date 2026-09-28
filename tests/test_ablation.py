from pathlib import Path
import unittest
import numpy as np
import torch
from ablation.contrast import ScatterMix
from ablation.instrument_response import analytic_residual_variance
from namt.momentum import ContinuousMomentumPrior

ROOT = Path(__file__).resolve().parents[1]
LAYER_Z = np.asarray([415.0, 215.0, -215.0, -415.0])

def straight_hits(count=3, tx=0.08, ty=-0.04):
    intercept = np.asarray([12.0, -7.0])
    slope = np.asarray([tx, ty])
    one = intercept[None, :] + LAYER_Z[:, None] * slope[None, :]
    return np.repeat(one[None, :, :], count, axis=0)

class MomentumAblationTest(unittest.TestCase):

    def test_fixed_node_is_quadrature_mean_scattering_scale(self):
        prior_path = ROOT / 'assets' / 'momentum_prior.npz'
        prior = ContinuousMomentumPrior.load(prior_path)
        nodes, weights = prior.quadrature(32)
        expected = float(np.dot(nodes, weights))
        model = ScatterMix(LAYER_Z, sigma_pos=1.0, dev='cpu', momentum_prior_path=prior_path, momentum_quad_order=32, momentum_mode='fixed_mean_g')
        self.assertEqual(model.momentum_prior_kind, 'fixed_mean_g')
        self.assertEqual(model.momentum_quad_order, 1)
        self.assertAlmostEqual(float(model.g_a.item()), expected, places=15)
        self.assertEqual(float(model.log_w_a.item()), 0.0)

class InstrumentAblationTest(unittest.TestCase):

    def test_zero_position_response_is_exactly_analytic(self):
        model = ScatterMix(LAYER_Z, sigma_pos=1.0, dev='cpu')
        response = model.analytic_response()
        self.assertTrue(model.validate_instrument_response(response))
        features = torch.zeros((5, 4), dtype=torch.float64)
        mean, variance, log_scale = response.evaluate_all_layers(features)
        expected = analytic_residual_variance(LAYER_Z, 1.0)
        np.testing.assert_array_equal(mean.numpy(), 0.0)
        np.testing.assert_array_equal(log_scale.numpy(), 0.0)
        np.testing.assert_allclose(variance.numpy(), np.broadcast_to(expected, (5, *expected.shape)))

class AngleLikelihoodTest(unittest.TestCase):

    def test_no_instrument_likelihood_uses_only_material_scattering(self):
        prior_path = ROOT / 'assets' / 'momentum_prior.npz'
        model = ScatterMix(LAYER_Z, sigma_pos=1.0, dev='cpu', momentum_prior_path=prior_path, momentum_quad_order=16, instrument_mode='none')
        prepared = model.prepare(straight_hits(count=2))
        self.assertNotIn('instrument_mean', prepared)
        self.assertNotIn('instrument_variance', prepared)
        lam = torch.full((2, len(model.zq)), 0.012, dtype=torch.float64, requires_grad=True)
        loss = model.nll_mix(prepared, lam)
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        self.assertTrue(bool(torch.isfinite(lam.grad).all()))
