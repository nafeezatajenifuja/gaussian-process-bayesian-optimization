"""Check acquisition behavior, bounded search, and a reproducible BO loop."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from src.acquisition import ThompsonAcquisition
from src.gp_model import SpectralGP
from src.optimization import minimize_acquisition, run_bayesian_optimization


class OptimizationTests(unittest.TestCase):
    def test_acquisition_outputs_and_gradient(self):
        model = SpectralGP()
        weights = np.random.default_rng(5).standard_normal(model.n_terms)
        acquisition = ThompsonAcquisition(model, weights)
        points = np.array([-1.0, 0.0, 0.8])
        first = acquisition(points)
        self.assertEqual(first.shape, (3,))
        self.assertTrue(np.all(np.isfinite(first)))
        assert_array_equal(first, acquisition(points))
        assert_allclose(first, model.evaluate(points, weights), atol=0, rtol=0)
        step = 1e-6
        numerical = (acquisition(points + step) - acquisition(points - step)) / (2 * step)
        assert_allclose(acquisition.derivative(points), numerical, atol=2e-8, rtol=2e-7)
        self.assertIsInstance(acquisition.scalar_value([0.0]), float)
        weights[:] = 0.0
        assert_array_equal(first, acquisition(points))

    def test_multistart_returns_finite_bounded_candidate(self):
        model = SpectralGP()
        acquisition = ThompsonAcquisition(model, np.random.default_rng(11).standard_normal(50))
        candidate = minimize_acquisition(acquisition)
        self.assertGreaterEqual(candidate.x, -2.0)
        self.assertLessEqual(candidate.x, 2.0)
        self.assertTrue(np.isfinite(candidate.value))
        self.assertLessEqual(candidate.value, np.min(acquisition([-2, -1, 0, 1, 2])))
        self.assertAlmostEqual(candidate.value, acquisition.scalar_value([candidate.x]))

    def test_fallback_uses_lowest_function_value(self):
        # Every mocked solver returns a worse point. The fallback must preserve
        # x=-2, the lowest-valued start for this positive Gaussian basis term.
        model = SpectralGP(n_terms=1)
        acquisition = ThompsonAcquisition(model, np.ones(1))
        starts = [-2.0, -1.0, 0.0, 0.5, 1.5]
        with patch("src.optimization.minimize", return_value=SimpleNamespace(x=np.array([0.0]), success=False)):
            candidate = minimize_acquisition(acquisition, starts)
        self.assertEqual(candidate.x, -2.0)
        self.assertEqual(candidate.successful_starts, 0)

    def test_same_seed_reproduces_sequential_history(self):
        first = run_bayesian_optimization(iterations=4, seed=87)
        second = run_bayesian_optimization(iterations=4, seed=87)
        assert_array_equal(first.x, second.x)
        assert_array_equal(first.y, second.y)
        assert_array_equal(first.sampled_weights, second.sampled_weights)
        self.assertTrue(np.all(np.isfinite(first.y)))

    def test_run_does_not_change_global_numpy_rng(self):
        np.random.seed(23)
        expected = np.random.standard_normal(3)
        np.random.seed(23)
        run_bayesian_optimization(iterations=1)
        assert_array_equal(np.random.standard_normal(3), expected)

    def test_invalid_optimizer_inputs_raise(self):
        acquisition = ThompsonAcquisition(SpectralGP(n_terms=1), np.ones(1))
        for starts, bounds in [([], (-2, 2)), ([3], (-2, 2)), ([0], (2, -2))]:
            with self.subTest(starts=starts, bounds=bounds):
                with self.assertRaises(ValueError):
                    minimize_acquisition(acquisition, starts, bounds)


if __name__ == "__main__":
    unittest.main()
