"""Check covariance, objective values, and analytic derivatives independently."""

import unittest

import numpy as np
from numpy.testing import assert_allclose

from src.gp_model import SpectralGP
from src.optimization import synthetic_objective


class GaussianProcessTests(unittest.TestCase):
    def test_covariance_approximates_squared_exponential_kernel(self):
        # Independent kernel expression, rather than a second basis recurrence.
        model = SpectralGP(n_terms=100)
        x = np.linspace(-2.0, 2.0, 17)
        expected = np.exp(-0.5 * ((x[:, None] - x[None, :]) / 0.4) ** 2)
        covariance = model.covariance(x, x)
        assert_allclose(covariance, expected, atol=1e-10, rtol=1e-10)
        self.assertGreaterEqual(np.linalg.eigvalsh(covariance).min(), -1e-12)

    def test_derivative_agrees_with_central_difference(self):
        model = SpectralGP()
        weights = np.random.default_rng(7).standard_normal(model.n_terms)
        points = np.array([-1.7, -0.3, 0.0, 0.6, 1.8])
        step = 1e-6
        numerical = (
            model.evaluate(points + step, weights)
            - model.evaluate(points - step, weights)
        ) / (2.0 * step)
        assert_allclose(model.derivative(points, weights), numerical, atol=2e-8, rtol=2e-7)

    def test_objective_matches_original_initial_values(self):
        # Noiseless values computed from the original notebook, before refactoring.
        expected = np.array([-0.31574020, 1.02235477, -0.34216602])
        assert_allclose(synthetic_objective([-2.0, 0.0, 2.0]), expected, atol=5e-9, rtol=0)

    def test_eigenvalues_are_positive_and_decreasing(self):
        values = SpectralGP().eigenvalues()
        self.assertTrue(np.all(values > 0))
        self.assertTrue(np.all(np.diff(values) < 0))


if __name__ == "__main__":
    unittest.main()
