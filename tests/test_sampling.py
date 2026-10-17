"""Compare the SVD sampler with direct Gaussian conditioning."""

import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from src.gp_model import SpectralGP
from src.posterior_sampling import condition_on_observations


class PosteriorSamplingTests(unittest.TestCase):
    def setUp(self):
        self.model = SpectralGP(n_terms=8)
        self.x = np.array([-1.0, 0.0, 1.0])
        self.y = np.array([0.3, -0.6, 0.4])
        self.noise = 0.03
        self.posterior = condition_on_observations(
            self.model, self.x, self.y, self.noise
        )

    def test_svd_mean_matches_direct_conditioning(self):
        design = self.model.design_matrix(self.x)
        system = design @ design.T + self.noise**2 * np.eye(len(self.x))
        direct_mean = design.T @ np.linalg.solve(system, self.y)
        assert_allclose(self.posterior.mean, direct_mean, atol=1e-12, rtol=1e-12)

    def test_sample_moments_match_direct_posterior(self):
        design = self.model.design_matrix(self.x)
        system = design @ design.T + self.noise**2 * np.eye(len(self.x))
        covariance = np.eye(self.model.n_terms) - design.T @ np.linalg.solve(system, design)
        samples = self.posterior.sample(np.random.default_rng(123), n_samples=20000)
        assert_allclose(samples.mean(axis=0), self.posterior.mean, atol=0.025, rtol=0)
        assert_allclose(np.cov(samples, rowvar=False), covariance, atol=0.03, rtol=0)

    def test_fixed_seed_reproduces_draws(self):
        first = self.posterior.sample(np.random.default_rng(42), n_samples=4)
        second = self.posterior.sample(np.random.default_rng(42), n_samples=4)
        assert_array_equal(first, second)
        different = self.posterior.sample(np.random.default_rng(43), n_samples=4)
        self.assertFalse(np.array_equal(first, different))

    def test_duplicate_observations_remain_finite(self):
        posterior = condition_on_observations(
            self.model, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 1e-3
        )
        self.assertTrue(np.all(np.isfinite(posterior.sample(np.random.default_rng(9), 3))))


if __name__ == "__main__":
    unittest.main()
