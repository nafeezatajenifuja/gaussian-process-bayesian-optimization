"""Check array contracts and invalid-input failures at module boundaries."""

import unittest

import numpy as np

from src.acquisition import ThompsonAcquisition
from src.gp_model import SpectralGP
from src.optimization import run_bayesian_optimization
from src.posterior_sampling import condition_on_observations


class ShapeTests(unittest.TestCase):
    def test_model_and_sampler_shapes_for_multiple_truncations(self):
        for terms in (1, 2, 8, 50):
            with self.subTest(n_terms=terms):
                model = SpectralGP(n_terms=terms)
                points = np.linspace(-2, 2, 7)
                self.assertEqual(model.basis(points).shape, (terms, 7))
                self.assertEqual(model.design_matrix(points).shape, (7, terms))
                self.assertEqual(model.basis_derivative(points).shape, (7, terms))
                self.assertEqual(model.evaluate(points, np.zeros(terms)).shape, (7,))
                self.assertEqual(model.evaluate(0.0, np.zeros(terms)).shape, (1,))
                posterior = condition_on_observations(model, [-1, 0, 1], [1, 0, -1])
                self.assertEqual(posterior.sample(np.random.default_rng(2), 4).shape, (4, terms))

    def test_bo_history_shapes(self):
        result = run_bayesian_optimization(iterations=2, seed=14)
        self.assertEqual(result.x.shape, (5,))
        self.assertEqual(result.y.shape, (5,))
        self.assertEqual(result.sampled_weights.shape, (2, 50))
        self.assertEqual(result.acquisition_values.shape, (2,))
        self.assertEqual(len(result.successful_starts), 2)
        empty = run_bayesian_optimization(iterations=0)
        self.assertEqual(empty.sampled_weights.shape, (0, 50))
        self.assertEqual(empty.x.shape, (3,))

    def test_invalid_shapes_and_parameters_raise(self):
        model = SpectralGP()
        invalid_calls = [
            lambda: model.basis([[1.0], [2.0]]),
            lambda: model.basis([]),
            lambda: model.basis([np.nan]),
            lambda: model.evaluate([0], np.ones(49)),
            lambda: ThompsonAcquisition(model, np.ones(49)),
            lambda: condition_on_observations(model, [0, 1], [0]),
            lambda: condition_on_observations(model, [0], [0], 0),
            lambda: condition_on_observations(model, [0], [0], np.nan),
            lambda: SpectralGP(n_terms=0),
            lambda: SpectralGP(length_scale=-1),
            lambda: SpectralGP(n_terms=2.5),
            lambda: run_bayesian_optimization(iterations=-1),
            lambda: run_bayesian_optimization(seed=-1),
        ]
        for call in invalid_calls:
            with self.subTest(call=call):
                with self.assertRaises(ValueError):
                    call()


if __name__ == "__main__":
    unittest.main()
