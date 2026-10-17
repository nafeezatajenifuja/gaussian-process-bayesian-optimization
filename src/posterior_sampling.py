"""SVD-based Gaussian conditioning and coefficient sampling from the notebook."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .gp_model import SpectralGP
from .utils import FloatArray, as_vector, nonnegative_integer


@dataclass(frozen=True)
class WeightPosterior:
    """Gaussian coefficient posterior using a low-rank square-root correction.

    With reduced SVD H = U diag(s) V^T, a sample is
    mean + z - V diag(shrinkage) V^T z, where z ~ N(0, I).
    """

    mean: FloatArray
    right_vectors: FloatArray
    shrinkage: FloatArray

    def sample(self, rng: np.random.Generator, n_samples: int = 1) -> FloatArray:
        """Return coefficient draws with shape (n_samples, n_terms)."""
        count = nonnegative_integer(n_samples, "n_samples")
        if count == 0:
            raise ValueError("n_samples must be positive")
        standard_normal = rng.standard_normal((count, self.mean.size))
        correction = (
            (standard_normal @ self.right_vectors) * self.shrinkage
        ) @ self.right_vectors.T
        return self.mean + standard_normal - correction


def condition_on_observations(
    model: SpectralGP,
    x: ArrayLike,
    y: ArrayLike,
    noise_std: float = 1e-3,
) -> WeightPosterior:
    """Condition standard-normal GP coefficients on Gaussian-noisy observations.

    ``noise_std`` is the observation-noise standard deviation. The variance
    entering the original SVD equations is its square. No hyperparameters
    are fitted here.
    """
    points = as_vector(x, "x")
    observations = as_vector(y, "y")
    if points.size != observations.size:
        raise ValueError("x and y must have the same length")
    if not np.isfinite(noise_std) or noise_std <= 0:
        raise ValueError("noise_std must be finite and positive")
    design = model.design_matrix(points)
    left, singular_values, right_transpose = np.linalg.svd(design, full_matrices=False)
    denominator = singular_values**2 + noise_std**2
    right = right_transpose.T
    mean = right @ ((singular_values / denominator) * (left.T @ observations))
    shrinkage = 1.0 - np.sqrt(noise_std**2 / denominator)
    return WeightPosterior(mean, right, shrinkage)
