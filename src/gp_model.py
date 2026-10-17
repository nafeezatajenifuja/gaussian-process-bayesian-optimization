"""The original one-dimensional Gaussian-process spectral representation.

The eigenfunction recurrence and eigenvalues are extracted from the course
notebook. The finite feature matrix H(x) gives the approximate covariance
K_m(x, x') = H(x) H(x')^T of the zero-mean prior.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .utils import FloatArray, as_vector, nonnegative_integer


@dataclass(frozen=True)
class SpectralGP:
    """Squared-exponential GP represented by a truncated Wiener series.

    ``measure_scale`` is the notebook's sigma in a = 1 / (2 sigma^2), the
    scale of the Gaussian reference measure used for the expansion. It is
    neither the observation-noise standard deviation nor a fitted amplitude.
    The original settings are sigma=1, length_scale=0.4, and n_terms=50.
    """

    measure_scale: float = 1.0
    length_scale: float = 0.4
    n_terms: int = 50

    def __post_init__(self) -> None:
        for name in ("measure_scale", "length_scale"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if nonnegative_integer(self.n_terms, "n_terms") == 0:
            raise ValueError("n_terms must be positive")

    @property
    def parameters(self) -> tuple[float, float, float, float]:
        """Return the notebook's expansion constants (a, b, c, A)."""
        a = 1.0 / (2.0 * self.measure_scale**2)
        b = 1.0 / (2.0 * self.length_scale**2)
        c = float(np.sqrt(a**2 + 4.0 * a * b))
        normalizer = 0.5 * a + b + 0.5 * c
        return a, b, c, normalizer

    def eigenvalues(self) -> FloatArray:
        """Return eigenvalues lambda_0 through lambda_(m-1), shape (m,)."""
        a, b, _, normalizer = self.parameters
        return np.sqrt(a / normalizer) * (b / normalizer) ** np.arange(self.n_terms)

    def basis(self, x: ArrayLike) -> FloatArray:
        """Evaluate eigenfunctions; return shape (n_terms, n_points)."""
        points = as_vector(x, "x")
        a, _, c, _ = self.parameters
        values = np.empty((self.n_terms, points.size))
        values[0] = (c / a) ** 0.25 * np.exp(-0.5 * (c - a) * points**2)
        if self.n_terms > 1:
            values[1] = np.sqrt(2.0 * c) * points * values[0]
        for index in range(2, self.n_terms):
            values[index] = (
                np.sqrt(2.0 * c / index) * points * values[index - 1]
                - np.sqrt((index - 1.0) / index) * values[index - 2]
            )
        return values

    def basis_derivative(self, x: ArrayLike) -> FloatArray:
        """Differentiate the recurrence analytically; shape (n_points, m)."""
        points = as_vector(x, "x")
        a, _, c, _ = self.parameters
        values = self.basis(points)
        derivatives = np.empty_like(values)
        derivatives[0] = -(c - a) * points * values[0]
        if self.n_terms > 1:
            derivatives[1] = np.sqrt(2.0 * c) * (
                values[0] + points * derivatives[0]
            )
        for index in range(2, self.n_terms):
            derivatives[index] = np.sqrt(2.0 * c / index) * (
                values[index - 1] + points * derivatives[index - 1]
            ) - np.sqrt((index - 1.0) / index) * derivatives[index - 2]
        return derivatives.T

    def design_matrix(self, x: ArrayLike) -> FloatArray:
        """Return H(x) = phi(x)^T sqrt(Lambda), shape (n_points, m)."""
        return self.basis(x).T * np.sqrt(self.eigenvalues())

    def evaluate(self, x: ArrayLike, weights: ArrayLike) -> FloatArray:
        """Evaluate one fixed coefficient draw, returning shape (n_points,)."""
        coefficients = self.validate_weights(weights)
        return self.design_matrix(x) @ coefficients

    def derivative(self, x: ArrayLike, weights: ArrayLike) -> FloatArray:
        """Evaluate the analytic derivative of one fixed function draw."""
        coefficients = self.validate_weights(weights)
        return (self.basis_derivative(x) * np.sqrt(self.eigenvalues())) @ coefficients

    def covariance(self, x: ArrayLike, other: ArrayLike) -> FloatArray:
        """Return the finite-expansion prior covariance between two point sets."""
        return self.design_matrix(x) @ self.design_matrix(other).T

    def validate_weights(self, weights: ArrayLike) -> FloatArray:
        """Return a finite coefficient vector of length ``n_terms``."""
        coefficients = as_vector(weights, "weights")
        if coefficients.size != self.n_terms:
            raise ValueError(f"weights must have length {self.n_terms}")
        return coefficients
