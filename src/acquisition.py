"""A Thompson-sampling acquisition formed from one fixed posterior draw."""

from __future__ import annotations

from dataclasses import dataclass

from numpy.typing import ArrayLike

from .gp_model import SpectralGP
from .utils import FloatArray


@dataclass(frozen=True)
class ThompsonAcquisition:
    """Minimize a sampled GP function using its analytic derivative.

    Coefficients are drawn once per BO iteration. Evaluating the acquisition
    or its derivative never resamples the coefficients.
    """

    model: SpectralGP
    weights: FloatArray

    def __post_init__(self) -> None:
        coefficients = self.model.validate_weights(self.weights).copy()
        coefficients.setflags(write=False)
        object.__setattr__(self, "weights", coefficients)

    def __call__(self, x: ArrayLike) -> FloatArray:
        """Return sampled function values, shape (n_points,)."""
        return self.model.evaluate(x, self.weights)

    def derivative(self, x: ArrayLike) -> FloatArray:
        """Return sampled function derivatives, shape (n_points,)."""
        return self.model.derivative(x, self.weights)

    def scalar_value(self, x: ArrayLike) -> float:
        """Adapt a single point to SciPy's scalar objective interface."""
        values = self(x)
        if values.size != 1:
            raise ValueError("the optimizer expects exactly one point")
        return float(values[0])
