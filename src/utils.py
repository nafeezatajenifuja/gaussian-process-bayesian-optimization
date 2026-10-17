"""Shared input validation and reproducible random-number generation."""

from __future__ import annotations

from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def as_vector(values: ArrayLike, name: str) -> FloatArray:
    """Convert a scalar or nonempty one-dimensional input to a finite vector.

    Column matrices are rejected so that accidental broadcasting cannot silently
    turn a vector calculation into a matrix calculation.
    """
    vector = np.asarray(values, dtype=float)
    if vector.ndim == 0:
        vector = vector.reshape(1)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{name} must be a scalar or a nonempty 1D array")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def nonnegative_integer(value: int, name: str) -> int:
    """Validate a count without accepting booleans or silently rounding floats."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def make_rng(seed: int = 100) -> np.random.Generator:
    """Create a local generator without changing NumPy's global random state."""
    return np.random.default_rng(nonnegative_integer(seed, "seed"))
