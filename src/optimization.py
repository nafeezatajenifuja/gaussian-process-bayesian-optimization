"""The original multistart acquisition minimization and sequential BO loop.

Run from the repository root with ``python -m src.optimization``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize

from .acquisition import ThompsonAcquisition
from .gp_model import SpectralGP
from .posterior_sampling import condition_on_observations
from .utils import FloatArray, as_vector, make_rng, nonnegative_integer

if TYPE_CHECKING:
    from matplotlib.figure import Figure

DEFAULT_STARTS = (-2.0, -1.0, 0.0, 1.0, 2.0)
DEFAULT_INITIAL_POINTS = (-2.0, 0.0, 2.0)
DEFAULT_BOUNDS = (-2.0, 2.0)


@dataclass(frozen=True)
class AcquisitionMinimum:
    """Best finite value among the starting points and local optimizer results."""

    x: float
    value: float
    successful_starts: int


@dataclass(frozen=True)
class BOResult:
    """Observed data and one posterior coefficient draw per BO iteration."""

    x: FloatArray
    y: FloatArray
    sampled_weights: FloatArray
    acquisition_values: FloatArray
    successful_starts: tuple[int, ...]
    n_initial: int
    seed: int


def synthetic_objective(x: ArrayLike, model: SpectralGP | None = None) -> FloatArray:
    """Evaluate the notebook's objective using basis indices 1 and 2.

    The two fixed coefficients are [-0.25, -2.5]. Basis index 0 is deliberately
    omitted, as in the original ``f`` function. This is a synthetic example,
    not an externally collected dataset.
    """
    source = model if model is not None else SpectralGP()
    objective_model = SpectralGP(source.measure_scale, source.length_scale, n_terms=3)
    return objective_model.design_matrix(x)[:, 1:3] @ np.array([-0.25, -2.5])


def observe(
    x: ArrayLike,
    model: SpectralGP,
    noise_std: float,
    rng: np.random.Generator,
) -> FloatArray:
    """Observe the synthetic objective with independent Gaussian noise."""
    if not np.isfinite(noise_std) or noise_std <= 0:
        raise ValueError("noise_std must be finite and positive")
    values = synthetic_objective(x, model)
    return values + noise_std * rng.standard_normal(values.size)


def minimize_acquisition(
    acquisition: ThompsonAcquisition,
    starts: ArrayLike = DEFAULT_STARTS,
    bounds: tuple[float, float] = DEFAULT_BOUNDS,
) -> AcquisitionMinimum:
    """Apply bounded L-BFGS-B from each starting point, using analytic gradients.

    The fallback selects the actual lowest-valued starting point with argmin.
    The notebook instead searches x coordinates for a function value; this is
    a documented indexing correction. Multistart local search does not certify
    a global minimum.
    """
    start_points = as_vector(starts, "starts")
    limits = as_vector(bounds, "bounds")
    if limits.size != 2 or limits[0] >= limits[1]:
        raise ValueError("bounds must contain a finite lower and upper limit")
    if np.any((start_points < limits[0]) | (start_points > limits[1])):
        raise ValueError("all starting points must lie inside bounds")
    values = acquisition(start_points)
    if not np.all(np.isfinite(values)):
        raise ValueError("acquisition returned nonfinite starting values")
    best_index = int(np.argmin(values))
    best_x = float(start_points[best_index])
    best_value = float(values[best_index])
    successful_starts = 0
    for start in start_points:
        result = minimize(
            acquisition.scalar_value,
            np.array([start]),
            method="L-BFGS-B",
            jac=acquisition.derivative,
            bounds=[(float(limits[0]), float(limits[1]))],
        )
        successful_starts += int(result.success)
        # Retain finite improvements even if a solver stopping flag is false,
        # matching the original policy, and expose success counts for inspection.
        value = acquisition.scalar_value(result.x)
        if np.isfinite(value) and value < best_value:
            best_x, best_value = float(result.x[0]), value
    return AcquisitionMinimum(best_x, best_value, successful_starts)


def run_bayesian_optimization(
    iterations: int = 30,
    seed: int = 100,
    model: SpectralGP | None = None,
    noise_std: float = 1e-3,
) -> BOResult:
    """Run the course experiment: three initial points and sequential updates.

    One local RNG drives initial noise, posterior draws, and subsequent noise
    in that order. A fixed seed reproduces a run in the same numerical
    environment; the unseeded historical notebook trajectory is not replayed.
    """
    count = nonnegative_integer(iterations, "iterations")
    surrogate = model if model is not None else SpectralGP()
    rng = make_rng(seed)
    x = np.array(DEFAULT_INITIAL_POINTS)
    y = observe(x, surrogate, noise_std, rng)
    weights = np.empty((count, surrogate.n_terms))
    acquisition_values = np.empty(count)
    successes: list[int] = []
    for iteration in range(count):
        posterior = condition_on_observations(surrogate, x, y, noise_std)
        weights[iteration] = posterior.sample(rng)[0]
        acquisition = ThompsonAcquisition(surrogate, weights[iteration])
        candidate = minimize_acquisition(acquisition)
        new_y = observe([candidate.x], surrogate, noise_std, rng)
        x = np.append(x, candidate.x)
        y = np.append(y, new_y)
        acquisition_values[iteration] = candidate.value
        successes.append(candidate.successful_starts)
    return BOResult(x, y, weights, acquisition_values, tuple(successes), 3, seed)


def plot_result(result: BOResult, model: SpectralGP | None = None) -> Figure:
    """Plot the last acquisition decision, matching the original plot timing.

    Observations exclude the last newly selected point; the selected point is
    shown on the sampled function, rather than at its noisy observed value.
    """
    import matplotlib.pyplot as plt

    surrogate = model if model is not None else SpectralGP()
    if result.sampled_weights.shape[0] == 0:
        raise ValueError("at least one iteration is needed for this plot")
    grid = np.arange(-2.0, 2.0, 0.01)
    acquisition = ThompsonAcquisition(surrogate, result.sampled_weights[-1])
    figure, axes = plt.subplots(figsize=(9, 5))
    axes.plot(grid, synthetic_objective(grid, surrogate), color="#252525", label="Objective")
    axes.scatter(result.x[:-1], result.y[:-1], color="#c94c4c", s=25, label="Observations")
    axes.plot(grid, acquisition(grid), color="#d59712", label="Posterior draw / acquisition")
    axes.scatter(
        result.x[-1], result.acquisition_values[-1], color="#2166ac", s=60,
        label="Selected candidate", zorder=4,
    )
    axes.set(
        xlabel="x", ylabel="f(x)", xlim=DEFAULT_BOUNDS,
        title=f"Thompson sampling: iteration {len(result.sampled_weights)}, seed {result.seed}",
    )
    axes.legend(loc="upper right", fontsize=8)
    figure.tight_layout()
    return figure


def main() -> None:
    """Run a seeded demonstration and write its figure and numerical history."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--figure", type=Path, default=Path("figures/seeded_demo.png"))
    parser.add_argument("--history", type=Path, default=Path("results/seeded_history.npz"))
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive for a plotted demonstration")
    import matplotlib

    matplotlib.use("Agg")
    result = run_bayesian_optimization(iterations=args.iterations, seed=args.seed)
    figure = plot_result(result)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.figure, dpi=160)
    import matplotlib.pyplot as plt

    plt.close(figure)
    args.history.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.history, x=result.x, y=result.y, sampled_weights=result.sampled_weights,
        acquisition_values=result.acquisition_values,
        successful_starts=np.array(result.successful_starts),
        seed=result.seed, n_initial=result.n_initial,
        measure_scale=1.0, length_scale=0.4, n_terms=50, noise_std=1e-3,
    )
    print(f"Completed {args.iterations} iterations; {len(result.x)} total observations.")
    print(f"Figure: {args.figure}\nNumerical history: {args.history}")


if __name__ == "__main__":
    main()
