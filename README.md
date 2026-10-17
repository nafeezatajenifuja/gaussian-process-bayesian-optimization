# Gaussian-Process Bayesian Optimization

Posterior function sampling for sequential optimization with a spectral Gaussian process.

![Objective, observations, posterior function draw, and Thompson-sampling selection](figures/representative_result.png)

When objective evaluations are expensive, choosing where to evaluate becomes part of the optimization problem. Bayesian optimization uses a probabilistic surrogate to guide those decisions as observations become available.

I construct the Gaussian-process surrogate explicitly using a spectral representation, update it as noisy observations become available, and draw entire functions from the resulting posterior. Each draw represents one plausible version of the unknown objective under the model.

With Thompson sampling, one posterior draw is treated as a plausible objective for the next decision. I search for a low point on that sampled function, evaluate the true objective there, add the new observation, and update the GP before repeating the process. The full sampling and sequential optimization loop is implemented directly in Python.

## Bayesian optimization in this project

1. Start with a small set of noisy observations.
2. Condition the spectral Gaussian process on those observations.
3. Draw one posterior function.
4. Search for its minimum using bounded optimization from several starting points.
5. Evaluate the objective at the selected location.
6. Add the new observation and repeat.

## What the experiment shows

The demonstration minimizes a one-dimensional synthetic objective on `[-2, 2]`, with Gaussian noise added to observations. It starts with three observations and adds 30 evaluations through the sequential loop.

In the figure above, the black curve is the underlying objective, the red points are observations collected so far, and the orange curve is one posterior function draw. The blue point shows the next location selected by Thompson sampling.

The figure connects the probabilistic model to the decision it produces: a sampled function becomes an optimization problem, and its selected location becomes the next observation.

## Implementation

A truncated spectral (Wiener) series approximates a squared-exponential Gaussian process. Posterior conditioning uses an SVD calculation, after which Gaussian coefficient draws define sampled functions. Each draw remains fixed throughout its acquisition search. Analytic derivatives guide multistart L-BFGS-B optimization, and the posterior is updated after each new observation.

The numerical implementation uses NumPy and SciPy, with Matplotlib for visualization.

- [gp_model.py](src/gp_model.py) — spectral basis, covariance, and derivatives.
- [posterior_sampling.py](src/posterior_sampling.py) — posterior conditioning and function draws.
- [acquisition.py](src/acquisition.py) — the sampled-function objective and its derivative.
- [optimization.py](src/optimization.py) — multistart search and sequential optimization.
- [cleaned_demo.ipynb](notebooks/cleaned_demo.ipynb) — a walkthrough of the experiment.

## Running the example

```bash
git clone https://github.com/nafeezatajenifuja/gaussian-process-bayesian-optimization.git
cd gaussian-process-bayesian-optimization
python -m pip install -r requirements.txt
python -m src.optimization --seed 100 --iterations 30
```

For an interactive walkthrough, open [notebooks/cleaned_demo.ipynb](notebooks/cleaned_demo.ipynb).

## Background

I developed the original implementation during the graduate Decision-Making with Uncertainty course at the University of Houston. I later reorganized the notebook into a modular Python workflow so that the GP sampling and sequential optimization steps could be reproduced, tested, and examined individually. The [original notebook](notebooks/original_project.ipynb) remains available for provenance.

[Implementation and provenance notes](docs/implementation_notes.md) document the numerical details, changes from the original notebook, and preserved results.
