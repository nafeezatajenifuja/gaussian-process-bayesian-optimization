# Implementation audit and scientific provenance

## Inspection scope

The complete readable course-project archive `Nafeezat_CIVE_7397_Project (1).zip` contains a main notebook, a checkpoint notebook, and `K_Iterations.pdf`. All notebook cell sources, output types and textual outputs were inspected; the PDF's text confirms iterations 1 through 30. The main notebook's first and last optimization plots were visually inspected. The checkpoint has exactly the same cell sources as the main notebook but different saved outputs. The two course-project archive copies have identical SHA-256 hashes.

A separate, readable `Nafeezat_Assignment.ipynb` in the parent workspace implements spectral GP prior and posterior examples. It is not the source of the BO refactoring. The cloud-only Taiwo notebook, its checkpoint, and the two HW2 ZIPs could not be retrieved from OneDrive; the author explicitly selected the complete Nafeezat course-project archive as the project source. Those other workspace files are outside this publishable repository and no claim is made about their contents.

## What the source implements

Cell numbers below are **zero-based JSON cell indices** in `notebooks/original_project.ipynb`, not execution counts.

| Original cells | Verified behavior | Extracted location |
| --- | --- | --- |
| 2–7 | Expansion constants; recursive eigenfunctions; geometric eigenvalues | `src/gp_model.py` |
| 9–15 | Objective using basis indices 1 and 2, weights [-0.25, -2.5]; Gaussian observation noise; initial locations [-2, 0, 2] | `synthetic_objective` and `observe` in `src/optimization.py` |
| 18 | Surrogate as a truncated spectral/Wiener series | `SpectralGP.evaluate` |
| 20 | Reduced-SVD Gaussian coefficient conditioning and posterior sample | `src/posterior_sampling.py` |
| 22, 25–26 | One posterior function and its derivative as the TS acquisition | `src/acquisition.py` |
| 23–24 | Analytic eigenfunction and function derivatives | `SpectralGP.basis_derivative` and `derivative` |
| 28 | Five-start bounded L-BFGS-B minimization over [-2, 2] | `minimize_acquisition` |
| 30 | Exactly 30 sequential posterior-draw / optimize / observe / append steps; 30 saved plots | `run_bayesian_optimization` |

The prior coefficient distribution is standard multivariate normal. For design matrix H and observation-noise standard deviation s_n, the notebook uses the reduced SVD H = U diag(d) V^T and computes

```text
mean_w = V diag(d / (d^2 + s_n^2)) U^T y
R z    = z - V diag(1 - sqrt(s_n^2 / (d^2 + s_n^2))) V^T z
w_draw = mean_w + R z,  z ~ N(0, I)
```

The extracted sampler uses those same equations, with vector scaling in place of explicit diagonal matrices. Its distribution is checked against direct Gaussian conditioning. The surrogate's finite covariance is H(x) H(x')^T. An independent test checks the squared-exponential kernel approximation with 100 terms; the experiment retains the original 50 terms. That test is a numerical sanity check, not a convergence study of the 50-term approximation.

## Documented changes

The original notebook and PDF are unchanged. The representative PNG is the original final-iteration PNG decoded from the notebook, with identical image bytes.

| Change in extracted code | Reason and effect |
| --- | --- |
| Label the objective as minimization | Original cell 29 says “global maximum,” but cells 28–30 minimize the posterior draw. No objective sign was changed. |
| Replace the starting-point lookup with `argmin` of acquisition values | Original cell 28 searches sorted x coordinates for a minimum function value. This can initialize the wrong point or raise an indexing error. The correction associates the lowest sampled value with its actual starting location. The acquisition, gradients, starts, bounds, and local solver are retained. A regression test forces the fallback path. |
| Use scalar eigenvalue expressions / vector scaling | Original cell 7 assigns a length-one NumPy array into a scalar slot, which fails on the validated NumPy 2.4.4 environment. The eigenvalue formula is unchanged. |
| Use a local `default_rng(seed)` and explicit generator arguments | Makes newly generated noise and coefficient draws repeatable without changing global random state. Seed 100 is new; it is not inferred from the historical outputs. |
| Use `n_terms` consistently | Original sampling and acquisition helpers hardcode 50 even when accepting a term-count parameter. The extracted implementation honors the chosen term count. The default experiment still uses 50. |
| Support a one-term expansion and reject invalid inputs | Original derivative branches for n=0 or n=1 index nonexistent basis entries. The valid n=1 case is handled; empty, mismatched, nonfinite, and invalid-scale inputs raise errors. The default experiment is unchanged. |
| Scalar SciPy objective and one-element gradient | Clarifies the optimizer interface without changing the function minimized. |
| Return history and expose local-solver success counts | Adds inspection data. Finite improving candidates are retained even when a stopping flag is false, as in the original policy. |
| Close generated figures; keep one concise notebook example | Avoids the original “more than 20 figures” warning. Original notebook outputs remain intact; the clean demo is committed with no outputs. |

No new acquisition policy, GP fitting routine, objective, benchmark, or convergence claim was introduced. Because the source is unseeded, the candidate-index correction can affect decisions, and numerical libraries changed, a new seeded history is not an exact replay of the source run.

## README claim-verification checklist for the author

These are all substantive scientific, numerical, and provenance claims in the README. Check them before public release; the code-supported items have already been checked against the selected archive.

| Claim | Evidence / qualification |
| --- | --- |
| One-dimensional minimization on [-2, 2] | Original cell 28 uses `minimize`, `L-BFGS-B`, and these bounds. The “maximum” heading is inconsistent with the executed policy. |
| Synthetic objective uses spectral indices 1 and 2 and weights [-0.25, -2.5] | Original cell 9 deletes the zeroth basis term and eigenvalue; cell 11 sets the weights. This is not a real-world dataset. |
| Independent Gaussian observation noise has standard deviation 0.001 | Original cells 13–14 and 30 draw fresh normal noise and set `sigma_n = 10**(-3)`. |
| Zero-mean squared-exponential GP, represented by a 50-term truncated Wiener series | Cells 2–7 define the SE spectral construction; cells 18 and 30 evaluate 50 terms with standard-normal prior coefficients. The separate readable GP assignment explicitly names the SE prior. The extracted implementation approximates the kernel; it does not use an infinite expansion. |
| Expansion measure scale 1 and kernel length scale 0.4 remain fixed | Original cell 3 calls `parameters(1, 2/5)`. No parameter-fitting step exists. Measure scale is not a learned signal amplitude. |
| Posterior coefficients are conditioned and sampled using a reduced SVD and low-rank square-root correction | Original cell 20; the extracted sampler is checked against direct Gaussian conditioning and sampled moments. |
| Thompson sampling uses one fixed posterior function per decision | Original cell 28 draws `post_w` once and passes it to every local search. |
| Analytic derivatives and five-start L-BFGS-B acquisition minimization | Original cells 23–28 and the five starting points supplied in cell 30. Local searches provide no global-optimality guarantee. |
| Three initial points [-2, 0, 2], 30 new evaluations, 33 observations total | Original cells 14 and 30. The archived final plot is drawn before appending the last observation, so it does not display all 33 as red observations. |
| Representative image is the original iteration-30 output | Last PNG output of original cell 30, copied without replotting. It is a historical illustration, not a seeded rerun or benchmark. |
| Original artifacts are preserved unchanged | Compare hashes in `provenance/manifest.json` with the source archive; notebook and PDF are byte-for-byte copies. |
| Fixed-seed behavior and no mutation of global NumPy RNG | Tests check this for the refactored code in one environment. No cross-platform bitwise reproducibility claim is made. |
| Tests cover dimensions, posterior sampling, covariance, derivatives, and acquisition behavior | Review `tests/`; passing checks are correctness evidence, not empirical optimization benchmarks. |
| No fitted hyperparameters, baselines, regret analysis, constraints, batch acquisition, or multidimensional experiment | Absent from all selected archive source cells. The work remains a course implementation. |
| Standard methods are implemented, rather than invented here | The project makes no novelty claim; GP, BO, Thompson sampling, SVD, and L-BFGS-B are established methods. |
| Graduate Decision-Making with Uncertainty course, University of Houston, and authorship | Supplied by the author. The archive alone does not independently establish the official course title, institutional attribution, authorship, or any instructor-provided code attribution. Confirm these and retain any required course-material credits before publication. |

## Validation commands

```bash
python -m pip check
python -m unittest discover -s tests -v
python -m src.optimization --seed 100 --iterations 30
python scripts/run_notebook.py
```

The reorganized project is validated in a newly created virtual environment with the pinned requirements. The unmodified original notebook is retained as an archival artifact and is not the runnable entry point for current NumPy.

Observed validation results on Python 3.13.5: dependency installation succeeded; `pip check` reported no broken requirements; all 17 unit tests passed; the CLI completed 30 iterations and saved a finite, bounded history with 33 observations; and the headless runner executed all six clean-demo code cells successfully. Both notebook formats were validated, the clean demo retained empty output cells, and all preserved-artifact hashes matched after execution. Other interpreters and operating systems were not tested locally.

A read-only comparison also evaluated the original notebook's function definitions against the extracted basis, eigenvalues, function values, analytic derivatives, synthetic objective, and posterior coefficient draw. With identical coefficients / standard-normal draws, all six comparisons agreed within absolute and relative tolerances of 2e-12. The comparison adapted only the obsolete scalar-assignment expression in memory; it did not edit the original notebook. This checks formula preservation, rather than claiming that whole BO trajectories are identical.
