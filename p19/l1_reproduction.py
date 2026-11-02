"""Qualitative reproduction of Tang et al. ICLR 2025, Figure 1(c) / Appendix A.2.

Choices that the paper leaves unspecified are read from the L1 config and are not
adjusted after the results are known.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import svd

from p19.estimators import fit_ridge_unnormalized


L1_DEVIATIONS = [
    "The reproduced target is Figure 1(c) and Appendix A.2 only. Panels (a) and (b) are not run: Appendix A.1 does not give an operational sampler for the random target blocks that realize the stated values of ||T|| and tr[U]/tr[V].",
    "The figure axis is labeled 'Log n' with ticks 5, 6, 7, and 8. The paper does not state the base. Base e is used because Appendix A.1 places the covariates in dimension k+n^2; reading the same axis as log10 would make the largest panel-(a) sample size 10^6 and the column count about 10^12, which cannot be the experiment the figure reports. This interpretation is applied unchanged to panel (c).",
    "Appendix A.2 says 'various' training sizes and does not list them. The grid is seven evenly spaced points of natural log n from 5 through 8, including the labeled ticks. It was fixed in the config before the run.",
    "The random-seed schedule is an L1 reproducibility choice. It is not the missing L2 seed list and it is not written into protocol_freeze.yaml as an approval.",
    "PCR uses the top k=10 right singular vectors of the same n training rows. Theorem 5 instead splits 2n samples into a PCA half and a regression half. Appendix A.2 does not mention that split, so the simulation text's training size n is followed. k=10 is the paper's known major-direction count, not a value selected after seeing the plot.",
    "The acceptance metric is the population excess risk ||beta_hat-beta_star||^2_{Sigma_T} from Section 2.2. With Sigma_T=I this is the squared Euclidean parameter error. Appendix A.2 also draws 1000 test rows; those rows are drawn and a Monte Carlo excess is stored, and it is not used to accept or reject the run.",
    "L1 ridge uses Tang et al. equation beta = X^T (XX^T + lambda I)^{-1} Y, implemented in the equivalent primal form (X^T X + lambda I)^{-1} X^T y. The denominator is s^2+lambda. The L2 objective's s^2+n*lambda scaling is a different estimator and is not used here.",
    "No author repository is cited in the ICLR 2025 PDF, and none was used.",
]


def n_grid_from_config(config: dict) -> list[int]:
    axis = config["log_n_axis"]
    if axis["base"] != "e":
        raise ValueError("this L1 runner only implements the predeclared natural-log axis")
    logs = np.linspace(float(axis["tick_min"]), float(axis["tick_max"]), int(axis["n_points"]))
    values = [int(round(math.exp(float(value)))) for value in logs]
    if len(set(values)) != len(values):
        raise ValueError(f"rounded n grid is not unique: {values}")
    return values


def _sample_design(n_rows: int, evals: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    standard = rng.standard_normal((n_rows, evals.shape[0]))
    return np.asarray(standard * np.sqrt(evals), dtype=np.float64)


def _pcr_top_k(X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    _left, singular, right_h = svd(X, full_matrices=False, lapack_driver="gesdd", overwrite_a=False)
    if singular.size < k or not np.all(singular[:k] > 0):
        raise RuntimeError(f"PCR requested k={k} but only {singular.size} positive singular values were available")
    basis = right_h[:k, :].T
    projected = X @ basis
    gram = projected.T @ projected
    coefficient_k = np.linalg.solve(gram, projected.T @ y)
    return basis @ coefficient_k


def _one_replicate(n: int, seed: int, config: dict) -> dict:
    rng = np.random.default_rng(seed)
    k = int(config["k_major"])
    noise_variance = float(config["noise_variance"])
    n_test = int(config["n_test"])
    dimension = k + int(math.floor(math.sqrt(n)))
    beta = np.zeros(dimension, dtype=np.float64)
    beta[:k] = 1.0 / math.sqrt(k)
    evals = np.ones(dimension, dtype=np.float64)
    evals[k:] = float(n) ** (-0.5)
    X = _sample_design(n, evals, rng)
    epsilon = rng.normal(0.0, math.sqrt(noise_variance), size=n)
    y = X @ beta + epsilon
    X_test = _sample_design(n_test, np.ones(dimension, dtype=np.float64), rng)
    epsilon_test = rng.normal(0.0, math.sqrt(noise_variance), size=n_test)
    y_test = X_test @ beta + epsilon_test
    signal_test = X_test @ beta

    estimates = {
        "ridgeless_lambda_1e-8": fit_ridge_unnormalized(X, y, 1e-8).coefficient,
        "ridge_lambda_n^0.5": fit_ridge_unnormalized(X, y, n ** 0.5).coefficient,
        "ridge_lambda_n^0.75": fit_ridge_unnormalized(X, y, n ** 0.75).coefficient,
        "ridge_lambda_n": fit_ridge_unnormalized(X, y, float(n)).coefficient,
        "pcr_top_k": _pcr_top_k(X, y, k),
    }
    row = {"n": n, "seed": seed, "dimension": dimension, "k": k}
    for name, coefficient in estimates.items():
        delta = coefficient - beta
        population = float(delta @ delta)
        prediction = X_test @ coefficient
        empirical = float(np.mean((y_test - prediction) ** 2 - (y_test - signal_test) ** 2))
        row[f"{name}_population_excess"] = population
        row[f"{name}_monte_carlo_excess"] = empirical
    return row


def _ols_slope(n_values: np.ndarray, risk: np.ndarray) -> float:
    if np.any(risk <= 0) or not np.isfinite(risk).all():
        raise RuntimeError("cannot fit a log-risk slope to nonpositive or nonfinite means")
    x = np.log(np.asarray(n_values, dtype=np.float64))
    y = np.log(np.asarray(risk, dtype=np.float64))
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    return float(np.dot(x_centered, y_centered) / np.dot(x_centered, x_centered))


def _limit_threads() -> None:
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=1)
    except Exception:
        return


def run_l1(config: dict, run_dir: Path) -> dict:
    _limit_threads()
    run_dir = Path(run_dir)
    if (run_dir / "replicates.csv").exists():
        raise FileExistsError(f"refusing to overwrite {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    names = [
        "ridgeless_lambda_1e-8",
        "ridge_lambda_n^0.5",
        "ridge_lambda_n^0.75",
        "ridge_lambda_n",
        "pcr_top_k",
    ]
    grid = n_grid_from_config(config)
    rows = []
    failures = []
    n_runs = int(config["n_runs"])
    for n in grid:
        for run_index in range(n_runs):
            seed = int(config["seed_base"]) + run_index * int(config["seed_stride"]) + int(n)
            try:
                rows.append(_one_replicate(n, seed, config))
            except Exception as exc:
                failures.append({"n": n, "run_index": run_index, "seed": seed, "error": type(exc).__name__, "message": str(exc)})
    fieldnames = ["n", "seed", "dimension", "k"]
    for name in names:
        fieldnames.append(f"{name}_population_excess")
        fieldnames.append(f"{name}_monte_carlo_excess")
    with (run_dir / "replicates.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary_rows = []
    slopes = {}
    for name in names:
        slope = None
        slope_error = None
        try:
            n_values = []
            means = []
            ses = []
            for n in grid:
                sample = [
                    float(row[f"{name}_population_excess"])
                    for row in rows
                    if row["n"] == n
                ]
                if len(sample) != n_runs:
                    raise RuntimeError(f"{name} n={n} has {len(sample)} replicates, expected {n_runs}")
                array = np.asarray(sample, dtype=np.float64)
                n_values.append(n)
                means.append(float(np.mean(array)))
                ses.append(float(np.std(array, ddof=1) / math.sqrt(array.size)))
            slope = _ols_slope(np.asarray(n_values), np.asarray(means))
            for n, mean, se in zip(n_values, means, ses):
                summary_rows.append({"estimator": name, "n": n, "mean_population_excess": mean, "se_population_excess": se, "slope": slope})
        except Exception as exc:
            slope_error = f"{type(exc).__name__}: {exc}"
            failures.append({"n": None, "run_index": None, "seed": None, "error": "slope", "message": f"{name}: {slope_error}"})
        slopes[name] = {"slope": slope, "error": slope_error}

    bands = config["acceptance_bands"]
    pcr_slope = slopes["pcr_top_k"]["slope"]
    ridge_slope = slopes["ridge_lambda_n^0.75"]["slope"]
    reasons = []
    if failures:
        reasons.append(f"{len(failures)} failure records")
    if pcr_slope is None or ridge_slope is None:
        reasons.append("a required slope is missing")
    else:
        lo, hi = bands["pcr_slope_interval"]
        if not (lo <= pcr_slope <= hi):
            reasons.append(f"PCR slope {pcr_slope} outside [{lo}, {hi}]")
        lo, hi = bands["ridge_n075_slope_interval"]
        if not (lo <= ridge_slope <= hi):
            reasons.append(f"ridge lambda=n^0.75 slope {ridge_slope} outside [{lo}, {hi}]")
        if pcr_slope - ridge_slope > float(bands["pcr_minus_ridge_n075_max"]):
            reasons.append("PCR slope is not more negative than the lambda=n^0.75 ridge slope by the predeclared margin")
    matched = not reasons
    result = {
        "n_grid": grid,
        "slopes": slopes,
        "qualitative_band_match": matched,
        "band_reasons": reasons,
        "claim": "qualitative reproduction" if matched else "partial reproduction",
        "deviations": L1_DEVIATIONS,
        "n_replicates_written": len(rows),
        "n_failure_records": len(failures),
    }
    (run_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (run_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["estimator", "n", "mean_population_excess", "se_population_excess", "slope"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    _write_figure(run_dir / "figure1c_population_excess.png", summary_rows, names)
    with (run_dir / "failures.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["n", "run_index", "seed", "error", "message"])
        writer.writeheader()
        writer.writerows(failures)
    result["n_failure_records"] = len(failures)
    (run_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _write_figure(path: Path, summary_rows: list[dict], names: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    for name in names:
        rows = [row for row in summary_rows if row["estimator"] == name]
        if not rows:
            continue
        n_values = np.asarray([row["n"] for row in rows], dtype=np.float64)
        means = np.asarray([row["mean_population_excess"] for row in rows], dtype=np.float64)
        ses = np.asarray([row["se_population_excess"] for row in rows], dtype=np.float64)
        if np.any(means <= 0):
            continue
        axis.errorbar(n_values, means, yerr=ses, marker="o", label=name)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("n")
    axis.set_ylabel("population excess risk")
    axis.set_title("Tang et al. 2025 Figure 1(c) qualitative run")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
