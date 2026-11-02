"""Task 4: training-prefix-mean null predictor."""

from __future__ import annotations

import numpy as np
from scipy.stats import t as student_t

from p19.metrics import mse
from p19.uncertainty import summarize_complete_replicates

from analysis.posthoc.common import (
    RESULTS,
    SUPER_SIZES,
    YEAR_SIZES,
    ensure_results,
    load_archived,
    load_super,
    load_year,
    print_map,
    super_prefix,
    write_csv,
    write_markdown,
    year_prefix,
)


def _null_cube(dataset: str) -> tuple[np.ndarray, float]:
    if dataset == "YearPredictionMSD":
        data = load_year()
        y_source = data["y_source"]
        y_holdout = data["y_holdout"]
        sizes = YEAR_SIZES
        prefix_fn = lambda repetition, sample_size: year_prefix(repetition, sample_size)
        select = lambda prefix: y_source[prefix]
    else:
        data = load_super()
        y_all = data["y"]
        y_holdout = y_all[data["holdout_rows"]]
        sizes = SUPER_SIZES
        prefix_fn = lambda repetition, sample_size: super_prefix(repetition, sample_size, data["train_rows"])
        select = lambda prefix: y_all[prefix]
    losses = np.empty((len(sizes), 100), dtype=np.float64)
    for repetition in range(100):
        # One permutation, then every nested prefix.
        longest = prefix_fn(repetition, sizes[-1])
        for index, sample_size in enumerate(sizes):
            losses[index, repetition] = mse(y_holdout, np.full(y_holdout.shape, float(np.mean(select(longest[:sample_size])))))
    variance = float(np.var(y_holdout, ddof=0))
    return losses, variance


def main() -> None:
    print_map()
    ensure_results()
    archived = load_archived()
    rows = []
    summaries = []
    t_critical = float(student_t.ppf(0.975, 99))
    for dataset, sizes in (("YearPredictionMSD", YEAR_SIZES), ("Superconductivity", SUPER_SIZES)):
        losses, variance = _null_cube(dataset)
        print(dataset, "holdout variance", variance, flush=True)
        for index, sample_size in enumerate(sizes):
            null_values = losses[index]
            ridge_values = np.array(
                [archived[(dataset, repetition, sample_size, "ridge")]["mse"] for repetition in range(100)],
                dtype=np.float64,
            )
            paired = summarize_complete_replicates(ridge_values - null_values)
            improvement = float((np.mean(null_values) - np.mean(ridge_values)) / np.mean(null_values))
            row = {
                "dataset": dataset,
                "sample_size": sample_size,
                "null_mean_mse": float(np.mean(null_values)),
                "null_median_mse": float(np.median(null_values)),
                "ridge_mean_mse": float(np.mean(ridge_values)),
                "holdout_target_variance_ddof0": variance,
                "ridge_minus_null_mean": paired["mean"],
                "ridge_minus_null_sd": paired["sample_sd"],
                "ridge_minus_null_se": paired["standard_error"],
                "ridge_minus_null_low": paired["interval_low"],
                "ridge_minus_null_high": paired["interval_high"],
                "degrees_of_freedom": 99,
                "t_critical": t_critical,
                "interval": "pointwise",
                "ridge_mean_improvement_over_null": improvement,
            }
            rows.append(row)
            for repetition in range(100):
                summaries.append(
                    {
                        "dataset": dataset,
                        "repetition": repetition,
                        "sample_size": sample_size,
                        "null_mse": float(null_values[repetition]),
                        "ridge_mse": float(ridge_values[repetition]),
                    }
                )
    write_csv(RESULTS / "null_baseline.csv", rows)
    write_csv(RESULTS / "null_baseline_repetitions.csv", summaries)
    lines = [
        "# Null baseline",
        "",
        "The null predictor is the training-prefix mean. The paired interval is ridge MSE minus null MSE, pointwise Student-t, df=99.",
        "",
        "| Dataset | n | Null mean | Ridge mean | Improvement | 95% interval for ridge − null |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        if row["sample_size"] in {45, 90, 900, 41, 81, 810}:
            lines.append(
                f"| {row['dataset']} | {row['sample_size']} | {row['null_mean_mse']:.4g} | {row['ridge_mean_mse']:.4g} | "
                f"{100 * row['ridge_mean_improvement_over_null']:.2f}% | "
                f"[{row['ridge_minus_null_low']:.4g}, {row['ridge_minus_null_high']:.4g}] |"
            )
    write_markdown(RESULTS / "null_baseline.md", "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
