"""Task 0: refit archived hyperparameters and compare holdout MSE."""

from __future__ import annotations

from analysis.posthoc.common import (
    ESTIMATORS,
    RESULTS,
    ensure_results,
    load_archived,
    load_super,
    load_year,
    print_map,
    refit_cell,
    relative_error,
    super_prefix,
    write_csv,
    write_markdown,
    year_prefix,
)

STOP_RELATIVE = 1e-6


def _cells(archived: dict) -> list[tuple[str, int, int, str]]:
    year_losses = [
        (rep, archived[("YearPredictionMSD", rep, 90, "min_norm")]["mse"])
        for rep in range(100)
    ]
    year_worst = max(year_losses, key=lambda item: item[1])[0]
    return [
        ("YearPredictionMSD", 0, 45, "smallest n"),
        ("YearPredictionMSD", 0, 90, "n near p"),
        ("YearPredictionMSD", year_worst, 90, "repetition with maximum minimum-norm loss at n=90"),
        ("YearPredictionMSD", 0, 900, "largest n"),
        ("Superconductivity", 0, 41, "smallest n"),
        ("Superconductivity", 0, 81, "n near p"),
        ("Superconductivity", 5, 73, "repetition 5 at n=73"),
        ("Superconductivity", 0, 810, "largest n"),
    ]


def main() -> int:
    print_map()
    ensure_results()
    archived = load_archived()
    year = load_year()
    super_data = load_super()
    rows = []
    worst = {name: 0.0 for name in ESTIMATORS}
    for dataset, repetition, sample_size, reason in _cells(archived):
        if dataset == "YearPredictionMSD":
            prefix = year_prefix(repetition, sample_size)
            x_train = year["X_source"][prefix]
            y_train = year["y_source"][prefix]
            x_holdout = year["X_holdout"]
            y_holdout = year["y_holdout"]
        else:
            prefix = super_prefix(repetition, sample_size, super_data["train_rows"])
            x_train = super_data["X"][prefix]
            y_train = super_data["y"][prefix]
            holdout = super_data["holdout_rows"]
            x_holdout = super_data["X"][holdout]
            y_holdout = super_data["y"][holdout]
        ridge_lambda = archived[(dataset, repetition, sample_size, "ridge")]["lambda"]
        pcr_fraction = archived[(dataset, repetition, sample_size, "pcr")]["pcr_fraction"]
        recomputed = refit_cell(x_train, y_train, x_holdout, y_holdout, ridge_lambda, pcr_fraction)
        for estimator in ESTIMATORS:
            archived_mse = archived[(dataset, repetition, sample_size, estimator)]["mse"]
            error = relative_error(recomputed[estimator], archived_mse)
            worst[estimator] = max(worst[estimator], error)
            rows.append(
                {
                    "dataset": dataset,
                    "repetition": repetition,
                    "sample_size": sample_size,
                    "reason": reason,
                    "estimator": estimator,
                    "archived_mse": archived_mse,
                    "recomputed_mse": recomputed[estimator],
                    "relative_error": error,
                }
            )
            print(dataset, repetition, sample_size, estimator, error, flush=True)
    write_csv(RESULTS / "regeneration_check.csv", rows)
    stopped = worst["min_norm"] > STOP_RELATIVE
    lines = [
        "# Regeneration sanity check",
        "",
        "Refits use the archived ridge lambda and PCR fraction. No hyperparameter was retuned.",
        "",
        "| Estimator | Maximum relative error |",
        "|---|---:|",
    ]
    for estimator in ESTIMATORS:
        lines.append(f"| {estimator} | {worst[estimator]:.3e} |")
    lines.append("")
    if stopped:
        lines.append(
            f"Minimum-norm relative error exceeds {STOP_RELATIVE:.0e}. "
            "Tasks 1–3 are not run until this discrepancy is resolved."
        )
    else:
        lines.append(
            f"Every minimum-norm comparison is within relative error {STOP_RELATIVE:.0e}. "
            "Tasks 1–3 may use the same prefix reconstruction."
        )
    write_markdown(RESULTS / "regeneration_check.md", "\n".join(lines) + "\n")
    print("MAX", worst, "STOP", stopped)
    return 2 if stopped else 0


if __name__ == "__main__":
    raise SystemExit(main())
