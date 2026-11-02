"""Task 5: archived tuning choices and inner-fold sizes."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from p19.l2_sampling import N_FOLDS, longest_prefix_folds, restrict_nested_folds
from p19.protocol import ridge_lambda_grid

from analysis.posthoc.common import (
    RESULTS,
    SUPER_P,
    SUPER_SIZES,
    YEAR_P,
    YEAR_SIZES,
    ensure_results,
    load_archived,
    print_map,
    staging_design,
    write_csv,
    write_markdown,
)


def _fold_rows() -> list[dict]:
    rows = []
    year_folds = longest_prefix_folds()
    for sample_size in YEAR_SIZES:
        sizes = [int(fold.size) for fold in restrict_nested_folds(year_folds, sample_size)]
        rows.append(
            {
                "dataset": "YearPredictionMSD",
                "sample_size": sample_size,
                "min_fold_size": min(sizes),
                "max_fold_size": max(sizes),
                "n_folds_below_5": sum(size < 5 for size in sizes),
                "n_empty_folds": sum(size == 0 for size in sizes),
                "same_for_every_repetition": True,
            }
        )
    design = staging_design()
    membership = design.length_810_fold_membership()
    for sample_size in SUPER_SIZES:
        sizes = [int(fold.size) for fold in design.restrict_length_810_folds(membership, sample_size)]
        rows.append(
            {
                "dataset": "Superconductivity",
                "sample_size": sample_size,
                "min_fold_size": min(sizes),
                "max_fold_size": max(sizes),
                "n_folds_below_5": sum(size < 5 for size in sizes),
                "n_empty_folds": sum(size == 0 for size in sizes),
                "same_for_every_repetition": True,
            }
        )
    if N_FOLDS != 5:
        raise RuntimeError("fold count drifted from 5")
    return rows


def main() -> None:
    print_map()
    ensure_results()
    archived = load_archived()
    lower = float(ridge_lambda_grid()[0])
    upper = float(ridge_lambda_grid()[-1])
    for dataset, sizes in (("YearPredictionMSD", YEAR_SIZES), ("Superconductivity", SUPER_SIZES)):
        rows = []
        for sample_size in sizes:
            for repetition in range(100):
                ridge = archived[(dataset, repetition, sample_size, "ridge")]
                pcr = archived[(dataset, repetition, sample_size, "pcr")]
                rank = pcr["numerical_rank"]
                components = pcr["pcr_components"]
                rows.append(
                    {
                        "dataset": dataset,
                        "repetition": repetition,
                        "sample_size": sample_size,
                        "sampling_seed": ridge["sampling_seed"],
                        "ridge_lambda": ridge["lambda"],
                        "log10_lambda": float(np.log10(ridge["lambda"])),
                        "pcr_fraction": pcr["pcr_fraction"],
                        "pcr_components": components,
                        "numerical_rank": rank if rank is not None else "",
                        "k_over_r": "" if not rank else float(components) / float(rank),
                        "lambda_at_lower_bound": bool(np.isclose(ridge["lambda"], lower, rtol=1e-9, atol=0.0)),
                        "lambda_at_upper_bound": bool(np.isclose(ridge["lambda"], upper, rtol=1e-9, atol=0.0)),
                    }
                )
        write_csv(RESULTS / f"hyperparams_{dataset}.csv", rows)
    fold_rows = _fold_rows()
    write_csv(RESULTS / "fold_sizes.csv", fold_rows)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for column, (dataset, sizes, p) in enumerate(
        (("YearPredictionMSD", YEAR_SIZES, YEAR_P), ("Superconductivity", SUPER_SIZES, SUPER_P))
    ):
        import csv
        from pathlib import Path

        table = list(csv.DictReader((RESULTS / f"hyperparams_{dataset}.csv").open(encoding="utf-8")))
        lambdas = [
            [float(row["log10_lambda"]) for row in table if int(row["sample_size"]) == sample_size]
            for sample_size in sizes
        ]
        fractions = [
            [float(row["pcr_fraction"]) for row in table if int(row["sample_size"]) == sample_size]
            for sample_size in sizes
        ]
        axes[0, column].boxplot(lambdas, tick_labels=[str(size) for size in sizes], showfliers=False)
        axes[1, column].boxplot(fractions, tick_labels=[str(size) for size in sizes], showfliers=False)
        for ax in (axes[0, column], axes[1, column]):
            boundary = sizes.index(max(size for size in sizes if size <= p)) + 1.5
            ax.axvline(boundary, color="#444444", ls="--", lw=0.8)
            ax.tick_params(axis="x", labelrotation=60, labelsize=7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        axes[0, column].set_title(dataset)
        axes[0, column].set_ylabel(r"$\log_{10}\lambda$")
        axes[1, column].set_ylabel("PCR fraction q")
        axes[1, column].set_xlabel("n")
    fig.savefig(RESULTS / "fig_S_hyperparams.pdf")
    plt.close(fig)
    lines = ["# Selected hyperparameters and fold sizes", "", "The dashed line is immediately before the first archived n > p.", ""]
    for row in fold_rows:
        lines.append(
            f"- {row['dataset']} n={row['sample_size']}: fold sizes {row['min_fold_size']}–{row['max_fold_size']}, "
            f"below 5: {row['n_folds_below_5']}, empty: {row['n_empty_folds']}."
        )
    write_markdown(RESULTS / "hyperparams.md", "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
