"""Tasks 6 and 7: audit displayed numbers and the exploratory sign/log checks."""

from __future__ import annotations

import csv

import numpy as np
from scipy.stats import binomtest

from p19.uncertainty import summarize_complete_replicates

from analysis.posthoc.common import (
    AUDIT_DIR,
    RESULTS,
    SUPER_SIZES,
    YEAR_SIZES,
    ensure_results,
    load_archived,
    print_map,
    write_csv,
    write_markdown,
)

SUPER_ZERO_SIZES = (41, 61, 73, 77, 79, 81, 83, 85)


def _read(name: str) -> list[dict]:
    with (AUDIT_DIR / name).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _wins(archived: dict, dataset: str, sample_size: int) -> tuple[int, int, int]:
    mn = ridge = ties = 0
    for repetition in range(100):
        left = archived[(dataset, repetition, sample_size, "min_norm")]["mse"]
        right = archived[(dataset, repetition, sample_size, "ridge")]["mse"]
        if left < right:
            mn += 1
        elif right < left:
            ridge += 1
        else:
            ties += 1
    return mn, ridge, ties


def main() -> None:
    print_map()
    ensure_results()
    archived = load_archived()
    comparison = _read("STAGE5N8_CROSS_BENCHMARK_COMPARISON.csv")
    paired = _read("STAGE5N8_PAIRED_CONTRAST_AUDIT.csv")
    lines = ["# Number audit against the Stage 5N-8 CSVs", ""]
    lines.append("The manuscript LaTeX and `P19_revisions.md` are not in this repository, so Tables 2, 4, 5, and 6 cannot be diffed against a `.tex` source. The checks below use the audited CSVs and the official repetition records.")
    lines.append("")
    means = {}
    for row in comparison:
        if row["estimator"] == "min_norm" and row["benchmark"] == "Superconductivity" and int(row["sample_size"]) in {79, 81}:
            means[int(row["sample_size"])] = float(row["mean_mse"])
    argmax = {}
    for sample_size in (79, 81):
        losses = [(rep, archived[("Superconductivity", rep, sample_size, "min_norm")]["mse"]) for rep in range(100)]
        argmax[sample_size] = max(losses, key=lambda item: item[1])
    lines.append("## Superconductivity minimum-norm means at n=79 and n=81")
    lines.append("")
    lines.append(f"- Full-precision mean at n=79: {means[79]:.17g}")
    lines.append(f"- Full-precision mean at n=81: {means[81]:.17g}")
    lines.append(f"- Relative difference: {abs(means[79] - means[81]) / max(means[79], means[81]):.6e}")
    lines.append(f"- Argmax repetition at n=79: {argmax[79][0]}, MSE {argmax[79][1]:.17g}")
    lines.append(f"- Argmax repetition at n=81: {argmax[81][0]}, MSE {argmax[81][1]:.17g}")
    lines.append("- Displaying both means as 2.698e17 is rounding at three significant figures. The stored means are not the same number, and the argmax repetitions differ. This is not a copied cell.")
    lines.append("")
    lines.append("## PCR minus minimum-norm intervals that exclude zero")
    lines.append("")
    for dataset in ("YearPredictionMSD", "Superconductivity"):
        for row in paired:
            if row["benchmark"] == dataset and row["contrast"] == "pcr_minus_min_norm" and row["excludes_zero"] == "True":
                lines.append(f"- {dataset} n={row['sample_size']}: mean {float(row['mean_difference']):.6g}, {row['direction']}")
    lines.append("")
    lines.append("## Minimum-norm versus ridge repetition wins")
    lines.append("")
    win_rows = []
    for dataset, sizes in (("YearPredictionMSD", YEAR_SIZES), ("Superconductivity", SUPER_SIZES)):
        for sample_size in sizes:
            mn, ridge, ties = _wins(archived, dataset, sample_size)
            win_rows.append(
                {
                    "dataset": dataset,
                    "sample_size": sample_size,
                    "min_norm_wins": mn,
                    "ridge_wins": ridge,
                    "ties": ties,
                }
            )
            lines.append(f"- {dataset} n={sample_size}: minimum-norm {mn}, ridge {ridge}, ties {ties}")
    write_csv(RESULTS / "win_counts.csv", win_rows)
    ratios = [
        float(row["relative_mean_mse_versus_ridge"])
        for row in comparison
        if row["benchmark"] == "Superconductivity" and row["estimator"] == "pcr"
    ]
    lines.extend(
        [
            "",
            "## Superconductivity PCR / ridge mean ratio",
            "",
            f"- Minimum across 15 sizes: {min(ratios):.3f}",
            f"- Maximum across 15 sizes: {max(ratios):.3f}",
        ]
    )
    # Locate which n.
    by_n = {
        int(row["sample_size"]): float(row["relative_mean_mse_versus_ridge"])
        for row in comparison
        if row["benchmark"] == "Superconductivity" and row["estimator"] == "pcr"
    }
    lines.append(f"- n=810: {by_n[810]:.3f}")
    lines.append(f"- n=85: {by_n[85]:.3f}")
    lines.append(f"- n at minimum: {min(by_n, key=by_n.get)}")
    lines.append(f"- n at maximum: {max(by_n, key=by_n.get)}")
    write_markdown(RESULTS / "number_audit.md", "\n".join(lines) + "\n")

    exploratory = []
    for sample_size in SUPER_ZERO_SIZES:
        difference = np.array(
            [
                archived[("Superconductivity", rep, sample_size, "min_norm")]["mse"]
                - archived[("Superconductivity", rep, sample_size, "ridge")]["mse"]
                for rep in range(100)
            ],
            dtype=np.float64,
        )
        positive = int(np.count_nonzero(difference > 0))
        log_difference = np.log10(
            np.array([archived[("Superconductivity", rep, sample_size, "min_norm")]["mse"] for rep in range(100)], dtype=np.float64)
        ) - np.log10(
            np.array([archived[("Superconductivity", rep, sample_size, "ridge")]["mse"] for rep in range(100)], dtype=np.float64)
        )
        summary = summarize_complete_replicates(log_difference)
        exploratory.append(
            {
                "dataset": "Superconductivity",
                "sample_size": sample_size,
                "contrast": "log10(min_norm MSE) - log10(ridge MSE)",
                "label": "Exploratory (not prespecified)",
                "min_norm_higher_count": positive,
                "sign_test_two_sided_p": float(binomtest(positive, 100, 0.5, alternative="two-sided").pvalue),
                "log_difference_mean": summary["mean"],
                "log_difference_low": summary["interval_low"],
                "log_difference_high": summary["interval_high"],
                "degrees_of_freedom": 99,
                "interval": "pointwise",
            }
        )
    write_csv(RESULTS / "exploratory_sign_log.csv", exploratory)
    text = ["# Exploratory (not prespecified)", "", "These checks are not part of the primary endpoint and do not replace any primary table.", ""]
    for row in exploratory:
        text.append(
            f"- n={row['sample_size']}: sign-test p={row['sign_test_two_sided_p']:.3e}, "
            f"mean log10 difference {row['log_difference_mean']:.4f} "
            f"[{row['log_difference_low']:.4f}, {row['log_difference_high']:.4f}]"
        )
    write_markdown(RESULTS / "exploratory.md", "\n".join(text) + "\n")


if __name__ == "__main__":
    main()
