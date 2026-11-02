"""Task 8: map manuscript placeholders to computed post-hoc values."""

from __future__ import annotations

import csv

from analysis.posthoc.common import PROJECT_ROOT, RESULTS, ensure_results, print_map, write_markdown


def _rows(name: str) -> list[dict]:
    path = RESULTS / name
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict, key: str) -> float:
    return float(row[key])


def _median_log10(rows: list[dict], sample_size: int) -> float:
    values = sorted(float(row["log10_lambda"]) for row in rows if int(row["sample_size"]) == sample_size)
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return 0.5 * (values[mid - 1] + values[mid])


def _lambda_sentence(year_rows: list[dict], super_rows: list[dict]) -> str:
    parts = []
    for label, rows, left, right in (
        ("YearPredictionMSD", year_rows, 90, 92),
        ("Superconductivity", super_rows, 81, 83),
    ):
        if not rows:
            continue
        parts.append(
            f"{label} median log10(λ) moves from {_median_log10(rows, left):.3f} at n={left} to {_median_log10(rows, right):.3f} at n={right}"
        )
    if not parts:
        return "Lambda summaries were not available."
    return "Near the first archived size above p, " + "; ".join(parts) + "."


def main() -> None:
    print_map()
    ensure_results()
    regen = _rows("regeneration_check.csv")
    spectra = _rows("spectra_Superconductivity.csv")
    correlations = _rows("spectra_correlations.csv")
    pool = _rows("pool_spectrum.csv")
    nulls = _rows("null_baseline.csv")
    wins = _rows("win_counts.csv")
    hyper_year = _rows("hyperparams_YearPredictionMSD.csv")
    hyper_super = _rows("hyperparams_Superconductivity.csv")
    focus = next((row for row in spectra if row["repetition"] == "5" and row["sample_size"] == "73"), None)
    rho = next(
        (
            row
            for row in correlations
            if row["dataset"] == "Superconductivity" and row["sample_size"] == "73"
        ),
        None,
    )
    x_value = max((_float(row, "relative_error") for row in regen), default=None)
    lines = [
        "# Post-hoc summary",
        "",
        "`P19_revisions.md` and the P19 LaTeX manuscript are not in this repository. Placeholder values are recorded here. Direct replacements, bibliography edits, the wording change from “numerical instability” to “ill-conditioning”, and PDF compilation were not applied.",
        "",
        "Primary endpoint tables were not edited.",
        "",
        "## Placeholder map",
        "",
    ]
    lines.append(f"- ⟨X⟩ maximum relative regeneration error, all checked cells and estimators: {x_value}")
    if focus:
        lines.append(f"- ⟨s_min⟩ Superconductivity rep 5, n=73: {focus['s_min']}")
        lines.append(f"- ⟨τ⟩: {focus['tau']}")
        lines.append(f"- ⟨s₁⟩: {focus['s1']}")
        lines.append(
            f"- ⟨Y⟩ percent of ‖β̂‖² in the smallest retained direction: {100 * float(focus['frac_energy_smallest_retained'])}"
        )
    if rho:
        lines.append(
            f"- ⟨ρ⟩ Pearson correlation of log10 MSE with log10(1/s_min) at Superconductivity n=73: "
            f"{rho.get('pearson_log10_mse_vs_log10_inv_smin')}"
        )
        lines.append(
            f"- Spearman correlation at the same cell: {rho.get('spearman_log10_mse_vs_log10_inv_smin')}"
        )
    if pool:
        lines.append("- Eigenvalue counts below c × λ_max:")
        parsed = []
        for row in pool:
            counts = {
                "1e-2": int(row["below_1e-02_lambda_max"]),
                "1e-4": int(row["below_1e-04_lambda_max"]),
                "1e-6": int(row["below_1e-06_lambda_max"]),
                "1e-8": int(row["below_1e-08_lambda_max"]),
            }
            parsed.append((row["dataset"], counts, row))
            lines.append(
                f"  - {row['dataset']}: 1e-2 → {counts['1e-2']}, "
                f"1e-4 → {counts['1e-4']}, "
                f"1e-6 → {counts['1e-6']}, "
                f"1e-8 → {counts['1e-8']}, "
                f"numerical rank {row['numerical_rank']} of {row['n_cols']}, "
                f"duplicated columns {row['n_exactly_duplicated_columns']}, "
                f"zero-variance columns {row['n_zero_variance_columns']}."
            )
        by_name = {name: counts for name, counts, _row in parsed}
        if "YearPredictionMSD" in by_name and "Superconductivity" in by_name:
            best = max(
                ("1e-2", "1e-4", "1e-6", "1e-8"),
                key=lambda key: abs(by_name["YearPredictionMSD"][key] - by_name["Superconductivity"][key]),
            )
            lines.append(
                f"- ⟨k⟩ YearPredictionMSD eigenvalues below {best} × λ_max: {by_name['YearPredictionMSD'][best]}."
            )
            lines.append(
                f"- ⟨k′⟩ Superconductivity eigenvalues below {best} × λ_max: {by_name['Superconductivity'][best]}."
            )
            lines.append(f"- Chosen threshold: {best}, the cutoff with the largest absolute count difference.")
    lines.append("- ⟨Z⟩ extended-precision relative change: not computed. mpmath is not installed and numpy.longdouble is float64 on this Windows build.")
    win = next((row for row in wins if row["dataset"] == "Superconductivity" and row["sample_size"] == "405"), None)
    if win:
        lines.append(
            f"- ⟨w⟩ Superconductivity n=405 wins: minimum-norm {win['min_norm_wins']}, ridge {win['ridge_wins']}, ties {win['ties']}."
        )
    for sample_size, token in (("45", "⟨a⟩"), ("900", "⟨b⟩")):
        row = next(
            (
                item
                for item in nulls
                if item["dataset"] == "YearPredictionMSD" and item["sample_size"] == sample_size
            ),
            None,
        )
        if row:
            lines.append(
                f"- {token} YearPredictionMSD ridge mean improvement over the null at n={sample_size}: "
                f"{100 * float(row['ridge_mean_improvement_over_null'])} percent. "
                f"Null mean MSE {row['null_mean_mse']}, ridge mean MSE {row['ridge_mean_mse']}."
            )
    for dataset, token in (("YearPredictionMSD", "⟨v_Y⟩"), ("Superconductivity", "⟨v_S⟩")):
        row = next((item for item in nulls if item["dataset"] == dataset), None)
        if row:
            lines.append(f"- {token} holdout target variance, ddof=0: {row['holdout_target_variance_ddof0']}")
    for label, rows in (("⟨m⟩ YearPredictionMSD", hyper_year), ("⟨m′⟩ Superconductivity", hyper_super)):
        if not rows:
            continue
        lower = sum(row["lambda_at_lower_bound"] == "True" for row in rows)
        upper = sum(row["lambda_at_upper_bound"] == "True" for row in rows)
        lines.append(f"- {label} cells at λ = 1e-8: {lower}; cells at λ = 1e4: {upper}; total cells {len(rows)}.")
    lines.append("")
    lines.append(_lambda_sentence(hyper_year, hyper_super))
    lines.append("")
    lines.append("Primary Tables 2, 4, 5, and 6 were not edited. No LaTeX compilation was run. No confirmatory artifact was modified.")
    write_markdown(RESULTS / "SUMMARY.md", "\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
