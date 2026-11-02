"""Task 1: repetition-level singular spectra of the minimum-norm training matrix."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

from p19.estimators import economy_svd, fit_min_norm, numerical_cutoff
from p19.preprocessing import fit_preprocessor
from p19.protocol import DRAFT_ZERO_VARIANCE

from analysis.posthoc.common import (
    RESULTS,
    SUPER_SIZES,
    YEAR_SIZES,
    ensure_results,
    load_archived,
    load_super,
    load_year,
    max_abs_prediction,
    print_map,
    super_prefix,
    write_csv,
    write_markdown,
    year_seeds,
)
from p19.l2_sampling import source_permutation

SUPER_N = (41, 61, 73, 77, 79, 81, 83, 85, 89)
YEAR_N = (81, 86, 88, 90, 92, 95, 99)


def _unique_rows(matrix: np.ndarray) -> int:
    contiguous = np.ascontiguousarray(matrix, dtype=np.float64)
    view = contiguous.view(np.dtype((np.void, contiguous.dtype.itemsize * contiguous.shape[1]))).reshape(-1)
    return int(np.unique(view).size)


def _duplicate_columns(matrix: np.ndarray) -> int:
    transposed = np.ascontiguousarray(matrix.T, dtype=np.float64)
    view = transposed.view(np.dtype((np.void, transposed.dtype.itemsize * transposed.shape[1]))).reshape(-1)
    _values, counts = np.unique(view, return_counts=True)
    return int(np.sum(counts[counts > 1]))


def _cell(x_raw: np.ndarray, y_raw: np.ndarray) -> dict:
    preprocessor = fit_preprocessor(x_raw, y_raw, DRAFT_ZERO_VARIANCE)
    design = preprocessor.transform_X(x_raw)
    target = preprocessor.transform_y(y_raw)
    left, singular, _right = economy_svd(design)
    n_rows, n_cols = design.shape
    cutoff = numerical_cutoff(singular, n_rows, n_cols)
    retained = singular > cutoff
    retained_values = singular[retained]
    fit = fit_min_norm(design, target)
    coeff = np.zeros(singular.shape, dtype=np.float64)
    if np.any(retained):
        coeff[retained] = (left[:, retained].T @ target) / singular[retained]
    energy = coeff ** 2
    total = float(np.sum(energy))
    ones = np.ones(n_rows, dtype=np.float64) / np.sqrt(n_rows)
    truncated = np.flatnonzero(~retained)
    alignment = 0.0
    if truncated.size:
        alignments = np.abs(left[:, truncated].T @ ones)
        alignment = float(np.max(alignments))
    smallest_share = 0.0
    if total > 0 and retained_values.size:
        smallest_index = int(np.flatnonzero(retained)[np.argmin(retained_values)])
        smallest_share = float(energy[smallest_index] / total)
    return {
        "s1": float(singular[0]) if singular.size else 0.0,
        "tau": float(cutoff),
        "n_truncated": int(truncated.size),
        "s_min": float(np.min(retained_values)) if retained_values.size else float("nan"),
        "condition_number": float(singular[0] / np.min(retained_values)) if retained_values.size else float("nan"),
        "frac_energy_largest_direction": float(np.max(energy) / total) if total > 0 else float("nan"),
        "frac_energy_smallest_retained": smallest_share,
        "centering_alignment": alignment,
        "centering_claim": bool(n_rows <= n_cols and truncated.size == 1 and alignment > 0.999),
        "n_unique_rows": _unique_rows(x_raw),
        "n_duplicated_columns": _duplicate_columns(x_raw),
        "n_zero_variance_columns": int(np.count_nonzero(preprocessor.zero_variance_columns)),
        "numerical_rank": int(fit.numerical_rank or 0),
    }


def _correlations(rows: list[dict], sizes: tuple[int, ...], dataset: str) -> list[dict]:
    output = []
    for sample_size in sizes:
        subset = [row for row in rows if row["sample_size"] == sample_size and np.isfinite(row["s_min"]) and row["s_min"] > 0]
        mse = np.array([row["archived_mse"] for row in subset], dtype=np.float64)
        inverse = np.array([1.0 / row["s_min"] for row in subset], dtype=np.float64)
        condition = np.array([row["condition_number"] for row in subset], dtype=np.float64)
        valid = (mse > 0) & (inverse > 0) & (condition > 0)
        record = {"dataset": dataset, "sample_size": sample_size, "n": int(np.count_nonzero(valid))}
        if int(np.count_nonzero(valid)) >= 3:
            log_mse = np.log10(mse[valid])
            log_inverse = np.log10(inverse[valid])
            log_condition = np.log10(condition[valid])
            record["pearson_log10_mse_vs_log10_inv_smin"] = float(pearsonr(log_mse, log_inverse).statistic)
            record["spearman_log10_mse_vs_log10_inv_smin"] = float(spearmanr(log_mse, log_inverse).statistic)
            record["pearson_log10_mse_vs_log10_condition"] = float(pearsonr(log_mse, log_condition).statistic)
            record["spearman_log10_mse_vs_log10_condition"] = float(spearmanr(log_mse, log_condition).statistic)
        output.append(record)
    return output


def _run_dataset(dataset: str, sizes: tuple[int, ...], archived: dict, predictions: dict) -> list[dict]:
    if dataset == "YearPredictionMSD":
        data = load_year()
    else:
        data = load_super()
    rows = []
    year_orders: dict[int, np.ndarray] = {}
    seeds = year_seeds() if dataset == "YearPredictionMSD" else None
    for sample_size in sizes:
        for repetition in range(100):
            record = archived[(dataset, repetition, sample_size, "min_norm")]
            if dataset == "YearPredictionMSD":
                if repetition not in year_orders:
                    year_orders[repetition] = source_permutation(seeds[repetition])
                prefix = year_orders[repetition][:sample_size]
                raw_x = data["X_source"][prefix]
                raw_y = data["y_source"][prefix]
            else:
                prefix = super_prefix(repetition, sample_size, data["train_rows"])
                raw_x = data["X"][prefix]
                raw_y = data["y"][prefix]
            spectral = _cell(raw_x, raw_y)
            spectral.update(
                {
                    "dataset": dataset,
                    "repetition": repetition,
                    "sample_size": sample_size,
                    "sampling_seed": record["sampling_seed"],
                    "archived_mse": record["mse"],
                    "max_abs_prediction": max_abs_prediction(dataset, record["prediction_row"], predictions),
                }
            )
            rows.append(spectral)
        print(dataset, sample_size, "done", flush=True)
    return rows


def _figure(year_rows: list[dict], super_rows: list[dict]) -> None:
    panels = [("Superconductivity", SUPER_N, super_rows), ("YearPredictionMSD", YEAR_N, year_rows)]
    fig, axes = plt.subplots(2, 9, figsize=(18, 6.2), constrained_layout=True)
    for row_index, (dataset, sizes, rows) in enumerate(panels):
        for column, sample_size in enumerate(range(9)):
            ax = axes[row_index, column]
            if column >= len(sizes):
                ax.axis("off")
                continue
            size = sizes[column]
            subset = [item for item in rows if item["sample_size"] == size and item["s_min"] > 0 and item["archived_mse"] > 0]
            ax.scatter(
                [np.log10(1.0 / item["s_min"]) for item in subset],
                [np.log10(item["archived_mse"]) for item in subset],
                s=12,
                color="#9b2226",
            )
            ax.set_title(f"{dataset[:5]} n={size}", fontsize=8)
            ax.set_xlabel(r"$\log_{10}(1/s_{\min})$", fontsize=8)
            ax.set_ylabel(r"$\log_{10}$ MSE", fontsize=8)
    fig.savefig(RESULTS / "fig_S_spectra.pdf")
    plt.close(fig)


def main() -> None:
    print_map()
    ensure_results()
    archived = load_archived()
    predictions: dict[str, np.ndarray] = {}
    year_rows = _run_dataset("YearPredictionMSD", YEAR_N, archived, predictions)
    super_rows = _run_dataset("Superconductivity", SUPER_N, archived, predictions)
    write_csv(RESULTS / "spectra_YearPredictionMSD.csv", year_rows)
    write_csv(RESULTS / "spectra_Superconductivity.csv", super_rows)
    correlations = _correlations(year_rows, YEAR_N, "YearPredictionMSD") + _correlations(super_rows, SUPER_N, "Superconductivity")
    write_csv(RESULTS / "spectra_correlations.csv", correlations)
    _figure(year_rows, super_rows)
    eligible = [row for row in year_rows + super_rows if row["sample_size"] <= (90 if row["dataset"] == "YearPredictionMSD" else 81)]
    held = sum(int(row["centering_claim"]) for row in eligible)
    focus = next(row for row in super_rows if row["repetition"] == 5 and row["sample_size"] == 73)
    n83 = [row for row in super_rows if row["sample_size"] == 83 and row["n_truncated"] > 0]
    lines = [
        "# Spectral diagnostics",
        "",
        f"Cells with n ≤ p where exactly one singular value is truncated and its left singular vector satisfies |uᵀ1/√n| > 0.999: {held} of {len(eligible)}.",
        "",
        "Superconductivity repetition 5, n=73:",
        "",
        f"- s_min = {focus['s_min']:.6e}",
        f"- tau = {focus['tau']:.6e}",
        f"- s1 = {focus['s1']:.6e}",
        f"- fraction of coefficient energy in the smallest retained direction = {100 * focus['frac_energy_smallest_retained']:.4f}%",
        f"- truncated singular values = {focus['n_truncated']}",
        f"- unique raw rows = {focus['n_unique_rows']}",
        f"- duplicated raw columns = {focus['n_duplicated_columns']}",
        f"- zero-variance columns = {focus['n_zero_variance_columns']}",
        "",
        f"Superconductivity n=83 truncating repetitions: {len(n83)}.",
        "",
    ]
    for row in n83:
        lines.append(
            f"- rep {row['repetition']}: truncated {row['n_truncated']}, unique rows {row['n_unique_rows']}, "
            f"duplicated columns {row['n_duplicated_columns']}, zero-variance columns {row['n_zero_variance_columns']}"
        )
    rho = next(row for row in correlations if row["dataset"] == "Superconductivity" and row["sample_size"] == 73)
    lines.extend(
        [
            "",
            f"At Superconductivity n=73, Pearson correlation of log10 MSE with log10(1/s_min) is {rho.get('pearson_log10_mse_vs_log10_inv_smin')}.",
            f"Spearman correlation is {rho.get('spearman_log10_mse_vs_log10_inv_smin')}.",
        ]
    )
    write_markdown(RESULTS / "spectra.md", "\n".join(lines) + "\n")
    write_csv(RESULTS / "spectra_super_n83_truncated.csv", n83)


if __name__ == "__main__":
    main()
