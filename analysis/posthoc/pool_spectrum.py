"""Task 3: descriptive spectrum of each full source pool."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from p19.estimators import numerical_cutoff, numerical_rank

from analysis.posthoc.common import RESULTS, ensure_results, load_super, load_year, print_map, write_csv, write_markdown

THRESHOLDS = (1e-2, 1e-4, 1e-6, 1e-8)


def _duplicate_columns(matrix: np.ndarray) -> int:
    transposed = np.ascontiguousarray(matrix.T, dtype=np.float64)
    view = transposed.view(np.dtype((np.void, transposed.dtype.itemsize * transposed.shape[1]))).reshape(-1)
    _values, counts = np.unique(view, return_counts=True)
    return int(np.sum(counts[counts > 1]))


def _spectrum(matrix: np.ndarray) -> dict:
    raw = np.asarray(matrix, dtype=np.float64)
    n_rows, n_cols = raw.shape
    scale = np.std(raw, axis=0, ddof=0)
    zero = scale == 0
    safe = np.where(zero, 1.0, scale)
    standardized = (raw - np.mean(raw, axis=0)) / safe
    gram = standardized.T @ standardized
    eigenvalues = np.sort(np.linalg.eigvalsh(gram))[::-1]
    eigenvalues = np.clip(eigenvalues, 0.0, None).astype(np.float64)
    singular = np.sqrt(eigenvalues)
    cutoff = numerical_cutoff(singular, n_rows, n_cols)
    largest = float(eigenvalues[0])
    counts = {
        f"below_{threshold:.0e}_lambda_max": int(np.count_nonzero(eigenvalues < threshold * largest))
        for threshold in THRESHOLDS
    }
    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "lambda_max": largest,
        "numerical_rank": numerical_rank(singular, cutoff),
        "tau": float(cutoff),
        "n_exactly_duplicated_columns": _duplicate_columns(raw),
        "n_zero_variance_columns": int(np.count_nonzero(zero)),
        "eigenvalues": eigenvalues,
        **counts,
    }


def main() -> None:
    print_map()
    ensure_results()
    year = load_year()
    super_data = load_super()
    spectra = {
        "YearPredictionMSD": _spectrum(year["X_source"]),
        "Superconductivity": _spectrum(super_data["X"][super_data["train_rows"]]),
    }
    rows = []
    for dataset, payload in spectra.items():
        row = {key: value for key, value in payload.items() if key != "eigenvalues"}
        row["dataset"] = dataset
        rows.append(row)
        print(dataset, row, flush=True)
    write_csv(RESULTS / "pool_spectrum.csv", rows)
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    for dataset, payload in spectra.items():
        values = payload["eigenvalues"]
        positive = values[values > 0]
        ax.plot(np.arange(1, positive.size + 1), positive, marker="o", ms=3, label=dataset)
    ax.set_yscale("log")
    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("Correlation eigenvalue")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(RESULTS / "fig_S_pool_spectrum.pdf")
    plt.close(fig)
    lines = ["# Source-pool correlation spectra", ""]
    for row in rows:
        lines.append(
            f"- {row['dataset']}: rank {row['numerical_rank']} of {row['n_cols']}, "
            f"duplicated columns {row['n_exactly_duplicated_columns']}, "
            f"zero-variance columns {row['n_zero_variance_columns']}."
        )
        for threshold in THRESHOLDS:
            lines.append(f"  - below {threshold:.0e} λ_max: {row[f'below_{threshold:.0e}_lambda_max']}")
    write_markdown(RESULTS / "pool_spectrum.md", "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
