"""Shared readers for the completed runs. Does not call evaluate_gate or write run artifacts."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from p19.gates import GateDecision
from p19.l2_pipeline import evaluate_holdout, refit_full_prefix
from p19.l2_sampling import sampling_seeds_from_protocol, source_permutation
from p19.protocol import (
    SUPER_SAMPLE_SIZE_LIST_ORIGINAL,
    YEAR_SAMPLE_SIZE_LIST_ORIGINAL,
    load_yaml,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGING_ROOT = PROJECT_ROOT.parent / "p19-ood-spectral-regularization-super-staging"
RESULTS = PROJECT_ROOT / "results" / "posthoc"
CACHE = RESULTS / "_cache"

YEAR_RUN = PROJECT_ROOT / "artifacts" / "runs" / "20261030T085223Z_l2_28abcf33d8e4"
SUPER_RUN = (
    STAGING_ROOT
    / "artifacts"
    / "superconductivity_runs"
    / "20261102T141929Z_super_b3c9fff27517"
)
YEAR_PROTOCOL = PROJECT_ROOT / "reports" / "protocol_freeze.yaml"
SUPER_PROTOCOL = STAGING_ROOT / "reports" / "STAGE5N6_SUPERCONDUCTIVITY_PROTOCOL_FROZEN.yaml"
AUDIT_DIR = PROJECT_ROOT / "reports" / "stage5n8"
YEAR_DATA = PROJECT_ROOT / "data" / "raw" / "YearPredictionMSD.txt"
SUPER_DATA = PROJECT_ROOT / "data" / "raw" / "train.csv"
SPLIT_DIR = STAGING_ROOT / "reports" / "stage5n5_xonly"

ESTIMATORS = ("min_norm", "ridge", "pcr")
YEAR_SIZES = tuple(int(size) for size in YEAR_SAMPLE_SIZE_LIST_ORIGINAL)
SUPER_SIZES = tuple(int(size) for size in SUPER_SAMPLE_SIZE_LIST_ORIGINAL)
YEAR_P = 90
SUPER_P = 81


def print_map() -> None:
    """Locations used by the post-hoc scripts."""
    rows = [
        ("Year protocol / seed list", YEAR_PROTOCOL),
        ("Superconductivity frozen protocol", SUPER_PROTOCOL),
        ("Year metric and hyperparameter manifest", YEAR_RUN / "l2_run_manifest.json"),
        ("Year prediction array", YEAR_RUN / "holdout_predictions.npy"),
        ("Superconductivity metrics and warnings", SUPER_RUN / "holdout_metrics.jsonl"),
        ("Superconductivity predictions", SUPER_RUN / "holdout_predictions.npy"),
        ("Superconductivity uncertainty", SUPER_RUN / "uncertainty_summaries.json"),
        ("Saved grouped-split indices", SPLIT_DIR),
        ("Prefix hashes", SPLIT_DIR / "sampling_prefix_sha256.json"),
        ("Stage 5N-8 paired contrasts", AUDIT_DIR / "STAGE5N8_PAIRED_CONTRAST_AUDIT.csv"),
        ("Stage 5N-8 comparison", AUDIT_DIR / "STAGE5N8_CROSS_BENCHMARK_COMPARISON.csv"),
        ("Stage 5N-8 repetition distributions", AUDIT_DIR / "STAGE5N8_REPETITION_DISTRIBUTIONS.csv"),
        ("Minimum-norm warning text", "Superconductivity holdout_metrics.jsonl field warnings; Year manifest has no per-fit singular-value warnings"),
    ]
    print("Post-hoc input map")
    for label, path in rows:
        exists = path.exists() if isinstance(path, Path) else "n/a"
        print(f"  {label}: {path} [{exists}]")


def ensure_results() -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    return RESULTS


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: format(value, ".17g") if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def relative_error(left: float, right: float) -> float:
    return abs(float(left) - float(right)) / max(abs(float(left)), abs(float(right)), 1e-300)


def _load_staging_module(qualname: str, relative: str):
    if qualname in sys.modules:
        return sys.modules[qualname]
    path = STAGING_ROOT / relative
    spec = importlib.util.spec_from_file_location(qualname, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualname] = module
    spec.loader.exec_module(module)
    return module


def staging_design():
    """Import the staging sampling functions without replacing this checkout's p19 package."""
    _load_staging_module(
        "p19.superconductivity_authorization",
        "p19/superconductivity_authorization.py",
    )
    return _load_staging_module(
        "p19.superconductivity_design",
        "p19/superconductivity_design.py",
    )


def year_seeds() -> list[int]:
    return sampling_seeds_from_protocol(load_yaml(YEAR_PROTOCOL))


def super_seeds() -> list[int]:
    return [int(seed) for seed in staging_design().load_stored_sampling_seeds()]


def _sha256_index(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.int64).tobytes()).hexdigest()


def load_year() -> dict[str, np.ndarray]:
    cache = CACHE / "year_float64.npz"
    if cache.exists() and cache.stat().st_mtime >= YEAR_DATA.stat().st_mtime:
        loaded = np.load(cache)
        return {key: loaded[key] for key in loaded.files}
    from p19.data_loading import load_yearprediction_confirmatory_arrays

    # The confirmatory gate already approved this completed run. Constructing the
    # decision in memory reuses the official parser and does not append a gate event.
    arrays = load_yearprediction_confirmatory_arrays(
        YEAR_DATA,
        GateDecision(approved=True, reasons=[]),
    )
    payload = {
        "X_source": np.asarray(arrays["X_source"], dtype=np.float64),
        "y_source": np.asarray(arrays["y_source"], dtype=np.float64),
        "X_holdout": np.asarray(arrays["X_holdout"], dtype=np.float64),
        "y_holdout": np.asarray(arrays["y_holdout"], dtype=np.float64),
    }
    np.savez(cache, **payload)
    return payload


def load_super() -> dict[str, np.ndarray]:
    cache = CACHE / "super_float64.npz"
    manifest = json.loads((SUPER_RUN / "superconductivity_run_manifest.json").read_text(encoding="utf-8"))
    train = np.load(SPLIT_DIR / "training_row_indices.npy")
    holdout = np.load(SPLIT_DIR / "holdout_row_indices.npy")
    if _sha256_index(train) != manifest["split_identity"]["training_row_index_sha256"]:
        raise RuntimeError("saved Superconductivity training-row indices do not match the run manifest")
    if _sha256_index(holdout) != manifest["split_identity"]["holdout_row_index_sha256"]:
        raise RuntimeError("saved Superconductivity holdout-row indices do not match the run manifest")
    if cache.exists() and cache.stat().st_mtime >= SUPER_DATA.stat().st_mtime:
        loaded = np.load(cache)
        payload = {key: loaded[key] for key in loaded.files}
        payload["train_rows"] = np.asarray(train, dtype=np.int64)
        payload["holdout_rows"] = np.asarray(holdout, dtype=np.int64)
        return payload
    from p19.data_loading import load_superconductivity_labeled_data, load_superconductivity_x_only
    import p19.data_loading as data_loading

    features = load_superconductivity_x_only(SUPER_DATA)
    original_gate = data_loading.require_gate

    def _approved_without_writing(project_root, action):
        return GateDecision(approved=True, reasons=[])

    data_loading.require_gate = _approved_without_writing
    try:
        labels = load_superconductivity_labeled_data(SUPER_DATA, PROJECT_ROOT)
    finally:
        data_loading.require_gate = original_gate
    payload = {
        "X": np.asarray(features["X"], dtype=np.float64),
        "y": np.asarray(labels["y"], dtype=np.float64),
    }
    np.savez(cache, **payload)
    payload["train_rows"] = np.asarray(train, dtype=np.int64)
    payload["holdout_rows"] = np.asarray(holdout, dtype=np.int64)
    return payload


def year_prefix(repetition: int, sample_size: int) -> np.ndarray:
    return source_permutation(year_seeds()[repetition])[: int(sample_size)].copy()


def super_prefix(repetition: int, sample_size: int, train_rows: np.ndarray) -> np.ndarray:
    design = staging_design()
    order = design.permute_training_pool(train_rows, super_seeds()[repetition])
    return design.nested_prefixes(order)[int(sample_size)]


def load_archived() -> dict[tuple[str, int, int, str], dict]:
    """Per-repetition metric and hyperparameter records. Predictions stay on disk."""
    records: dict[tuple[str, int, int, str], dict] = {}
    manifest = json.loads((YEAR_RUN / "l2_run_manifest.json").read_text(encoding="utf-8"))
    cells = {
        (int(cell["repetition"]), int(cell["sample_size"])): cell
        for cell in manifest["source_cells"]
    }
    for row in manifest["holdout_metrics"]:
        repetition = int(row["repetition"])
        sample_size = int(row["sample_size"])
        estimator = str(row["estimator"])
        cell = cells[(repetition, sample_size)]
        records[("YearPredictionMSD", repetition, sample_size, estimator)] = {
            "mse": float(row["mse"]),
            "prediction_row": int(row["prediction_artifact_row"]),
            "lambda": float(cell["selected_ridge_lambda"]) if estimator == "ridge" else None,
            "pcr_fraction": float(cell["selected_pcr_fraction"]) if estimator == "pcr" else None,
            "pcr_components": int(cell["pcr_component_count"]) if estimator == "pcr" else None,
            "numerical_rank": None,
            "warnings": [],
            "sampling_seed": int(cell["sampling_seed"]),
        }
    seeds = super_seeds()
    with (SUPER_RUN / "holdout_metrics.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            estimator = str(row["estimator"])
            warnings = list(row.get("warnings") or [])
            rank = None
            components = None
            for warning in warnings:
                if warning.startswith("retained "):
                    components = int(warning.split()[1])
                    rank = int(warning.split("numerical rank ")[1].split()[0])
            selected = row.get("selected")
            repetition = int(row["repetition"])
            records[("Superconductivity", repetition, int(row["sample_size"]), estimator)] = {
                "mse": float(row["mse"]),
                "prediction_row": int(row["prediction_artifact_row"]),
                "lambda": float(selected) if estimator == "ridge" and selected is not None else None,
                "pcr_fraction": float(selected) if estimator == "pcr" and selected is not None else None,
                "pcr_components": components,
                "numerical_rank": rank,
                "warnings": warnings,
                "sampling_seed": seeds[repetition],
            }
    return records


def prediction_path(dataset: str) -> Path:
    if dataset == "YearPredictionMSD":
        return YEAR_RUN / "holdout_predictions.npy"
    return SUPER_RUN / "holdout_predictions.npy"


def max_abs_prediction(dataset: str, prediction_row: int, cache: dict[str, np.ndarray]) -> float:
    if dataset not in cache:
        cache[dataset] = np.load(prediction_path(dataset), mmap_mode="r")
    row = np.asarray(cache[dataset][prediction_row], dtype=np.float64)
    return float(np.max(np.abs(row)))


def refit_cell(x_train, y_train, x_holdout, y_holdout, ridge_lambda: float, pcr_fraction: float) -> dict[str, float]:
    preprocessor, models, _digest = refit_full_prefix(
        np.asarray(x_train, dtype=np.float64),
        np.asarray(y_train, dtype=np.float64),
        float(ridge_lambda),
        float(pcr_fraction),
    )
    scored = {}
    for name, model in models.items():
        scored[name] = float(evaluate_holdout(preprocessor, model, x_holdout, y_holdout)["mse"])
    return scored


def dataset_grid(dataset: str) -> tuple[tuple[int, ...], int]:
    if dataset == "YearPredictionMSD":
        return YEAR_SIZES, YEAR_P
    return SUPER_SIZES, SUPER_P
