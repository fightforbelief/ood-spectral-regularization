"""Isolated Superconductivity source phase, scoring, and prediction artifacts.

Official rows are not fitted here. A synthetic result cannot claim confirmation.
The completed YearPredictionMSD run directory is refused.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

from p19.l2_pipeline import (
    HoldoutScoringBlocked,
    PrefixFits,
    RunAborted,
    SourcePhaseResult,
    evaluate_holdout,
    fit_training_prefix,
)
from p19.l2_runner import (
    PREDICTION_ARTIFACT_NAME,
    PREDICTION_ESTIMATOR_ORDER,
    PredictionArtifactError,
    align_prediction_vectors,
    prediction_identity_order,
)
from p19.l2_sampling import ProtocolValidationError
from p19.protocol import SUPER_SAMPLE_SIZE_LIST_ORIGINAL
from p19.superconductivity_authorization import require_superconductivity_authorization
from p19.superconductivity_design import (
    DECLARED_REPETITIONS,
    ESTIMATOR_ORDER,
    MAX_PREFIX,
    assert_prefix_inside_pool,
    length_810_fold_membership,
    load_stored_sampling_seeds,
    nested_prefixes,
    permute_training_pool,
    require_archived_sample_sizes,
    restrict_length_810_folds,
)
from p19.uncertainty import (
    APPROVED_TRAINING_SUBSET_PROCEDURE,
    paired_estimator_loss_differences,
    summarize_complete_replicates,
)

ORIGINAL_PROJECT = Path(r"C:\Users\hanqi\Desktop\projects\p19-ood-spectral-regularization")
FORBIDDEN_RUN_NAME = "20261030T085223Z_l2_28abcf33d8e4"
PARTIAL_NAME = "holdout_predictions.partial.npy"
RUN_PREFIX = "superconductivity_runs"
MANIFEST_NAME = "superconductivity_run_manifest.json"
METRICS_ARTIFACT_NAME = "holdout_metrics.jsonl"
UNCERTAINTY_ARTIFACT_NAME = "uncertainty_summaries.json"
OFFICIAL_COMPLETE_STATUS = "SUPERCONDUCTIVITY_CONFIRMATORY_COMPLETED"
SYNTHETIC_VERIFIED_STATUS = "SYNTHETIC_PERSISTENCE_VERIFIED"
INCOMPLETE_STATUS = "INCOMPLETE"
METRIC_SCALAR_FIELDS = ("primary_endpoint", "mse", "rmse", "mae", "r2", "n_rows", "training_target_mean", "selected")
FROZEN_GROUPED_HOLDOUT_ROWS = 4392
YEAR_GROUPED_HOLDOUT_ROWS = 51630
OFFICIAL_PREDICTION_SHAPE = (4500, FROZEN_GROUPED_HOLDOUT_ROWS)


class SuperconductivityIsolationError(Exception):
    """The requested output would touch a protected or existing run."""


class SuperconductivityExecutionBlocked(Exception):
    """Official Superconductivity fitting or artifact creation is not authorized."""


class SuperconductivityPersistenceError(Exception):
    """A required result artifact is missing, duplicated, or failed validation."""


@dataclass
class SuperconductivitySourceResult:
    complete: bool
    cells: list = field(default_factory=list)
    failure: dict | None = None
    synthetic: bool = True
    confirmatory_result: bool = False


def assert_not_protected(path: Path) -> Path:
    """Refuse the completed YearPredictionMSD run and any path inside the original project."""
    resolved = Path(path).resolve()
    original = ORIGINAL_PROJECT.resolve()
    if resolved == original or original in resolved.parents:
        raise SuperconductivityIsolationError("refusing to write into the original project")
    if FORBIDDEN_RUN_NAME in resolved.parts or resolved.name == FORBIDDEN_RUN_NAME:
        raise SuperconductivityIsolationError(f"refusing protected run directory {FORBIDDEN_RUN_NAME}")
    return resolved


def assert_output_isolated(path: Path) -> Path:
    """Refuse a protected path and any directory that already exists."""
    resolved = assert_not_protected(path)
    if resolved.exists():
        raise SuperconductivityIsolationError("refusing to overwrite an existing run directory")
    return resolved


def source_cell_key_problems(cells) -> list[str]:
    """Exact adopted grid, each pair once. A partial grid is not complete."""
    expected = {
        (repetition, int(sample_size))
        for repetition in range(DECLARED_REPETITIONS)
        for sample_size in SUPER_SAMPLE_SIZE_LIST_ORIGINAL
    }
    found: list[tuple[int, int]] = []
    for cell in cells:
        repetition = getattr(cell, "repetition", None)
        sample_size = getattr(cell, "sample_size", None)
        if isinstance(repetition, bool) or not isinstance(repetition, (int, np.integer)):
            return ["source cell repetition is not an integer"]
        if isinstance(sample_size, bool) or not isinstance(sample_size, (int, np.integer)):
            return ["source cell sample size is not an integer"]
        found.append((int(repetition), int(sample_size)))
    problems: list[str] = []
    if len(found) != len(set(found)):
        problems.append("duplicated source cell")
    missing = expected - set(found)
    unexpected = set(found) - expected
    if missing:
        problems.append(f"missing source cells: {len(missing)}")
    if unexpected:
        problems.append(f"unexpected source cells: {len(unexpected)}")
    if len(cells) != len(expected):
        problems.append(f"source phase produced {len(cells)} cells, not {len(expected)}")
    return problems


def incomplete_confirmatory_record(source: SuperconductivitySourceResult) -> dict:
    """Failure stops the run. The required repetition count stays 100."""
    failure = dict(source.failure or {})
    return {
        "status": "INCOMPLETE",
        "complete": False,
        "confirmatory_result": False,
        "synthetic": bool(source.synthetic),
        "holdout_scored": False,
        "failure": failure,
        "completed_source_cells": len(source.cells),
        "required_source_cells": DECLARED_REPETITIONS * len(SUPER_SAMPLE_SIZE_LIST_ORIGINAL),
        "required_repetitions": DECLARED_REPETITIONS,
        "repetition_count_reduced": False,
        "replacement_repetition": False,
        "estimator_substitution": False,
    }


def run_superconductivity_source_phase(
    training_row_indices: np.ndarray,
    execute_cell: Callable[..., object],
    holdout_row_indices: np.ndarray | None = None,
) -> SuperconductivitySourceResult:
    """Drive the adopted 100 by 15 grid. This function does not score a holdout."""
    seeds = load_stored_sampling_seeds()
    sizes = require_archived_sample_sizes(SUPER_SAMPLE_SIZE_LIST_ORIGINAL)
    membership = length_810_fold_membership()
    cells: list = []
    for repetition, sampling_seed in enumerate(seeds):
        if int(sampling_seed) != seeds[repetition]:
            raise ProtocolValidationError("sampling seed order changed during the source phase")
        order = permute_training_pool(training_row_indices, sampling_seed)
        prefixes = nested_prefixes(order, sizes)
        for sample_size in sizes:
            prefix = prefixes[sample_size]
            folds = restrict_length_810_folds(membership, sample_size)
            try:
                if holdout_row_indices is not None:
                    assert_prefix_inside_pool(prefix, training_row_indices, holdout_row_indices)
                cell = execute_cell(
                    repetition=repetition,
                    sampling_seed=sampling_seed,
                    sample_size=sample_size,
                    prefix_indices=prefix,
                    folds=folds,
                    estimators=ESTIMATOR_ORDER,
                )
            except RunAborted as exc:
                return SuperconductivitySourceResult(complete=False, cells=cells, failure=exc.failure)
            except Exception as exc:
                return SuperconductivitySourceResult(
                    complete=False,
                    cells=cells,
                    failure={
                        "repetition": repetition,
                        "sample_size": sample_size,
                        "sampling_seed": sampling_seed,
                        "error": type(exc).__name__,
                        "message": str(exc),
                    },
                )
            if cell is None or int(cell.repetition) != repetition or int(cell.sample_size) != sample_size:
                return SuperconductivitySourceResult(
                    complete=False,
                    cells=cells,
                    failure={
                        "repetition": repetition,
                        "sample_size": sample_size,
                        "sampling_seed": sampling_seed,
                        "message": "source cell identity does not match the Superconductivity grid",
                    },
                )
            cells.append(cell)
    expected = DECLARED_REPETITIONS * len(sizes)
    if len(cells) != expected:
        return SuperconductivitySourceResult(
            complete=False,
            cells=cells,
            failure={"message": f"source phase produced {len(cells)} cells, not {expected}"},
        )
    return SuperconductivitySourceResult(complete=True, cells=cells, failure=None)


def score_superconductivity_source_phase(
    source: SuperconductivitySourceResult,
    X_holdout: np.ndarray,
    y_holdout: np.ndarray,
    evaluate: Callable = evaluate_holdout,
) -> list[dict]:
    """Score only after every adopted source cell succeeded. Failure does not call evaluate."""
    if not isinstance(source, SuperconductivitySourceResult):
        raise HoldoutScoringBlocked("malformed Superconductivity source-phase result")
    problems = source_cell_key_problems(source.cells)
    if not source.complete or source.failure is not None or problems:
        raise HoldoutScoringBlocked(
            "refusing Superconductivity holdout scoring until the adopted source phase succeeds"
        )
    scored: list[dict] = []
    for cell in source.cells:
        if not isinstance(cell, PrefixFits):
            raise HoldoutScoringBlocked("source cell is not a fitted prefix")
        for name in ESTIMATOR_ORDER:
            row = evaluate(cell.preprocessor, cell.models[name], X_holdout, y_holdout)
            row["repetition"] = int(cell.repetition)
            row["sample_size"] = int(cell.sample_size)
            row["estimator"] = name
            scored.append(row)
    return scored


def traverse_estimator_scoring_identities(source: SuperconductivitySourceResult, callback: Callable) -> list:
    """Visit all 4,500 estimator identities after a complete source phase.

    The callback is not invoked when the source phase is incomplete, duplicated,
    or outside the adopted grid. This path does not fit a model.
    """
    if not isinstance(source, SuperconductivitySourceResult):
        raise HoldoutScoringBlocked("malformed Superconductivity source-phase result")
    problems = source_cell_key_problems(source.cells)
    if not source.complete or source.failure is not None or problems:
        raise HoldoutScoringBlocked(
            "refusing Superconductivity holdout scoring until the adopted source phase succeeds"
        )
    visited = []
    for cell in source.cells:
        for name in ESTIMATOR_ORDER:
            visited.append(
                callback(
                    repetition=int(cell.repetition),
                    sample_size=int(cell.sample_size),
                    estimator=name,
                )
            )
    expected = DECLARED_REPETITIONS * len(SUPER_SAMPLE_SIZE_LIST_ORIGINAL) * len(ESTIMATOR_ORDER)
    if len(visited) != expected:
        raise HoldoutScoringBlocked(f"scoring traversal produced {len(visited)} identities, not {expected}")
    return visited


def fit_synthetic_prefix(
    X_prefix: np.ndarray,
    y_prefix: np.ndarray,
    folds: list[np.ndarray],
    *,
    repetition: int = 0,
    sampling_seed: int = 1,
) -> PrefixFits:
    """Fit the existing estimators on caller-supplied synthetic rows."""
    size = int(np.asarray(X_prefix).shape[0])
    return fit_training_prefix(
        X_prefix,
        y_prefix,
        repetition=repetition,
        sampling_seed=sampling_seed,
        sample_size=size,
        prefix_indices=np.arange(size, dtype=np.int64),
        folds=folds,
    )


def synthetic_confirmation_record(payload: dict) -> dict:
    """Synthetic completion is never an official confirmatory result."""
    record = dict(payload)
    record["synthetic"] = True
    record["confirmatory_result"] = False
    record["confirmatory_l2"] = "SYNTHETIC_NOT_CONFIRMATION"
    return record


def superconductivity_uncertainty_from_arrays(
    min_norm: np.ndarray,
    ridge: np.ndarray,
    pcr: np.ndarray,
    risk_by_sample_size: dict[int, np.ndarray],
) -> dict:
    """Stage 5D summaries on the adopted grid.

    The year adjacent-pair function stays on the year grid. Adjacent contrasts
    here use the same replicate summary on consecutive archived Superconductivity sizes.
    """
    procedure = APPROVED_TRAINING_SUBSET_PROCEDURE
    paired = paired_estimator_loss_differences(min_norm, ridge, pcr, procedure)
    sizes = list(SUPER_SAMPLE_SIZE_LIST_ORIGINAL)
    adjacent = {}
    for smaller, larger in zip(sizes, sizes[1:]):
        difference = np.asarray(risk_by_sample_size[larger], dtype=np.float64) - np.asarray(
            risk_by_sample_size[smaller], dtype=np.float64
        )
        adjacent[f"{larger}_minus_{smaller}"] = summarize_complete_replicates(difference)
    levels = {
        "min_norm": summarize_complete_replicates(min_norm),
        "ridge": summarize_complete_replicates(ridge),
        "pcr": summarize_complete_replicates(pcr),
    }
    return {"procedure": procedure, "levels": levels, "paired": paired, "adjacent": adjacent}


def summarize_superconductivity_holdout(scored_rows: list[dict]) -> dict:
    """Pointwise Student-t summaries on the adopted grid.

    Adjacent contrasts are computed on the p=81 sizes. The YearPredictionMSD
    adjacent-pair function is not called.
    """
    from p19.uncertainty import UncertaintyNotFrozen, training_subset_variability

    expected = {
        (repetition, int(sample_size), estimator)
        for repetition in range(DECLARED_REPETITIONS)
        for sample_size in SUPER_SAMPLE_SIZE_LIST_ORIGINAL
        for estimator in ESTIMATOR_ORDER
    }
    located: dict[tuple[int, int, str], float] = {}
    for row in scored_rows:
        try:
            key = (int(row["repetition"]), int(row["sample_size"]), str(row["estimator"]))
            mse = float(row["mse"])
        except (KeyError, TypeError, ValueError) as exc:
            raise UncertaintyNotFrozen("scored row is missing Superconductivity identity") from exc
        if key in located or key not in expected or not np.isfinite(mse):
            raise UncertaintyNotFrozen("scored rows are not the Superconductivity grid")
        located[key] = mse
    if set(located) != expected:
        raise UncertaintyNotFrozen("scored rows are not the Superconductivity grid")
    paired = {}
    by_estimator = {}
    sizes = [int(size) for size in SUPER_SAMPLE_SIZE_LIST_ORIGINAL]
    for size in sizes:
        paired[str(size)] = paired_estimator_loss_differences(
            np.array([located[(rep, size, "min_norm")] for rep in range(DECLARED_REPETITIONS)]),
            np.array([located[(rep, size, "ridge")] for rep in range(DECLARED_REPETITIONS)]),
            np.array([located[(rep, size, "pcr")] for rep in range(DECLARED_REPETITIONS)]),
            APPROVED_TRAINING_SUBSET_PROCEDURE,
        )
    for estimator in ESTIMATOR_ORDER:
        risks = {
            size: np.array([located[(rep, size, estimator)] for rep in range(DECLARED_REPETITIONS)])
            for size in sizes
        }
        adjacent = {}
        for smaller, larger in zip(sizes, sizes[1:]):
            adjacent[f"{larger}_minus_{smaller}"] = summarize_complete_replicates(risks[larger] - risks[smaller])
        by_estimator[estimator] = {
            "levels": {
                str(size): training_subset_variability(risks[size], APPROVED_TRAINING_SUBSET_PROCEDURE)
                for size in sizes
            },
            "adjacent": adjacent,
        }
    return {
        "procedure": APPROVED_TRAINING_SUBSET_PROCEDURE,
        "paired_by_sample_size": paired,
        "by_estimator": by_estimator,
        "year_adjacent_function_used": False,
    }


def _filesystem_error(operation: str, path: Path, exc: OSError) -> PredictionArtifactError:
    return PredictionArtifactError(f"{operation} failed for {path}: {type(exc).__name__}: {exc}")


def frozen_prediction_schema() -> dict:
    """Schema for the frozen grouped holdout. This does not allocate the array."""
    return {
        "dtype": "float64",
        "shape": [int(OFFICIAL_PREDICTION_SHAPE[0]), int(OFFICIAL_PREDICTION_SHAPE[1])],
        "row_order": (
            "repetition ascending from 0; within each repetition the archived sample sizes; "
            "within each sample size the estimators min_norm, ridge, pcr"
        ),
        "estimator_order": list(ESTIMATOR_ORDER),
        "frozen_grouped_holdout_rows": FROZEN_GROUPED_HOLDOUT_ROWS,
        "refused_yearprediction_holdout_rows": YEAR_GROUPED_HOLDOUT_ROWS,
        "source_split_order": "frozen_splitter_group_permutation_concatenation",
    }


def write_superconductivity_prediction_artifact(
    destination: Path,
    vectors: Sequence[np.ndarray],
    *,
    sample_sizes: Sequence[int],
    n_repetitions: int,
    holdout_row_count: int | None = None,
    split_identity: dict | None = None,
    official_write: bool = False,
) -> tuple[dict, list[str]]:
    """Write float64 predictions. The holdout width must match the split record."""
    directory = assert_output_isolated(Path(destination))
    if not vectors:
        raise PredictionArtifactError("prediction grid is empty")
    order = prediction_identity_order(n_repetitions, sample_sizes)
    if len(vectors) != len(order):
        raise PredictionArtifactError("prediction vectors do not match the requested identity order")
    declared_holdout = int(np.asarray(vectors[0]).shape[0] if holdout_row_count is None else holdout_row_count)
    if declared_holdout == YEAR_GROUPED_HOLDOUT_ROWS:
        raise PredictionArtifactError("refusing YearPredictionMSD holdout length 51630")
    if declared_holdout == FROZEN_GROUPED_HOLDOUT_ROWS and len(vectors) == OFFICIAL_PREDICTION_SHAPE[0]:
        if not official_write:
            raise SuperconductivityExecutionBlocked(
                "official 4500 by 4392 Superconductivity prediction artifact is not written without authorization"
            )
    if declared_holdout != FROZEN_GROUPED_HOLDOUT_ROWS and official_write:
        raise PredictionArtifactError("official write requires the frozen grouped holdout width 4392")
    try:
        directory.mkdir(parents=True)
    except OSError as exc:
        raise _filesystem_error("output directory creation", directory, exc) from exc
    final_path = directory / PREDICTION_ARTIFACT_NAME
    partial_path = directory / PARTIAL_NAME
    if final_path.exists() or partial_path.exists():
        raise PredictionArtifactError("prediction artifact already exists")
    width = declared_holdout
    shape = (len(vectors), width)
    try:
        memmap = open_memmap(partial_path, mode="w+", dtype=np.float64, shape=shape)
    except OSError as exc:
        raise _filesystem_error("open_memmap", partial_path, exc) from exc
    hashes: list[str] = []
    finished = False
    try:
        for index, vector in enumerate(vectors):
            row = np.ascontiguousarray(vector, dtype=np.float64)
            if row.shape != (width,) or not np.isfinite(row).all():
                raise PredictionArtifactError("prediction row failed validation during write")
            memmap[index, :] = row
            hashes.append(hashlib.sha256(row.tobytes()).hexdigest())
        memmap.flush()
        finished = True
    finally:
        mapped = getattr(memmap, "_mmap", None)
        if mapped is not None and not getattr(mapped, "closed", False):
            mapped.close()
    del memmap
    if not finished:
        raise PredictionArtifactError("prediction artifact write did not finish")
    if final_path.exists():
        raise PredictionArtifactError("completed prediction artifact already exists")
    digest = hashlib.sha256()
    with partial_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    os.replace(partial_path, final_path)
    record = {
        "relative_path": PREDICTION_ARTIFACT_NAME,
        "format": "npy",
        "dtype": "float64",
        "shape": [int(shape[0]), int(shape[1])],
        "n_prediction_vectors": int(shape[0]),
        "holdout_row_count": int(shape[1]),
        "official_artifact": bool(official_write),
        "synthetic": not bool(official_write),
        "confirmatory_result": False,
        "row_order": (
            "repetition ascending from 0; within each repetition the requested sample sizes; "
            "within each sample size the estimators min_norm, ridge, pcr"
        ),
        "estimator_order": list(PREDICTION_ESTIMATOR_ORDER),
        "future_official_shape": list(frozen_prediction_schema()["shape"]),
        "frozen_grouped_holdout_rows": FROZEN_GROUPED_HOLDOUT_ROWS,
        "split_identity": None if split_identity is None else dict(split_identity),
        "sha256": digest.hexdigest(),
        "vector_sha256": "sha256 of np.ascontiguousarray(vector, dtype=float64).tobytes()",
    }
    return record, hashes


def attach_synthetic_prediction_artifact(
    destination: Path,
    scored_rows: list[dict],
    *,
    sample_sizes: Sequence[int],
    n_repetitions: int,
) -> tuple[dict, list[dict]]:
    order = prediction_identity_order(n_repetitions, sample_sizes)
    vectors = align_prediction_vectors(scored_rows, order, None)
    record, hashes = write_superconductivity_prediction_artifact(
        destination,
        vectors,
        sample_sizes=sample_sizes,
        n_repetitions=n_repetitions,
    )
    located = {}
    for row in scored_rows:
        located[(int(row["repetition"]), int(row["sample_size"]), str(row["estimator"]))] = row
    metrics = []
    for index, key in enumerate(order):
        metric = dict(located[key])
        metric.pop("predictions", None)
        metric["prediction_artifact_row"] = index
        metric["prediction_sha256"] = hashes[index]
        metrics.append(metric)
    record["metrics"] = [
        {"prediction_artifact_row": row["prediction_artifact_row"], "prediction_sha256": row["prediction_sha256"]}
        for row in metrics
    ]
    return record, metrics


def _atomic_bytes(directory: Path, name: str, payload: bytes) -> Path:
    """Write a new file in the run directory. An existing final path is never replaced."""
    directory = assert_not_protected(directory)
    final = directory / name
    partial = directory / f"{name}.partial"
    if final.exists():
        raise SuperconductivityIsolationError(f"refusing to overwrite {name}")
    partial.write_bytes(payload)
    with partial.open("rb+") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, final)
    return final


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_number(value):
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise SuperconductivityPersistenceError("metric value is not finite")
        return number
    raise SuperconductivityPersistenceError(f"metric value {type(value).__name__} is not serializable")


def metric_disk_record(row: dict, artifact_row: int, prediction_sha256: str) -> dict:
    """Scalar metric identity. Prediction vectors stay in the npy artifact."""
    record = {
        "repetition": int(row["repetition"]),
        "sample_size": int(row["sample_size"]),
        "estimator": str(row["estimator"]),
        "prediction_artifact_row": int(artifact_row),
        "prediction_sha256": str(prediction_sha256),
    }
    for field in METRIC_SCALAR_FIELDS:
        if field in row:
            record[field] = _json_number(row[field])
    if "warnings" in row:
        record["warnings"] = list(row["warnings"])
    for field in ("mse", "rmse", "mae", "r2", "n_rows"):
        if field not in record:
            raise SuperconductivityPersistenceError(f"metric record is missing {field}")
    if int(record["n_rows"]) < 1:
        raise SuperconductivityPersistenceError("grouped-holdout row count is empty")
    return record


def metric_identity_problems(records: list[dict], expected_order: Sequence[tuple[int, int, str]]) -> list[str]:
    found = []
    for record in records:
        found.append((int(record["repetition"]), int(record["sample_size"]), str(record["estimator"])))
    problems = []
    if len(found) != len(set(found)):
        problems.append("duplicated metric identity")
    expected = [(int(repetition), int(size), str(estimator)) for repetition, size, estimator in expected_order]
    if len(found) != len(expected):
        problems.append(f"metric record count is {len(found)}, not {len(expected)}")
    missing = set(expected) - set(found)
    unexpected = set(found) - set(expected)
    if missing:
        problems.append(f"missing metric identities: {len(missing)}")
    if unexpected:
        problems.append(f"unexpected metric identities: {len(unexpected)}")
    if found != expected:
        problems.append("metric identity order does not match the prediction artifact")
    return problems


def _finite_tree(value, label: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _finite_tree(item, f"{label}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _finite_tree(item, f"{label}[{index}]")
        return
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)) and not math.isfinite(float(value)):
        raise SuperconductivityPersistenceError(f"{label} is not finite")


def uncertainty_record_counts(summary: dict) -> dict[str, int]:
    levels = sum(len(block["levels"]) for block in summary["by_estimator"].values())
    paired = sum(len(block) for block in summary["paired_by_sample_size"].values())
    adjacent = sum(len(block["adjacent"]) for block in summary["by_estimator"].values())
    return {"estimator_sample_size_summaries": levels, "paired_estimator_contrasts": paired, "adjacent_sample_size_contrasts": adjacent}


def assert_full_uncertainty_grid(summary: dict) -> dict[str, int]:
    counts = uncertainty_record_counts(summary)
    expected = {
        "estimator_sample_size_summaries": 45,
        "paired_estimator_contrasts": 45,
        "adjacent_sample_size_contrasts": 42,
    }
    if counts != expected:
        raise SuperconductivityPersistenceError(f"uncertainty record counts are {counts}, not {expected}")
    _finite_tree(summary, "uncertainty")
    return counts


def load_metric_records(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line == "":
            continue
        records.append(json.loads(line))
    return records


def verify_metric_artifact(path: Path, recorded_sha256: str, expected_order: Sequence[tuple[int, int, str]]) -> list[dict]:
    actual = _sha256_file(path)
    if actual != recorded_sha256:
        raise SuperconductivityPersistenceError("metric artifact hash does not match the recorded hash")
    records = load_metric_records(path)
    problems = metric_identity_problems(records, expected_order)
    if problems:
        raise SuperconductivityPersistenceError("; ".join(problems))
    _finite_tree(records, "metrics")
    return records


def verify_uncertainty_artifact(path: Path, recorded_sha256: str) -> dict:
    actual = _sha256_file(path)
    if actual != recorded_sha256:
        raise SuperconductivityPersistenceError("uncertainty artifact hash does not match the recorded hash")
    summary = json.loads(path.read_text(encoding="utf-8"))
    assert_full_uncertainty_grid(summary)
    return summary


def persist_metric_records(
    directory: Path,
    ordered_rows: Sequence[dict],
    vector_hashes: Sequence[str],
    expected_order: Sequence[tuple[int, int, str]],
) -> dict:
    if len(ordered_rows) != len(vector_hashes) or len(ordered_rows) != len(expected_order):
        raise SuperconductivityPersistenceError("metric rows, hashes, and identity order have different lengths")
    records = [
        metric_disk_record(row, index, vector_hashes[index]) for index, row in enumerate(ordered_rows)
    ]
    problems = metric_identity_problems(records, expected_order)
    if problems:
        raise SuperconductivityPersistenceError("; ".join(problems))
    payload = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records).encode("utf-8")
    path = _atomic_bytes(directory, METRICS_ARTIFACT_NAME, payload)
    digest = hashlib.sha256(payload).hexdigest()
    reopened = verify_metric_artifact(path, digest, expected_order)
    if reopened != records:
        raise SuperconductivityPersistenceError("reopened metric records do not match the written records")
    return {
        "relative_path": METRICS_ARTIFACT_NAME,
        "format": "jsonl",
        "record_count": len(records),
        "sha256": digest,
        "schema": "scalar metric identity; prediction vectors remain in holdout_predictions.npy",
    }


def persist_uncertainty_artifact(directory: Path, summary: dict) -> dict:
    counts = assert_full_uncertainty_grid(summary)
    payload = (json.dumps(summary, indent=2, sort_keys=True, default=_json_default) + "\n").encode("utf-8")
    path = _atomic_bytes(directory, UNCERTAINTY_ARTIFACT_NAME, payload)
    digest = hashlib.sha256(payload).hexdigest()
    reopened = verify_uncertainty_artifact(path, digest)
    if uncertainty_record_counts(reopened) != counts:
        raise SuperconductivityPersistenceError("reopened uncertainty counts do not match")
    return {"relative_path": UNCERTAINTY_ARTIFACT_NAME, "format": "json", "sha256": digest, "record_counts": counts}


def publish_verified_results(
    directory: Path,
    ordered_rows: Sequence[dict],
    vector_hashes: Sequence[str],
    expected_order: Sequence[tuple[int, int, str]],
    uncertainty: dict,
    prediction_record: dict,
    *,
    synthetic: bool,
    protocol_sha256: str,
    source_identity_sha256: str,
    split_identity: dict | None = None,
) -> dict:
    """Persist metrics and uncertainty, validate them, then write the manifest last."""
    directory = assert_not_protected(directory)
    prediction_path = directory / PREDICTION_ARTIFACT_NAME
    if not prediction_path.is_file():
        raise SuperconductivityPersistenceError("prediction artifact is missing")
    prediction_sha = _sha256_file(prediction_path)
    if prediction_sha != prediction_record.get("sha256"):
        raise SuperconductivityPersistenceError("prediction artifact hash does not match the writer record")
    metric_info = persist_metric_records(directory, ordered_rows, vector_hashes, expected_order)
    uncertainty_info = persist_uncertainty_artifact(directory, uncertainty)
    verify_metric_artifact(directory / METRICS_ARTIFACT_NAME, metric_info["sha256"], expected_order)
    verify_uncertainty_artifact(directory / UNCERTAINTY_ARTIFACT_NAME, uncertainty_info["sha256"])
    status = SYNTHETIC_VERIFIED_STATUS if synthetic else OFFICIAL_COMPLETE_STATUS
    manifest = {
        "status": status,
        "synthetic": bool(synthetic),
        "confirmatory_result": False if synthetic else True,
        "artifacts_verified": True,
        "run_id": directory.name,
        "output_dir": str(directory),
        "protocol_sha256": protocol_sha256,
        "source_identity_sha256": source_identity_sha256,
        "prediction_artifact": {
            "relative_path": prediction_record.get("relative_path", PREDICTION_ARTIFACT_NAME),
            "sha256": prediction_sha,
            "shape": prediction_record.get("shape"),
            "dtype": prediction_record.get("dtype"),
        },
        "metric_artifact": metric_info,
        "uncertainty_artifact": uncertainty_info,
        "split_identity": None if split_identity is None else dict(split_identity),
        "official_target_labels_loaded": not synthetic,
        "official_models_fit": not synthetic,
        "holdout_scored": True,
    }
    path = write_superconductivity_run_manifest(directory, manifest)
    reopened = json.loads(path.read_text(encoding="utf-8"))
    if reopened.get("status") != status or reopened.get("artifacts_verified") is not True:
        raise SuperconductivityPersistenceError("reopened manifest does not confirm completion")
    if reopened["metric_artifact"]["sha256"] != _sha256_file(directory / METRICS_ARTIFACT_NAME):
        raise SuperconductivityPersistenceError("manifest metric hash does not match the file")
    if reopened["uncertainty_artifact"]["sha256"] != _sha256_file(directory / UNCERTAINTY_ARTIFACT_NAME):
        raise SuperconductivityPersistenceError("manifest uncertainty hash does not match the file")
    if reopened["prediction_artifact"]["sha256"] != _sha256_file(prediction_path):
        raise SuperconductivityPersistenceError("manifest prediction hash does not match the file")
    reopened["output_dir"] = str(directory)
    return reopened


def finish_superconductivity_cli(result: dict) -> int:
    """Return 0 only for a verified complete result. Synthetic completion is not official."""
    if not isinstance(result, dict):
        print(INCOMPLETE_STATUS)
        return 3
    status = result.get("status")
    output_dir = result.get("output_dir", "")
    run_id = result.get("run_id", "")
    verified = result.get("artifacts_verified") is True
    if result.get("synthetic") is True:
        if status == SYNTHETIC_VERIFIED_STATUS and verified and result.get("confirmatory_result") is False:
            print(SYNTHETIC_VERIFIED_STATUS)
            print(run_id)
            print(output_dir)
            return 0
        print("SYNTHETIC_NOT_CONFIRMATION")
        print(status or INCOMPLETE_STATUS)
        return 3
    if (
        status == OFFICIAL_COMPLETE_STATUS
        and verified
        and result.get("synthetic") is False
        and result.get("confirmatory_result") is True
    ):
        print(OFFICIAL_COMPLETE_STATUS)
        print(run_id)
        print(output_dir)
        return 0
    print(status or INCOMPLETE_STATUS)
    return 3


def write_superconductivity_run_manifest(destination: Path, payload: dict) -> Path:
    """Write one manifest into an existing isolated run directory. Do not overwrite it."""
    directory = assert_not_protected(destination)
    document = dict(payload)
    if document.get("synthetic") is True:
        document["confirmatory_result"] = False
        if document.get("status") == OFFICIAL_COMPLETE_STATUS:
            document["status"] = SYNTHETIC_VERIFIED_STATUS
    text = (json.dumps(document, indent=2, sort_keys=True, default=_json_default) + "\n").encode("utf-8")
    path = _atomic_bytes(directory, MANIFEST_NAME, text)
    return path


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"manifest value is not JSON serializable: {type(value).__name__}")


def run_superconductivity_confirmatory(project_root: Path, output_dir: Path | None = None) -> dict:
    """Official entry point. Authorization is checked before any labeled file opens."""
    if output_dir is not None:
        assert_output_isolated(output_dir)
    require_superconductivity_authorization(project_root, "run-superconductivity-confirmatory")
    from p19.superconductivity_execution import execute_official_superconductivity

    return execute_official_superconductivity(project_root, output_dir)


def default_superconductivity_output_dir(staging_root: Path, stamp: str, protocol_hash: str) -> Path:
    path = Path(staging_root) / "artifacts" / RUN_PREFIX / f"{stamp}_super_{protocol_hash[:12]}"
    return assert_output_isolated(path)
