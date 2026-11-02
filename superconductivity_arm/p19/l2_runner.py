"""Gated YearPredictionMSD confirmatory runner.

``evaluate_gate`` is the only authorization check. Data files stay closed until
that check returns an approved decision. Tests can inject the check, the loader,
and the cell executor; an injected dependency cannot be written out as a
confirmatory result.
"""

from __future__ import annotations

import hashlib
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

from p19.estimators import pcr_component_count
from p19.gates import GateDecision, GateRefusal, evaluate_gate
from p19.l2_pipeline import (
    declared_source_cell_keys,
    evaluate_holdout,
    fit_training_prefix,
    run_declared_source_phase,
    score_completed_source_phase,
)
from p19.l2_sampling import rows_for_prefix
from p19.protocol import (
    CODE_IDENTITY_KIND,
    DRAFT_PCR_ROUNDING,
    YEAR_OFFICIAL_SOURCE_ROWS,
    YEAR_OFFICIAL_TARGET_ROWS,
    YEAR_SELECTION_PREDICTOR_COUNT,
    YEAR_SAMPLE_SIZE_LIST_ORIGINAL,
    load_yaml,
)
from p19.reporting import write_json
from p19.uncertainty import (
    APPROVED_TRAINING_SUBSET_PROCEDURE,
    UncertaintyNotFrozen,
    adjacent_sample_size_differences,
    paired_estimator_loss_differences,
    training_subset_variability,
)

YEAR_TEXT_NAME = "YearPredictionMSD.txt"
PREDICTION_ARTIFACT_NAME = "holdout_predictions.npy"
PREDICTION_PARTIAL_NAME = "holdout_predictions.partial.npy"
PREDICTION_ESTIMATOR_ORDER = ("min_norm", "ridge", "pcr")


class RunnerError(Exception):
    """The gated runner stopped before a confirmatory result existed."""


class PredictionArtifactError(Exception):
    """Holdout predictions were scored but the artifact contract was not satisfied."""

    def __init__(
        self,
        message: str,
        *,
        operation: str | None = None,
        path: str | None = None,
        original_type: str | None = None,
    ) -> None:
        super().__init__(message)
        self.operation = operation
        self.path = path
        self.original_type = original_type


def _filesystem_artifact_error(operation: str, path: Path, exc: OSError) -> PredictionArtifactError:
    """Describe one filesystem failure. Programming errors stay unchanged."""
    return PredictionArtifactError(
        f"{operation} failed for {path}: {type(exc).__name__}: {exc}",
        operation=operation,
        path=str(path),
        original_type=type(exc).__name__,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def runtime_environment() -> dict:
    """Versions and BLAS thread settings. This does not open a dataset."""
    import scipy

    try:
        from threadpoolctl import threadpool_info

        threads = threadpool_info()
    except Exception as exc:
        threads = [{"error": type(exc).__name__, "message": str(exc)}]
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "threadpool_info": threads,
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
    }


def validate_yearprediction_arrays(payload: dict) -> dict:
    """Require the official source/holdout shapes and finite values before any fit."""
    required = ("X_source", "y_source", "X_holdout", "y_holdout")
    if any(name not in payload for name in required):
        raise RunnerError("yearprediction arrays are missing source or holdout values")
    x_source = np.asarray(payload["X_source"], dtype=np.float64)
    y_source = np.asarray(payload["y_source"], dtype=np.float64).reshape(-1)
    x_holdout = np.asarray(payload["X_holdout"], dtype=np.float64)
    y_holdout = np.asarray(payload["y_holdout"], dtype=np.float64).reshape(-1)
    expected = {
        "X_source": (YEAR_OFFICIAL_SOURCE_ROWS, YEAR_SELECTION_PREDICTOR_COUNT),
        "y_source": (YEAR_OFFICIAL_SOURCE_ROWS,),
        "X_holdout": (YEAR_OFFICIAL_TARGET_ROWS, YEAR_SELECTION_PREDICTOR_COUNT),
        "y_holdout": (YEAR_OFFICIAL_TARGET_ROWS,),
    }
    observed = {
        "X_source": x_source.shape,
        "y_source": y_source.shape,
        "X_holdout": x_holdout.shape,
        "y_holdout": y_holdout.shape,
    }
    if observed != expected:
        raise RunnerError(f"yearprediction array shapes {observed} are not {expected}")
    for name, array in (
        ("X_source", x_source),
        ("y_source", y_source),
        ("X_holdout", x_holdout),
        ("y_holdout", y_holdout),
    ):
        if not np.isfinite(array).all():
            raise RunnerError(f"{name} contains a non-finite value")
    return {
        "X_source": x_source,
        "y_source": y_source,
        "X_holdout": x_holdout,
        "y_holdout": y_holdout,
    }


def _production_executor(x_source: np.ndarray, y_source: np.ndarray):
    def execute_cell(**kwargs):
        prefix_x, prefix_y = rows_for_prefix(x_source, y_source, kwargs["prefix_indices"])
        return fit_training_prefix(
            prefix_x,
            prefix_y,
            repetition=kwargs["repetition"],
            sampling_seed=kwargs["sampling_seed"],
            sample_size=kwargs["sample_size"],
            prefix_indices=kwargs["prefix_indices"],
            folds=kwargs["folds"],
        )

    return execute_cell


def _selected(cell, tune_name: str):
    tune = getattr(cell, tune_name, None)
    if tune is None:
        return None
    selected = getattr(tune, "selected", None)
    return None if selected is None else float(selected)


def _failure_rows(cell, tune_name: str):
    tune = getattr(cell, tune_name, None)
    if tune is None:
        return None
    return list(getattr(tune, "failures", []) or [])


def _pcr_component_count(cell):
    tune = getattr(cell, "pcr_tune", None)
    models = getattr(cell, "models", None)
    if tune is None or getattr(tune, "selected", None) is None or not isinstance(models, dict):
        return None
    model = models.get("pcr")
    rank = getattr(model, "numerical_rank", None)
    if rank is None:
        return None
    return int(pcr_component_count(tune.selected, int(rank), DRAFT_PCR_ROUNDING))


def _cell_log(cell, requested: dict) -> dict:
    return {
        "repetition": int(requested["repetition"]),
        "sample_size": int(requested["sample_size"]),
        "sampling_seed": int(requested["sampling_seed"]),
        "selected_ridge_lambda": _selected(cell, "ridge_tune"),
        "selected_pcr_fraction": _selected(cell, "pcr_tune"),
        "pcr_component_count": _pcr_component_count(cell),
        "ridge_failures": _failure_rows(cell, "ridge_tune"),
        "pcr_failures": _failure_rows(cell, "pcr_tune"),
    }


def _metric_row(row: dict) -> dict:
    kept = {}
    for key in (
        "repetition",
        "sample_size",
        "estimator",
        "primary_endpoint",
        "mse",
        "rmse",
        "mae",
        "r2",
        "n_rows",
        "selected",
        "training_target_mean",
    ):
        if key in row:
            value = row[key]
            if isinstance(value, np.ndarray):
                continue
            kept[key] = value.item() if isinstance(value, np.generic) else value
    return kept


def prediction_identity_order(n_repetitions: int, sample_sizes) -> list[tuple[int, int, str]]:
    """Deterministic artifact rows. This does not change execution order."""
    if isinstance(n_repetitions, bool) or not isinstance(n_repetitions, (int, np.integer)):
        raise PredictionArtifactError("repetition count is not an integer")
    if int(n_repetitions) < 1:
        raise PredictionArtifactError("repetition count is empty")
    order: list[tuple[int, int, str]] = []
    for repetition in range(int(n_repetitions)):
        for sample_size in sample_sizes:
            size = int(sample_size)
            for estimator in PREDICTION_ESTIMATOR_ORDER:
                order.append((repetition, size, estimator))
    if len(order) != len(set(order)):
        raise PredictionArtifactError("prediction identity order contains a duplicate")
    return order


def _prediction_key(row: dict) -> tuple[int, int, str]:
    try:
        repetition = row["repetition"]
        sample_size = row["sample_size"]
        estimator = row["estimator"]
    except KeyError as exc:
        raise PredictionArtifactError("scored row is missing prediction identity") from exc
    if isinstance(repetition, bool) or not isinstance(repetition, (int, np.integer)):
        raise PredictionArtifactError("repetition is not an integer")
    if isinstance(sample_size, bool) or not isinstance(sample_size, (int, np.integer)):
        raise PredictionArtifactError("sample size is not an integer")
    if not isinstance(estimator, str):
        raise PredictionArtifactError("estimator is not a string")
    return int(repetition), int(sample_size), estimator


def _finite_prediction_vector(value, required_length: int | None) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise PredictionArtifactError("prediction vector is missing")
    if value.ndim != 1:
        raise PredictionArtifactError("prediction vector must be one-dimensional")
    try:
        vector = np.ascontiguousarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise PredictionArtifactError("prediction vector is not numeric") from exc
    if required_length is not None and vector.shape[0] != required_length:
        raise PredictionArtifactError("prediction vector has the wrong length")
    if vector.shape[0] < 1 or not np.isfinite(vector).all():
        raise PredictionArtifactError("prediction vector is not finite")
    return vector


def align_prediction_vectors(scored_rows, order, required_length: int | None) -> list[np.ndarray]:
    """Return vectors in artifact order. Missing, duplicate, or extra keys fail."""
    expected = set(order)
    located: dict[tuple[int, int, str], np.ndarray] = {}
    for row in scored_rows:
        key = _prediction_key(row)
        if key not in expected or key in located:
            raise PredictionArtifactError("duplicate or unexpected prediction identity")
        located[key] = _finite_prediction_vector(row.get("predictions"), required_length)
    if set(located) != expected:
        raise PredictionArtifactError("prediction grid is incomplete")
    lengths = {int(located[key].shape[0]) for key in order}
    if len(lengths) != 1:
        raise PredictionArtifactError("prediction vectors have different lengths")
    return [located[key] for key in order]


def _release_memmap(array) -> None:
    mapped = getattr(array, "_mmap", None)
    if mapped is not None and not getattr(mapped, "closed", False):
        mapped.close()


def _close_memmap_quietly(array) -> None:
    try:
        _release_memmap(array)
    except OSError:
        return


def write_holdout_prediction_artifact(destination: Path, vectors: list[np.ndarray]) -> tuple[dict, list[str]]:
    """Write one float64 row at a time. A finished file is never overwritten."""
    if not vectors:
        raise PredictionArtifactError("prediction grid is empty")
    if any(np.asarray(vector).ndim != 1 for vector in vectors):
        raise PredictionArtifactError("prediction vector must be one-dimensional")
    width = int(vectors[0].shape[0])
    if width < 1 or any(int(vector.shape[0]) != width for vector in vectors):
        raise PredictionArtifactError("prediction vectors have different lengths")
    directory = Path(destination)
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise _filesystem_artifact_error("output directory creation", directory, exc) from exc
    final_path = directory / PREDICTION_ARTIFACT_NAME
    partial_path = directory / PREDICTION_PARTIAL_NAME
    try:
        final_exists = final_path.exists()
        partial_exists = partial_path.exists()
    except OSError as exc:
        raise _filesystem_artifact_error("path existence check", directory, exc) from exc
    if final_exists:
        raise PredictionArtifactError("completed prediction artifact already exists")
    if partial_exists:
        raise PredictionArtifactError("incomplete prediction artifact already exists")
    shape = (len(vectors), width)
    try:
        memmap = open_memmap(partial_path, mode="w+", dtype=np.float64, shape=shape)
    except OSError as exc:
        raise _filesystem_artifact_error("open_memmap", partial_path, exc) from exc
    hashes: list[str] = []
    writes_finished = False
    try:
        for index, vector in enumerate(vectors):
            row = np.ascontiguousarray(vector, dtype=np.float64)
            if row.shape != (width,) or row.dtype != np.float64 or not np.isfinite(row).all():
                raise PredictionArtifactError("prediction row failed validation during write")
            try:
                memmap[index, :] = row
            except OSError as exc:
                raise _filesystem_artifact_error("memory-mapped row write", partial_path, exc) from exc
            hashes.append(hashlib.sha256(row.tobytes()).hexdigest())
        try:
            memmap.flush()
        except OSError as exc:
            raise _filesystem_artifact_error("memmap flush", partial_path, exc) from exc
        writes_finished = True
    finally:
        if writes_finished:
            try:
                _release_memmap(memmap)
            except OSError as exc:
                raise _filesystem_artifact_error("memmap close", partial_path, exc) from exc
        else:
            _close_memmap_quietly(memmap)
    del memmap
    try:
        final_exists = final_path.exists()
    except OSError as exc:
        raise _filesystem_artifact_error("path existence check", final_path, exc) from exc
    if final_exists:
        raise PredictionArtifactError("completed prediction artifact already exists")
    try:
        file_sha256 = _sha256_file(partial_path)
    except OSError as exc:
        raise _filesystem_artifact_error("full-file SHA-256", partial_path, exc) from exc
    try:
        os.replace(partial_path, final_path)
    except OSError as exc:
        raise _filesystem_artifact_error("atomic rename", partial_path, exc) from exc
    record = {
        "relative_path": PREDICTION_ARTIFACT_NAME,
        "format": "npy",
        "dtype": "float64",
        "shape": [int(shape[0]), int(shape[1])],
        "n_prediction_vectors": int(shape[0]),
        "holdout_row_count": int(shape[1]),
        "holdout_file_row_start": int(YEAR_OFFICIAL_SOURCE_ROWS),
        "holdout_file_row_end_exclusive": int(YEAR_OFFICIAL_SOURCE_ROWS + YEAR_OFFICIAL_TARGET_ROWS),
        "row_order": (
            "repetition ascending from 0; within each repetition the archived "
            "YearPredictionMSD sample sizes 45, 68, 81, 86, 88, 90, 92, 95, 99, "
            "113, 135, 180, 270, 450, 900; within each sample size the estimators "
            "min_norm, ridge, pcr"
        ),
        "sha256": file_sha256,
        "vector_sha256": "sha256 of np.ascontiguousarray(vector, dtype=float64).tobytes()",
    }
    return record, hashes


def _rows_have_prediction_vectors(scored_rows) -> bool:
    flags = [isinstance(row.get("predictions"), np.ndarray) for row in scored_rows]
    if not flags or not any(flags):
        return False
    if not all(flags):
        raise PredictionArtifactError("prediction vectors do not cover every scored row")
    return True


def attach_prediction_artifact(destination: Path, scored_rows, *, production: bool) -> tuple[dict, list[dict]]:
    """Validate, write, and return manifest metrics that still contain the scalar losses."""
    order = prediction_identity_order(100, YEAR_SAMPLE_SIZE_LIST_ORIGINAL)
    required_length = YEAR_OFFICIAL_TARGET_ROWS if production else None
    vectors = align_prediction_vectors(scored_rows, order, required_length)
    located = {_prediction_key(row): row for row in scored_rows}
    record, hashes = write_holdout_prediction_artifact(destination, vectors)
    metrics: list[dict] = []
    for index, key in enumerate(order):
        metric = _metric_row(located[key])
        metric["prediction_artifact_row"] = index
        metric["prediction_sha256"] = hashes[index]
        metrics.append(metric)
    return record, metrics


def uncertainty_from_scored_rows(scored_rows: list[dict]) -> dict:
    """Align holdout MSE by repetition identifier before any paired difference.

    Input order is not used. A missing or duplicate repetition is rejected.
    """
    expected = {
        (repetition, sample_size, estimator)
        for repetition, sample_size in declared_source_cell_keys()
        for estimator in ("min_norm", "ridge", "pcr")
    }
    located: dict[tuple[int, int, str], float] = {}
    for row in scored_rows:
        try:
            repetition = int(row["repetition"])
            sample_size = int(row["sample_size"])
            estimator = str(row["estimator"])
            mse = float(row["mse"])
        except (KeyError, TypeError, ValueError) as exc:
            raise UncertaintyNotFrozen("scored row is missing repetition identity") from exc
        if not np.isfinite(mse):
            raise UncertaintyNotFrozen("scored MSE is not finite")
        key = (repetition, sample_size, estimator)
        if key in located or key not in expected:
            raise UncertaintyNotFrozen("scored rows are not the declared repetition grid")
        located[key] = mse
    if set(located) != expected:
        raise UncertaintyNotFrozen("scored rows are not the declared repetition grid")
    repetition_ids = np.arange(100, dtype=np.int64)
    paired = {}
    by_estimator = {}
    for sample_size in YEAR_SAMPLE_SIZE_LIST_ORIGINAL:
        size = int(sample_size)
        min_norm = np.array([located[(rep, size, "min_norm")] for rep in range(100)])
        ridge = np.array([located[(rep, size, "ridge")] for rep in range(100)])
        pcr = np.array([located[(rep, size, "pcr")] for rep in range(100)])
        paired[str(size)] = paired_estimator_loss_differences(
            min_norm,
            ridge,
            pcr,
            APPROVED_TRAINING_SUBSET_PROCEDURE,
            repetition_ids={
                "min_norm": repetition_ids,
                "ridge": repetition_ids,
                "pcr": repetition_ids,
            },
        )
    for estimator in ("min_norm", "ridge", "pcr"):
        risks = {
            int(sample_size): np.array(
                [located[(rep, int(sample_size), estimator)] for rep in range(100)]
            )
            for sample_size in YEAR_SAMPLE_SIZE_LIST_ORIGINAL
        }
        by_estimator[estimator] = {
            "levels": {
                str(int(sample_size)): training_subset_variability(
                    risks[int(sample_size)],
                    APPROVED_TRAINING_SUBSET_PROCEDURE,
                )
                for sample_size in YEAR_SAMPLE_SIZE_LIST_ORIGINAL
            },
            "adjacent": adjacent_sample_size_differences(
                risks,
                APPROVED_TRAINING_SUBSET_PROCEDURE,
                repetition_ids=repetition_ids,
            ),
        }
    return {
        "procedure": APPROVED_TRAINING_SUBSET_PROCEDURE,
        "pairing": "repetition_identifier",
        "paired_by_sample_size": paired,
        "by_estimator": by_estimator,
    }


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_manifest(output_dir: Path, payload: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "l2_run_manifest.json"
    write_json(path, _jsonable(payload))
    return path


def _artifact_error_record(exc: PredictionArtifactError) -> dict:
    record = {"error": type(exc).__name__, "message": str(exc)}
    if exc.operation is not None:
        record["operation"] = exc.operation
    if exc.path is not None:
        record["path"] = exc.path
    if exc.original_type is not None:
        record["original_type"] = exc.original_type
    return record


def _incomplete_prediction_result(base: dict, destination: Path, scored, exc: PredictionArtifactError) -> dict:
    """Record an artifact failure. Do not claim a manifest that was not written."""
    base["status"] = "INCOMPLETE"
    base["confirmatory_l2"] = "INCOMPLETE_PREDICTION_ARTIFACT"
    base["confirmatory_result"] = False
    base["target_labels_loaded"] = True
    base["holdout_scored"] = True
    base["prediction_artifact_satisfied"] = False
    base["prediction_artifact_error"] = _artifact_error_record(exc)
    partial = destination / PREDICTION_PARTIAL_NAME
    try:
        partial_is_file = partial.is_file()
    except OSError as io_exc:
        print(
            "incomplete prediction artifact could not be identified: "
            f"path existence check failed for {partial}: {type(io_exc).__name__}: {io_exc}",
            file=sys.stderr,
        )
        partial_is_file = False
    if partial_is_file:
        base["incomplete_prediction_artifact"] = PREDICTION_PARTIAL_NAME
    base["holdout_metrics"] = [_metric_row(row) for row in scored]
    manifest_path = destination / "l2_run_manifest.json"
    try:
        written = _write_manifest(destination, base)
    except OSError as io_exc:
        base.pop("manifest", None)
        base["manifest_write_failed"] = True
        print(
            "prediction artifact failure manifest was not written: "
            f"manifest write failed for {manifest_path}: {type(io_exc).__name__}: {io_exc}",
            file=sys.stderr,
        )
        return base
    base["manifest"] = str(written)
    return base


def run_yearprediction_confirmatory(
    project_root: Path,
    *,
    gate_check=None,
    load_arrays=None,
    execute_cell=None,
    evaluate=None,
    output_dir: Path | None = None,
) -> dict:
    """Run the declared YearPredictionMSD plan after the existing gate approves.

    The default ``gate_check`` is ``evaluate_gate``. The default loader opens
    ``YearPredictionMSD.txt`` only after that call approves. Injected callables
    are test seams. Their results are marked synthetic and are not confirmatory.
    """
    root = Path(project_root)
    injected = any(value is not None for value in (gate_check, load_arrays, execute_cell, evaluate))
    checker = evaluate_gate if gate_check is None else gate_check
    decision = checker(root, "run-l2-confirmatory")
    approved = isinstance(decision, GateDecision) and decision.approved is True
    if not approved:
        reasons = list(getattr(decision, "reasons", []) or ["gate refused"])
        raise GateRefusal(reasons)

    protocol_path = root / "reports" / "protocol_freeze.yaml"
    document = load_yaml(protocol_path)
    data_path = root / "data" / "raw" / YEAR_TEXT_NAME
    if load_arrays is None:
        from p19.data_loading import load_yearprediction_confirmatory_arrays

        payload = load_yearprediction_confirmatory_arrays(data_path, decision)
    else:
        payload = load_arrays(data_path, decision)
    try:
        arrays = validate_yearprediction_arrays(payload)
    except RunnerError as exc:
        rejected = {
            "status": "INCOMPLETE",
            "confirmatory_result": False,
            "synthetic": injected,
            "holdout_scored": False,
            "official_data_opened_by_this_call": load_arrays is None,
            "confirmatory_l2": "INCOMPLETE_NO_HOLDOUT_SCORE",
            "target_labels_loaded": bool(payload.get("labels_loaded")),
            "gate_approved": True,
            "protocol_file_sha256": _sha256_file(protocol_path),
            "protocol_status": document.get("status"),
            "confirmatory_l2_authorized": document.get("confirmatory_l2_authorized"),
            "code_identity_sha256": document.get("code_identity_sha256"),
            "recorded_utc": _utc(),
            "source_complete": False,
            "source_failure": {"error": "RunnerError", "message": str(exc)},
            "n_source_cells": 0,
            "source_cells": [],
        }
        destination = Path(output_dir) if output_dir is not None else _default_output_dir(root, decision)
        rejected["manifest"] = str(_write_manifest(destination, rejected))
        return rejected
    cell_log: list[dict] = []
    user_executor = execute_cell

    def recorded_executor(**kwargs):
        if user_executor is None:
            cell = _production_executor(arrays["X_source"], arrays["y_source"])(**kwargs)
        else:
            cell = user_executor(**kwargs)
        cell_log.append(_cell_log(cell, kwargs))
        return cell

    source = run_declared_source_phase(document, recorded_executor)
    destination = Path(output_dir) if output_dir is not None else _default_output_dir(root, decision)
    base = {
        "status": "INCOMPLETE",
        "confirmatory_result": False,
        "synthetic": injected,
        "holdout_scored": False,
        "official_data_opened_by_this_call": load_arrays is None,
        "gate_action": "run-l2-confirmatory",
        "gate_approved": True,
        "injected_gate_protocol_sha256": getattr(decision, "protocol_sha256", None),
        "protocol_file_sha256": _sha256_file(protocol_path),
        "protocol_status": document.get("status"),
        "confirmatory_l2_authorized": document.get("confirmatory_l2_authorized"),
        "code_identity_kind": document.get("code_identity_kind", CODE_IDENTITY_KIND),
        "code_identity_sha256": document.get("code_identity_sha256"),
        "recorded_utc": _utc(),
        "environment": runtime_environment(),
        "source_cells": cell_log,
        "source_complete": bool(source.complete),
        "source_failure": source.failure,
        "n_source_cells": len(source.cells),
    }
    if not source.complete or source.failure is not None:
        base["confirmatory_l2"] = "INCOMPLETE_NO_HOLDOUT_SCORE"
        base["target_labels_loaded"] = True
        manifest_path = _write_manifest(destination, base)
        base["manifest"] = str(manifest_path)
        return base

    scorer = evaluate_holdout if evaluate is None else evaluate
    try:
        scored = score_completed_source_phase(
            source,
            arrays["X_holdout"],
            arrays["y_holdout"],
            evaluate=scorer,
        )
    except Exception as exc:
        base["confirmatory_l2"] = "INCOMPLETE_NO_HOLDOUT_SCORE"
        base["target_labels_loaded"] = True
        base["source_failure"] = {"error": type(exc).__name__, "message": str(exc)}
        manifest_path = _write_manifest(destination, base)
        base["manifest"] = str(manifest_path)
        return base
    artifact = None
    try:
        if (not injected) or _rows_have_prediction_vectors(scored):
            artifact, metrics = attach_prediction_artifact(
                destination,
                scored,
                production=not injected,
            )
        else:
            metrics = [_metric_row(row) for row in scored]
    except PredictionArtifactError as exc:
        return _incomplete_prediction_result(base, destination, scored, exc)
    try:
        uncertainty = uncertainty_from_scored_rows(metrics)
    except UncertaintyNotFrozen as exc:
        base["confirmatory_l2"] = "INCOMPLETE_NO_UNCERTAINTY"
        base["target_labels_loaded"] = True
        base["holdout_scored"] = True
        base["holdout_metrics"] = metrics
        base["uncertainty_error"] = str(exc)
        if artifact is not None:
            base["prediction_artifact"] = artifact
            base["prediction_artifact_satisfied"] = True
        manifest_path = _write_manifest(destination, base)
        base["manifest"] = str(manifest_path)
        return base
    complete = {
        **base,
        "status": "COMPLETE" if not injected else "SYNTHETIC_COMPLETE",
        "confirmatory_result": not injected,
        "confirmatory_l2": "COMPLETE" if not injected else "SYNTHETIC_COMPLETE",
        "target_labels_loaded": True,
        "holdout_scored": True,
        "holdout_metrics": metrics,
        "uncertainty": uncertainty,
    }
    if artifact is not None:
        complete["prediction_artifact"] = artifact
        complete["prediction_artifact_satisfied"] = True
    manifest_path = _write_manifest(destination, complete)
    complete["manifest"] = str(manifest_path)
    return complete


def _default_output_dir(root: Path, decision: GateDecision) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    digest = str(getattr(decision, "protocol_sha256", None) or "nohash")
    path = root / "artifacts" / "runs" / f"{stamp}_l2_{digest[:12]}"
    if path.exists():
        raise FileExistsError(path)
    return path
