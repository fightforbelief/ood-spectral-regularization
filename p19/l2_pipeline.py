"""YearPredictionMSD L2 source phase and original-scale holdout scoring.

Source-side tuning and full-prefix refits finish for every declared repetition
and archived sample size before any holdout scoring function is called.
This module does not open a dataset and does not call ``evaluate_gate``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from p19.estimators import (
    EstimatorError,
    NumericalRankError,
    fit_min_norm,
    fit_pcr,
    fit_ridge,
)
from p19.l2_sampling import (
    DECLARED_REPETITIONS,
    NESTED_FOLD_SEED,
    ProtocolValidationError,
    longest_prefix_folds,
    nested_folds_for_sample_size,
    restrict_nested_folds,
    sampling_seeds_from_protocol,
    source_permutation,
)
from p19.metrics import metric_bundle
from p19.preprocessing import PreprocessError, Preprocessor, fit_preprocessor
from p19.protocol import (
    DRAFT_PCR_ROUNDING,
    DRAFT_ZERO_VARIANCE,
    PCR_FRACTION_GRID,
    YEAR_OFFICIAL_SOURCE_ROWS,
    YEAR_SAMPLE_SIZE_LIST_ORIGINAL,
    ridge_lambda_grid,
)
from p19.tuning import TuneResult, TuningError, tune_pcr, tune_ridge


class RunAborted(Exception):
    """A required source-side fit failed. Holdout scoring must not start."""

    def __init__(self, failure: dict):
        super().__init__(failure.get("message", "L2 source phase aborted"))
        self.failure = failure


class HoldoutScoringBlocked(Exception):
    """Official-holdout scoring was requested before a complete source phase."""


@dataclass
class PrefixFits:
    repetition: int
    sample_size: int
    sampling_seed: int
    prefix_indices: np.ndarray
    folds: list[np.ndarray]
    ridge_tune: TuneResult
    pcr_tune: TuneResult
    preprocessor: Preprocessor
    models: dict
    transformed_training_sha256: str
    ok: bool = True


@dataclass
class SourcePhaseResult:
    complete: bool
    cells: list[PrefixFits] = field(default_factory=list)
    failure: dict | None = None


def _matrix_sha256(matrix: np.ndarray) -> str:
    payload = np.ascontiguousarray(matrix, dtype=np.float64).tobytes()
    return hashlib.sha256(payload).hexdigest()


def refit_full_prefix(
    X_prefix: np.ndarray,
    y_prefix: np.ndarray,
    ridge_lambda: float,
    pcr_fraction: float,
    rounding_rule: str = DRAFT_PCR_ROUNDING,
    zero_variance_policy: str = DRAFT_ZERO_VARIANCE,
) -> tuple[Preprocessor, dict, str]:
    """Fit one preprocessor on the entire prefix and refit all three estimators on it.

    Ridge uses the L2 objective implemented by ``fit_ridge``.
    """
    preprocessor = fit_preprocessor(X_prefix, y_prefix, zero_variance_policy)
    transformed_x = preprocessor.transform_X(X_prefix)
    transformed_y = preprocessor.transform_y(y_prefix)
    models = {
        "min_norm": fit_min_norm(transformed_x, transformed_y),
        "ridge": fit_ridge(transformed_x, transformed_y, float(ridge_lambda)),
        "pcr": fit_pcr(transformed_x, transformed_y, pcr_fraction, rounding_rule),
    }
    return preprocessor, models, _matrix_sha256(transformed_x)


def fit_training_prefix(
    X_prefix: np.ndarray,
    y_prefix: np.ndarray,
    *,
    repetition: int,
    sampling_seed: int,
    sample_size: int,
    prefix_indices: np.ndarray,
    folds: list[np.ndarray] | None = None,
    fold_seed: int = NESTED_FOLD_SEED,
) -> PrefixFits:
    """Tune ridge and PCR on one shared prefix, then refit all three estimators."""
    matrix = np.asarray(X_prefix, dtype=np.float64)
    target = np.asarray(y_prefix, dtype=np.float64).reshape(-1)
    if matrix.shape[0] != sample_size or target.shape[0] != sample_size:
        raise ProtocolValidationError("prefix matrix rows must equal the archived sample size")
    assigned = folds if folds is not None else nested_folds_for_sample_size(sample_size, fold_seed)
    ridge_tune = tune_ridge(
        matrix,
        target,
        ridge_lambda_grid(),
        assigned,
        DRAFT_ZERO_VARIANCE,
    )
    pcr_tune = tune_pcr(
        matrix,
        target,
        list(PCR_FRACTION_GRID),
        assigned,
        DRAFT_ZERO_VARIANCE,
        DRAFT_PCR_ROUNDING,
    )
    if ridge_tune.selected is None or pcr_tune.selected is None:
        failures = list(ridge_tune.failures) + list(pcr_tune.failures)
        raise RunAborted(
            {
                "repetition": repetition,
                "sample_size": sample_size,
                "sampling_seed": sampling_seed,
                "message": "a required tuned estimator has no scorable candidate",
                "ridge_selected": ridge_tune.selected,
                "pcr_selected": pcr_tune.selected,
                "failures": failures,
            }
        )
    try:
        preprocessor, models, digest = refit_full_prefix(
            matrix, target, ridge_tune.selected, pcr_tune.selected
        )
    except (EstimatorError, NumericalRankError, PreprocessError, ValueError, FloatingPointError) as exc:
        raise RunAborted(
            {
                "repetition": repetition,
                "sample_size": sample_size,
                "sampling_seed": sampling_seed,
                "message": str(exc),
                "error": type(exc).__name__,
            }
        ) from exc
    return PrefixFits(
        repetition=repetition,
        sample_size=sample_size,
        sampling_seed=sampling_seed,
        prefix_indices=np.asarray(prefix_indices, dtype=np.int64).copy(),
        folds=[np.asarray(fold, dtype=np.int64).copy() for fold in assigned],
        ridge_tune=ridge_tune,
        pcr_tune=pcr_tune,
        preprocessor=preprocessor,
        models=models,
        transformed_training_sha256=digest,
    )


def run_source_phase(
    seeds: Sequence[int],
    sample_sizes: Sequence[int],
    execute_cell: Callable[..., PrefixFits],
    *,
    fold_seed: int = NESTED_FOLD_SEED,
) -> SourcePhaseResult:
    """Run every declared cell. Never call a holdout scorer from this function."""
    from p19.l2_sampling import validate_sampling_seed_list

    values = validate_sampling_seed_list(seeds, DECLARED_REPETITIONS)
    if tuple(int(size) for size in sample_sizes) != tuple(YEAR_SAMPLE_SIZE_LIST_ORIGINAL):
        raise ProtocolValidationError("the archived YearPredictionMSD sample-size list cannot change")
    if len(values) != DECLARED_REPETITIONS:
        raise ProtocolValidationError("the declared repetition count cannot change")
    cells: list[PrefixFits] = []
    for repetition, sampling_seed in enumerate(values):
        order = source_permutation(sampling_seed)
        folds_900 = longest_prefix_folds(fold_seed)
        for sample_size in YEAR_SAMPLE_SIZE_LIST_ORIGINAL:
            prefix = order[:sample_size]
            folds = restrict_nested_folds(folds_900, sample_size)
            try:
                cell = execute_cell(
                    repetition=repetition,
                    sampling_seed=sampling_seed,
                    sample_size=sample_size,
                    prefix_indices=prefix,
                    folds=folds,
                )
            except RunAborted as exc:
                return SourcePhaseResult(complete=False, cells=cells, failure=exc.failure)
            except (
                EstimatorError,
                NumericalRankError,
                PreprocessError,
                TuningError,
                ValueError,
                FloatingPointError,
            ) as exc:
                return SourcePhaseResult(
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
            if cell is None:
                return SourcePhaseResult(
                    complete=False,
                    cells=cells,
                    failure={
                        "repetition": repetition,
                        "sample_size": sample_size,
                        "sampling_seed": sampling_seed,
                        "message": "source cell returned no fit",
                    },
                )
            if not _cell_identity_matches(cell, repetition, sample_size):
                return SourcePhaseResult(
                    complete=False,
                    cells=cells,
                    failure={
                        "repetition": repetition,
                        "sample_size": sample_size,
                        "sampling_seed": sampling_seed,
                        "message": "source cell identity does not match the declared grid",
                    },
                )
            cells.append(cell)
    expected = DECLARED_REPETITIONS * len(YEAR_SAMPLE_SIZE_LIST_ORIGINAL)
    if len(cells) != expected:
        return SourcePhaseResult(
            complete=False,
            cells=cells,
            failure={"message": f"source phase produced {len(cells)} cells, not {expected}"},
        )
    return SourcePhaseResult(complete=True, cells=cells, failure=None)


def run_declared_source_phase(protocol_document: dict, execute_cell: Callable[..., PrefixFits]) -> SourcePhaseResult:
    """Use the protocol document's stored seed list and the archived sample sizes."""
    seeds = sampling_seeds_from_protocol(protocol_document)
    return run_source_phase(seeds, YEAR_SAMPLE_SIZE_LIST_ORIGINAL, execute_cell)


def evaluate_holdout(
    preprocessor: Preprocessor,
    model,
    X_holdout: np.ndarray,
    y_holdout: np.ndarray,
) -> dict:
    """Original-scale metrics for one fitted prefix. The caller supplies the rows.

    Predictions are restored with the training-target mean before scoring.
    """
    restored = preprocessor.restore_y(model.predict(preprocessor.transform_X(X_holdout)))
    bundle = metric_bundle(y_holdout, restored)
    return {
        "primary_endpoint": "unweighted_holdout_mse_original_scale",
        "mse": bundle["mse"],
        "rmse": bundle["rmse"],
        "mae": bundle["mae"],
        "r2": bundle["r2"],
        "n_rows": bundle["n_rows"],
        "predictions": restored,
        "training_target_mean": float(preprocessor.y_mean),
        "estimator": model.name,
        "selected": model.hyperparameter,
        "warnings": list(model.warnings),
        "transformed_feature_columns": int(preprocessor.n_cols),
    }


def declared_source_cell_keys() -> set[tuple[int, int]]:
    """Every required ``(repetition, archived sample size)`` pair, each once."""
    return {
        (repetition, int(sample_size))
        for repetition in range(DECLARED_REPETITIONS)
        for sample_size in YEAR_SAMPLE_SIZE_LIST_ORIGINAL
    }


def _integer_identifier(value, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise HoldoutScoringBlocked(f"{label} is not an integer")
    return int(value)


def _cell_identity_matches(cell, repetition: int, sample_size: int) -> bool:
    try:
        returned_repetition = _integer_identifier(cell.repetition, "repetition")
        returned_size = _integer_identifier(cell.sample_size, "sample size")
    except (AttributeError, HoldoutScoringBlocked):
        return False
    return returned_repetition == repetition and returned_size == sample_size


def _ready_source_cells(source) -> list:
    """Return cells only when they are exactly the declared grid. Do not score otherwise."""
    if source is None or isinstance(source, (str, bytes, dict, list, tuple)):
        raise HoldoutScoringBlocked("malformed source-phase result")
    try:
        complete = source.complete
        failure = source.failure
        cells = source.cells
    except AttributeError as exc:
        raise HoldoutScoringBlocked("malformed source-phase result") from exc
    if isinstance(cells, (str, bytes)) or not isinstance(cells, Sequence):
        raise HoldoutScoringBlocked("malformed source-phase result")
    expected = DECLARED_REPETITIONS * len(YEAR_SAMPLE_SIZE_LIST_ORIGINAL)
    if not complete or failure is not None or len(cells) != expected:
        raise HoldoutScoringBlocked(
            "refusing holdout scoring until all 100 repetitions and archived sample sizes succeed"
        )
    keys: list[tuple[int, int]] = []
    for cell in cells:
        try:
            repetition = _integer_identifier(cell.repetition, "repetition")
            sample_size = _integer_identifier(cell.sample_size, "sample size")
            models = cell.models
            if cell.preprocessor is None:
                raise HoldoutScoringBlocked("source cell is missing its preprocessor")
        except AttributeError as exc:
            raise HoldoutScoringBlocked("malformed source cell") from exc
        if not isinstance(models, dict) or any(name not in models or models[name] is None for name in ("min_norm", "ridge", "pcr")):
            raise HoldoutScoringBlocked("source cell is missing a required estimator")
        keys.append((repetition, sample_size))
    if len(keys) != len(set(keys)) or set(keys) != declared_source_cell_keys():
        raise HoldoutScoringBlocked(
            "source cells are not the exact declared repetition and sample-size grid"
        )
    return list(cells)


def score_completed_source_phase(
    source: SourcePhaseResult,
    X_holdout: np.ndarray,
    y_holdout: np.ndarray,
    evaluate: Callable = evaluate_holdout,
) -> list[dict]:
    """Score holdout rows only after every required source cell succeeded."""
    cells = _ready_source_cells(source)
    scored: list[dict] = []
    for cell in cells:
        for name in ("min_norm", "ridge", "pcr"):
            scored.append(
                evaluate(cell.preprocessor, cell.models[name], X_holdout, y_holdout)
            )
            scored[-1]["repetition"] = int(cell.repetition)
            scored[-1]["sample_size"] = int(cell.sample_size)
            scored[-1]["estimator"] = name
    return scored


def pool_row_count() -> int:
    """Source-pool length used by the permutation. Holdout rows are not included."""
    return int(YEAR_OFFICIAL_SOURCE_ROWS)
