"""Source-only five-fold selection. Test labels are not a parameter."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from p19.estimators import EstimatorError, NumericalRankError, fit_pcr, fit_ridge
from p19.metrics import mse
from p19.preprocessing import fit_preprocessor
from p19.protocol import DRAFT_FOLD_RULE, DRAFT_TIE_BREAK, DRAFT_CV_AGGREGATION


class TuningError(Exception):
    pass


@dataclass
class TuneResult:
    estimator: str
    selected: float | None
    selected_index: int | None
    fold_indices: list[np.ndarray]
    mean_scores: list[float | None]
    failures: list[dict]
    tie_break: str
    aggregation: str
    warnings: list[str] = field(default_factory=list)


def fold_training_feature_means(
    X: np.ndarray,
    y: np.ndarray,
    folds: list[np.ndarray],
    zero_variance_policy: str,
) -> list[np.ndarray]:
    """Fit preprocessing on each fold's training rows and return those feature means."""
    matrix = np.asarray(X, dtype=np.float64)
    target = np.asarray(y, dtype=np.float64).reshape(-1)
    means = []
    n_rows = matrix.shape[0]
    for validation_index in folds:
        mask = np.ones(n_rows, dtype=bool)
        mask[validation_index] = False
        pre = fit_preprocessor(matrix[mask], target[mask], zero_variance_policy)
        means.append(pre.x_mean.copy())
    return means


def assign_source_folds(
    n_source_rows: int,
    n_folds: int = 5,
    seed: int = 0,
    rule: str = DRAFT_FOLD_RULE,
) -> list[np.ndarray]:
    if rule != DRAFT_FOLD_RULE:
        raise TuningError(f"fold rule {rule!r} is not an implemented named rule")
    if n_folds != 5:
        raise TuningError("the selection report specifies five source folds")
    if n_source_rows < n_folds:
        raise TuningError("not enough source rows for five folds")
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(n_source_rows)
    folds = [np.sort(permutation[i::n_folds].astype(np.int64)) for i in range(n_folds)]
    covered = np.concatenate(folds)
    if covered.size != n_source_rows or np.unique(covered).size != n_source_rows:
        raise TuningError("fold assignment did not partition the source rows")
    if np.any(covered < 0) or np.any(covered >= n_source_rows):
        raise TuningError("fold index escaped the source row range")
    return folds


def _select_index(scores: list[float | None], tie_break: str) -> int | None:
    if tie_break != DRAFT_TIE_BREAK:
        raise TuningError(f"tie break {tie_break!r} is not an implemented named rule")
    best_index = None
    best_score = None
    for index, score in enumerate(scores):
        if score is None or not np.isfinite(score):
            continue
        if best_score is None or score < best_score:
            best_score = score
            best_index = index
        # Equal scores keep the earlier ascending-grid index.
    return best_index


def _cv_scores(
    X: np.ndarray,
    y: np.ndarray,
    folds: list[np.ndarray],
    candidates: list,
    fit_one,
    zero_variance_policy: str,
    aggregation: str,
) -> tuple[list[float | None], list[dict], list[np.ndarray]]:
    if aggregation != DRAFT_CV_AGGREGATION:
        raise TuningError(f"CV aggregation {aggregation!r} is not an implemented named rule")
    matrix = np.asarray(X, dtype=np.float64)
    target = np.asarray(y, dtype=np.float64).reshape(-1)
    n_rows = matrix.shape[0]
    for fold in folds:
        if np.any(fold < 0) or np.any(fold >= n_rows):
            raise TuningError("refusing folds that point outside the source matrix")
    mean_scores: list[float | None] = []
    failures: list[dict] = []
    fold_means: list[np.ndarray] = []
    # Record preprocessing means from the first candidate's folds so tests can
    # see that each fold was fit on its own training rows. The means do not
    # depend on the candidate.
    recorded = False
    for candidate_index, candidate in enumerate(candidates):
        fold_scores: list[float] = []
        failed = False
        for fold_id, validation_index in enumerate(folds):
            mask = np.ones(n_rows, dtype=bool)
            mask[validation_index] = False
            train_index = np.flatnonzero(mask)
            try:
                pre = fit_preprocessor(
                    matrix[train_index], target[train_index], zero_variance_policy
                )
                if not recorded:
                    fold_means.append(pre.x_mean.copy())
                model = fit_one(
                    pre.transform_X(matrix[train_index]),
                    pre.transform_y(target[train_index]),
                    candidate,
                )
                prediction = pre.restore_y(
                    model.predict(pre.transform_X(matrix[validation_index]))
                )
                fold_scores.append(mse(target[validation_index], prediction))
            except (EstimatorError, NumericalRankError, ValueError, FloatingPointError) as exc:
                failed = True
                failures.append(
                    {
                        "candidate_index": candidate_index,
                        "candidate": str(candidate),
                        "fold": fold_id,
                        "error": type(exc).__name__,
                        "message": str(exc),
                    }
                )
                break
        recorded = True
        if failed or len(fold_scores) != len(folds):
            mean_scores.append(None)
        else:
            mean_scores.append(float(np.mean(np.asarray(fold_scores, dtype=np.float64))))
    return mean_scores, failures, fold_means


def tune_ridge(
    X: np.ndarray,
    y: np.ndarray,
    lambdas: np.ndarray,
    folds: list[np.ndarray],
    zero_variance_policy: str,
    tie_break: str = DRAFT_TIE_BREAK,
    aggregation: str = DRAFT_CV_AGGREGATION,
) -> TuneResult:
    grid = [float(value) for value in np.asarray(lambdas, dtype=np.float64)]
    scores, failures, _means = _cv_scores(
        X, y, folds, grid,
        lambda xt, yt, lam: fit_ridge(xt, yt, lam),
        zero_variance_policy,
        aggregation,
    )
    chosen = _select_index(scores, tie_break)
    return TuneResult(
        estimator="ridge",
        selected=None if chosen is None else grid[chosen],
        selected_index=chosen,
        fold_indices=[np.asarray(fold).copy() for fold in folds],
        mean_scores=scores,
        failures=failures,
        tie_break=tie_break,
        aggregation=aggregation,
        warnings=[] if chosen is not None else ["all ridge candidates failed"],
    )


def tune_pcr(
    X: np.ndarray,
    y: np.ndarray,
    fractions: list,
    folds: list[np.ndarray],
    zero_variance_policy: str,
    rounding_rule: str,
    tie_break: str = DRAFT_TIE_BREAK,
    aggregation: str = DRAFT_CV_AGGREGATION,
) -> TuneResult:
    scores, failures, _means = _cv_scores(
        X, y, folds, list(fractions),
        lambda xt, yt, q: fit_pcr(xt, yt, q, rounding_rule),
        zero_variance_policy,
        aggregation,
    )
    chosen = _select_index(scores, tie_break)
    selected = None if chosen is None else float(fractions[chosen])
    return TuneResult(
        estimator="pcr",
        selected=selected,
        selected_index=chosen,
        fold_indices=[np.asarray(fold).copy() for fold in folds],
        mean_scores=scores,
        failures=failures,
        tie_break=tie_break,
        aggregation=aggregation,
        warnings=[] if chosen is not None else ["all PCR candidates failed"],
    )
