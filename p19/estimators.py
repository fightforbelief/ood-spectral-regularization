"""Explicit float64 SVD estimators. No library ridge scaling is used."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
from scipy.linalg import svd

from p19.protocol import DRAFT_PCR_ROUNDING


class EstimatorError(Exception):
    """A fit failed in a way the caller must record, not hide."""


class NumericalRankError(EstimatorError):
    """PCR was asked to retain components when the numerical rank is zero."""


@dataclass
class FitResult:
    name: str
    n: int
    p: int
    singular_values: np.ndarray
    cutoff: float | None
    numerical_rank: int | None
    hyperparameter: float | None
    coefficient: np.ndarray
    intercept: float
    warnings: list[str] = field(default_factory=list)
    dtype: str = "float64"

    def predict(self, X: np.ndarray) -> np.ndarray:
        matrix = _as_float64_matrix(X)
        if matrix.shape[1] != self.p:
            raise EstimatorError(
                f"{self.name} expected {self.p} columns, received {matrix.shape[1]}"
            )
        prediction = matrix @ self.coefficient + self.intercept
        if not np.isfinite(prediction).all():
            raise EstimatorError(f"{self.name} produced nonfinite predictions")
        return prediction


def _as_float64_matrix(X: np.ndarray) -> np.ndarray:
    matrix = np.asarray(X, dtype=np.float64, order="C")
    if matrix.ndim != 2:
        raise EstimatorError("design matrix must be two-dimensional")
    if matrix.dtype != np.float64:
        raise EstimatorError("design matrix is not float64")
    if not np.isfinite(matrix).all():
        raise EstimatorError("design matrix contains nonfinite values")
    return matrix


def _as_float64_target(y: np.ndarray, n_rows: int) -> np.ndarray:
    target = np.asarray(y, dtype=np.float64).reshape(-1)
    if target.shape != (n_rows,):
        raise EstimatorError(
            f"target length {target.shape[0]} does not match n={n_rows}"
        )
    if not np.isfinite(target).all():
        raise EstimatorError("target contains nonfinite values")
    return target


def economy_svd(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Float64 economy SVD via SciPy/LAPACK (gesdd), full_matrices=False."""
    matrix = _as_float64_matrix(X)
    left, singular, right_h = svd(
        matrix,
        full_matrices=False,
        lapack_driver="gesdd",
        check_finite=True,
        overwrite_a=False,
    )
    if left.dtype != np.float64 or singular.dtype != np.float64 or right_h.dtype != np.float64:
        raise EstimatorError("SVD did not return float64 factors")
    return left, singular, right_h


def numerical_cutoff(singular_values: np.ndarray, n_rows: int, n_cols: int) -> float:
    """tau = eps64 * max(n, p) * sigma_max. Zero when the matrix is identically zero."""
    if singular_values.size == 0:
        return 0.0
    sigma_max = float(singular_values[0])
    eps = float(np.finfo(np.float64).eps)
    return eps * float(max(n_rows, n_cols)) * sigma_max


def numerical_rank(singular_values: np.ndarray, cutoff: float) -> int:
    if singular_values.size == 0:
        return 0
    return int(np.count_nonzero(singular_values > cutoff))


def pcr_component_count(q: Fraction | float, rank: int, rounding_rule: str) -> int:
    """Apply an explicit rounding rule. Rank zero is an error, not a silent model."""
    if rank <= 0:
        raise NumericalRankError(
            "PCR numerical rank is 0; refusing to return a coefficient vector"
        )
    if rounding_rule != DRAFT_PCR_ROUNDING:
        raise EstimatorError(
            f"PCR rounding rule {rounding_rule!r} is not an implemented named rule"
        )
    fraction = q if isinstance(q, Fraction) else Fraction(q).limit_denominator(1000)
    if fraction < 0 or fraction > 1:
        raise EstimatorError(f"PCR fraction {fraction} is outside [0, 1]")
    if fraction == 1:
        return int(rank)
    # ceil(q * rank) in exact integer arithmetic, then at least 1.
    numerator = fraction.numerator * int(rank)
    denominator = fraction.denominator
    rounded = (numerator + denominator - 1) // denominator
    count = max(1, int(rounded))
    return min(count, int(rank))


def _pack(
    name: str,
    n_rows: int,
    n_cols: int,
    singular: np.ndarray,
    cutoff: float | None,
    rank: int | None,
    hyperparameter: float | None,
    coefficient: np.ndarray,
    intercept: float,
    warnings: list[str],
) -> FitResult:
    if coefficient.dtype != np.float64 or coefficient.shape != (n_cols,):
        raise EstimatorError("coefficient vector is not a float64 column match")
    if not np.isfinite(coefficient).all():
        raise EstimatorError(f"{name} produced a nonfinite coefficient")
    return FitResult(
        name=name,
        n=n_rows,
        p=n_cols,
        singular_values=np.asarray(singular, dtype=np.float64).copy(),
        cutoff=cutoff,
        numerical_rank=rank,
        hyperparameter=hyperparameter,
        coefficient=np.asarray(coefficient, dtype=np.float64).copy(),
        intercept=float(intercept),
        warnings=list(warnings),
        dtype="float64",
    )


def fit_min_norm(X: np.ndarray, y: np.ndarray) -> FitResult:
    """Minimum-Euclidean-norm least squares from the economy SVD pseudoinverse."""
    matrix = _as_float64_matrix(X)
    n_rows, n_cols = matrix.shape
    target = _as_float64_target(y, n_rows)
    left, singular, right_h = economy_svd(matrix)
    cutoff = numerical_cutoff(singular, n_rows, n_cols)
    keep = singular > cutoff
    rank = int(np.count_nonzero(keep))
    warnings: list[str] = []
    if rank < singular.size:
        warnings.append(
            f"truncated {int(singular.size - rank)} singular values at or below cutoff"
        )
    coeff_svd = np.zeros(singular.shape, dtype=np.float64)
    if rank:
        coeff_svd[keep] = (left[:, keep].T @ target) / singular[keep]
    coefficient = right_h.T @ coeff_svd
    return _pack(
        "min_norm", n_rows, n_cols, singular, cutoff, rank, None, coefficient, 0.0, warnings
    )


def fit_ridge(X: np.ndarray, y: np.ndarray, lam: float) -> FitResult:
    """Ridge for (1/n)||y-Xb||^2 + lam||b||^2. Filter denominator is s^2 + n*lam."""
    if not np.isfinite(lam) or lam < 0:
        raise EstimatorError(f"ridge lambda must be finite and nonnegative, got {lam}")
    matrix = _as_float64_matrix(X)
    n_rows, n_cols = matrix.shape
    target = _as_float64_target(y, n_rows)
    left, singular, right_h = economy_svd(matrix)
    cutoff = numerical_cutoff(singular, n_rows, n_cols)
    rank = numerical_rank(singular, cutoff)
    # The objective's stationarity condition uses n*lam even for tiny singular values.
    # Those directions are shrunk by the penalty; they are not hard-thresholded.
    filter_weights = singular / (singular * singular + n_rows * float(lam))
    coefficient = right_h.T @ (filter_weights * (left.T @ target))
    warnings: list[str] = []
    if lam == 0:
        warnings.append("lambda 0 uses the project objective; prefer fit_min_norm for the pseudoinverse")
    return _pack(
        "ridge",
        n_rows,
        n_cols,
        singular,
        cutoff,
        rank,
        float(lam),
        coefficient,
        0.0,
        warnings,
    )


def fit_ridge_unnormalized(X: np.ndarray, y: np.ndarray, lam: float) -> FitResult:
    """Tang et al. ridge: ||y-Xb||^2 + lam||b||^2. Denominator s^2 + lam, not s^2 + n*lam."""
    if not np.isfinite(lam) or lam <= 0:
        raise EstimatorError("paper ridge lambda must be finite and positive")
    matrix = _as_float64_matrix(X)
    n_rows, n_cols = matrix.shape
    target = _as_float64_target(y, n_rows)
    left, singular, right_h = economy_svd(matrix)
    cutoff = numerical_cutoff(singular, n_rows, n_cols)
    rank = numerical_rank(singular, cutoff)
    filter_weights = singular / (singular * singular + float(lam))
    coefficient = right_h.T @ (filter_weights * (left.T @ target))
    return _pack(
        "ridge_unnormalized",
        n_rows,
        n_cols,
        singular,
        cutoff,
        rank,
        float(lam),
        coefficient,
        0.0,
        ["Tang et al. unnormalized ridge; not the L2 (1/n) objective"],
    )


def fit_pcr(
    X: np.ndarray,
    y: np.ndarray,
    q: Fraction | float,
    rounding_rule: str,
) -> FitResult:
    """PCR on the leading right singular vectors that survive the numerical cutoff."""
    matrix = _as_float64_matrix(X)
    n_rows, n_cols = matrix.shape
    target = _as_float64_target(y, n_rows)
    left, singular, right_h = economy_svd(matrix)
    cutoff = numerical_cutoff(singular, n_rows, n_cols)
    rank = numerical_rank(singular, cutoff)
    count = pcr_component_count(q, rank, rounding_rule)
    # Leading components are the largest singular values. gesdd returns them descending.
    if singular.size > 1 and np.any(np.diff(singular) > 0):
        raise EstimatorError("SVD singular values were not in descending order")
    coeff_svd = np.zeros(singular.shape, dtype=np.float64)
    coeff_svd[:count] = (left[:, :count].T @ target) / singular[:count]
    coefficient = right_h.T @ coeff_svd
    fraction = q if isinstance(q, Fraction) else Fraction(q).limit_denominator(1000)
    return _pack(
        "pcr",
        n_rows,
        n_cols,
        singular,
        cutoff,
        rank,
        float(fraction),
        coefficient,
        0.0,
        [f"retained {count} of numerical rank {rank} using {rounding_rule}"],
    )


def fit_intercept_only(y: np.ndarray) -> FitResult:
    """Training-mean predictor. Coefficient is zero in the supplied feature space."""
    target = np.asarray(y, dtype=np.float64).reshape(-1)
    if target.size == 0 or not np.isfinite(target).all():
        raise EstimatorError("intercept-only target is empty or nonfinite")
    n_rows = int(target.shape[0])
    mean = float(np.mean(target))
    return FitResult(
        name="intercept_only",
        n=n_rows,
        p=0,
        singular_values=np.zeros(0, dtype=np.float64),
        cutoff=None,
        numerical_rank=0,
        hyperparameter=None,
        coefficient=np.zeros(0, dtype=np.float64),
        intercept=mean,
        warnings=[],
        dtype="float64",
    )
