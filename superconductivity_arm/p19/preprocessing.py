"""Training-only centering and scaling. The zero-variance rule must be named."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from p19.protocol import DRAFT_ZERO_VARIANCE


class PreprocessError(Exception):
    pass


@dataclass
class Preprocessor:
    x_mean: np.ndarray
    scale: np.ndarray
    y_mean: float
    zero_variance_columns: np.ndarray
    policy: str
    n_rows: int
    n_cols: int

    def transform_X(self, X: np.ndarray) -> np.ndarray:
        matrix = np.asarray(X, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != self.n_cols:
            raise PreprocessError("transform_X received an incompatible matrix")
        return (matrix - self.x_mean) / self.scale

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        target = np.asarray(y, dtype=np.float64).reshape(-1)
        return target - self.y_mean

    def restore_y(self, prediction: np.ndarray) -> np.ndarray:
        values = np.asarray(prediction, dtype=np.float64).reshape(-1)
        return values + self.y_mean


def fit_preprocessor(X: np.ndarray, y: np.ndarray, zero_variance_policy: str) -> Preprocessor:
    if zero_variance_policy != DRAFT_ZERO_VARIANCE:
        raise PreprocessError(
            "zero-variance policy is not frozen; pass an explicit implemented policy name"
        )
    matrix = np.asarray(X, dtype=np.float64)
    target = np.asarray(y, dtype=np.float64).reshape(-1)
    if matrix.ndim != 2:
        raise PreprocessError("X must be two-dimensional")
    if target.shape[0] != matrix.shape[0]:
        raise PreprocessError("X and y have different row counts")
    if matrix.shape[0] == 0:
        raise PreprocessError("cannot fit preprocessing on zero rows")
    if not np.isfinite(matrix).all() or not np.isfinite(target).all():
        raise PreprocessError("nonfinite values in preprocessing fit")
    # Copies so later mutation of the caller arrays cannot change the fit.
    x_mean = np.mean(matrix, axis=0).astype(np.float64, copy=True)
    # Population standard deviation. This divisor is part of the unapproved draft.
    x_std = np.std(matrix, axis=0, ddof=0).astype(np.float64, copy=True)
    zero_variance = x_std == 0
    scale = np.where(zero_variance, 1.0, x_std).astype(np.float64, copy=True)
    return Preprocessor(
        x_mean=x_mean,
        scale=scale,
        y_mean=float(np.mean(target)),
        zero_variance_columns=zero_variance.copy(),
        policy=zero_variance_policy,
        n_rows=int(matrix.shape[0]),
        n_cols=int(matrix.shape[1]),
    )
