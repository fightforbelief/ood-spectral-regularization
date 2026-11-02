"""Finite-valued regression metrics. Rows are never dropped."""

from __future__ import annotations

import numpy as np


class MetricError(Exception):
    pass


def _pair(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(y_true, dtype=np.float64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if truth.shape != pred.shape:
        raise MetricError(f"metric shape mismatch {truth.shape} vs {pred.shape}")
    if truth.size == 0:
        raise MetricError("metric received no rows")
    if not np.isfinite(truth).all() or not np.isfinite(pred).all():
        raise MetricError("metric received nonfinite values")
    return truth, pred


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    truth, pred = _pair(y_true, y_pred)
    return float(np.mean((truth - pred) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    truth, pred = _pair(y_true, y_pred)
    return float(np.mean(np.abs(truth - pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    truth, pred = _pair(y_true, y_pred)
    residual = float(np.sum((truth - pred) ** 2))
    total = float(np.sum((truth - np.mean(truth)) ** 2))
    if total == 0.0:
        raise MetricError("R^2 is undefined when the target has zero variance")
    value = 1.0 - residual / total
    if not np.isfinite(value):
        raise MetricError("R^2 was nonfinite")
    return float(value)


def metric_bundle(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    truth, pred = _pair(y_true, y_pred)
    return {
        "n_rows": int(truth.size),
        "mse": mse(truth, pred),
        "rmse": rmse(truth, pred),
        "mae": mae(truth, pred),
        "r2": r2(truth, pred),
    }
