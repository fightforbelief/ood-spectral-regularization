import numpy as np
import pytest

from p19.protocol import DRAFT_PCR_ROUNDING, DRAFT_ZERO_VARIANCE, PCR_FRACTION_GRID
from p19.tuning import assign_source_folds, fold_training_feature_means, tune_pcr, tune_ridge


def test_folds_are_source_only_and_shared():
    folds = assign_source_folds(20, n_folds=5, seed=7)
    covered = np.sort(np.concatenate(folds))
    assert np.array_equal(covered, np.arange(20))
    X = np.random.default_rng(0).normal(size=(20, 3))
    y = np.random.default_rng(1).normal(size=20)
    lambdas = np.array([1e-3, 1e-1, 1.0])
    ridge = tune_ridge(X, y, lambdas, folds, DRAFT_ZERO_VARIANCE)
    pcr = tune_pcr(X, y, [PCR_FRACTION_GRID[0], PCR_FRACTION_GRID[-1]], folds, DRAFT_ZERO_VARIANCE, DRAFT_PCR_ROUNDING)
    assert len(ridge.fold_indices) == 5
    for left, right in zip(ridge.fold_indices, pcr.fold_indices):
        assert np.array_equal(left, right)
    again = assign_source_folds(20, n_folds=5, seed=7)
    for left, right in zip(folds, again):
        assert np.array_equal(left, right)


def test_each_fold_fits_its_own_preprocessing():
    X = np.column_stack([np.arange(10, dtype=float), np.linspace(0, 1, 10)])
    y = np.arange(10, dtype=float)
    folds = assign_source_folds(10, seed=3)
    means = fold_training_feature_means(X, y, folds, DRAFT_ZERO_VARIANCE)
    full_mean = X.mean(axis=0)
    assert len(means) == 5
    assert any(not np.allclose(mean, full_mean) for mean in means)
    mask = np.ones(10, dtype=bool)
    mask[folds[0]] = False
    assert np.allclose(means[0], X[mask].mean(axis=0))


def test_tie_break_is_deterministic_and_keeps_the_smaller_lambda():
    X = np.zeros((10, 2))
    y = np.array([1.0, -1.0] * 5)
    folds = assign_source_folds(10, seed=1)
    grid = np.array([1e-2, 1.0, 10.0])
    first = tune_ridge(X, y, grid, folds, DRAFT_ZERO_VARIANCE)
    second = tune_ridge(X, y, grid, folds, DRAFT_ZERO_VARIANCE)
    assert first.selected == second.selected == pytest.approx(1e-2)
    assert first.mean_scores[0] == pytest.approx(first.mean_scores[1])


def test_tuning_rejects_target_labels_as_an_argument():
    X = np.ones((10, 2))
    y = np.ones(10)
    folds = assign_source_folds(10, seed=1)
    with pytest.raises(TypeError):
        tune_ridge(X, y, np.array([0.1]), folds, DRAFT_ZERO_VARIANCE, y_test=y)
    with pytest.raises(TypeError):
        tune_pcr(X, y, PCR_FRACTION_GRID[:1], folds, DRAFT_ZERO_VARIANCE, DRAFT_PCR_ROUNDING, test_labels=y)
