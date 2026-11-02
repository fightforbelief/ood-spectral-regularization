import numpy as np
import pytest
from fractions import Fraction

from p19.estimators import (
    NumericalRankError,
    economy_svd,
    fit_intercept_only,
    fit_min_norm,
    fit_pcr,
    fit_ridge,
    numerical_cutoff,
    pcr_component_count,
)
from p19.protocol import DRAFT_PCR_ROUNDING, ridge_lambda_grid


def test_min_norm_matches_least_squares_on_full_column_rank():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(30, 6))
    y = rng.normal(size=30)
    reference = np.linalg.lstsq(X, y, rcond=None)[0]
    fit = fit_min_norm(X, y)
    assert fit.dtype == "float64"
    assert fit.coefficient.dtype == np.float64
    assert np.allclose(fit.coefficient, reference, atol=1e-8)
    assert np.allclose(fit.predict(X), X @ reference, atol=1e-8)


def test_underdetermined_solution_has_minimum_euclidean_norm():
    rng = np.random.default_rng(12)
    X = rng.normal(size=(4, 7))
    y = rng.normal(size=4)
    fit = fit_min_norm(X, y)
    assert np.linalg.norm(X @ fit.coefficient - y) < 1e-8
    _u, _s, vt = np.linalg.svd(X, full_matrices=True)
    null = vt[4:].T
    alternative = fit.coefficient + null[:, 0]
    assert np.linalg.norm(X @ alternative - y) < 1e-7
    assert np.linalg.norm(fit.coefficient) < np.linalg.norm(alternative) - 1e-8


def test_cutoff_formula_and_rank_deficient_are_finite():
    singular = np.array([3.0, 1e-20], dtype=np.float64)
    tau = numerical_cutoff(singular, 8, 5)
    assert tau == np.finfo(np.float64).eps * max(8, 5) * 3.0
    X = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    y = np.array([1.0, 0.0, 0.0, 2.0])
    fit = fit_min_norm(X, y)
    assert np.isfinite(fit.coefficient).all()
    assert np.isfinite(fit.predict(X)).all()
    zero = fit_min_norm(np.zeros((5, 3)), np.ones(5))
    assert zero.numerical_rank == 0
    assert np.allclose(zero.coefficient, 0)
    assert np.isfinite(zero.predict(np.zeros((2, 3)))).all()


def test_svd_factors_are_float64():
    left, singular, right = economy_svd(np.ones((4, 3)))
    assert left.dtype == np.float64
    assert singular.dtype == np.float64
    assert right.dtype == np.float64


def test_ridge_matches_scaled_objective_not_unnormalized_library_form():
    rng = np.random.default_rng(13)
    X = rng.normal(size=(12, 4))
    y = rng.normal(size=12)
    lam = 0.37
    fit = fit_ridge(X, y, lam)
    n = X.shape[0]
    solved = np.linalg.solve((X.T @ X) / n + lam * np.eye(X.shape[1]), (X.T @ y) / n)
    wrong = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)
    assert np.allclose(fit.coefficient, solved, atol=1e-8)
    assert not np.allclose(fit.coefficient, wrong, atol=1e-6)
    left, singular, right = economy_svd(X)
    filt = singular / (singular ** 2 + n * lam)
    svd_beta = right.T @ (filt * (left.T @ y))
    assert np.allclose(fit.coefficient, svd_beta, atol=1e-8)


def test_positive_lambda_is_finite_for_rank_deficient_design():
    X = np.ones((6, 3))
    y = np.arange(6, dtype=np.float64)
    fit = fit_ridge(X, y, 1e-2)
    assert np.isfinite(fit.predict(X)).all()


def test_ridge_grid_has_25_decade_endpoints():
    grid = ridge_lambda_grid()
    assert grid.shape == (25,)
    assert grid.dtype == np.float64
    assert float(grid[0]) == pytest.approx(1e-8)
    assert float(grid[-1]) == pytest.approx(1e4)
    assert np.allclose(grid, 10 ** np.linspace(-8, 4, 25))


def test_pcr_keeps_leading_components_and_full_rank_at_q_one():
    rng = np.random.default_rng(14)
    X = rng.normal(size=(20, 5))
    y = rng.normal(size=20)
    full = fit_pcr(X, y, Fraction(1), DRAFT_PCR_ROUNDING)
    reference = fit_min_norm(X, y)
    assert full.numerical_rank == min(X.shape)
    assert np.allclose(full.coefficient, reference.coefficient, atol=1e-8)
    partial = fit_pcr(X, y, Fraction(2, 5), DRAFT_PCR_ROUNDING)
    assert partial.hyperparameter == pytest.approx(0.4)
    assert "retained" in partial.warnings[0]


def test_pcr_rounding_rule_and_rank_zero():
    assert pcr_component_count(Fraction(1, 20), 10, DRAFT_PCR_ROUNDING) == 1
    assert pcr_component_count(Fraction(19, 20), 10, DRAFT_PCR_ROUNDING) == 10
    assert pcr_component_count(Fraction(1), 7, DRAFT_PCR_ROUNDING) == 7
    with pytest.raises(NumericalRankError):
        fit_pcr(np.zeros((4, 3)), np.ones(4), Fraction(1), DRAFT_PCR_ROUNDING)


def test_intercept_only_is_the_training_mean():
    y = np.array([2.0, 4.0, 9.0])
    fit = fit_intercept_only(y)
    assert fit.intercept == pytest.approx(5.0)
    assert fit.coefficient.shape == (0,)
