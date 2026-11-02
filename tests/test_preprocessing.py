import numpy as np
import pytest

from p19.estimators import fit_intercept_only
from p19.preprocessing import PreprocessError, fit_preprocessor
from p19.protocol import DRAFT_ZERO_VARIANCE


def test_preprocessing_uses_training_rows_only_and_restores_target_mean():
    X = np.array([[1.0, 2.0], [1.0, 4.0], [1.0, 6.0]])
    y = np.array([10.0, 12.0, 14.0])
    pre = fit_preprocessor(X, y, DRAFT_ZERO_VARIANCE)
    assert pre.policy == DRAFT_ZERO_VARIANCE
    assert np.allclose(pre.zero_variance_columns, [True, False])
    transformed = pre.transform_X(X)
    assert np.allclose(transformed[:, 0], 0.0)
    assert pre.scale[0] == 1.0
    held_out = np.array([[1.0, 100.0]])
    held = pre.transform_X(held_out)
    original_mean = pre.x_mean.copy()
    mutated = X.copy()
    mutated += 50
    _ = pre.transform_X(mutated)
    assert np.allclose(pre.x_mean, original_mean)
    centered = pre.transform_y(y)
    model = fit_intercept_only(centered)
    restored = pre.restore_y(np.array([model.intercept]))
    assert restored[0] == pytest.approx(y.mean())
    assert held.shape == (1, 2)


def test_unknown_zero_variance_policy_is_refused():
    X = np.ones((3, 2))
    y = np.arange(3, dtype=float)
    with pytest.raises(PreprocessError):
        fit_preprocessor(X, y, "silently_drop")
    with pytest.raises(TypeError):
        fit_preprocessor(X, y)
