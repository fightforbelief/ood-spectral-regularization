import numpy as np
import pytest

from p19.metrics import MetricError, mae, mse, r2, rmse
from p19.uncertainty import UncertaintyNotFrozen, distinguish_sources, target_row_uncertainty, training_subset_variability


def test_metrics_match_definitions_and_do_not_drop_rows():
    y = np.array([1.0, 2.0, 3.0])
    pred = np.array([1.0, 0.0, 5.0])
    assert mse(y, pred) == pytest.approx((0 + 4 + 4) / 3)
    assert rmse(y, pred) == pytest.approx(mse(y, pred) ** 0.5)
    assert mae(y, pred) == pytest.approx((0 + 2 + 2) / 3)
    assert np.isfinite(r2(y, pred))
    with pytest.raises(MetricError):
        mse(y, np.array([1.0, np.nan, 3.0]))
    with pytest.raises(MetricError):
        r2(np.array([2.0, 2.0]), np.array([2.0, 2.0]))


def test_uncertainty_interfaces_stay_closed_and_distinct():
    with pytest.raises(UncertaintyNotFrozen):
        training_subset_variability(np.array([1.0, 2.0]), None)
    with pytest.raises(UncertaintyNotFrozen):
        target_row_uncertainty(np.array([1.0, 2.0]), None)
    described = distinguish_sources()
    assert "training_subset" in described["training_subset_variability"] or "source" in described["training_subset_variability"]
    assert described["training_subset_variability"] != described["target_row_uncertainty"]
