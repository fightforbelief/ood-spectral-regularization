import numpy as np

from p19.estimators import fit_min_norm


def test_repeated_fit_is_identical():
    rng = np.random.default_rng(21)
    design = rng.normal(size=(9, 4))
    target = rng.normal(size=9)
    first = fit_min_norm(design, target)
    second = fit_min_norm(design, target)
    assert first.coefficient.dtype == np.float64
    assert np.array_equal(first.coefficient, second.coefficient)
    assert first.singular_values.dtype == np.float64
