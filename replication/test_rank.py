"""Small exact-decimal examples for the certificate used in the paper."""
import numpy as np

from cutoff import PRIME, rank_certificate


def test_nearly_collinear_rows():
    rows = [["0", "0"], ["1", "1"], ["2", "2.00000000000000000001"]]
    assert np.linalg.matrix_rank(np.array(rows, dtype=float)) == 1
    assert rank_certificate(rows) == (2, 2)


def test_duplicate_and_affine_dependence():
    assert rank_certificate([["1","2"],["1.0","2e0"],["0","7"]]) == (1,1)
    assert rank_certificate([["1","2"],["2","4"],["3","6"]]) == (1,2)


def test_lower_bound_can_be_strict():
    assert rank_certificate([["0"], [str(PRIME)]]) == (0,1)
