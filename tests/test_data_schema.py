import numpy as np
import pytest

from p19.data_audit import audit_yearprediction_x_only, feature_count_status, original_sample_size_list
from p19.data_loading import load_superconductivity_x_only, load_yearprediction_x_only
from p19.protocol import YEAR_SAMPLE_SIZE_LIST_ORIGINAL


def _year_line(target: str, n_features: int) -> str:
    features = ",".join(str(index + 1) for index in range(n_features))
    return f"{target},{features}\n"


def test_parser_reports_columns_and_predictors_separately_and_skips_target_token(tmp_path):
    path = tmp_path / "tiny.txt"
    path.write_text("DO_NOT_PARSE,1.5,2.5\nALSO_SKIP,3.5,4.5\n", encoding="utf-8")
    loaded = load_yearprediction_x_only(path, source_rows=1, target_rows=1)
    assert loaded["labels_loaded"] is False
    assert "y" not in loaded
    assert loaded["column_counts"] == [3]
    assert loaded["predictor_count"] == 2
    assert loaded["target_index"] == 0
    assert np.allclose(loaded["X_source"], [[1.5, 2.5]])
    assert np.allclose(loaded["X_target"], [[3.5, 4.5]])


def test_feature_count_mismatch_blocks_and_preserves_original_grid(tmp_path):
    path = tmp_path / "ninety_columns.txt"
    path.write_text(_year_line("YEAR_TOKEN", 89) + _year_line("OTHER_TOKEN", 89), encoding="utf-8")
    audit = audit_yearprediction_x_only(path, source_rows=1, target_rows=1, enforce_official_rows=False)
    assert audit["total_columns"] == 90
    assert audit["target_index"] == 0
    assert audit["predictor_count"] == 89
    assert audit["feature_count_status"] == "BLOCKED_FEATURE_COUNT_MISMATCH"
    assert audit["status"] == "BLOCKED_FEATURE_COUNT_MISMATCH"
    assert audit["original_sample_size_list"] == list(YEAR_SAMPLE_SIZE_LIST_ORIGINAL)
    assert audit["amended_sample_size_list"] is None
    assert audit["labels_loaded"] is False
    assert feature_count_status(89) == "BLOCKED_FEATURE_COUNT_MISMATCH"
    assert original_sample_size_list("yearpredictionmsd") == list(YEAR_SAMPLE_SIZE_LIST_ORIGINAL)


def test_matching_predictor_count_does_not_invent_a_new_grid(tmp_path):
    path = tmp_path / "ninety_one_columns.txt"
    path.write_text(_year_line("1922", 90), encoding="utf-8")
    audit = audit_yearprediction_x_only(path, source_rows=1, target_rows=0, enforce_official_rows=False)
    assert audit["total_columns"] == 91
    assert audit["predictor_count"] == 90
    assert audit["feature_count_status"] == "MATCHES_SELECTION_REPORT_P90"
    assert audit["amended_sample_size_list"] is None
    assert audit["confirmatory_l2"] == "BLOCKED"
    assert audit["uci_text_resolution"] == "UNRESOLVED"


def test_superconductivity_x_only_skips_the_named_target(tmp_path):
    path = tmp_path / "train.csv"
    path.write_text(
        "number_of_elements,mean_atomic_mass,critical_temp\n1,10.5,NOT_A_LABEL\n2,11.5,ALSO_NOT\n",
        encoding="utf-8",
    )
    loaded = load_superconductivity_x_only(path)
    assert loaded["labels_loaded"] is False
    assert loaded["total_columns"] == 3
    assert loaded["predictor_count"] == 2
    assert loaded["target_index"] == 2
    assert loaded["feature_names"] == ["number_of_elements", "mean_atomic_mass"]
    assert np.allclose(loaded["X"], [[1.0, 10.5], [2.0, 11.5]])


def test_superconductivity_quoted_uci_header_drops_target_before_parse(tmp_path):
    path = tmp_path / "train.csv"
    path.write_text(
        '"number_of_elements","mean_atomic_mass","critical_temp"\n'
        '"4","55.85","DO_NOT_PARSE"\n',
        encoding="utf-8",
    )
    loaded = load_superconductivity_x_only(path)
    assert loaded["labels_loaded"] is False
    assert loaded["total_columns"] == 3
    assert loaded["predictor_count"] == 2
    assert loaded["target_index"] == 2
    assert loaded["feature_names"] == ["number_of_elements", "mean_atomic_mass"]
    assert np.allclose(loaded["X"], [[4.0, 55.85]])
