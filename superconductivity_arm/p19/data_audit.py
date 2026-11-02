"""Schema and X-only diagnostics. Does not rewrite the sample-size list."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import numpy as np

from p19.data_loading import (
    _unquote_cell,
    iter_year_feature_rows,
    load_formula_column,
    load_superconductivity_x_only,
)
from p19.protocol import (
    SUPER_OFFICIAL_ROWS,
    SUPER_PREDICTOR_COUNT_UCI,
    SUPER_SAMPLE_SIZE_LIST_ORIGINAL,
    YEAR_OFFICIAL_ROWS,
    YEAR_OFFICIAL_SOURCE_ROWS,
    YEAR_OFFICIAL_TARGET_ROWS,
    YEAR_SAMPLE_SIZE_LIST_ORIGINAL,
    YEAR_SELECTION_PREDICTOR_COUNT,
    YEAR_TARGET_INDEX,
)


def feature_count_status(observed_predictors: int, protocol_predictors: int = YEAR_SELECTION_PREDICTOR_COUNT) -> str:
    if int(observed_predictors) != int(protocol_predictors):
        return "BLOCKED_FEATURE_COUNT_MISMATCH"
    return "MATCHES_SELECTION_REPORT_P90"


def original_sample_size_list(dataset: str) -> list[int]:
    if dataset == "yearpredictionmsd":
        return list(YEAR_SAMPLE_SIZE_LIST_ORIGINAL)
    if dataset == "superconductivity":
        return list(SUPER_SAMPLE_SIZE_LIST_ORIGINAL)
    raise ValueError(f"unknown dataset {dataset}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _welford_update(count, mean, m2, values):
    for value in values:
        count += 1
        delta = value - mean
        mean = mean + delta / count
        m2 = m2 + delta * (value - mean)
    return count, mean, m2


def audit_yearprediction_x_only(
    path: Path,
    source_rows: int = YEAR_OFFICIAL_SOURCE_ROWS,
    target_rows: int = YEAR_OFFICIAL_TARGET_ROWS,
    enforce_official_rows: bool = True,
) -> dict:
    """Stream features. Column 0 is counted and then discarded, never stored as a label."""
    path = Path(path)
    byte_size = path.stat().st_size
    file_hash = _sha256(path)
    column_counts: Counter[int] = Counter()
    predictor_counts: Counter[int] = Counter()
    nonfinite_rows = 0
    unparsed_feature_rows = 0
    empty_rows = 0
    n_rows = 0
    source_seen = 0
    target_seen = 0
    source_hashes: dict[bytes, int] = {}
    target_hashes: dict[bytes, int] = {}
    cross_overlap = 0
    mean_s = None
    m2_s = None
    mean_t = None
    m2_t = None
    n_s = 0
    n_t = 0

    for row_index, total_columns, features in iter_year_feature_rows(path):
        n_rows += 1
        column_counts[total_columns] += 1
        is_source = row_index < source_rows
        if is_source:
            source_seen += 1
        else:
            target_seen += 1
        if total_columns == 0:
            empty_rows += 1
            continue
        predictor_count = total_columns - 1 if total_columns >= 1 else 0
        predictor_counts[int(features.shape[0])] += 1
        if features.shape[0] != predictor_count:
            unparsed_feature_rows += 1
            continue
        if features.size and not np.isfinite(features).all():
            nonfinite_rows += 1
        # Hash the raw feature substring by reconstructing parsed values is NOT exact-text.
        # The iterator does not return the raw substring. Re-read is wasteful; hash the
        # float64 bytes only for numeric equality of parsed features. Exact token duplicates
        # of values that parse equally are the duplicate definition used here.
        digest = hashlib.blake2s(features.tobytes(), digest_size=16).digest()
        if is_source:
            source_hashes[digest] = source_hashes.get(digest, 0) + 1
            if mean_s is None:
                mean_s = np.zeros(features.shape[0], dtype=np.float64)
                m2_s = np.zeros(features.shape[0], dtype=np.float64)
            if features.shape[0] == mean_s.shape[0] and np.isfinite(features).all():
                n_s += 1
                delta = features - mean_s
                mean_s = mean_s + delta / n_s
                m2_s = m2_s + delta * (features - mean_s)
        else:
            target_hashes[digest] = target_hashes.get(digest, 0) + 1
            if digest in source_hashes:
                cross_overlap += 1
            if mean_t is None and features.size:
                mean_t = np.zeros(features.shape[0], dtype=np.float64)
                m2_t = np.zeros(features.shape[0], dtype=np.float64)
            if mean_t is not None and features.shape[0] == mean_t.shape[0] and np.isfinite(features).all():
                n_t += 1
                delta = features - mean_t
                mean_t = mean_t + delta / n_t
                m2_t = m2_t + delta * (features - mean_t)

    observed_predictors = None
    if len(predictor_counts) == 1 and unparsed_feature_rows == 0:
        observed_predictors = next(iter(predictor_counts))
    total_columns = next(iter(column_counts)) if len(column_counts) == 1 else None
    status_flags: list[str] = []
    if len(column_counts) != 1 or unparsed_feature_rows or empty_rows:
        status_flags.append("BLOCKED_SCHEMA")
    if nonfinite_rows:
        status_flags.append("BLOCKED_MISSINGNESS")
    if observed_predictors is None:
        status_flags.append("BLOCKED_SCHEMA")
        feature_status = "BLOCKED_SCHEMA"
    else:
        feature_status = feature_count_status(observed_predictors)
        if feature_status == "BLOCKED_FEATURE_COUNT_MISMATCH":
            status_flags.append(feature_status)
    if enforce_official_rows and (
        n_rows != YEAR_OFFICIAL_ROWS
        or source_seen != YEAR_OFFICIAL_SOURCE_ROWS
        or target_seen != YEAR_OFFICIAL_TARGET_ROWS
    ):
        status_flags.append("BLOCKED_SCHEMA")
    if not status_flags:
        schema_status = feature_status
    elif "BLOCKED_FEATURE_COUNT_MISMATCH" in status_flags and status_flags.count("BLOCKED_SCHEMA") == 0 and "BLOCKED_MISSINGNESS" not in status_flags:
        schema_status = "BLOCKED_FEATURE_COUNT_MISMATCH"
    elif "BLOCKED_FEATURE_COUNT_MISMATCH" in status_flags:
        schema_status = "BLOCKED_FEATURE_COUNT_MISMATCH"
    elif "BLOCKED_MISSINGNESS" in status_flags and "BLOCKED_SCHEMA" not in status_flags:
        schema_status = "BLOCKED_MISSINGNESS"
    else:
        schema_status = status_flags[0]

    zero_variance = []
    standardized_mean_diff = None
    if mean_s is not None and n_s > 0:
        variance = m2_s / n_s
        zero_variance = [int(i) for i, value in enumerate(variance) if value == 0.0]
        if mean_t is not None and variance.shape == mean_t.shape:
            comparable = variance > 0
            if np.any(comparable):
                scale = np.sqrt(variance[comparable])
                diff = np.abs(mean_s[comparable] - mean_t[comparable]) / scale
                standardized_mean_diff = {
                    "n_columns_compared": int(np.sum(comparable)),
                    "median_abs_smd": float(np.median(diff)),
                    "max_abs_smd": float(np.max(diff)),
                }
            else:
                standardized_mean_diff = {
                    "n_columns_compared": 0,
                    "median_abs_smd": None,
                    "max_abs_smd": None,
                }

    def _extra(counter: dict[bytes, int]) -> int:
        return int(sum(count - 1 for count in counter.values() if count > 1))

    return {
        "dataset": "yearpredictionmsd",
        "path_name": path.name,
        "byte_size": byte_size,
        "sha256": file_hash,
        "n_rows": n_rows,
        "source_rows_observed": source_seen,
        "target_rows_observed": target_seen,
        "total_columns": total_columns,
        "target_index": YEAR_TARGET_INDEX,
        "predictor_count": observed_predictors,
        "protocol_predictor_count": YEAR_SELECTION_PREDICTOR_COUNT,
        "feature_count_status": feature_status,
        "status": schema_status,
        "status_flags": status_flags,
        "nonfinite_feature_rows": nonfinite_rows,
        "unparsed_feature_rows": unparsed_feature_rows,
        "empty_rows": empty_rows,
        "zero_variance_feature_columns": zero_variance,
        "source_extra_duplicate_feature_rows": _extra(source_hashes),
        "target_extra_duplicate_feature_rows": _extra(target_hashes),
        "target_rows_whose_features_match_a_source_row": cross_overlap,
        "x_only_shift": standardized_mean_diff,
        "original_sample_size_list": list(YEAR_SAMPLE_SIZE_LIST_ORIGINAL),
        "amended_sample_size_list": None,
        "labels_loaded": False,
        "human_amendment_required": True,
        "confirmatory_l2": "BLOCKED",
        "uci_text_resolution": "UNRESOLVED",
    }


def audit_superconductivity_x_only(train_csv: Path, unique_m_csv: Path | None = None) -> dict:
    loaded = load_superconductivity_x_only(train_csv)
    matrix = loaded["X"]
    zero_variance = []
    near_duplicate_extra = None
    exact_extra = None
    if matrix.size:
        variance = np.var(matrix, axis=0)
        zero_variance = [int(i) for i, value in enumerate(variance) if value == 0.0]
        exact = {}
        near = {}
        for row in matrix:
            exact_key = hashlib.blake2s(np.ascontiguousarray(row).tobytes(), digest_size=16).digest()
            rounded = np.round(row, 6).astype(np.float64, copy=False)
            near_key = hashlib.blake2s(np.ascontiguousarray(rounded).tobytes(), digest_size=16).digest()
            exact[exact_key] = exact.get(exact_key, 0) + 1
            near[near_key] = near.get(near_key, 0) + 1
        exact_extra = int(sum(count - 1 for count in exact.values() if count > 1))
        near_duplicate_extra = int(sum(count - 1 for count in near.values() if count > 1))
    formula_report = None
    if unique_m_csv is not None:
        with Path(unique_m_csv).open("r", encoding="utf-8", newline="") as handle:
            header = handle.readline().strip("\r\n")
        names = [_unquote_cell(name) for name in header.split(",")]
        formula_name = None
        for candidate in ("material", "formula", "chemical_formula"):
            if candidate in names:
                formula_name = candidate
                break
        if formula_name is None:
            # Last column is the formula when the official layout holds and it is not the target.
            if names and names[-1] != "critical_temp":
                formula_name = names[-1]
        if formula_name is None:
            formula_report = {"status": "BLOCKED_SCHEMA", "reason": "formula column not identified"}
        else:
            formulas = load_formula_column(unique_m_csv, formula_name)
            counts = Counter(formulas["formulas"])
            repeated = {key: value for key, value in counts.items() if value > 1}
            formula_report = {
                "formula_column": formula_name,
                "n_rows": formulas["n_rows"],
                "n_unique_formulas": len(counts),
                "n_formulas_with_repeats": len(repeated),
                "extra_repeated_formula_rows": int(sum(value - 1 for value in repeated.values())),
                "labels_loaded": False,
            }
    predictor_count = loaded["predictor_count"]
    status_flags = []
    if predictor_count != SUPER_PREDICTOR_COUNT_UCI:
        status_flags.append("BLOCKED_SCHEMA")
    if loaded["n_rows"] != SUPER_OFFICIAL_ROWS:
        status_flags.append("BLOCKED_SCHEMA")
    if not np.isfinite(matrix).all():
        status_flags.append("BLOCKED_MISSINGNESS")
    status = "X_ONLY_SCHEMA_RECORDED" if not status_flags else status_flags[0]
    duplicate_issue = bool(exact_extra or (formula_report or {}).get("n_formulas_with_repeats"))
    return {
        "dataset": "superconductivity",
        "train_name": Path(train_csv).name,
        "byte_size": Path(train_csv).stat().st_size,
        "sha256": _sha256(Path(train_csv)),
        "unique_m_sha256": None if unique_m_csv is None else _sha256(Path(unique_m_csv)),
        "unique_m_byte_size": None if unique_m_csv is None else Path(unique_m_csv).stat().st_size,
        "n_rows": loaded["n_rows"],
        "total_columns": loaded["total_columns"],
        "target_column": loaded["target_column"],
        "target_index": loaded["target_index"],
        "predictor_count": predictor_count,
        "official_predictor_count": SUPER_PREDICTOR_COUNT_UCI,
        "status": status,
        "status_flags": status_flags,
        "zero_variance_feature_columns": zero_variance,
        "exact_extra_duplicate_feature_rows": exact_extra,
        "rounded6_extra_near_duplicate_feature_rows": near_duplicate_extra,
        "formula_audit": formula_report,
        "duplicate_or_formula_issue_requires_human_decision": duplicate_issue,
        "original_sample_size_list": list(SUPER_SAMPLE_SIZE_LIST_ORIGINAL),
        "amended_sample_size_list": None,
        "split_frozen": False,
        "labels_loaded": False,
        "confirmatory_l2": "BLOCKED",
    }
