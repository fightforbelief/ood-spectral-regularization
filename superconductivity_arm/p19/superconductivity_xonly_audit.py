"""X-only grouped split, sampling manifest, and within-CV duplicate exposure.

The official critical_temp column is never parsed. This module does not fit a model.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from p19.data_loading import load_superconductivity_x_only
from p19.protocol import SUPER_SAMPLE_SIZE_LIST_ORIGINAL
from p19.superconductivity_design import (
    N_FOLDS,
    fold_counts,
    length_810_fold_membership,
    load_stored_sampling_seeds,
    nested_prefixes,
    permute_training_pool,
    prefix_sha256,
)
from p19.superconductivity_split import split_superconductivity_groups

ORIGINAL_TRAIN = Path(
    r"C:\Users\hanqi\Desktop\projects\p19-ood-spectral-regularization\data\raw\train.csv"
)
EXPECTED_ROWS = 21263
EXPECTED_EXTRA = 6093
EXPECTED_GROUPS = 15170
EXPECTED_TRAIN_GROUPS = 12136
EXPECTED_HOLDOUT_GROUPS = 3034


class XOnlyAuditError(Exception):
    """Saved X-only facts or the frozen splitter disagree. Do not repair the split."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    return _sha256_bytes(np.ascontiguousarray(array).tobytes())


def _group_overlap(groups: list[np.ndarray], train_rows: np.ndarray, holdout_rows: np.ndarray) -> int:
    train = set(train_rows.tolist())
    holdout = set(holdout_rows.tolist())
    crossing = 0
    for group in groups:
        members = set(group.tolist())
        in_train = members & train
        in_holdout = members & holdout
        if in_train and in_holdout:
            crossing += 1
    return crossing


def _prefix_overlap(keys: list[bytes], labels: np.ndarray) -> dict:
    exposures = 0
    affected_folds = 0
    maximum = 0.0
    validation_rows = 0
    for fold in range(N_FOLDS):
        validation = [index for index, label in enumerate(labels.tolist()) if int(label) == fold]
        training_keys = {keys[index] for index, label in enumerate(labels.tolist()) if int(label) != fold}
        overlap = sum(1 for index in validation if keys[index] in training_keys)
        validation_rows += len(validation)
        proportion = overlap / len(validation) if validation else 0.0
        exposures += overlap
        if overlap:
            affected_folds += 1
        if proportion > maximum:
            maximum = proportion
    return {
        "affected_folds": affected_folds,
        "validation_exposures": exposures,
        "validation_rows": validation_rows,
        "max_overlap_proportion": maximum,
    }


def run_xonly_audit(train_csv: Path, output_dir: Path, protocol_text: str) -> dict:
    """Apply the frozen splitter to features only and measure the adopted CV exposure."""
    payload = load_superconductivity_x_only(train_csv)
    if payload["labels_loaded"] is not False:
        raise XOnlyAuditError("X-only loader reported labels_loaded")
    if payload["target_column"] != "critical_temp" or "critical_temp" in payload["feature_names"]:
        raise XOnlyAuditError("target column was not dropped before feature parsing")
    features = payload["X"]
    if features.ndim != 2 or features.shape[1] != 81 or features.dtype != np.float64:
        raise XOnlyAuditError("feature matrix is not 81 float64 predictors")
    n_rows = int(features.shape[0])
    split = split_superconductivity_groups(features, protocol_text=protocol_text)
    groups = split["groups"]
    n_groups = int(split["n_groups"])
    extra = n_rows - n_groups
    train_rows = np.asarray(split["train_rows"], dtype=np.int64)
    holdout_rows = np.asarray(split["test_rows"], dtype=np.int64)
    train_set = set(train_rows.tolist())
    holdout_set = set(holdout_rows.tolist())
    crossing = _group_overlap(groups, train_rows, holdout_rows)
    problems = []
    if n_rows != EXPECTED_ROWS:
        problems.append(f"n_rows {n_rows} != {EXPECTED_ROWS}")
    if extra != EXPECTED_EXTRA:
        problems.append(f"extra exact duplicates {extra} != {EXPECTED_EXTRA}")
    if n_groups != EXPECTED_GROUPS:
        problems.append(f"G {n_groups} != {EXPECTED_GROUPS}")
    if int(split["n_train_groups"]) != EXPECTED_TRAIN_GROUPS:
        problems.append("training group count does not match the saved audit")
    if int(split["n_test_groups"]) != EXPECTED_HOLDOUT_GROUPS:
        problems.append("holdout group count does not match the saved audit")
    if train_set & holdout_set:
        problems.append("training and holdout row positions intersect")
    if train_set | holdout_set != set(range(n_rows)):
        problems.append("training and holdout row positions do not cover the file")
    if crossing:
        problems.append(f"{crossing} exact feature groups cross the outer split")
    if train_rows.size < 810:
        problems.append("training row count is below 810")
    if problems:
        raise XOnlyAuditError("; ".join(problems))

    membership = length_810_fold_membership()
    seeds = load_stored_sampling_seeds()
    feature_key = {int(index): features[int(index)].tobytes() for index in train_rows}
    by_size = {
        int(size): {
            "affected_repetitions": 0,
            "affected_fold_evaluations": 0,
            "fold_evaluations": 0,
            "validation_rows": 0,
            "overlapping_validation_rows": 0,
            "max_validation_overlap_proportion": 0.0,
        }
        for size in SUPER_SAMPLE_SIZE_LIST_ORIGINAL
    }
    prefix_hashes = []
    affected_cells = 0
    total_fold_evaluations = 0
    affected_fold_evaluations = 0
    total_exposures = 0
    max_proportion = 0.0
    for repetition, seed in enumerate(seeds):
        order = permute_training_pool(train_rows, seed)
        prefixes = nested_prefixes(order)
        for size, prefix in prefixes.items():
            if set(prefix.tolist()) - train_set or set(prefix.tolist()) & holdout_set:
                raise XOnlyAuditError("a sampling prefix left the training pool")
            labels = membership[:size]
            keys = [feature_key[int(index)] for index in prefix]
            overlap = _prefix_overlap(keys, labels)
            bucket = by_size[int(size)]
            bucket["fold_evaluations"] += N_FOLDS
            bucket["affected_fold_evaluations"] += overlap["affected_folds"]
            bucket["validation_rows"] += overlap["validation_rows"]
            bucket["overlapping_validation_rows"] += overlap["validation_exposures"]
            if overlap["validation_exposures"]:
                bucket["affected_repetitions"] += 1
                affected_cells += 1
            if overlap["max_overlap_proportion"] > bucket["max_validation_overlap_proportion"]:
                bucket["max_validation_overlap_proportion"] = overlap["max_overlap_proportion"]
            if overlap["max_overlap_proportion"] > max_proportion:
                max_proportion = overlap["max_overlap_proportion"]
            total_fold_evaluations += N_FOLDS
            affected_fold_evaluations += overlap["affected_folds"]
            total_exposures += overlap["validation_exposures"]
            prefix_hashes.append(
                {
                    "repetition": repetition,
                    "sampling_seed": int(seed),
                    "sample_size": int(size),
                    "prefix_sha256": prefix_sha256(prefix),
                }
            )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_path = output / "training_row_indices.npy"
    holdout_path = output / "holdout_row_indices.npy"
    np.save(train_path, train_rows)
    np.save(holdout_path, holdout_rows)
    manifest = {
        "labels_loaded": False,
        "models_fit": False,
        "row_index_order": "frozen_splitter_group_permutation_concatenation",
        "training_rows_globally_ascending": bool(np.array_equal(train_rows, np.sort(train_rows))),
        "holdout_rows_globally_ascending": bool(np.array_equal(holdout_rows, np.sort(holdout_rows))),
        "n_rows": n_rows,
        "predictor_count": 81,
        "exact_extra_duplicate_feature_rows": extra,
        "n_groups": n_groups,
        "n_train_groups": int(split["n_train_groups"]),
        "n_holdout_groups": int(split["n_test_groups"]),
        "n_train_rows": int(train_rows.size),
        "n_holdout_rows": int(holdout_rows.size),
        "group_crossings": crossing,
        "positions_disjoint": True,
        "positions_cover_file": True,
        "training_row_index_sha256": _sha256_array(train_rows),
        "holdout_row_index_sha256": _sha256_array(holdout_rows),
        "training_row_index_file_sha256": _sha256_bytes(train_path.read_bytes()),
        "holdout_row_index_file_sha256": _sha256_bytes(holdout_path.read_bytes()),
        "fold_counts": {str(size): counts for size, counts in fold_counts(membership).items()},
        "cv_overlap_label": "WITHIN_POOL_EXACT_FEATURE_CV_OVERLAP_MEASURED",
        "folds_are_feature_disjoint": False,
        "repetition_size_cells": 1500,
        "affected_repetition_size_cells": affected_cells,
        "fold_evaluations": total_fold_evaluations,
        "affected_fold_evaluations": affected_fold_evaluations,
        "validation_exposures": total_exposures,
        "max_validation_overlap_proportion": max_proportion,
        "by_sample_size": {str(size): bucket for size, bucket in by_size.items()},
    }
    (output / "xonly_audit.json").write_bytes(
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    (output / "sampling_prefix_sha256.json").write_bytes(
        (json.dumps(prefix_hashes, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    _write_csv(output / "STAGE5N5_X_ONLY_SPLIT_AND_CV_AUDIT.csv", manifest)
    manifest["csv"] = str(output / "STAGE5N5_X_ONLY_SPLIT_AND_CV_AUDIT.csv")
    return manifest


def _write_csv(path: Path, manifest: dict) -> None:
    rows = [
        ("split", "", "n_rows", manifest["n_rows"]),
        ("split", "", "exact_extra_duplicate_feature_rows", manifest["exact_extra_duplicate_feature_rows"]),
        ("split", "", "n_groups", manifest["n_groups"]),
        ("split", "", "n_train_groups", manifest["n_train_groups"]),
        ("split", "", "n_holdout_groups", manifest["n_holdout_groups"]),
        ("split", "", "n_train_rows", manifest["n_train_rows"]),
        ("split", "", "n_holdout_rows", manifest["n_holdout_rows"]),
        ("split", "", "group_crossings", manifest["group_crossings"]),
        ("split", "", "training_rows_globally_ascending", manifest["training_rows_globally_ascending"]),
        ("split", "", "holdout_rows_globally_ascending", manifest["holdout_rows_globally_ascending"]),
        ("split", "", "row_index_order", manifest["row_index_order"]),
        ("cv_overlap", "ALL", "affected_repetition_size_cells", manifest["affected_repetition_size_cells"]),
        ("cv_overlap", "ALL", "repetition_size_cells", manifest["repetition_size_cells"]),
        ("cv_overlap", "ALL", "affected_fold_evaluations", manifest["affected_fold_evaluations"]),
        ("cv_overlap", "ALL", "fold_evaluations", manifest["fold_evaluations"]),
        ("cv_overlap", "ALL", "validation_exposures", manifest["validation_exposures"]),
        ("cv_overlap", "ALL", "max_validation_overlap_proportion", manifest["max_validation_overlap_proportion"]),
        ("cv_overlap", "ALL", "label", manifest["cv_overlap_label"]),
        ("cv_overlap", "ALL", "folds_are_feature_disjoint", manifest["folds_are_feature_disjoint"]),
    ]
    for size, bucket in manifest["by_sample_size"].items():
        evaluations = bucket["fold_evaluations"]
        exposures = bucket["overlapping_validation_rows"]
        validation_rows = bucket["validation_rows"]
        proportion = exposures / validation_rows if validation_rows else 0.0
        rows.extend(
            [
                ("cv_overlap", size, "affected_repetitions", bucket["affected_repetitions"]),
                ("cv_overlap", size, "affected_fold_evaluations", bucket["affected_fold_evaluations"]),
                ("cv_overlap", size, "fold_evaluations", evaluations),
                ("cv_overlap", size, "overlapping_validation_rows", exposures),
                ("cv_overlap", size, "validation_rows", validation_rows),
                ("cv_overlap", size, "mean_validation_overlap_proportion", proportion),
                ("cv_overlap", size, "max_validation_overlap_proportion", bucket["max_validation_overlap_proportion"]),
            ]
        )
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["record_type", "sample_size", "name", "value"])
        writer.writerows(rows)
