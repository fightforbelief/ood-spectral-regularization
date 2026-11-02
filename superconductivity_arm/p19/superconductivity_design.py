"""Adopted Superconductivity seeds, nested prefixes, and length-810 folds.

The candidate YAML is authoritative. This module does not regenerate the seed
list and does not call the YearPredictionMSD permutation or fold functions.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from p19.l2_sampling import ProtocolValidationError
from p19.protocol import SUPER_SAMPLE_SIZE_LIST_ORIGINAL, load_yaml
from p19.superconductivity_authorization import CANDIDATE_SHA256, candidate_path

FOLD_SEED = 820000001
MAX_PREFIX = 810
N_FOLDS = 5
DECLARED_REPETITIONS = 100
ESTIMATOR_ORDER = ("min_norm", "ridge", "pcr")


def load_adopted_candidate(path: Path | None = None) -> dict:
    candidate = Path(path) if path is not None else candidate_path()
    digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
    if digest != CANDIDATE_SHA256:
        raise ProtocolValidationError("refusing a Superconductivity candidate whose hash changed")
    document = load_yaml(candidate)
    if document.get("status") != "CANDIDATE_NOT_AUTHORIZED":
        raise ProtocolValidationError("historical candidate status must stay CANDIDATE_NOT_AUTHORIZED")
    if document.get("execution_authorized") is not False:
        raise ProtocolValidationError("historical candidate execution_authorized must stay false")
    return document


def load_stored_sampling_seeds(path: Path | None = None) -> list[int]:
    """Return the 100 stored seeds in file order. Do not sort or redraw them."""
    document = load_adopted_candidate(path)
    seeds = document.get("random_seed_list")
    if isinstance(seeds, (str, bytes)) or not isinstance(seeds, list):
        raise ProtocolValidationError("candidate random_seed_list is missing")
    values: list[int] = []
    for seed in seeds:
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise ProtocolValidationError(f"candidate sampling seed {seed!r} is not an integer")
        values.append(int(seed))
    if len(values) != DECLARED_REPETITIONS or len(set(values)) != DECLARED_REPETITIONS:
        raise ProtocolValidationError("candidate random_seed_list is not 100 distinct integers")
    if values != list(document["random_seed_list"]):
        raise ProtocolValidationError("candidate seed order was not preserved")
    return values


def require_archived_sample_sizes(sample_sizes) -> tuple[int, ...]:
    sizes = tuple(int(size) for size in sample_sizes)
    if sizes != tuple(SUPER_SAMPLE_SIZE_LIST_ORIGINAL):
        raise ProtocolValidationError(
            "the archived Superconductivity sample-size list cannot change"
        )
    return sizes


def canonical_training_pool(training_row_indices: np.ndarray) -> np.ndarray:
    """Keep the frozen splitter's row order. Do not sort it into file order."""
    pool = np.asarray(training_row_indices, dtype=np.int64).reshape(-1)
    if pool.size < MAX_PREFIX:
        raise ProtocolValidationError("Superconductivity training pool is shorter than 810")
    if len(set(pool.tolist())) != int(pool.size):
        raise ProtocolValidationError("training-pool row index vector contains a duplicate")
    if np.any(pool < 0):
        raise ProtocolValidationError("training-pool row index is negative")
    return pool


def permute_training_pool(training_row_indices: np.ndarray, sampling_seed: int) -> np.ndarray:
    """One permutation of the canonical training-pool row indices."""
    if isinstance(sampling_seed, bool) or not isinstance(sampling_seed, (int, np.integer)):
        raise ProtocolValidationError("sampling seed must be an integer")
    pool = canonical_training_pool(training_row_indices)
    order = np.random.default_rng(int(sampling_seed)).permutation(pool)
    return np.asarray(order, dtype=np.int64)


def nested_prefixes(order: np.ndarray, sample_sizes=SUPER_SAMPLE_SIZE_LIST_ORIGINAL) -> dict[int, np.ndarray]:
    sizes = require_archived_sample_sizes(sample_sizes)
    permutation = np.asarray(order, dtype=np.int64).reshape(-1)
    prefixes: dict[int, np.ndarray] = {}
    previous = 0
    for size in sizes:
        prefix = permutation[:size].copy()
        if prefix.shape != (size,) or len(set(prefix.tolist())) != size:
            raise ProtocolValidationError(f"prefix of length {size} is not n unique positions")
        if previous and not np.array_equal(prefix[:previous], prefixes[previous]):
            raise ProtocolValidationError("nested prefix relation failed")
        prefixes[size] = prefix
        previous = size
    return prefixes


def length_810_fold_membership(fold_seed: int = FOLD_SEED, max_n: int = MAX_PREFIX) -> np.ndarray:
    """Positions 0..809. Fold i receives permutation[i::5]. This is not length 900."""
    if int(fold_seed) != FOLD_SEED or int(max_n) != MAX_PREFIX:
        raise ProtocolValidationError("Superconductivity folds are fixed at seed 820000001 and length 810")
    permutation = np.random.default_rng(FOLD_SEED).permutation(MAX_PREFIX)
    membership = np.empty(MAX_PREFIX, dtype=np.int64)
    for fold in range(N_FOLDS):
        membership[permutation[fold::5]] = fold
    if membership.min() != 0 or membership.max() != N_FOLDS - 1:
        raise ProtocolValidationError("length-810 fold assignment is not a five-fold partition")
    return membership


def restrict_length_810_folds(membership: np.ndarray, sample_size: int) -> list[np.ndarray]:
    """Fold ids of prefix positions 0..n-1. These are not original dataset row indices."""
    labels = np.asarray(membership, dtype=np.int64).reshape(-1)
    if labels.shape != (MAX_PREFIX,):
        raise ProtocolValidationError("fold membership must be the length-810 assignment")
    size = int(sample_size)
    if size not in SUPER_SAMPLE_SIZE_LIST_ORIGINAL:
        raise ProtocolValidationError(f"sample size {size} is not in the archived Superconductivity list")
    restricted: list[np.ndarray] = []
    for fold in range(N_FOLDS):
        positions = np.flatnonzero(labels[:size] == fold).astype(np.int64, copy=False)
        if positions.size == 0:
            raise ProtocolValidationError(f"restricted fold {fold} is empty at sample size {size}")
        restricted.append(positions)
    covered = np.concatenate(restricted)
    if covered.size != size or not np.array_equal(np.sort(covered), np.arange(size, dtype=np.int64)):
        raise ProtocolValidationError(f"restricted folds do not partition positions 0..{size - 1}")
    return restricted


def fold_counts(membership: np.ndarray) -> dict[int, list[int]]:
    counts: dict[int, list[int]] = {}
    for size in SUPER_SAMPLE_SIZE_LIST_ORIGINAL:
        labels = np.asarray(membership[:size], dtype=np.int64)
        counts[int(size)] = [int(np.sum(labels == fold)) for fold in range(N_FOLDS)]
    return counts


def candidate_fold_counts(document: dict | None = None) -> dict[int, list[int]]:
    payload = document if document is not None else load_adopted_candidate()
    raw = payload["cv_fold_assignment"]["fold_counts_folds_0_to_4"]
    return {int(size): [int(value) for value in tally] for size, tally in raw.items()}


def assert_folds_match_candidate(membership: np.ndarray, document: dict | None = None) -> None:
    if fold_counts(membership) != candidate_fold_counts(document):
        raise ProtocolValidationError("length-810 fold counts do not match the adopted candidate")


def prefix_sha256(prefix: np.ndarray) -> str:
    array = np.ascontiguousarray(prefix, dtype=np.int64)
    return hashlib.sha256(array.tobytes()).hexdigest()


def assert_prefix_inside_pool(prefix: np.ndarray, training_row_indices: np.ndarray, holdout_row_indices: np.ndarray) -> None:
    pool = set(np.asarray(training_row_indices, dtype=np.int64).tolist())
    holdout = set(np.asarray(holdout_row_indices, dtype=np.int64).tolist())
    values = np.asarray(prefix, dtype=np.int64)
    if len(set(values.tolist())) != int(values.size):
        raise ProtocolValidationError("prefix repeats a row position")
    if any(int(value) not in pool for value in values):
        raise ProtocolValidationError("prefix escaped the training pool")
    if any(int(value) in holdout for value in values):
        raise ProtocolValidationError("prefix contains an outer-holdout position")


def count_exact_feature_fold_overlap(
    features: np.ndarray,
    prefix: np.ndarray,
    membership: np.ndarray,
) -> dict:
    """Count validation rows whose exact feature key also appears in that fold's training rows.

    No target column is read. Fold ids are prefix positions.
    """
    matrix = np.asarray(features)
    if matrix.ndim != 2:
        raise ProtocolValidationError("overlap audit requires a feature matrix")
    rows = np.asarray(prefix, dtype=np.int64).reshape(-1)
    labels = np.asarray(membership[: rows.size], dtype=np.int64)
    if np.any(rows < 0) or np.any(rows >= matrix.shape[0]):
        raise ProtocolValidationError("prefix row escaped the feature matrix")
    keys = [matrix[int(index)].tobytes() for index in rows]
    folds = []
    exposures = 0
    affected = 0
    maximum = 0.0
    for fold in range(N_FOLDS):
        validation = [index for index, label in enumerate(labels.tolist()) if label == fold]
        training = [index for index, label in enumerate(labels.tolist()) if label != fold]
        training_keys = {keys[index] for index in training}
        overlap = sum(1 for index in validation if keys[index] in training_keys)
        proportion = float(overlap / len(validation)) if validation else 0.0
        exposures += overlap
        if overlap:
            affected += 1
        maximum = max(maximum, proportion)
        folds.append(
            {
                "fold": fold,
                "n_validation_rows": len(validation),
                "n_overlapping_validation_rows": overlap,
                "overlap_proportion": proportion,
            }
        )
    return {
        "affected_folds": affected,
        "validation_exposures": exposures,
        "max_overlap_proportion": maximum,
        "folds": folds,
    }
