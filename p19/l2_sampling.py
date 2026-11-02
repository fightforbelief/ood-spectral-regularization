"""YearPredictionMSD sampling and nested folds for the Stage 5D candidate.

The stored ``random_seed_list`` is authoritative. This module does not
reconstruct that list from a master seed.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from p19.protocol import YEAR_OFFICIAL_SOURCE_ROWS, YEAR_SAMPLE_SIZE_LIST_ORIGINAL

DECLARED_REPETITIONS = 100
NESTED_FOLD_SEED = 810000001
LONGEST_PREFIX = 900
N_FOLDS = 5


class ProtocolValidationError(Exception):
    """The stored protocol values cannot drive the declared L2 plan."""


def validate_sampling_seed_list(seeds: Sequence[int], repetitions: int = DECLARED_REPETITIONS) -> list[int]:
    """Require the declared repetition count and a list of distinct integers, in order."""
    if repetitions != DECLARED_REPETITIONS:
        raise ProtocolValidationError(
            f"n_training_subset_repetitions must stay {DECLARED_REPETITIONS}, got {repetitions}"
        )
    if isinstance(seeds, (str, bytes)) or not isinstance(seeds, Sequence):
        raise ProtocolValidationError("random_seed_list must be a sequence of integers")
    values: list[int] = []
    for seed in seeds:
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise ProtocolValidationError(f"sampling seed {seed!r} is not an integer")
        values.append(int(seed))
    if len(values) != DECLARED_REPETITIONS:
        raise ProtocolValidationError(
            f"random_seed_list must contain {DECLARED_REPETITIONS} seeds, got {len(values)}"
        )
    if len(set(values)) != DECLARED_REPETITIONS:
        raise ProtocolValidationError("random_seed_list contains duplicate seeds")
    return values


def sampling_seeds_from_protocol(document: dict) -> list[int]:
    """Read ``random_seed_list`` in its stored order. Do not sort or regenerate it."""
    repetitions = document.get("n_training_subset_repetitions")
    seeds = document.get("random_seed_list")
    if seeds is None:
        raise ProtocolValidationError("random_seed_list is missing")
    return validate_sampling_seed_list(seeds, repetitions)


def sampling_seed_for_repetition(seeds: Sequence[int], repetition: int) -> int:
    values = validate_sampling_seed_list(seeds)
    if repetition < 0 or repetition >= len(values):
        raise ProtocolValidationError(f"repetition {repetition} is outside 0..{len(values) - 1}")
    return values[repetition]


def source_permutation(sampling_seed: int, pool_rows: int = YEAR_OFFICIAL_SOURCE_ROWS) -> np.ndarray:
    """Uniform permutation of source-pool positions ``0 .. pool_rows-1``.

    The Stage 5D call is ``numpy.random.default_rng(sampling_seed).permutation(463715)``.
    """
    if pool_rows != YEAR_OFFICIAL_SOURCE_ROWS:
        raise ProtocolValidationError(
            f"source permutation length is fixed at {YEAR_OFFICIAL_SOURCE_ROWS}"
        )
    if isinstance(sampling_seed, bool) or not isinstance(sampling_seed, (int, np.integer)):
        raise ProtocolValidationError("sampling seed must be an integer")
    order = np.random.default_rng(int(sampling_seed)).permutation(pool_rows)
    return np.asarray(order, dtype=np.int64)


def training_prefix(sampling_seed: int, sample_size: int) -> np.ndarray:
    """Prefix of the full source-pool permutation. The same prefix is shared by all estimators."""
    if sample_size not in YEAR_SAMPLE_SIZE_LIST_ORIGINAL:
        raise ProtocolValidationError(f"sample size {sample_size} is not in the archived list")
    return source_permutation(sampling_seed)[: int(sample_size)].copy()


def rows_for_prefix(X_source: np.ndarray, y_source: np.ndarray, prefix_indices: np.ndarray):
    """Select source-pool rows in prefix order. The arrays must already be in memory.

    This does not open a data file. The official source pool has 463,715 rows.
    """
    matrix = np.asarray(X_source)
    target = np.asarray(y_source, dtype=np.float64).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[0] != YEAR_OFFICIAL_SOURCE_ROWS or target.shape[0] != YEAR_OFFICIAL_SOURCE_ROWS:
        raise ProtocolValidationError(
            f"source pool arrays must have {YEAR_OFFICIAL_SOURCE_ROWS} rows"
        )
    indices = np.asarray(prefix_indices, dtype=np.int64)
    if np.any(indices < 0) or np.any(indices >= YEAR_OFFICIAL_SOURCE_ROWS):
        raise ProtocolValidationError("prefix index escaped the source pool")
    return matrix[indices], target[indices]


def longest_prefix_folds(fold_seed: int = NESTED_FOLD_SEED) -> list[np.ndarray]:
    """Five folds on the 900 positions of the longest archived prefix.

    Fold ``i`` receives ``permutation(900)[i::5]``. Seed 0 is the historical
    default of ``assign_source_folds`` and is not this protocol seed.
    """
    if fold_seed == 0:
        raise ProtocolValidationError("nested L2 folds do not use the default seed 0")
    if isinstance(fold_seed, bool) or not isinstance(fold_seed, (int, np.integer)):
        raise ProtocolValidationError("fold seed must be an integer")
    permutation = np.random.default_rng(int(fold_seed)).permutation(LONGEST_PREFIX)
    folds = [np.sort(np.asarray(permutation[i::N_FOLDS], dtype=np.int64)) for i in range(N_FOLDS)]
    covered = np.concatenate(folds)
    if covered.size != LONGEST_PREFIX or np.unique(covered).size != LONGEST_PREFIX:
        raise ProtocolValidationError("length-900 fold assignment is not a partition")
    return folds


def restrict_nested_folds(folds_900: Sequence[np.ndarray], sample_size: int) -> list[np.ndarray]:
    """Keep the fold membership of positions ``0 .. sample_size-1``."""
    if sample_size not in YEAR_SAMPLE_SIZE_LIST_ORIGINAL:
        raise ProtocolValidationError(f"sample size {sample_size} is not in the archived list")
    if len(folds_900) != N_FOLDS:
        raise ProtocolValidationError("nested CV requires five folds")
    restricted: list[np.ndarray] = []
    for fold_id, fold in enumerate(folds_900):
        positions = np.asarray(fold, dtype=np.int64)
        kept = np.sort(positions[positions < sample_size])
        if kept.size == 0:
            raise ProtocolValidationError(
                f"restricted fold {fold_id} is empty at sample size {sample_size}"
            )
        restricted.append(kept)
    covered = np.concatenate(restricted)
    expected = np.arange(sample_size, dtype=np.int64)
    if covered.size != sample_size or not np.array_equal(np.sort(covered), expected):
        raise ProtocolValidationError(
            f"restricted folds do not partition positions 0..{sample_size - 1}"
        )
    return restricted


def nested_folds_for_sample_size(
    sample_size: int,
    fold_seed: int = NESTED_FOLD_SEED,
) -> list[np.ndarray]:
    """Restrict the single length-900 assignment. Do not draw a new permutation of ``sample_size``."""
    return restrict_nested_folds(longest_prefix_folds(fold_seed), sample_size)
