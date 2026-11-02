"""Superconductivity exact-feature group split from the Stage 5D candidate.

The stored split seed and the 80/20 group rule are implemented. Nested prefixes
inside the training pool are not seeded in the protocol, so that draw is refused
instead of being invented or copied from YearPredictionMSD.
"""

from __future__ import annotations

import numpy as np

STORED_SPLIT_SEED = 732111676
DECLARED_MARKERS = (
    "732111676",
    "(G * 4) // 5",
    "81 predictor",
    "not seeded here",
    "19000003",
)


class SuperconductivityProtocolError(Exception):
    """The superconductivity control text does not support the requested operation."""


def require_declared_split_text(protocol_text: str) -> None:
    """Refuse a missing or rewritten split instead of falling back to another policy."""
    if not isinstance(protocol_text, str) or protocol_text == "":
        raise SuperconductivityProtocolError("superconductivity split text is missing")
    for marker in DECLARED_MARKERS:
        if marker not in protocol_text:
            raise SuperconductivityProtocolError(
                f"superconductivity split text is missing the declared marker {marker!r}"
            )
    if "random_seed_list" in protocol_text and "not seeded here" not in protocol_text:
        raise SuperconductivityProtocolError("superconductivity split text no longer blocks an unseeded nested draw")


def group_exact_feature_rows(features: np.ndarray) -> list[np.ndarray]:
    """Group row indices by exact float64 feature vectors.

    Groups are ordered by increasing minimum file row index. Membership does not
    use a target column. Non-finite rows are rejected rather than grouped by an
    unstated NaN rule.
    """
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise SuperconductivityProtocolError("feature matrix must be two-dimensional and non-empty")
    if not np.isfinite(matrix).all():
        raise SuperconductivityProtocolError("non-finite feature rows have no declared group rule")
    order: list[bytes] = []
    members: dict[bytes, list[int]] = {}
    for row_index, row in enumerate(matrix):
        key = row.tobytes()
        if key not in members:
            order.append(key)
            members[key] = []
        members[key].append(row_index)
    groups = [np.asarray(members[key], dtype=np.int64) for key in order]
    groups.sort(key=lambda rows: int(rows.min()))
    return groups


def split_superconductivity_groups(
    features: np.ndarray,
    *,
    protocol_text: str,
    split_seed: int = STORED_SPLIT_SEED,
) -> dict:
    """Apply the stored group split. ``protocol_text`` is required and is not defaulted."""
    require_declared_split_text(protocol_text)
    if isinstance(split_seed, bool) or not isinstance(split_seed, (int, np.integer)):
        raise SuperconductivityProtocolError("split seed must be the stored integer")
    if int(split_seed) != STORED_SPLIT_SEED:
        raise SuperconductivityProtocolError(
            "the runtime split seed must be the stored value 732111676"
        )
    groups = group_exact_feature_rows(features)
    n_groups = len(groups)
    n_train_groups = (n_groups * 4) // 5
    n_test_groups = n_groups - n_train_groups
    permutation = np.random.default_rng(STORED_SPLIT_SEED).permutation(n_groups)
    train_positions = np.asarray(permutation[:n_train_groups], dtype=np.int64)
    test_positions = np.asarray(permutation[n_train_groups:], dtype=np.int64)
    train_rows = (
        np.concatenate([groups[int(position)] for position in train_positions])
        if train_positions.size
        else np.zeros(0, dtype=np.int64)
    )
    test_rows = (
        np.concatenate([groups[int(position)] for position in test_positions])
        if test_positions.size
        else np.zeros(0, dtype=np.int64)
    )
    return {
        "split_seed": STORED_SPLIT_SEED,
        "n_groups": n_groups,
        "n_train_groups": n_train_groups,
        "n_test_groups": n_test_groups,
        "train_group_positions": train_positions,
        "test_group_positions": test_positions,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "groups": groups,
    }


def nested_training_prefixes(*args, **kwargs):
    """The protocol does not contain a seed list for this draw."""
    raise SuperconductivityProtocolError(
        "Nested training-subset draws inside the Superconductivity training pool "
        "are not seeded. Refusing to invent a seed list or to reuse the "
        "YearPredictionMSD sampling seeds."
    )
