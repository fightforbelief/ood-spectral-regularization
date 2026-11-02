"""Training-subset uncertainty for the Stage 5D candidate.

Intervals describe permutation variability of the fixed-holdout empirical risk.
They are pointwise. They do not generalize beyond the official holdout rows.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import t

from p19.protocol import YEAR_SAMPLE_SIZE_LIST_ORIGINAL


class UncertaintyNotFrozen(Exception):
    pass


APPROVED_TRAINING_SUBSET_PROCEDURE = "STAGE5D_TRAINING_SUBSET_UNCERTAINTY_V1"
DECLARED_REPETITIONS = 100
ADJACENT_SAMPLE_SIZE_PAIRS = (
    (68, 45),
    (81, 68),
    (86, 81),
    (88, 86),
    (90, 88),
    (92, 90),
    (95, 92),
    (99, 95),
    (113, 99),
    (135, 113),
    (180, 135),
    (270, 180),
    (450, 270),
    (900, 450),
)


def _approved_procedure(procedure: str | None) -> None:
    if procedure is None or procedure == "":
        raise UncertaintyNotFrozen(
            "training-subset uncertainty procedure is not in protocol_freeze.yaml"
        )
    if not str(procedure).startswith(APPROVED_TRAINING_SUBSET_PROCEDURE):
        raise UncertaintyNotFrozen(
            f"training-subset procedure {procedure!r} has no approved implementation"
        )


def _complete_replicates(replicate_metrics: np.ndarray) -> np.ndarray:
    values = np.asarray(replicate_metrics, dtype=np.float64).reshape(-1)
    if values.size != DECLARED_REPETITIONS or not np.isfinite(values).all():
        raise UncertaintyNotFrozen(
            "training-subset summary requires all 100 finite repetition values; "
            "a partial set of successful repetitions is not summarized"
        )
    return values


def summarize_complete_replicates(replicate_metrics: np.ndarray) -> dict:
    """Mean, sample SD with divisor R-1, SE, and a pointwise 95 percent Student-t interval."""
    values = _complete_replicates(replicate_metrics)
    mean = float(np.mean(values))
    sample_sd = float(np.std(values, ddof=1))
    standard_error = float(sample_sd / np.sqrt(DECLARED_REPETITIONS))
    degrees_of_freedom = DECLARED_REPETITIONS - 1
    t_critical = float(t.ppf(0.975, degrees_of_freedom))
    half_width = t_critical * standard_error
    return {
        "R": DECLARED_REPETITIONS,
        "mean": mean,
        "sample_sd": sample_sd,
        "standard_error": standard_error,
        "degrees_of_freedom": degrees_of_freedom,
        "t_critical": t_critical,
        "interval_low": float(mean - half_width),
        "interval_high": float(mean + half_width),
        "interval": "pointwise",
        "generalizes_beyond_fixed_holdout": False,
    }


def training_subset_variability(replicate_metrics: np.ndarray, procedure: str | None) -> dict:
    """Variability across the 100 source-pool permutations. Not target-row uncertainty."""
    _approved_procedure(procedure)
    summary = summarize_complete_replicates(replicate_metrics)
    summary["procedure"] = APPROVED_TRAINING_SUBSET_PROCEDURE
    return summary


def _align_replicates(replicate_metrics: np.ndarray, repetition_ids: Sequence[int] | None) -> np.ndarray:
    """Put values in repetition order ``0 .. 99``.

    Omitted identifiers mean position ``i`` is already repetition ``i``.
    Supplied identifiers must be exactly that set once each. Pairing then follows
    the identifier, not the incoming list order.
    """
    values = np.asarray(replicate_metrics, dtype=np.float64).reshape(-1)
    if repetition_ids is None:
        return _complete_replicates(values)
    identifiers: list[int] = []
    for item in repetition_ids:
        if isinstance(item, bool) or not isinstance(item, (int, np.integer)):
            raise UncertaintyNotFrozen("repetition identifier is not an integer")
        identifiers.append(int(item))
    if (
        len(identifiers) != values.size
        or values.size != DECLARED_REPETITIONS
        or not np.isfinite(values).all()
        or set(identifiers) != set(range(DECLARED_REPETITIONS))
    ):
        raise UncertaintyNotFrozen(
            "misaligned repetition identifiers; refusing to pair by list order"
        )
    ordered = np.empty(DECLARED_REPETITIONS, dtype=np.float64)
    for identifier, value in zip(identifiers, values):
        ordered[identifier] = value
    return ordered


def _estimator_repetition_ids(repetition_ids) -> dict[str, Sequence[int] | None]:
    names = ("min_norm", "ridge", "pcr")
    if repetition_ids is None:
        return {name: None for name in names}
    if isinstance(repetition_ids, dict):
        if set(repetition_ids) != set(names):
            raise UncertaintyNotFrozen(
                "estimator repetition identifiers must name min_norm, ridge, and pcr"
            )
        return {name: repetition_ids[name] for name in names}
    return {name: repetition_ids for name in names}


def paired_estimator_loss_differences(
    min_norm: np.ndarray,
    ridge: np.ndarray,
    pcr: np.ndarray,
    procedure: str | None,
    *,
    repetition_ids=None,
) -> dict[str, dict]:
    """Within-repetition paired differences, then the Stage 5D summary across repetitions.

    Signs are PCR minus ridge, minimum-norm minus ridge, and PCR minus minimum-norm.
    When repetition identifiers are supplied, each series is aligned by those
    identifiers before subtraction. Mismatched identifiers are rejected.
    """
    _approved_procedure(procedure)
    identifiers = _estimator_repetition_ids(repetition_ids)
    min_norm_values = _align_replicates(min_norm, identifiers["min_norm"])
    ridge_values = _align_replicates(ridge, identifiers["ridge"])
    pcr_values = _align_replicates(pcr, identifiers["pcr"])
    contrasts = {
        "pcr_minus_ridge": pcr_values - ridge_values,
        "min_norm_minus_ridge": min_norm_values - ridge_values,
        "pcr_minus_min_norm": pcr_values - min_norm_values,
    }
    return {
        name: training_subset_variability(values, procedure)
        for name, values in contrasts.items()
    }


def adjacent_sample_size_differences(
    risk_by_sample_size: dict[int, np.ndarray],
    procedure: str | None,
    *,
    repetition_ids=None,
) -> dict[str, dict]:
    """Larger archived size minus the next smaller size, within each repetition."""
    _approved_procedure(procedure)
    observed = {int(size) for size in risk_by_sample_size}
    if observed != set(YEAR_SAMPLE_SIZE_LIST_ORIGINAL):
        raise UncertaintyNotFrozen(
            "adjacent comparisons require every archived sample size and no others"
        )
    if isinstance(repetition_ids, dict) and set(int(size) for size in repetition_ids) != observed:
        raise UncertaintyNotFrozen(
            "adjacent repetition identifiers must name every archived sample size"
        )
    summaries = {}
    for larger, smaller in ADJACENT_SAMPLE_SIZE_PAIRS:
        if repetition_ids is None:
            larger_ids = None
            smaller_ids = None
        elif isinstance(repetition_ids, dict):
            larger_ids = repetition_ids[larger]
            smaller_ids = repetition_ids[smaller]
        else:
            larger_ids = repetition_ids
            smaller_ids = repetition_ids
        difference = _align_replicates(risk_by_sample_size[larger], larger_ids) - _align_replicates(
            risk_by_sample_size[smaller], smaller_ids
        )
        summaries[f"{larger}_minus_{smaller}"] = training_subset_variability(difference, procedure)
    return summaries


def target_row_uncertainty(row_residuals: np.ndarray, procedure: str | None) -> dict:
    """No approved interval for a target population beyond the fixed official holdout."""
    if procedure is None or procedure == "":
        raise UncertaintyNotFrozen(
            "target-row uncertainty procedure is not in protocol_freeze.yaml"
        )
    raise UncertaintyNotFrozen(
        f"target-row procedure {procedure!r} has no approved implementation"
    )


def distinguish_sources() -> dict[str, str]:
    return {
        "training_subset_variability": (
            "Variation of fixed-holdout empirical risk across the declared "
            "independent source-pool permutations."
        ),
        "target_row_uncertainty": (
            "No approved confidence interval for a target population beyond "
            "the fixed official holdout rows."
        ),
    }
