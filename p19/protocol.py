"""Protocol constants that are explicitly reported, and fields that are not frozen."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import yaml

# Selection-report values. Archived, not a human-approved confirmatory grid.
YEAR_SELECTION_PREDICTOR_COUNT = 90
YEAR_SAMPLE_SIZE_LIST_ORIGINAL = (
    45, 68, 81, 86, 88, 90, 92, 95, 99, 113, 135, 180, 270, 450, 900,
)
SUPER_PREDICTOR_COUNT_UCI = 81
SUPER_SAMPLE_SIZE_LIST_ORIGINAL = (
    41, 61, 73, 77, 79, 81, 83, 85, 89, 101, 122, 162, 243, 405, 810,
)
ASPECT_RATIO_GRID = (
    0.50, 0.75, 0.90, 0.95, 0.98, 1.00, 1.02, 1.05, 1.10, 1.25, 1.50, 2, 3, 5, 10,
)

YEAR_OFFICIAL_ROWS = 515345
YEAR_OFFICIAL_SOURCE_ROWS = 463715
YEAR_OFFICIAL_TARGET_ROWS = 51630
YEAR_TARGET_INDEX = 0

SUPER_OFFICIAL_ROWS = 21263
SUPER_TARGET_COLUMN = "critical_temp"

PCR_FRACTION_GRID = (
    Fraction(1, 20),
    Fraction(1, 10),
    Fraction(1, 5),
    Fraction(3, 10),
    Fraction(2, 5),
    Fraction(1, 2),
    Fraction(3, 5),
    Fraction(7, 10),
    Fraction(4, 5),
    Fraction(9, 10),
    Fraction(19, 20),
    Fraction(1, 1),
)

# Draft names. They are not written into protocol_freeze.yaml as approvals.
DRAFT_PCR_ROUNDING = "draft_max1_ceil"
DRAFT_ZERO_VARIANCE = "draft_center_unit_scale_if_zero"
DRAFT_TIE_BREAK = "draft_ascending_grid_first"
DRAFT_CV_AGGREGATION = "draft_unweighted_fold_mean"
DRAFT_FOLD_RULE = "draft_seeded_interleave"

# Accepted file-schema record. Not confirmatory authorization.
ACCEPTED_FEATURE_COUNT_AMENDMENT = "ACCEPTED_OBSERVED_P90_ORIGINAL_SAMPLE_SIZE_LIST_RETAINED"
P90_DECISION_SENTENCE = (
    "The observed official archive file has 91 total columns, with the target in column 0 "
    "and 90 predictors. For this project, the observed file schema is accepted as P90, "
    "matching the selection report. The original YearPredictionMSD sample-size list is "
    "retained without recalculation. This decision records the observed schema; it does "
    "not constitute a full protocol freeze or authorization for confirmatory L2."
)
CODE_IDENTITY_KIND = "SOURCE_TREE_SNAPSHOT_SHA256_V1"
ACCEPTED_CONFIRMATORY_SCOPE = "YEARPREDICTION_MSD_ONLY"
ACCEPTED_SUPERCONDUCTIVITY_CONFIRMATORY = "DEFERRED_MISSING_NESTED_PREFIX_SEEDS"

REQUIRED_FREEZE_FIELDS = (
    "feature_count_amendment",
    "yearprediction_predictor_count",
    "random_seed_list",
    "n_training_subset_repetitions",
    "primary_endpoint",
    "uncertainty_procedure",
    "preprocessing_rule",
    "zero_variance_rule",
    "pcr_rounding_rule",
    "superconductivity_split",
    "cv_fold_assignment_rule",
    "cv_tie_break_rule",
    "cv_score_aggregation",
    "failed_fit_policy",
    "training_subset_policy",
    "confirmatory_scope",
    "superconductivity_confirmatory",
    "code_identity_kind",
    "code_identity_sha256",
)


def ridge_lambda_grid() -> np.ndarray:
    """Exactly 25 log-spaced values from 10^{-8} to 10^{4}, inclusive."""
    grid = np.power(10.0, np.linspace(-8.0, 4.0, 25, dtype=np.float64))
    grid = np.asarray(grid, dtype=np.float64)
    if grid.shape != (25,):
        raise RuntimeError("ridge grid does not have 25 values")
    return grid


def load_yaml(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def freeze_field_status(document: dict) -> dict[str, str]:
    status = {}
    for field in REQUIRED_FREEZE_FIELDS:
        value = document.get(field, None)
        if value is None or value == "" or value == []:
            status[field] = "UNRESOLVED"
        else:
            status[field] = "PRESENT"
    return status


def scope_declaration_problems(document: dict) -> list[str]:
    """Reject a present scope value that contradicts the frozen declaration.

    A missing or empty value is reported by the required-field check.
    """
    reasons = []
    scope = document.get("confirmatory_scope", None)
    if scope not in (None, "", []) and scope != ACCEPTED_CONFIRMATORY_SCOPE:
        reasons.append(f"confirmatory_scope is not {ACCEPTED_CONFIRMATORY_SCOPE}")
    control = document.get("superconductivity_confirmatory", None)
    if control not in (None, "", []) and control != ACCEPTED_SUPERCONDUCTIVITY_CONFIRMATORY:
        reasons.append(
            f"superconductivity_confirmatory is not {ACCEPTED_SUPERCONDUCTIVITY_CONFIRMATORY}"
        )
    return reasons
