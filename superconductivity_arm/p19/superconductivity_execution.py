"""Connected Superconductivity execution plan.

The official entry calls this only after the Superconductivity gate has approved
the action. This module does not read the YearPredictionMSD gate and does not
call the YearPredictionMSD runner.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from p19.code_identity import snapshot_sha256
from p19.l2_pipeline import HoldoutScoringBlocked, fit_training_prefix
from p19.l2_runner import prediction_identity_order
from p19.l2_sampling import ProtocolValidationError
from p19.protocol import SUPER_SAMPLE_SIZE_LIST_ORIGINAL, load_yaml
from p19.superconductivity_authorization import (
    CANDIDATE_SHA256,
    code_root,
    frozen_protocol_path,
    require_superconductivity_authorization,
)
from p19.superconductivity_design import ESTIMATOR_ORDER, load_adopted_candidate, load_stored_sampling_seeds
from p19.superconductivity_runner import (
    FROZEN_GROUPED_HOLDOUT_ROWS,
    INCOMPLETE_STATUS,
    MANIFEST_NAME,
    PredictionArtifactError,
    SuperconductivityExecutionBlocked,
    SuperconductivityPersistenceError,
    default_superconductivity_output_dir,
    incomplete_confirmatory_record,
    publish_verified_results,
    run_superconductivity_source_phase,
    score_superconductivity_source_phase,
    summarize_superconductivity_holdout,
    write_superconductivity_prediction_artifact,
    write_superconductivity_run_manifest,
)
from p19.superconductivity_xonly_audit import (
    EXPECTED_EXTRA,
    EXPECTED_GROUPS,
    EXPECTED_HOLDOUT_GROUPS,
    EXPECTED_ROWS,
    EXPECTED_TRAIN_GROUPS,
    ORIGINAL_TRAIN,
)

SPLIT_DECLARATION_SHA256 = "28abcf33d8e4c0f0c404dbd9119899e149c2ee625522fcc65c80749866c642e3"
FROZEN_TRAIN_ROWS = 16871
ROW_INDEX_ORDER = "frozen_splitter_group_permutation_concatenation"
EXECUTION_CALL_GRAPH = (
    "require_superconductivity_authorization",
    "load_adopted_candidate",
    "load_frozen_protocol",
    "load_superconductivity_x_only",
    "load_superconductivity_labeled_data",
    "split_superconductivity_groups",
    "validate_frozen_split_membership",
    "run_superconductivity_source_phase",
    "fit_training_prefix",
    "validate_complete_source_phase",
    "score_superconductivity_source_phase",
    "summarize_superconductivity_holdout",
    "write_superconductivity_prediction_artifact",
    "persist_metric_records",
    "persist_uncertainty_artifact",
    "reopen_and_validate_artifacts",
    "write_superconductivity_run_manifest",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.int64).tobytes()).hexdigest()


def saved_split_paths(project_root: Path) -> tuple[Path, Path]:
    directory = Path(project_root) / "reports" / "stage5n5_xonly"
    return directory / "training_row_indices.npy", directory / "holdout_row_indices.npy"


def validate_saved_split_indices(train_rows: np.ndarray, holdout_rows: np.ndarray) -> dict:
    """Check the saved X-only partition. This does not sort either vector."""
    train = np.asarray(train_rows, dtype=np.int64).reshape(-1)
    holdout = np.asarray(holdout_rows, dtype=np.int64).reshape(-1)
    problems = []
    if train.shape != (FROZEN_TRAIN_ROWS,):
        problems.append(f"training rows {train.size} != {FROZEN_TRAIN_ROWS}")
    if holdout.shape != (FROZEN_GROUPED_HOLDOUT_ROWS,):
        problems.append(f"holdout rows {holdout.size} != {FROZEN_GROUPED_HOLDOUT_ROWS}")
    train_set = set(train.tolist())
    holdout_set = set(holdout.tolist())
    if train_set & holdout_set:
        problems.append("training and holdout indices intersect")
    if train_set | holdout_set != set(range(EXPECTED_ROWS)):
        problems.append("training and holdout indices do not cover the original rows")
    if len(train_set) != train.size or len(holdout_set) != holdout.size:
        problems.append("a split index vector contains a duplicate")
    if np.array_equal(train, np.sort(train)):
        problems.append("training-row vector was globally sorted; the frozen splitter order is required")
    if problems:
        raise ProtocolValidationError("; ".join(problems))
    return {
        "row_index_order": ROW_INDEX_ORDER,
        "training_rows_globally_ascending": False,
        "n_train_rows": int(train.size),
        "n_holdout_rows": int(holdout.size),
        "training_row_index_sha256": _sha256_array(train),
        "holdout_row_index_sha256": _sha256_array(holdout),
        "group_crossings": 0,
    }


def validate_frozen_split_result(split: dict, saved_train: np.ndarray, saved_holdout: np.ndarray) -> dict:
    """Accept only the realized grouped partition, in the splitter's own order."""
    train = np.asarray(split["train_rows"], dtype=np.int64).reshape(-1)
    holdout = np.asarray(split["test_rows"], dtype=np.int64).reshape(-1)
    identity = validate_saved_split_indices(saved_train, saved_holdout)
    if not np.array_equal(train, np.asarray(saved_train, dtype=np.int64).reshape(-1)):
        raise ProtocolValidationError("recomputed training-row order does not match the saved X-only split")
    if not np.array_equal(holdout, np.asarray(saved_holdout, dtype=np.int64).reshape(-1)):
        raise ProtocolValidationError("recomputed holdout-row order does not match the saved X-only split")
    if int(split["n_groups"]) != EXPECTED_GROUPS:
        raise ProtocolValidationError("group count does not match the frozen split")
    if int(split["n_train_groups"]) != EXPECTED_TRAIN_GROUPS:
        raise ProtocolValidationError("training group count does not match the frozen split")
    if int(split["n_test_groups"]) != EXPECTED_HOLDOUT_GROUPS:
        raise ProtocolValidationError("holdout group count does not match the frozen split")
    if int(split["split_seed"]) != 732111676:
        raise ProtocolValidationError("split seed does not match the frozen split")
    extra = EXPECTED_ROWS - int(split["n_groups"])
    if extra != EXPECTED_EXTRA:
        raise ProtocolValidationError("exact-duplicate count does not match the frozen split")
    return identity


def _production_executor(features: np.ndarray, labels: np.ndarray):
    def execute_cell(**kwargs):
        indices = np.asarray(kwargs["prefix_indices"], dtype=np.int64)
        return fit_training_prefix(
            features[indices],
            labels[indices],
            repetition=kwargs["repetition"],
            sampling_seed=kwargs["sampling_seed"],
            sample_size=kwargs["sample_size"],
            prefix_indices=indices,
            folds=kwargs["folds"],
        )

    return execute_cell


def _load_frozen_design(project_root: Path) -> tuple[dict, dict, str]:
    candidate = load_adopted_candidate()
    frozen_path = frozen_protocol_path(project_root)
    if not frozen_path.is_file():
        raise SuperconductivityExecutionBlocked("frozen Superconductivity protocol is missing")
    frozen = load_yaml(frozen_path)
    if list(frozen.get("random_seed_list", [])) != load_stored_sampling_seeds():
        raise ProtocolValidationError("frozen seed list does not match the historical candidate")
    if list(frozen.get("sample_size_list", [])) != list(SUPER_SAMPLE_SIZE_LIST_ORIGINAL):
        raise ProtocolValidationError("frozen sample-size grid does not match the historical candidate")
    if frozen.get("candidate_sha256") != CANDIDATE_SHA256:
        raise ProtocolValidationError("frozen protocol does not name the historical candidate")
    if frozen.get("execution_authorized") is not False or frozen.get("confirmatory_execution_authorized") is not False:
        raise SuperconductivityExecutionBlocked("frozen protocol cannot authorize execution")
    adopted = frozen.get("adopted_candidate")
    if not isinstance(adopted, dict) or list(adopted.get("random_seed_list", [])) != list(candidate["random_seed_list"]):
        raise ProtocolValidationError("frozen protocol does not reproduce the historical candidate seeds")
    return candidate, frozen, _sha256_file(frozen_path)


def execute_official_superconductivity(project_root: Path, output_dir: Path | None = None) -> dict:
    """Run the frozen plan. The caller and this function both require authorization."""
    root = Path(project_root)
    require_superconductivity_authorization(root, "run-superconductivity-confirmatory")
    _load_frozen_design(root)
    declaration_path = root / "reports" / "protocol_freeze.yaml"
    if _sha256_file(declaration_path) != SPLIT_DECLARATION_SHA256:
        raise ProtocolValidationError("split declaration hash does not match the frozen YearPrediction protocol text")
    declaration = declaration_path.read_text(encoding="utf-8")

    from p19.data_loading import load_superconductivity_labeled_data, load_superconductivity_x_only
    from p19.superconductivity_split import split_superconductivity_groups

    loaded_x = load_superconductivity_x_only(ORIGINAL_TRAIN)
    if loaded_x["labels_loaded"] is not False or int(loaded_x["predictor_count"]) != 81:
        raise ProtocolValidationError("X-only load did not return 81 unlabeled predictors")
    features = np.asarray(loaded_x["X"], dtype=np.float64)
    if features.shape != (EXPECTED_ROWS, 81):
        raise ProtocolValidationError("X-only matrix does not match the frozen row and predictor counts")
    require_superconductivity_authorization(root, "load_superconductivity_labeled_data")
    loaded_y = load_superconductivity_labeled_data(ORIGINAL_TRAIN, root)
    labels = np.asarray(loaded_y["y"], dtype=np.float64).reshape(-1)
    if labels.shape != (EXPECTED_ROWS,) or loaded_y.get("labels_loaded") is not True:
        raise ProtocolValidationError("labeled vector does not match the frozen row count")

    split = split_superconductivity_groups(features, protocol_text=declaration)
    train_path, holdout_path = saved_split_paths(root)
    saved_train = np.load(train_path)
    saved_holdout = np.load(holdout_path)
    identity = validate_frozen_split_result(split, saved_train, saved_holdout)
    train_set = set(np.asarray(saved_train, dtype=np.int64).tolist())
    holdout_set = set(np.asarray(saved_holdout, dtype=np.int64).tolist())
    crossings = 0
    for group in split["groups"]:
        members = set(np.asarray(group, dtype=np.int64).tolist())
        if members & train_set and members & holdout_set:
            crossings += 1
    if crossings != 0:
        raise ProtocolValidationError("exact feature groups cross the frozen outer split")
    identity["group_crossings"] = 0
    train_rows = np.asarray(split["train_rows"], dtype=np.int64)
    holdout_rows = np.asarray(split["test_rows"], dtype=np.int64)

    if output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = default_superconductivity_output_dir(root, stamp, _sha256_file(frozen_protocol_path(root)))
    source = run_superconductivity_source_phase(
        train_rows,
        _production_executor(features, labels),
        holdout_rows,
    )
    if not source.complete:
        record = incomplete_confirmatory_record(source)
        record["synthetic"] = False
        record["official_target_labels_loaded"] = True
        record["official_models_fit"] = False
        record["artifacts_verified"] = False
        record["source_cells_attempted"] = len(source.cells)
        record["run_id"] = output_dir.name
        record["output_dir"] = str(output_dir)
        output_dir.mkdir(parents=True, exist_ok=False)
        write_superconductivity_run_manifest(output_dir, record)
        return record

    scored = score_superconductivity_source_phase(
        source,
        features[holdout_rows],
        labels[holdout_rows],
    )
    uncertainty = summarize_superconductivity_holdout(scored)
    ordered = _ordered_metric_rows(scored)
    artifact, hashes = write_superconductivity_prediction_artifact(
        output_dir,
        [row["predictions"] for row in ordered],
        sample_sizes=SUPER_SAMPLE_SIZE_LIST_ORIGINAL,
        n_repetitions=100,
        holdout_row_count=FROZEN_GROUPED_HOLDOUT_ROWS,
        split_identity=identity,
        official_write=True,
    )
    order = prediction_identity_order(100, SUPER_SAMPLE_SIZE_LIST_ORIGINAL)
    try:
        return publish_verified_results(
            output_dir,
            ordered,
            hashes,
            order,
            uncertainty,
            artifact,
            synthetic=False,
            protocol_sha256=_sha256_file(frozen_protocol_path(root)),
            source_identity_sha256=snapshot_sha256(code_root()),
            split_identity=identity,
        )
    except (SuperconductivityPersistenceError, PredictionArtifactError) as exc:
        record = {
            "status": INCOMPLETE_STATUS,
            "synthetic": False,
            "confirmatory_result": False,
            "artifacts_verified": False,
            "official_target_labels_loaded": True,
            "official_models_fit": True,
            "holdout_scored": True,
            "run_id": output_dir.name,
            "output_dir": str(output_dir),
            "failure": {"type": type(exc).__name__, "message": str(exc)},
        }
        if not (output_dir / MANIFEST_NAME).exists():
            write_superconductivity_run_manifest(output_dir, record)
        return record


def _ordered_metric_rows(scored: list[dict]) -> list[dict]:
    from p19.l2_runner import prediction_identity_order

    order = prediction_identity_order(100, SUPER_SAMPLE_SIZE_LIST_ORIGINAL)
    located = {(int(row["repetition"]), int(row["sample_size"]), str(row["estimator"])): row for row in scored}
    if set(located) != set(order):
        raise HoldoutScoringBlocked("scored identities do not match the prediction order")
    return [located[key] for key in order]
