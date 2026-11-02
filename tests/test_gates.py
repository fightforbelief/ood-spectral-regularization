import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml

from p19.data_loading import (
    DataLoadError,
    load_superconductivity_labeled_data,
    load_yearprediction_labeled_source,
    load_yearprediction_target_labels,
)
from p19.code_identity import snapshot_sha256
from p19.gates import GateRefusal, evaluate_gate, sha256_file
from p19.l2_evaluation import run_confirmatory
from p19.protocol import (
    ACCEPTED_CONFIRMATORY_SCOPE,
    ACCEPTED_FEATURE_COUNT_AMENDMENT,
    ACCEPTED_SUPERCONDUCTIVITY_CONFIRMATORY,
    CODE_IDENTITY_KIND,
    P90_DECISION_SENTENCE,
    REQUIRED_FREEZE_FIELDS,
    YEAR_SAMPLE_SIZE_LIST_ORIGINAL,
    YEAR_SELECTION_PREDICTOR_COUNT,
)


ROOT = Path(__file__).resolve().parents[1]


def _touch(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _supporting_files(root: Path) -> None:
    _touch(root / "p19" / "marker.py", "marker = 1\n")
    _touch(root / "reports" / "environment_audit.md", "environment\n")
    _touch(
        root / "reports" / "data_schema_audit.md",
        "status: MATCHES_SELECTION_REPORT_P90\n"
        "feature_count_amendment: UNRESOLVED\n"
        "x_only_diagnostics: COMPLETE\n",
    )
    _touch(root / "reports" / "yearprediction_schema_decision.md", P90_DECISION_SENTENCE + "\n")
    _touch(root / "data" / "manifests" / "source_manifest.json", "{}\n")
    _touch(root / "reports" / "l1_reproduction_report.md", "l1\n")


def _protocol_document(root: Path) -> dict:
    document = {name: "filled-for-test" for name in REQUIRED_FREEZE_FIELDS}
    document["confirmatory_l2_authorized"] = True
    document["random_seed_list"] = [1, 2]
    document["feature_count_amendment"] = ACCEPTED_FEATURE_COUNT_AMENDMENT
    document["yearprediction_predictor_count"] = YEAR_SELECTION_PREDICTOR_COUNT
    document["original_yearprediction_sample_size_list_archived"] = list(YEAR_SAMPLE_SIZE_LIST_ORIGINAL)
    document["code_identity_kind"] = CODE_IDENTITY_KIND
    document["code_identity_sha256"] = snapshot_sha256(root)
    document["confirmatory_scope"] = ACCEPTED_CONFIRMATORY_SCOPE
    document["superconductivity_confirmatory"] = ACCEPTED_SUPERCONDUCTIVITY_CONFIRMATORY
    return document


def _complete_freeze(root: Path) -> Path:
    path = root / "reports" / "protocol_freeze.yaml"
    _touch(path, yaml.safe_dump(_protocol_document(root)))
    return path


def _seal(root: Path) -> None:
    protocol = root / "reports" / "protocol_freeze.yaml"
    _touch(
        root / "reports" / "confirmatory_gate.md",
        f"status: APPROVED\nprotocol_freeze_sha256: {sha256_file(protocol)}\n",
    )


def test_missing_and_malformed_gates_fail_closed(tmp_path):
    missing = evaluate_gate(tmp_path, "missing-gate")
    assert missing.approved is False
    _supporting_files(tmp_path)
    _complete_freeze(tmp_path)
    _touch(tmp_path / "reports" / "confirmatory_gate.md", "")
    blank = evaluate_gate(tmp_path, "blank-gate")
    assert blank.approved is False
    _touch(tmp_path / "reports" / "confirmatory_gate.md", "status: NOT_APPROVED\n")
    negated = evaluate_gate(tmp_path, "negated-gate")
    assert negated.approved is False
    assert any("NOT_APPROVED" in reason or "not APPROVED" in reason for reason in negated.reasons)


def test_stale_hash_invalidates_an_otherwise_complete_gate(tmp_path):
    _supporting_files(tmp_path)
    protocol = _complete_freeze(tmp_path)
    _touch(
        tmp_path / "reports" / "confirmatory_gate.md",
        "status: APPROVED\nprotocol_freeze_sha256: " + ("0" * 64) + "\n",
    )
    decision = evaluate_gate(tmp_path, "stale-hash")
    assert decision.approved is False
    assert any("stale" in reason for reason in decision.reasons)
    assert decision.protocol_sha256 == sha256_file(protocol)
    _touch(
        tmp_path / "reports" / "confirmatory_gate.md",
        f"status: APPROVED\nprotocol_freeze_sha256: {decision.protocol_sha256}\n",
    )
    opened = evaluate_gate(tmp_path, "matching-hash")
    assert opened.approved is True


def _gate_field(text: str, name: str) -> list[str]:
    prefix = f"{name}:"
    return [line.split(":", 1)[1].strip() for line in text.splitlines() if line.startswith(prefix)]


def _authorization_pair_is_consistent(authorized: bool, gate_status: list[str]) -> bool:
    if authorized is False:
        return gate_status == ["NOT_APPROVED"]
    if authorized is True:
        return gate_status == ["APPROVED"]
    return False


def test_live_gate_status_matches_protocol_authorization():
    protocol_path = ROOT / "reports" / "protocol_freeze.yaml"
    document = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    authorized = document["confirmatory_l2_authorized"]
    assert isinstance(authorized, bool)
    assert document["status"] == "FROZEN"
    gate = (ROOT / "reports" / "confirmatory_gate.md").read_text(encoding="utf-8")
    assert _authorization_pair_is_consistent(authorized, _gate_field(gate, "status"))
    assert _gate_field(gate, "protocol_freeze_sha256") == [sha256_file(protocol_path)]


def test_closed_temporary_gate_refuses_before_reading_a_sentinel(tmp_path):
    _supporting_files(tmp_path)
    _complete_freeze(tmp_path)
    _touch(tmp_path / "reports" / "confirmatory_gate.md", "status: NOT_APPROVED\n")
    sentinel = tmp_path / "data" / "raw" / "YearPredictionMSD.txt"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_bytes(b"DO_NOT_READ,1.0\n")
    with pytest.raises(GateRefusal):
        run_confirmatory(tmp_path)
    assert sentinel.read_bytes() == b"DO_NOT_READ,1.0\n"
    audit = (tmp_path / "reports" / "gate_audit.jsonl").read_text(encoding="utf-8")
    assert "run-l2-confirmatory" in audit
    assert '"approved": false' in audit


def _approve(root: Path) -> None:
    _complete_freeze(root)
    _seal(root)


def test_missing_freeze_field_does_not_default_to_approved(tmp_path):
    _supporting_files(tmp_path)
    document = _protocol_document(tmp_path)
    document["primary_endpoint"] = None
    path = tmp_path / "reports" / "protocol_freeze.yaml"
    _touch(path, yaml.safe_dump(document))
    _seal(tmp_path)
    decision = evaluate_gate(tmp_path, "missing-primary-endpoint")
    assert decision.approved is False
    assert any("primary_endpoint" in reason for reason in decision.reasons)


def test_approved_synthetic_gate_rejects_a_sentinel_before_a_confirmatory_result(tmp_path):
    _supporting_files(tmp_path)
    _approve(tmp_path)
    sentinel = tmp_path / "data" / "raw" / "YearPredictionMSD.txt"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_bytes(b"DO_NOT_READ,1.0\n")
    with pytest.raises(DataLoadError, match="row count"):
        run_confirmatory(tmp_path)
    assert sentinel.read_bytes() == b"DO_NOT_READ,1.0\n"


def test_loader_refuses_before_opening_a_missing_path(tmp_path):
    _supporting_files(tmp_path)
    _complete_freeze(tmp_path)
    _touch(tmp_path / "reports" / "confirmatory_gate.md", "status: NOT_APPROVED\n")
    missing = tmp_path / "data" / "raw" / "not-a-real-label-file.csv"
    assert not missing.exists()
    with pytest.raises(GateRefusal):
        load_yearprediction_target_labels(missing, tmp_path)
    assert not missing.exists()


def test_open_synthetic_gate_parses_only_the_target_token(tmp_path):
    _supporting_files(tmp_path)
    _approve(tmp_path)
    year = tmp_path / "tiny_year.csv"
    year.write_text("7,100.0\n8,200.0\n", encoding="utf-8")
    labels = load_yearprediction_target_labels(year, tmp_path)
    assert np.allclose(labels, [7.0, 8.0])
    source = load_yearprediction_labeled_source(year, tmp_path, source_rows=1)
    assert np.allclose(source["y"], [7.0])
    assert source["labels_loaded"] is True
    super_path = tmp_path / "tiny_super.csv"
    super_path.write_text(
        '"a","critical_temp"\n"1.5","NOT_PARSED_AS_FEATURE"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_superconductivity_labeled_data(super_path, tmp_path)
    super_path.write_text('"a","critical_temp"\n"9.0","4.5"\n', encoding="utf-8")
    loaded = load_superconductivity_labeled_data(super_path, tmp_path)
    assert loaded["labels_loaded"] is True
    assert loaded["target_index"] == 1
    assert np.allclose(loaded["y"], [4.5])


def test_unresolved_measurement_line_allows_a_separate_p90_decision(tmp_path):
    _supporting_files(tmp_path)
    audit = (tmp_path / "reports" / "data_schema_audit.md").read_text(encoding="utf-8")
    assert "feature_count_amendment: UNRESOLVED" in audit
    assert "feature_count_amendment: APPROVED" not in audit
    _approve(tmp_path)
    sentinel = tmp_path / "data" / "raw" / "YearPredictionMSD.txt"
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text("DO_NOT_READ,1.0\n", encoding="utf-8")
    decision = evaluate_gate(tmp_path, "p90-measurement-unresolved")
    assert decision.approved is True
    assert sentinel.read_text(encoding="utf-8") == "DO_NOT_READ,1.0\n"
    assert not any("feature-count amendment is not approved" in reason for reason in decision.reasons)


def test_missing_or_contradictory_p90_decision_is_rejected(tmp_path):
    _supporting_files(tmp_path)
    _approve(tmp_path)
    decision_path = tmp_path / "reports" / "yearprediction_schema_decision.md"
    decision_path.unlink()
    missing = evaluate_gate(tmp_path, "missing-p90-decision")
    assert missing.approved is False
    assert any("schema decision" in reason for reason in missing.reasons)

    _touch(decision_path, P90_DECISION_SENTENCE + "\n")
    document = _protocol_document(tmp_path)
    document["yearprediction_predictor_count"] = 89
    protocol = tmp_path / "reports" / "protocol_freeze.yaml"
    _touch(protocol, yaml.safe_dump(document))
    _seal(tmp_path)
    contradictory = evaluate_gate(tmp_path, "contradictory-predictor-count")
    assert contradictory.approved is False
    assert any("yearprediction_predictor_count" in reason for reason in contradictory.reasons)

    document = _protocol_document(tmp_path)
    document["original_yearprediction_sample_size_list_archived"] = [1]
    _touch(protocol, yaml.safe_dump(document))
    _seal(tmp_path)
    amended = evaluate_gate(tmp_path, "changed-sample-size-list")
    assert amended.approved is False
    assert any("sample-size list" in reason for reason in amended.reasons)

    _complete_freeze(tmp_path)
    _seal(tmp_path)
    audit = tmp_path / "reports" / "data_schema_audit.md"
    audit.write_text(audit.read_text(encoding="utf-8") + "BLOCKED_FEATURE_COUNT_MISMATCH\n", encoding="utf-8")
    blocked = evaluate_gate(tmp_path, "blocked-feature-count")
    assert blocked.approved is False
    assert any("BLOCKED_FEATURE_COUNT_MISMATCH" in reason for reason in blocked.reasons)


def test_changing_included_source_fails_the_open_gate(tmp_path):
    _supporting_files(tmp_path)
    _approve(tmp_path)
    assert evaluate_gate(tmp_path, "before-source-edit").approved is True
    (tmp_path / "p19" / "marker.py").write_text("marker = 2\n", encoding="utf-8")
    changed = evaluate_gate(tmp_path, "after-source-edit")
    assert changed.approved is False
    assert any("does not match" in reason for reason in changed.reasons)


def test_real_protocol_identity_matches_the_source_tree_and_the_gate_hash():
    document = yaml.safe_load((ROOT / "reports" / "protocol_freeze.yaml").read_text(encoding="utf-8"))
    identity = document["code_identity_sha256"]
    assert isinstance(identity, str) and len(identity) == 64 and identity == identity.lower()
    int(identity, 16)
    assert identity == snapshot_sha256(ROOT)
    assert document["code_identity_kind"] == CODE_IDENTITY_KIND
    assert document["status"] == "FROZEN"
    authorized = document["confirmatory_l2_authorized"]
    assert isinstance(authorized, bool)
    assert document["confirmatory_scope"] == ACCEPTED_CONFIRMATORY_SCOPE
    assert document["superconductivity_confirmatory"] == ACCEPTED_SUPERCONDUCTIVITY_CONFIRMATORY
    protocol_path = ROOT / "reports" / "protocol_freeze.yaml"
    gate = (ROOT / "reports" / "confirmatory_gate.md").read_text(encoding="utf-8")
    assert _authorization_pair_is_consistent(authorized, _gate_field(gate, "status"))
    assert _gate_field(gate, "protocol_freeze_sha256") == [sha256_file(protocol_path)]
    assert _gate_field(gate, "code_identity_sha256") == ["null"]



def test_missing_or_contradictory_scope_fails_closed(tmp_path):
    _supporting_files(tmp_path)
    path = tmp_path / "reports" / "protocol_freeze.yaml"

    document = _protocol_document(tmp_path)
    document.pop("confirmatory_scope")
    _touch(path, yaml.safe_dump(document))
    _seal(tmp_path)
    missing_scope = evaluate_gate(tmp_path, "missing-confirmatory-scope")
    assert missing_scope.approved is False
    assert any(reason == "unresolved freeze field confirmatory_scope" for reason in missing_scope.reasons)
    assert not any("confirmatory_scope is not" in reason for reason in missing_scope.reasons)

    document = _protocol_document(tmp_path)
    document["confirmatory_scope"] = "BOTH_DATASETS"
    _touch(path, yaml.safe_dump(document))
    _seal(tmp_path)
    wrong_scope = evaluate_gate(tmp_path, "wrong-confirmatory-scope")
    assert wrong_scope.approved is False
    assert any(reason == "confirmatory_scope is not YEARPREDICTION_MSD_ONLY" for reason in wrong_scope.reasons)
    assert not any(reason == "unresolved freeze field confirmatory_scope" for reason in wrong_scope.reasons)

    document = _protocol_document(tmp_path)
    document.pop("superconductivity_confirmatory")
    _touch(path, yaml.safe_dump(document))
    _seal(tmp_path)
    missing_control = evaluate_gate(tmp_path, "missing-superconductivity-confirmatory")
    assert missing_control.approved is False
    assert any(
        reason == "unresolved freeze field superconductivity_confirmatory"
        for reason in missing_control.reasons
    )
    assert not any("superconductivity_confirmatory is not" in reason for reason in missing_control.reasons)

    document = _protocol_document(tmp_path)
    document["superconductivity_confirmatory"] = "AUTHORIZED"
    _touch(path, yaml.safe_dump(document))
    _seal(tmp_path)
    wrong_control = evaluate_gate(tmp_path, "wrong-superconductivity-confirmatory")
    assert wrong_control.approved is False
    assert any(
        reason == "superconductivity_confirmatory is not DEFERRED_MISSING_NESTED_PREFIX_SEEDS"
        for reason in wrong_control.reasons
    )
    assert not any(
        reason == "unresolved freeze field superconductivity_confirmatory"
        for reason in wrong_control.reasons
    )

def _install_authorization_state(root: Path, *, authorized: bool, gate_status: str) -> None:
    _supporting_files(root)
    document = _protocol_document(root)
    document["confirmatory_l2_authorized"] = authorized
    protocol = root / "reports" / "protocol_freeze.yaml"
    _touch(protocol, yaml.safe_dump(document))
    _touch(
        root / "reports" / "confirmatory_gate.md",
        f"status: {gate_status}\nprotocol_freeze_sha256: {sha256_file(protocol)}\n",
    )


def test_authorization_state_matrix_uses_only_temporary_roots(tmp_path):
    closed = tmp_path / "closed"
    _install_authorization_state(closed, authorized=False, gate_status="NOT_APPROVED")
    sentinel = closed / "data" / "raw" / "YearPredictionMSD.txt"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_bytes(b"DO_NOT_READ,1.0\n")
    closed_gate = (closed / "reports" / "confirmatory_gate.md").read_text(encoding="utf-8")
    closed_document = yaml.safe_load((closed / "reports" / "protocol_freeze.yaml").read_text(encoding="utf-8"))
    assert _authorization_pair_is_consistent(
        closed_document["confirmatory_l2_authorized"],
        _gate_field(closed_gate, "status"),
    )
    refused = evaluate_gate(closed, "matrix-closed")
    assert refused.approved is False
    with pytest.raises(GateRefusal):
        run_confirmatory(closed)
    assert sentinel.read_bytes() == b"DO_NOT_READ,1.0\n"

    opened = tmp_path / "opened"
    _install_authorization_state(opened, authorized=True, gate_status="APPROVED")
    opened_decision = evaluate_gate(opened, "matrix-opened")
    assert opened_decision.approved is True
    assert opened_decision.protocol_sha256 == sha256_file(opened / "reports" / "protocol_freeze.yaml")

    mismatched_open = tmp_path / "false-approved"
    _install_authorization_state(mismatched_open, authorized=False, gate_status="APPROVED")
    false_open = evaluate_gate(mismatched_open, "matrix-false-approved")
    assert false_open.approved is False
    assert any("confirmatory_l2_authorized is not true" in reason for reason in false_open.reasons)

    mismatched_closed = tmp_path / "true-not-approved"
    _install_authorization_state(mismatched_closed, authorized=True, gate_status="NOT_APPROVED")
    true_closed = evaluate_gate(mismatched_closed, "matrix-true-not-approved")
    assert true_closed.approved is False
    assert any("not APPROVED" in reason for reason in true_closed.reasons)


def test_raw_data_paths_are_gitignored():
    probe = ROOT / "data" / "raw" / "example.txt"
    completed = subprocess.run(
        ["git", "check-ignore", "-q", "--", "data/raw/example.txt"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert not probe.exists() or True
