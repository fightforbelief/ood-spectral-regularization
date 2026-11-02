import yaml

from p19.code_identity import (
    ALGORITHM,
    CodeIdentityError,
    identity_problems,
    serialize_records,
    snapshot_sha256,
)


def _marker(root):
    path = root / "p19" / "marker.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("marker = 1\n", encoding="utf-8")
    return path


def test_missing_and_malformed_identity_hash_are_rejected(tmp_path):
    _marker(tmp_path)
    missing = identity_problems(tmp_path, {"code_identity_kind": ALGORITHM, "code_identity_sha256": None})
    assert any("missing or malformed" in reason for reason in missing)
    malformed = identity_problems(
        tmp_path,
        {"code_identity_kind": ALGORITHM, "code_identity_sha256": "abc"},
    )
    assert any("missing or malformed" in reason for reason in malformed)
    uppercase = identity_problems(
        tmp_path,
        {"code_identity_kind": ALGORITHM, "code_identity_sha256": "A" * 64},
    )
    assert any("missing or malformed" in reason for reason in uppercase)
    substituted = identity_problems(
        tmp_path,
        {"code_commit_sha": "a" * 64, "code_identity_sha256": None},
    )
    assert any("code_identity_kind" in reason for reason in substituted)
    assert any("missing or malformed" in reason for reason in substituted)


def test_correct_snapshot_is_accepted_and_a_source_edit_mismatches(tmp_path):
    marker = _marker(tmp_path)
    digest = snapshot_sha256(tmp_path)
    assert identity_problems(
        tmp_path,
        {"code_identity_kind": ALGORITHM, "code_identity_sha256": digest},
    ) == []
    marker.write_text("marker = 2\n", encoding="utf-8")
    assert snapshot_sha256(tmp_path) != digest
    mismatched = identity_problems(
        tmp_path,
        {"code_identity_kind": ALGORITHM, "code_identity_sha256": digest},
    )
    assert any("does not match" in reason for reason in mismatched)


def test_excluded_report_and_artifact_do_not_change_the_snapshot(tmp_path):
    _marker(tmp_path)
    before = snapshot_sha256(tmp_path)
    report = tmp_path / "reports" / "not_source.md"
    report.parent.mkdir(parents=True)
    report.write_text("excluded report\n", encoding="utf-8")
    generated = tmp_path / "artifacts" / "run.txt"
    generated.parent.mkdir()
    generated.write_text("generated\n", encoding="utf-8")
    assert snapshot_sha256(tmp_path) == before


def test_record_order_does_not_change_the_canonical_bytes():
    later = {"path": "p19/b.py", "sha256": "bb", "size_bytes": 2}
    earlier = {"path": "p19/a.py", "sha256": "aa", "size_bytes": 1}
    forward = serialize_records([later, earlier])
    backward = serialize_records([earlier, later])
    assert forward == backward
    text = forward.decode("utf-8")
    assert text.index("p19/a.py") < text.index("p19/b.py")
    assert " " not in text
    assert not text.endswith("\n")


def test_empty_tree_fails_closed(tmp_path):
    try:
        snapshot_sha256(tmp_path)
    except CodeIdentityError as exc:
        assert "no files" in str(exc)
    else:
        raise AssertionError("empty snapshot was accepted")


def test_protocol_file_is_outside_the_snapshot(tmp_path):
    _marker(tmp_path)
    before = snapshot_sha256(tmp_path)
    protocol = tmp_path / "reports" / "protocol_freeze.yaml"
    protocol.parent.mkdir(parents=True, exist_ok=True)
    protocol.write_text(yaml.safe_dump({"code_identity_sha256": "ab"}), encoding="utf-8")
    assert snapshot_sha256(tmp_path) == before
