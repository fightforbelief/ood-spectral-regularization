"""Fail-closed confirmatory gate. A blank or stale file is not approval."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from p19.code_identity import identity_problems
from p19.protocol import (
    ACCEPTED_FEATURE_COUNT_AMENDMENT,
    P90_DECISION_SENTENCE,
    REQUIRED_FREEZE_FIELDS,
    YEAR_SAMPLE_SIZE_LIST_ORIGINAL,
    YEAR_SELECTION_PREDICTOR_COUNT,
    load_yaml,
    scope_declaration_problems,
)


class GateRefusal(Exception):
    def __init__(self, reasons: list[str]):
        self.reasons = list(reasons)
        super().__init__("; ".join(self.reasons))


@dataclass
class GateDecision:
    approved: bool
    reasons: list[str] = field(default_factory=list)
    protocol_sha256: str | None = None
    recorded_sha256: str | None = None


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_gate_text(text: str) -> tuple[str | None, str | None]:
    status = None
    recorded = None
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("status:"):
            status = line.split(":", 1)[1].strip()
        elif line.startswith("protocol_freeze_sha256:"):
            recorded = line.split(":", 1)[1].strip()
    return status, recorded


def _append_event(project_root: Path, event: dict) -> None:
    path = project_root / "reports" / "gate_audit.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    event = dict(event)
    event["utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def schema_decision_reasons(project_root: Path, document: dict, schema_text: str | None) -> list[str]:
    """Check the formal P90 decision. The measurement file is not an approval form."""
    root = Path(project_root)
    reasons: list[str] = []
    decision_path = root / "reports" / "yearprediction_schema_decision.md"
    if not decision_path.is_file():
        reasons.append("missing yearprediction schema decision")
    else:
        decision_text = decision_path.read_text(encoding="utf-8")
        if P90_DECISION_SENTENCE not in decision_text:
            reasons.append("schema decision report does not contain the accepted P90 decision")
    if document.get("feature_count_amendment") != ACCEPTED_FEATURE_COUNT_AMENDMENT:
        reasons.append("feature_count_amendment is not the accepted P90 decision")
    if document.get("yearprediction_predictor_count") != YEAR_SELECTION_PREDICTOR_COUNT:
        reasons.append("yearprediction_predictor_count is not 90")
    archived = document.get("original_yearprediction_sample_size_list_archived")
    if not isinstance(archived, list) or archived != list(YEAR_SAMPLE_SIZE_LIST_ORIGINAL):
        reasons.append("original YearPredictionMSD sample-size list does not match the archived list")
    if schema_text is None:
        return reasons
    if "MATCHES_SELECTION_REPORT_P90" not in schema_text:
        reasons.append("schema audit does not report MATCHES_SELECTION_REPORT_P90")
    if "BLOCKED_FEATURE_COUNT_MISMATCH" in schema_text:
        reasons.append("schema audit contains BLOCKED_FEATURE_COUNT_MISMATCH")
    return reasons


def evaluate_gate(project_root: Path, action: str) -> GateDecision:
    root = Path(project_root)
    reasons: list[str] = []
    protocol_path = root / "reports" / "protocol_freeze.yaml"
    gate_path = root / "reports" / "confirmatory_gate.md"
    protocol_hash = None
    recorded_hash = None
    status = None
    document: dict = {}
    schema_text = None

    if not (root / "reports" / "environment_audit.md").is_file():
        reasons.append("missing environment audit")
    schema_path = root / "reports" / "data_schema_audit.md"
    if not schema_path.is_file():
        reasons.append("missing data schema audit")
    else:
        schema_text = schema_path.read_text(encoding="utf-8")
        if "x_only_diagnostics: COMPLETE" not in schema_text:
            reasons.append("X-only diagnostics are not marked complete")
    manifest = root / "data" / "manifests" / "source_manifest.json"
    if not manifest.is_file():
        reasons.append("missing raw-data source manifest")
    if not (root / "reports" / "l1_reproduction_report.md").is_file():
        reasons.append("missing L1 report")

    if not protocol_path.is_file():
        reasons.append("missing protocol_freeze.yaml")
    else:
        protocol_hash = sha256_file(protocol_path)
        try:
            document = load_yaml(protocol_path)
        except Exception as exc:
            document = {}
            reasons.append(f"protocol_freeze.yaml is malformed ({type(exc).__name__})")
        if document.get("confirmatory_l2_authorized") is not True:
            reasons.append("confirmatory_l2_authorized is not true")
        for name in REQUIRED_FREEZE_FIELDS:
            value = document.get(name, None)
            if value is None or value == "" or value == []:
                reasons.append(f"unresolved freeze field {name}")
        reasons.extend(scope_declaration_problems(document))

    if not gate_path.is_file():
        reasons.append("missing confirmatory_gate.md")
        status = None
    else:
        status, recorded_hash = _parse_gate_text(gate_path.read_text(encoding="utf-8"))
        if status != "APPROVED":
            reasons.append(f"gate status is {status!r}, not APPROVED")
        if not recorded_hash or recorded_hash in {"", "null", "None"}:
            reasons.append("gate file has no protocol hash")
        elif protocol_hash is not None and recorded_hash != protocol_hash:
            reasons.append("stale protocol hash")

    reasons.extend(schema_decision_reasons(root, document, schema_text))
    reasons.extend(identity_problems(root, document))

    approved = len(reasons) == 0
    decision = GateDecision(
        approved=approved,
        reasons=reasons,
        protocol_sha256=protocol_hash,
        recorded_sha256=recorded_hash,
    )
    _append_event(
        root,
        {
            "action": action,
            "approved": approved,
            "reasons": reasons,
            "protocol_sha256": protocol_hash,
            "recorded_sha256": recorded_hash,
            "gate_status": status,
        },
    )
    return decision


def require_gate(project_root: Path, action: str) -> GateDecision:
    decision = evaluate_gate(project_root, action)
    if not decision.approved:
        raise GateRefusal(decision.reasons)
    return decision
