"""Action-scoped Superconductivity authorization.

A YearPredictionMSD approval is not an authorization to parse critical_temp.
The real staging gate stays closed. Temporary test roots may simulate a later
approval. This module does not call the original project's evaluate_gate.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from p19.code_identity import snapshot_sha256
from p19.gates import GateDecision, GateRefusal
from p19.protocol import load_yaml

CANDIDATE_NAME = "STAGE5N4_SUPERCONDUCTIVITY_PROTOCOL_CANDIDATE.yaml"
CANDIDATE_SHA256 = "829bcb399e9338afd8797d5666440d05785dffc2ee913020be3dc53489819dcb"
FROZEN_PROTOCOL_NAME = "STAGE5N6_SUPERCONDUCTIVITY_PROTOCOL_FROZEN.yaml"
AUTHORIZATION_NAME = "superconductivity_execution_authorization.yaml"
GATE_NAME = "superconductivity_execution_gate.md"
AUDIT_NAME = "superconductivity_gate_audit.jsonl"
ADOPTED_DESIGN_STATUS = "ADOPTED_FOR_ENGINEERING"
CONFIRMATORY_SCOPE = "SUPERCONDUCTIVITY_CONFIRMATORY"
ENGINEERING_SCOPE = "SUPERCONDUCTIVITY_ENGINEERING_ONLY"
SUPERCONDUCTIVITY_ACTIONS = frozenset(
    {
        "load_superconductivity_labeled_data",
        "run-superconductivity-confirmatory",
        "score-superconductivity-holdout",
    }
)


def code_root() -> Path:
    return Path(__file__).resolve().parents[1]


def is_superconductivity_action(action: str) -> bool:
    if not isinstance(action, str) or action == "":
        return False
    return action in SUPERCONDUCTIVITY_ACTIONS or action.startswith("superconductivity:")


def candidate_path(root: Path | None = None) -> Path:
    return Path(root or code_root()) / "reports" / CANDIDATE_NAME


def frozen_protocol_path(project_root: Path | None = None) -> Path:
    return Path(project_root or code_root()) / "reports" / FROZEN_PROTOCOL_NAME


def authorization_path(project_root: Path) -> Path:
    return Path(project_root) / "reports" / AUTHORIZATION_NAME


def gate_path(project_root: Path) -> Path:
    return Path(project_root) / "reports" / GATE_NAME


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bool_token(value: str | None) -> bool | None:
    if value == "true":
        return True
    if value == "false":
        return False
    return None


def parse_superconductivity_gate_text(text: str) -> dict:
    """Parse an explicit gate. A bare APPROVED line does not create a scope."""
    found: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        found[key.strip()] = value.strip()
    return found


def _append_audit(project_root: Path, event: dict) -> None:
    path = Path(project_root) / "reports" / AUDIT_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(event)
    payload["utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def evaluate_superconductivity_authorization(project_root: Path, action: str) -> GateDecision:
    """Fail closed unless this action has its own approved Superconductivity scope.

    The YearPredictionMSD confirmatory_gate.md file is not read. Generic APPROVED
    text, a missing file, a stale hash, or execution_authorized false cannot open
    critical_temp.
    """
    root = Path(project_root)
    reasons: list[str] = []
    protocol_hash = None
    recorded_hash = None
    status = None
    if not isinstance(action, str) or action == "" or not is_superconductivity_action(action):
        reasons.append("action is not a Superconductivity authorization action")

    gate_file = gate_path(root)
    protocol_file = authorization_path(root)
    parsed: dict[str, str] = {}
    if not gate_file.is_file():
        reasons.append("missing superconductivity execution gate")
    else:
        parsed = parse_superconductivity_gate_text(gate_file.read_text(encoding="utf-8"))
        status = parsed.get("status")
        if status != "APPROVED":
            reasons.append(f"superconductivity gate status is {status!r}, not APPROVED")
        if "scope" not in parsed:
            reasons.append("generic APPROVED is not a Superconductivity scope")
        elif parsed.get("scope") != CONFIRMATORY_SCOPE:
            reasons.append(
                f"superconductivity scope is {parsed.get('scope')!r}, not {CONFIRMATORY_SCOPE}"
            )
        gate_execution = _bool_token(parsed.get("execution_authorized"))
        gate_confirmatory = _bool_token(parsed.get("confirmatory_execution_authorized"))
        if gate_execution is None or gate_confirmatory is None:
            reasons.append("superconductivity gate execution flags are missing or malformed")
        elif gate_execution is not True or gate_confirmatory is not True:
            reasons.append("superconductivity execution is not authorized")
            if status == "APPROVED":
                reasons.append("contradictory superconductivity gate approves execution while execution_authorized is false")
        recorded_hash = parsed.get("protocol_sha256")
        if not recorded_hash or recorded_hash in {"", "null", "None"}:
            reasons.append("superconductivity gate has no protocol hash")

    document: dict = {}
    frozen_file = frozen_protocol_path(root)
    frozen_hash = _sha256_file(frozen_file) if frozen_file.is_file() else None
    frozen_document: dict = {}
    if not protocol_file.is_file():
        reasons.append("missing superconductivity execution authorization")
    else:
        authorization_hash = _sha256_file(protocol_file)
        if frozen_hash is None:
            protocol_hash = authorization_hash
            if recorded_hash and protocol_hash != recorded_hash:
                reasons.append("stale superconductivity protocol hash")
        else:
            protocol_hash = frozen_hash
            if recorded_hash != frozen_hash:
                reasons.append("stale superconductivity protocol hash")
            declared_authorization = parsed.get("authorization_sha256")
            if declared_authorization and declared_authorization != authorization_hash:
                reasons.append("stale superconductivity authorization hash")
        try:
            document = load_yaml(protocol_file)
        except Exception as exc:
            document = {}
            reasons.append(f"superconductivity authorization is malformed ({type(exc).__name__})")
        if document.get("research_design_status") != ADOPTED_DESIGN_STATUS:
            reasons.append("research design status is not ADOPTED_FOR_ENGINEERING")
        if document.get("scope") != parsed.get("scope"):
            reasons.append("superconductivity gate scope contradicts the authorization record")
        for name in ("execution_authorized", "confirmatory_execution_authorized"):
            value = document.get(name, None)
            gate_value = _bool_token(parsed.get(name))
            if value is not True:
                reasons.append(f"authorization field {name} is not true")
            if gate_value is not None and value is not None and bool(value) != gate_value:
                reasons.append(f"contradictory authorization field {name}")
        if document.get("candidate_sha256") != CANDIDATE_SHA256:
            reasons.append("authorization does not name the adopted candidate hash")
        if frozen_hash is not None:
            try:
                frozen_document = load_yaml(frozen_file)
            except Exception as exc:
                frozen_document = {}
                reasons.append(f"frozen superconductivity protocol is malformed ({type(exc).__name__})")
            if frozen_document.get("status") != "FROZEN":
                reasons.append("superconductivity protocol is not FROZEN")
            if frozen_document.get("dataset_scope") != "SUPERCONDUCTIVITY":
                reasons.append("superconductivity protocol dataset scope is not SUPERCONDUCTIVITY")
            if frozen_document.get("execution_authorized") is not False or frozen_document.get(
                "confirmatory_execution_authorized"
            ) is not False:
                reasons.append("frozen superconductivity protocol cannot carry execution authorization")
            if frozen_document.get("confirmatory_l2_authorized") is True:
                reasons.append("frozen protocol copied YearPredictionMSD confirmatory authorization")
            if frozen_document.get("candidate_sha256") != CANDIDATE_SHA256:
                reasons.append("frozen protocol does not name the adopted candidate")
            if frozen_document.get("research_design_status") != "APPROVED":
                reasons.append("frozen protocol research design is not APPROVED")

    candidate = candidate_path()
    if not candidate.is_file():
        reasons.append("missing adopted Superconductivity candidate")
    elif _sha256_file(candidate) != CANDIDATE_SHA256:
        reasons.append("adopted Superconductivity candidate hash does not match")

    declared_identity = document.get("staging_code_identity_sha256")
    gate_identity = parsed.get("code_identity_sha256")
    if not isinstance(declared_identity, str) or declared_identity != gate_identity:
        reasons.append("superconductivity code identity is missing or contradictory")
    else:
        try:
            actual_identity = snapshot_sha256(code_root())
        except Exception as exc:
            actual_identity = None
            reasons.append(f"staging source snapshot failed closed ({type(exc).__name__})")
        if actual_identity is not None and declared_identity != actual_identity:
            reasons.append("stale superconductivity source-tree identity")
        if frozen_hash is not None and frozen_document.get("code_identity_sha256") != declared_identity:
            reasons.append("frozen protocol source identity does not match the gate")

    approved = len(reasons) == 0
    decision = GateDecision(
        approved=approved,
        reasons=reasons,
        protocol_sha256=protocol_hash,
        recorded_sha256=recorded_hash,
    )
    _append_audit(
        root,
        {
            "action": action,
            "approved": approved,
            "reasons": reasons,
            "protocol_sha256": protocol_hash,
            "recorded_sha256": recorded_hash,
            "gate_status": status,
            "scope": parsed.get("scope"),
        },
    )
    return decision


def require_superconductivity_authorization(project_root: Path, action: str) -> GateDecision:
    """Check the Superconductivity gate before any labeled file is opened."""
    decision = evaluate_superconductivity_authorization(project_root, action)
    if not decision.approved:
        raise GateRefusal(decision.reasons)
    return decision
