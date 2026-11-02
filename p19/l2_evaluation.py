"""Confirmatory L2 entry point. The gate is checked before any target label is read."""

from __future__ import annotations

from pathlib import Path

from p19.gates import evaluate_gate
from p19.protocol import REQUIRED_FREEZE_FIELDS, freeze_field_status, load_yaml


def unresolved_freeze_fields(project_root: Path) -> dict[str, str]:
    path = Path(project_root) / "reports" / "protocol_freeze.yaml"
    if not path.is_file():
        return {name: "MISSING_FILE" for name in REQUIRED_FREEZE_FIELDS}
    return freeze_field_status(load_yaml(path))


def l2_preflight(project_root: Path) -> dict:
    root = Path(project_root)
    decision = evaluate_gate(root, "l2-preflight")
    fields = unresolved_freeze_fields(root)
    return {
        "confirmatory_l2": "BLOCKED",
        "gate_approved": decision.approved,
        "gate_reasons": decision.reasons,
        "freeze_fields": fields,
        "unresolved_fields": [name for name, state in fields.items() if state != "PRESENT"],
        "target_labels_loaded": False,
    }


def run_confirmatory(project_root: Path) -> dict:
    """Production YearPredictionMSD entry. ``evaluate_gate`` runs before any data file is opened.

    The stored gate still refuses, so this function does not reach a loader until a
    future valid approval. There is no second authorization check.
    """
    from p19.l2_runner import run_yearprediction_confirmatory

    return run_yearprediction_confirmatory(Path(project_root))
