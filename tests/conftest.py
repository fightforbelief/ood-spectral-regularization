"""The suite may read the live protocol and gate. It must not rewrite them."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_PATHS = (
    _ROOT / "reports" / "protocol_freeze.yaml",
    _ROOT / "reports" / "confirmatory_gate.md",
    _ROOT / "reports" / "gate_audit.jsonl",
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="session", autouse=True)
def live_governance_files_stay_byte_for_byte_unchanged():
    before = {path: _digest(path) for path in _PATHS}
    yield
    after = {path: _digest(path) for path in _PATHS}
    assert after == before
