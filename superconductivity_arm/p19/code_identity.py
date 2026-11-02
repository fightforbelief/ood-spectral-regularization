"""Deterministic source-tree identity. This is not a Git commit id.

Algorithm SOURCE_TREE_SNAPSHOT_SHA256_V1 hashes a declared set of executable
and configuration files. Reports, data, artifacts, and this manifest are outside
that set. The resulting digest is an observed snapshot until a later protocol
freeze copies it into reports/protocol_freeze.yaml.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ALGORITHM = "SOURCE_TREE_SNAPSHOT_SHA256_V1"

EXCLUDED_DIRECTORY_NAMES = frozenset(
    {
        ".git",
        ".venv",
        ".ssh",
        ".pytest_cache",
        "__pycache__",
        "data",
        "artifacts",
        "results",
        "logs",
        "reports",
        "docs",
    }
)

ROOT_PACKAGING_NAMES = frozenset(
    {
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "requirements.txt",
        "pytest.ini",
        "conftest.py",
        "Pipfile",
        "Pipfile.lock",
        "poetry.lock",
        "uv.lock",
        "pdm.lock",
    }
)

INCLUSION_POLICY = (
    "p19/**/*.py",
    "tests/**/*.py",
    "configs/**",
    "root/*.py",
    "root packaging and dependency files: pyproject.toml, setup.py, setup.cfg, "
    "requirements*.txt, pytest.ini, conftest.py, Pipfile, Pipfile.lock, "
    "poetry.lock, uv.lock, pdm.lock",
)

EXCLUSION_POLICY = (
    ".git/",
    ".venv/",
    ".ssh/",
    ".pytest_cache/",
    "__pycache__/",
    "data/",
    "artifacts/",
    "results/",
    "logs/",
    "reports/ including the identity manifest, protocol_freeze.yaml, confirmatory_gate.md, and gate_audit.jsonl",
    "docs/",
    "non-executable documentation and other root files that are not Python entry points or packaging files",
    "symbolic links",
)


class CodeIdentityError(Exception):
    pass


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def is_included_relative(relative: str) -> bool:
    """Return whether a repository-relative POSIX path is in the snapshot."""
    if relative == "" or relative.endswith("/"):
        return False
    parts = relative.split("/")
    if any(part in EXCLUDED_DIRECTORY_NAMES or part.startswith(".") for part in parts):
        return False
    name = parts[-1]
    if len(parts) == 1:
        if name.endswith(".py"):
            return True
        if name in ROOT_PACKAGING_NAMES:
            return True
        return name.startswith("requirements") and name.endswith(".txt")
    top = parts[0]
    if top in {"p19", "tests"} and name.endswith(".py"):
        return True
    return top == "configs"


def serialize_records(records: list[dict]) -> bytes:
    """Canonical UTF-8 JSON. Path order and object key order are both fixed."""
    ordered = sorted(records, key=lambda row: row["path"])
    text = json.dumps(ordered, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return text.encode("utf-8")


def collect_records(project_root: Path) -> list[dict]:
    """Hash included files. A symlink or an empty set fails closed."""
    root = Path(project_root)
    if not root.is_dir():
        raise CodeIdentityError(f"project root is not a directory: {root}")
    records: list[dict] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(dirpath)
        kept: list[str] = []
        for name in dirnames:
            child = current / name
            if child.is_symlink():
                raise CodeIdentityError(f"refusing to follow symlink {child}")
            if name in EXCLUDED_DIRECTORY_NAMES or name.startswith("."):
                continue
            kept.append(name)
        dirnames[:] = kept
        for name in filenames:
            path = current / name
            if path.is_symlink():
                raise CodeIdentityError(f"refusing to hash symlink {path}")
            relative = path.relative_to(root).as_posix()
            if not is_included_relative(relative):
                continue
            payload = path.read_bytes()
            records.append(
                {
                    "path": relative,
                    "size_bytes": len(payload),
                    "sha256": _sha256(payload),
                }
            )
    if not records:
        raise CodeIdentityError("source snapshot matched no files")
    records.sort(key=lambda row: row["path"])
    return records


def snapshot_sha256(project_root: Path) -> str:
    """SHA-256 of the canonical record array for SOURCE_TREE_SNAPSHOT_SHA256_V1."""
    return _sha256(serialize_records(collect_records(project_root)))


def identity_problems(project_root: Path, document: dict) -> list[str]:
    """Reasons the declared identity does not match a fresh snapshot.

    A non-empty string is not enough. code_commit_sha is ignored.
    """
    reasons: list[str] = []
    kind = document.get("code_identity_kind")
    if kind != ALGORITHM:
        reasons.append(f"code_identity_kind is {kind!r}, not {ALGORITHM}")
    declared = document.get("code_identity_sha256")
    if not isinstance(declared, str) or len(declared) != 64 or any(ch not in "0123456789abcdef" for ch in declared):
        reasons.append("code identity hash is missing or malformed")
        return reasons
    try:
        actual = snapshot_sha256(project_root)
    except CodeIdentityError as exc:
        reasons.append(f"source snapshot failed closed ({exc})")
        return reasons
    if declared != actual:
        reasons.append("source snapshot hash does not match the tree")
    return reasons


def write_manifest(project_root: Path, destination: Path | None = None) -> Path:
    """Write an observed snapshot manifest. This does not edit the protocol."""
    root = Path(project_root)
    path = Path(destination) if destination is not None else root / "reports" / "code_identity_manifest.json"
    records = collect_records(root)
    digest = _sha256(serialize_records(records))
    payload = {
        "algorithm": ALGORITHM,
        "warning": (
            "Observed source-tree snapshot only. This hash is not a frozen protocol "
            "identity. reports/protocol_freeze.yaml code_identity_sha256 stays null "
            "until a later explicit protocol freeze."
        ),
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "inclusion": list(INCLUSION_POLICY),
        "exclusion": list(EXCLUSION_POLICY),
        "files": records,
        "file_count": len(records),
        "snapshot_sha256": digest,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8", newline="\n")
    return path
