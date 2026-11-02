"""Markdown and JSON writers for audit reports. No target values are accepted."""

from __future__ import annotations

import json
from pathlib import Path


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _lines(title: str, mapping: dict) -> list[str]:
    rows = [f"# {title}", ""]
    for key, value in mapping.items():
        rows.append(f"- **{key}:** `{value}`")
    rows.append("")
    return rows
