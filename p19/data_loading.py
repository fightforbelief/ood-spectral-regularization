"""Feature parsers, plus a confirmatory loader that runs only after an approved gate decision.

X-only parsers never convert the official target token. The confirmatory loader
parses labels only when the caller already holds an approved ``GateDecision``.
It does not call ``evaluate_gate`` again.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from p19.gates import GateDecision, GateRefusal, require_gate
from p19.protocol import (
    SUPER_TARGET_COLUMN,
    YEAR_OFFICIAL_ROWS,
    YEAR_OFFICIAL_SOURCE_ROWS,
    YEAR_OFFICIAL_TARGET_ROWS,
    YEAR_SELECTION_PREDICTOR_COUNT,
    YEAR_TARGET_INDEX,
)


class DataLoadError(Exception):
    pass


def _feature_array(feature_text: str) -> np.ndarray:
    if feature_text == "":
        return np.zeros(0, dtype=np.float64)
    try:
        values = np.fromstring(feature_text, sep=",", dtype=np.float64)
    except ValueError:
        # A nonnumeric feature token must not be coerced. Length 1 marks the row unparsed.
        return np.array([np.nan], dtype=np.float64)
    return np.asarray(values, dtype=np.float64)


def iter_year_feature_rows(path: Path):
    """Yield (row_index, total_columns, feature_vector) without parsing column 0."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row_index, line in enumerate(handle):
            text = line.strip("\r\n")
            if text.startswith("\ufeff"):
                text = text.lstrip("\ufeff")
            if text == "":
                yield row_index, 0, np.zeros(0, dtype=np.float64)
                continue
            comma = text.find(",")
            if comma < 0:
                total_columns = 1
                features = np.zeros(0, dtype=np.float64)
            else:
                total_columns = text.count(",") + 1
                features = _feature_array(text[comma + 1 :])
            # text[:comma] is the target token and is intentionally discarded.
            yield row_index, total_columns, features


def load_yearprediction_x_only(
    path: Path,
    source_rows: int = YEAR_OFFICIAL_SOURCE_ROWS,
    target_rows: int = YEAR_OFFICIAL_TARGET_ROWS,
    target_index: int = YEAR_TARGET_INDEX,
) -> dict:
    """Load source and official-holdout features. The target token is ignored on every row."""
    if target_index != 0:
        raise DataLoadError(
            "this parser only implements the official first-column target position"
        )
    source_rows_list: list[np.ndarray] = []
    target_rows_list: list[np.ndarray] = []
    column_counts: set[int] = set()
    n_rows = 0
    for row_index, total_columns, features in iter_year_feature_rows(path):
        n_rows += 1
        column_counts.add(total_columns)
        if row_index < source_rows:
            source_rows_list.append(features)
        else:
            target_rows_list.append(features)
    if n_rows != source_rows + target_rows:
        raise DataLoadError(
            f"row count {n_rows} does not equal requested source+target "
            f"{source_rows}+{target_rows}"
        )
    if len(source_rows_list) != source_rows or len(target_rows_list) != target_rows:
        raise DataLoadError("source/target split counts do not match the request")
    predictor_counts = {int(row.shape[0]) for row in source_rows_list + target_rows_list}
    return {
        "X_source": np.vstack(source_rows_list) if source_rows_list else np.zeros((0, 0)),
        "X_target": np.vstack(target_rows_list) if target_rows_list else np.zeros((0, 0)),
        "n_rows": n_rows,
        "column_counts": sorted(column_counts),
        "target_index": target_index,
        "predictor_count": None if len(predictor_counts) != 1 else next(iter(predictor_counts)),
        "labels_loaded": False,
    }


def _year_target_token(line: str) -> str | None:
    text = line.strip("\r\n")
    if text.startswith("\ufeff"):
        text = text.lstrip("\ufeff")
    if text == "":
        return None
    comma = text.find(",")
    return text if comma < 0 else text[:comma]


def _parse_year_target_column(path: Path, max_rows: int | None) -> np.ndarray:
    labels: list[float] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for line in handle:
            if max_rows is not None and len(labels) >= max_rows:
                break
            token = _year_target_token(line)
            if token is None:
                continue
            labels.append(float(token))
    if max_rows is not None and len(labels) != max_rows:
        raise DataLoadError(f"expected {max_rows} source labels, parsed {len(labels)}")
    return np.asarray(labels, dtype=np.float64)


def load_yearprediction_labeled_source(path: Path, project_root: Path, source_rows: int) -> dict:
    """Source labels. The gate is required before the file is opened.

    Only the first `source_rows` target tokens are parsed. Later rows, including
    an official test partition in the same file, are not read.
    """
    require_gate(project_root, "load_yearprediction_labeled_source")
    labels = _parse_year_target_column(path, max_rows=int(source_rows))
    return {"y": labels, "n": int(labels.shape[0]), "labels_loaded": True}


def load_yearprediction_confirmatory_arrays(path: Path, decision: GateDecision) -> dict:
    """Source and holdout features and labels. The file is opened only after approval.

    ``decision`` must be the ``GateDecision`` already returned by ``evaluate_gate``.
    This function does not evaluate the gate a second time and does not treat any
    other object as approval.
    """
    if not isinstance(decision, GateDecision) or decision.approved is not True:
        raise GateRefusal(["YearPredictionMSD confirmatory arrays require an approved gate decision"])
    features = load_yearprediction_x_only(path)
    labels = _parse_year_target_column(path, max_rows=None)
    if int(labels.shape[0]) != YEAR_OFFICIAL_ROWS:
        raise DataLoadError(
            f"parsed {int(labels.shape[0])} target labels, not {YEAR_OFFICIAL_ROWS}"
        )
    y_source = labels[:YEAR_OFFICIAL_SOURCE_ROWS]
    y_holdout = labels[YEAR_OFFICIAL_SOURCE_ROWS:]
    return {
        "X_source": features["X_source"],
        "y_source": y_source,
        "X_holdout": features["X_target"],
        "y_holdout": y_holdout,
        "predictor_count": features["predictor_count"],
        "expected_predictor_count": YEAR_SELECTION_PREDICTOR_COUNT,
        "labels_loaded": True,
    }


def load_yearprediction_target_labels(path: Path, project_root: Path) -> np.ndarray:
    """Target tokens from column 0. The gate is required before the file is opened."""
    require_gate(project_root, "load_yearprediction_target_labels")
    return _parse_year_target_column(path, max_rows=None)


def _unquote_cell(cell: str) -> str:
    """Strip a matching wrapping quote. UCI superconductivity headers are quoted."""
    text = cell.strip().lstrip("\ufeff")
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        return text[1:-1]
    return text


def _split_csv_line(text: str) -> list[str]:
    # Official files are simple numeric CSV without quoted commas in feature fields.
    # The formula field is read as a whole remainder only when it is the last column.
    return [_unquote_cell(part) for part in text.split(",")]


def load_superconductivity_x_only(path: Path, target_column: str = SUPER_TARGET_COLUMN) -> dict:
    """Load feature columns. The named target column is never converted to a number."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        header_line = handle.readline()
        if header_line == "":
            raise DataLoadError("superconductivity file has no header")
        header = _split_csv_line(header_line.strip("\r\n"))
        if header.count(target_column) != 1:
            raise DataLoadError(
                f"expected exactly one {target_column!r} column, found {header.count(target_column)}"
            )
        target_pos = header.index(target_column)
        feature_names = [name for i, name in enumerate(header) if i != target_pos]
        rows: list[np.ndarray] = []
        column_counts: set[int] = set()
        for line in handle:
            text = line.strip("\r\n")
            if text == "":
                continue
            parts = _split_csv_line(text)
            column_counts.add(len(parts))
            if len(parts) != len(header):
                raise DataLoadError("superconductivity row width does not match the header")
            # The target cell is dropped before numeric parsing. It is never passed to float().
            feature_text = ",".join(part for i, part in enumerate(parts) if i != target_pos)
            values = _feature_array(feature_text)
            if values.shape[0] != len(feature_names):
                raise DataLoadError("a feature token could not be parsed as a float")
            rows.append(values)
    matrix = np.vstack(rows) if rows else np.zeros((0, len(feature_names)), dtype=np.float64)
    return {
        "X": matrix,
        "feature_names": feature_names,
        "target_column": target_column,
        "target_index": target_pos,
        "total_columns": len(header),
        "predictor_count": len(feature_names),
        "n_rows": int(matrix.shape[0]),
        "column_counts": sorted(column_counts | {len(header)}),
        "labels_loaded": False,
    }


def load_superconductivity_labeled_data(path: Path, project_root: Path) -> dict:
    """Parse only the named target column, and only after the gate opens."""
    require_gate(project_root, "load_superconductivity_labeled_data")
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        header = _split_csv_line(handle.readline().strip("\r\n"))
        if header.count(SUPER_TARGET_COLUMN) != 1:
            raise DataLoadError(f"expected one {SUPER_TARGET_COLUMN!r} column")
        target_pos = header.index(SUPER_TARGET_COLUMN)
        labels: list[float] = []
        for line in handle:
            text = line.strip("\r\n")
            if text == "":
                continue
            parts = _split_csv_line(text)
            if len(parts) != len(header):
                raise DataLoadError("labeled superconductivity row width does not match the header")
            labels.append(float(parts[target_pos]))
    return {
        "y": np.asarray(labels, dtype=np.float64),
        "target_column": SUPER_TARGET_COLUMN,
        "target_index": target_pos,
        "labels_loaded": True,
    }


def load_formula_column(path: Path, formula_column: str, target_column: str = SUPER_TARGET_COLUMN) -> dict:
    """Formula identifiers only. The critical-temperature column is not parsed."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        header = _split_csv_line(handle.readline().strip("\r\n"))
        if formula_column not in header:
            raise DataLoadError(f"formula column {formula_column!r} is not in the header")
        if target_column not in header:
            raise DataLoadError(f"target column {target_column!r} is not in the header")
        formula_pos = header.index(formula_column)
        target_pos = header.index(target_column)
        formulas: list[str] = []
        for line in handle:
            text = line.strip("\r\n")
            if text == "":
                continue
            parts = _split_csv_line(text)
            if len(parts) != len(header):
                # Chemical formulas do not contain commas in this UCI file.
                raise DataLoadError("formula file row width does not match the header")
            formulas.append(parts[formula_pos])
            if parts[target_pos] == "":
                raise DataLoadError("formula file is missing the ignored target cell")
    return {
        "formulas": formulas,
        "n_rows": len(formulas),
        "formula_column": formula_column,
        "target_index_ignored": target_pos,
        "labels_loaded": False,
    }
