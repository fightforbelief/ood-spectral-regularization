"""Small-array tests for holdout prediction artifacts. No official file is opened."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from p19.gates import GateDecision
from p19.l2_runner import (
    PREDICTION_ARTIFACT_NAME,
    PREDICTION_PARTIAL_NAME,
    PredictionArtifactError,
    _incomplete_prediction_result,
    align_prediction_vectors,
    attach_prediction_artifact,
    prediction_identity_order,
    run_yearprediction_confirmatory,
    write_holdout_prediction_artifact,
)
from p19.protocol import YEAR_OFFICIAL_TARGET_ROWS, YEAR_SAMPLE_SIZE_LIST_ORIGINAL

PROJECT = Path(__file__).resolve().parents[1]
ESTIMATOR_CODE = {"min_norm": 0.0, "ridge": 1.0, "pcr": 2.0}


@pytest.fixture(autouse=True)
def block_real_gate_and_loaders(monkeypatch):
    import p19.data_loading as data_loading
    import p19.l2_runner as runner

    def refuse(*args, **kwargs):
        raise AssertionError("synthetic test called a real loader or the real gate")

    monkeypatch.setattr(runner, "evaluate_gate", refuse)
    for name in dir(data_loading):
        if name == "load_yearprediction_confirmatory_arrays":
            continue
        if name.startswith("load_") or name.startswith("iter_"):
            monkeypatch.setattr(data_loading, name, refuse)


def _vector(repetition: int, sample_size: int, estimator: str, length: int) -> np.ndarray:
    values = np.arange(length, dtype=np.float64)
    values[0] = repetition
    if length > 1:
        values[1] = sample_size
    if length > 2:
        values[2] = ESTIMATOR_CODE[estimator]
    return values


def _rows(order, length=4):
    rows = []
    for repetition, sample_size, estimator in order:
        vector = _vector(repetition, sample_size, estimator, length)
        rows.append(
            {
                "repetition": repetition,
                "sample_size": sample_size,
                "estimator": estimator,
                "mse": 1.5,
                "rmse": 1.2,
                "mae": 0.7,
                "r2": 0.3,
                "n_rows": length,
                "selected": 0.25,
                "training_target_mean": 4.0,
                "predictions": vector,
            }
        )
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _vector_sha256(vector: np.ndarray) -> str:
    payload = np.ascontiguousarray(vector, dtype=np.float64).tobytes()
    return hashlib.sha256(payload).hexdigest()


def test_production_identity_order_has_4500_rows():
    order = prediction_identity_order(100, YEAR_SAMPLE_SIZE_LIST_ORIGINAL)
    assert len(order) == 4500
    assert len(set(order)) == 4500
    assert order[0] == (0, 45, "min_norm")
    assert order[1] == (0, 45, "ridge")
    assert order[2] == (0, 45, "pcr")
    assert order[3] == (0, 68, "min_norm")
    assert order[-1] == (99, 900, "pcr")
    assert YEAR_OFFICIAL_TARGET_ROWS == 51630
    source = inspect.getsource(attach_prediction_artifact)
    assert "YEAR_OFFICIAL_TARGET_ROWS if production else None" in source


def test_small_artifact_shape_order_hashes_and_numpy_load(tmp_path):
    order = prediction_identity_order(2, (45, 68))
    rows = list(reversed(_rows(order, length=5)))
    vectors = align_prediction_vectors(rows, order, None)
    record, hashes = write_holdout_prediction_artifact(tmp_path, vectors)
    path = tmp_path / PREDICTION_ARTIFACT_NAME
    loaded = np.load(path)
    assert loaded.dtype == np.float64
    assert loaded.shape == (12, 5)
    assert record["format"] == "npy"
    assert record["dtype"] == "float64"
    assert record["shape"] == [12, 5]
    assert record["n_prediction_vectors"] == 12
    assert record["holdout_row_count"] == 5
    assert record["holdout_file_row_start"] == 463715
    assert record["holdout_file_row_end_exclusive"] == 515345
    assert record["sha256"] == _sha256(path)
    assert not (tmp_path / PREDICTION_PARTIAL_NAME).exists()
    located = {
        (row["repetition"], row["sample_size"], row["estimator"]): row
        for row in rows
    }
    for index, key in enumerate(order):
        assert np.array_equal(loaded[index], located[key]["predictions"])
        assert hashes[index] == _vector_sha256(loaded[index])
        assert located[key]["mse"] == 1.5
        assert located[key]["n_rows"] == 5


def test_wrong_length_nonfinite_duplicate_and_incomplete_grids_are_rejected(tmp_path):
    order = prediction_identity_order(1, (45, 68))
    rows = _rows(order, length=4)
    with pytest.raises(PredictionArtifactError):
        align_prediction_vectors(rows, order, 51630)
    broken = _rows(order, length=4)
    broken[1]["predictions"] = np.array([np.nan, 1.0, 2.0, 3.0], dtype=np.float64)
    with pytest.raises(PredictionArtifactError):
        align_prediction_vectors(broken, order, None)
    duplicated = _rows(order, length=4)
    duplicated.append(dict(duplicated[0]))
    with pytest.raises(PredictionArtifactError):
        align_prediction_vectors(duplicated, order, None)
    missing = _rows(order, length=4)[:-1]
    with pytest.raises(PredictionArtifactError):
        align_prediction_vectors(missing, order, None)
    assert not (tmp_path / PREDICTION_ARTIFACT_NAME).exists()
    assert not (tmp_path / PREDICTION_PARTIAL_NAME).exists()


def test_completed_and_partial_artifacts_are_not_overwritten(tmp_path):
    order = prediction_identity_order(1, (45,))
    vectors = align_prediction_vectors(_rows(order, length=4), order, None)
    record, _hashes = write_holdout_prediction_artifact(tmp_path, vectors)
    before = _sha256(tmp_path / PREDICTION_ARTIFACT_NAME)
    with pytest.raises(PredictionArtifactError):
        write_holdout_prediction_artifact(tmp_path, vectors)
    assert _sha256(tmp_path / PREDICTION_ARTIFACT_NAME) == before == record["sha256"]
    partial_dir = tmp_path / "partial"
    partial_dir.mkdir()
    partial = partial_dir / PREDICTION_PARTIAL_NAME
    partial.write_bytes(b"incomplete")
    with pytest.raises(PredictionArtifactError):
        write_holdout_prediction_artifact(partial_dir, vectors)
    assert partial.read_bytes() == b"incomplete"
    assert not (partial_dir / PREDICTION_ARTIFACT_NAME).exists()


def test_failed_rename_leaves_only_an_incomplete_partial(tmp_path, monkeypatch):
    order = prediction_identity_order(1, (45,))
    vectors = align_prediction_vectors(_rows(order, length=4), order, None)

    def fail_replace(*args, **kwargs):
        raise OSError("injected replace failure")

    monkeypatch.setattr("p19.l2_runner.os.replace", fail_replace)
    with pytest.raises(PredictionArtifactError):
        write_holdout_prediction_artifact(tmp_path, vectors)
    assert not (tmp_path / PREDICTION_ARTIFACT_NAME).exists()
    assert (tmp_path / PREDICTION_PARTIAL_NAME).is_file()


def _official_shapes():
    return {
        "X_source": np.zeros((463715, 90), dtype=np.float64),
        "y_source": np.zeros(463715, dtype=np.float64),
        "X_holdout": np.zeros((51630, 90), dtype=np.float64),
        "y_holdout": np.ones(51630, dtype=np.float64),
        "labels_loaded": True,
    }


def _executor(**kwargs):
    models = {}
    for name in ("min_norm", "ridge", "pcr"):
        models[name] = SimpleNamespace(
            repetition=kwargs["repetition"],
            sample_size=kwargs["sample_size"],
            code=ESTIMATOR_CODE[name],
        )
    return SimpleNamespace(
        repetition=kwargs["repetition"],
        sample_size=kwargs["sample_size"],
        preprocessor=object(),
        models=models,
    )


def _scorer(_preprocessor, model, _x_holdout, _y_holdout):
    vector = np.array(
        [float(model.repetition), float(model.sample_size), float(model.code)],
        dtype=np.float64,
    )
    return {
        "mse": 1.5,
        "rmse": 1.2,
        "mae": 0.7,
        "r2": 0.3,
        "n_rows": 3,
        "selected": 0.25,
        "training_target_mean": 4.0,
        "predictions": vector,
    }


def test_synthetic_runner_persists_a_small_artifact_without_a_confirmatory_result(tmp_path):
    result = run_yearprediction_confirmatory(
        PROJECT,
        gate_check=lambda root, action: GateDecision(approved=True, reasons=[], protocol_sha256="synthetic"),
        load_arrays=lambda path, decision: _official_shapes(),
        execute_cell=_executor,
        evaluate=_scorer,
        output_dir=tmp_path / "synthetic",
    )
    assert result["status"] == "SYNTHETIC_COMPLETE"
    assert result["confirmatory_l2"] == "SYNTHETIC_COMPLETE"
    assert result["confirmatory_result"] is False
    assert result["synthetic"] is True
    assert result["official_data_opened_by_this_call"] is False
    assert result["prediction_artifact_satisfied"] is True
    artifact = result["prediction_artifact"]
    assert artifact["shape"] == [4500, 3]
    assert artifact["holdout_row_count"] == 3
    path = Path(result["manifest"]).parent / artifact["relative_path"]
    loaded = np.load(path)
    assert loaded.shape == (4500, 3)
    assert loaded.dtype == np.float64
    assert artifact["sha256"] == _sha256(path)
    assert path.stat().st_size < 1_000_000
    by_identity = {
        (row["repetition"], row["sample_size"], row["estimator"]): row
        for row in result["holdout_metrics"]
    }
    assert len(by_identity) == 4500
    for row in result["holdout_metrics"]:
        assert row["mse"] == 1.5
        assert row["rmse"] == 1.2
        assert row["mae"] == 0.7
        assert row["r2"] == 0.3
        assert row["n_rows"] == 3
        assert row["selected"] == 0.25
        assert row["training_target_mean"] == 4.0
        assert "predictions" not in row
        index = row["prediction_artifact_row"]
        assert _vector_sha256(loaded[index]) == row["prediction_sha256"]
    last = by_identity[(99, 900, "pcr")]
    assert np.array_equal(loaded[last["prediction_artifact_row"]], np.array([99.0, 900.0, 2.0]))
    assert loaded[0, 0] == 0.0
    assert loaded[0, 1] == 45.0
    assert loaded[0, 2] == 0.0


def test_runner_rename_failure_cannot_be_a_complete_manifest(tmp_path, monkeypatch):
    def fail_replace(*args, **kwargs):
        raise OSError("injected replace failure")

    monkeypatch.setattr("p19.l2_runner.os.replace", fail_replace)
    result = run_yearprediction_confirmatory(
        PROJECT,
        gate_check=lambda root, action: GateDecision(approved=True, reasons=[], protocol_sha256="synthetic"),
        load_arrays=lambda path, decision: _official_shapes(),
        execute_cell=_executor,
        evaluate=_scorer,
        output_dir=tmp_path / "failed-artifact",
    )
    assert result["status"] == "INCOMPLETE"
    assert result["confirmatory_l2"] == "INCOMPLETE_PREDICTION_ARTIFACT"
    assert result["confirmatory_result"] is False
    assert result["synthetic"] is True
    assert result["prediction_artifact_satisfied"] is False
    assert "prediction_artifact" not in result
    assert result["incomplete_prediction_artifact"] == PREDICTION_PARTIAL_NAME
    assert result["holdout_metrics"][0]["mse"] == 1.5
    assert result["status"] != "COMPLETE"
    assert result["confirmatory_result"] is not True
    error = result["prediction_artifact_error"]
    assert error["operation"] == "atomic rename"
    assert error["original_type"] == "OSError"
    assert "OSError" in error["message"]
    assert "injected replace failure" in error["message"]
    directory = Path(result["manifest"]).parent
    assert not (directory / PREDICTION_ARTIFACT_NAME).exists()
    assert (directory / PREDICTION_PARTIAL_NAME).is_file()


def _tiny_vectors():
    order = prediction_identity_order(1, (45,))
    return align_prediction_vectors(_rows(order, length=4), order, None)


def _assert_filesystem_error(error: PredictionArtifactError, operation: str, detail: str) -> None:
    assert error.operation == operation
    assert error.original_type == "OSError"
    assert error.path
    assert operation in str(error)
    assert "OSError" in str(error)
    assert detail in str(error)


class _RowFailure:
    def __setitem__(self, key, value):
        raise OSError("injected row write failure")

    def flush(self):
        raise AssertionError("flush should not run after a row-write failure")


class _FlushFailure:
    def __setitem__(self, key, value):
        return None

    def flush(self):
        raise OSError("injected flush failure")


class _CloseFailure:
    closed = False

    def __setitem__(self, key, value):
        return None

    def flush(self):
        return None

    def close(self):
        raise OSError("injected close failure")


def test_open_memmap_oserror_becomes_prediction_artifact_error(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("injected open_memmap failure")

    monkeypatch.setattr("p19.l2_runner.open_memmap", boom)
    with pytest.raises(PredictionArtifactError) as caught:
        write_holdout_prediction_artifact(tmp_path, _tiny_vectors())
    _assert_filesystem_error(caught.value, "open_memmap", "injected open_memmap failure")
    assert not (tmp_path / PREDICTION_ARTIFACT_NAME).exists()


def test_row_write_oserror_becomes_prediction_artifact_error(tmp_path, monkeypatch):
    monkeypatch.setattr("p19.l2_runner.open_memmap", lambda *args, **kwargs: _RowFailure())
    with pytest.raises(PredictionArtifactError) as caught:
        write_holdout_prediction_artifact(tmp_path, _tiny_vectors())
    _assert_filesystem_error(caught.value, "memory-mapped row write", "injected row write failure")
    assert not (tmp_path / PREDICTION_ARTIFACT_NAME).exists()


def test_flush_oserror_becomes_prediction_artifact_error(tmp_path, monkeypatch):
    monkeypatch.setattr("p19.l2_runner.open_memmap", lambda *args, **kwargs: _FlushFailure())
    with pytest.raises(PredictionArtifactError) as caught:
        write_holdout_prediction_artifact(tmp_path, _tiny_vectors())
    _assert_filesystem_error(caught.value, "memmap flush", "injected flush failure")
    assert not (tmp_path / PREDICTION_ARTIFACT_NAME).exists()


def test_memmap_close_oserror_becomes_prediction_artifact_error(tmp_path, monkeypatch):
    mapped = _CloseFailure()
    mapped._mmap = mapped
    monkeypatch.setattr("p19.l2_runner.open_memmap", lambda *args, **kwargs: mapped)
    with pytest.raises(PredictionArtifactError) as caught:
        write_holdout_prediction_artifact(tmp_path, _tiny_vectors())
    _assert_filesystem_error(caught.value, "memmap close", "injected close failure")
    assert not (tmp_path / PREDICTION_ARTIFACT_NAME).exists()


def test_sha256_oserror_becomes_prediction_artifact_error(tmp_path, monkeypatch):
    def boom(path):
        raise OSError("injected hash failure")

    monkeypatch.setattr("p19.l2_runner._sha256_file", boom)
    with pytest.raises(PredictionArtifactError) as caught:
        write_holdout_prediction_artifact(tmp_path, _tiny_vectors())
    _assert_filesystem_error(caught.value, "full-file SHA-256", "injected hash failure")
    assert not (tmp_path / PREDICTION_ARTIFACT_NAME).exists()
    assert (tmp_path / PREDICTION_PARTIAL_NAME).is_file()


def test_atomic_rename_oserror_names_the_operation(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("injected replace failure")

    monkeypatch.setattr("p19.l2_runner.os.replace", boom)
    with pytest.raises(PredictionArtifactError) as caught:
        write_holdout_prediction_artifact(tmp_path, _tiny_vectors())
    _assert_filesystem_error(caught.value, "atomic rename", "injected replace failure")
    assert not (tmp_path / PREDICTION_ARTIFACT_NAME).exists()
    assert (tmp_path / PREDICTION_PARTIAL_NAME).is_file()


def test_output_directory_oserror_becomes_prediction_artifact_error(tmp_path, monkeypatch):
    def boom(self, *args, **kwargs):
        raise OSError("injected directory failure")

    monkeypatch.setattr(Path, "mkdir", boom)
    with pytest.raises(PredictionArtifactError) as caught:
        write_holdout_prediction_artifact(tmp_path / "missing", _tiny_vectors())
    _assert_filesystem_error(caught.value, "output directory creation", "injected directory failure")
    assert not (tmp_path / "missing").exists()


def test_programming_errors_are_not_relabeled_as_filesystem_failures(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise ValueError("not a filesystem failure")

    monkeypatch.setattr("p19.l2_runner.open_memmap", boom)
    with pytest.raises(ValueError, match="not a filesystem failure"):
        write_holdout_prediction_artifact(tmp_path, _tiny_vectors())


def test_runner_open_memmap_failure_stays_incomplete(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("injected open_memmap failure")

    monkeypatch.setattr("p19.l2_runner.open_memmap", boom)
    result = run_yearprediction_confirmatory(
        PROJECT,
        gate_check=lambda root, action: GateDecision(approved=True, reasons=[], protocol_sha256="synthetic"),
        load_arrays=lambda path, decision: _official_shapes(),
        execute_cell=_executor,
        evaluate=_scorer,
        output_dir=tmp_path / "open-failed",
    )
    assert result["status"] == "INCOMPLETE"
    assert result["status"] != "COMPLETE"
    assert result["confirmatory_l2"] == "INCOMPLETE_PREDICTION_ARTIFACT"
    assert result["confirmatory_result"] is False
    assert result["synthetic"] is True
    assert result["prediction_artifact_satisfied"] is False
    assert "prediction_artifact" not in result
    assert result["prediction_artifact_error"]["operation"] == "open_memmap"
    assert result["prediction_artifact_error"]["original_type"] == "OSError"
    assert result["holdout_metrics"][0]["mse"] == 1.5
    assert "predictions" not in result["holdout_metrics"][0]
    directory = Path(result["manifest"]).parent
    assert not (directory / PREDICTION_ARTIFACT_NAME).exists()


def test_unwritable_failure_manifest_is_not_claimed(tmp_path, monkeypatch, capsys):
    def boom(*args, **kwargs):
        raise OSError("injected manifest failure")

    monkeypatch.setattr("p19.l2_runner._write_manifest", boom)
    result = _incomplete_prediction_result(
        {"status": "INCOMPLETE", "synthetic": True},
        tmp_path,
        [{"mse": 1.5, "repetition": 0, "sample_size": 45, "estimator": "min_norm"}],
        PredictionArtifactError(
            "open_memmap failed for partial.npy: OSError: injected open_memmap failure",
            operation="open_memmap",
            path=str(tmp_path / PREDICTION_PARTIAL_NAME),
            original_type="OSError",
        ),
    )
    assert result["status"] == "INCOMPLETE"
    assert result["status"] != "COMPLETE"
    assert result["confirmatory_l2"] == "INCOMPLETE_PREDICTION_ARTIFACT"
    assert result["confirmatory_result"] is False
    assert result["prediction_artifact_satisfied"] is False
    assert "manifest" not in result
    assert result["manifest_write_failed"] is True
    assert result["prediction_artifact_error"]["operation"] == "open_memmap"
    assert result["holdout_metrics"][0]["mse"] == 1.5
    captured = capsys.readouterr()
    assert "prediction artifact failure manifest was not written" in captured.err
    assert "OSError" in captured.err
    assert "injected manifest failure" in captured.err
    assert not (tmp_path / "l2_run_manifest.json").exists()


def test_cli_returns_nonzero_without_a_confirmatory_result(monkeypatch):
    monkeypatch.setattr(
        "p19.cli.run_confirmatory",
        lambda root: {
            "confirmatory_l2": "INCOMPLETE_PREDICTION_ARTIFACT",
            "confirmatory_result": False,
            "target_labels_loaded": True,
        },
    )
    from p19.cli import main

    assert main(["run-l2", "--confirmatory"]) == 2
