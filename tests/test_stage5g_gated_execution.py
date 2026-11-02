"""Stage 5G synthetic tests. The real gate and the official files are not used."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from p19.code_identity import snapshot_sha256
from p19.estimators import EstimatorError
from p19.gates import GateDecision, GateRefusal
from p19.l2_evaluation import run_confirmatory
from p19.l2_runner import (
    run_yearprediction_confirmatory,
    uncertainty_from_scored_rows,
    validate_yearprediction_arrays,
)
from p19.l2_sampling import rows_for_prefix
from p19.protocol import YEAR_SAMPLE_SIZE_LIST_ORIGINAL, load_yaml
from p19.superconductivity_split import (
    STORED_SPLIT_SEED,
    SuperconductivityProtocolError,
    nested_training_prefixes,
    split_superconductivity_groups,
)
from p19.uncertainty import UncertaintyNotFrozen

PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def year_arrays():
    source = np.zeros((463715, 90), dtype=np.float64)
    source[:, 0] = np.arange(463715, dtype=np.float64)
    holdout = np.zeros((51630, 90), dtype=np.float64)
    return {
        "X_source": source,
        "y_source": np.arange(463715, dtype=np.float64),
        "X_holdout": holdout,
        "y_holdout": np.ones(51630, dtype=np.float64),
        "labels_loaded": True,
    }


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


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _approved_decision() -> GateDecision:
    return GateDecision(
        approved=True,
        reasons=[],
        protocol_sha256="synthetic-injected-gate",
        recorded_sha256="synthetic-injected-gate",
    )


def _cell(**kwargs):
    return SimpleNamespace(
        repetition=kwargs["repetition"],
        sample_size=kwargs["sample_size"],
        preprocessor=object(),
        models={"min_norm": object(), "ridge": object(), "pcr": object()},
    )


def test_production_call_checks_the_gate_before_the_loader():
    runner_source = inspect.getsource(run_yearprediction_confirmatory)
    confirmatory_source = inspect.getsource(run_confirmatory)
    assert "run_yearprediction_confirmatory" in confirmatory_source
    assert runner_source.index("if not approved") < runner_source.index("load_yearprediction_confirmatory_arrays")
    cli_source = (PROJECT / "p19" / "cli.py").read_text(encoding="utf-8")
    assert "result = run_confirmatory(root)" in cli_source
    assert 'result.get("confirmatory_result") is not True' in cli_source


def test_gate_refusal_calls_no_loader_executor_or_scorer(tmp_path):
    audit = PROJECT / "reports" / "gate_audit.jsonl"
    before = _file_sha256(audit)
    events = []

    def refuse_gate(root, action):
        events.append(("gate", action))
        return GateDecision(approved=False, reasons=["synthetic refusal"])

    def loader(*args, **kwargs):
        events.append(("load",))
        raise AssertionError("loader called after refusal")

    def executor(**kwargs):
        events.append(("cell",))
        raise AssertionError("executor called after refusal")

    def scorer(*args, **kwargs):
        events.append(("score",))
        raise AssertionError("scorer called after refusal")

    output = tmp_path / "refusal"
    with pytest.raises(GateRefusal):
        run_yearprediction_confirmatory(
            PROJECT,
            gate_check=refuse_gate,
            load_arrays=loader,
            execute_cell=executor,
            evaluate=scorer,
            output_dir=output,
        )
    assert events == [("gate", "run-l2-confirmatory")]
    assert not output.exists()
    assert _file_sha256(audit) == before


def test_unapproved_decision_does_not_open_the_data_file():
    from p19.data_loading import load_yearprediction_confirmatory_arrays

    with pytest.raises(GateRefusal):
        load_yearprediction_confirmatory_arrays(
            Path("/no/such/YearPredictionMSD.txt"),
            GateDecision(approved=False, reasons=["synthetic refusal"]),
        )


def test_wrong_shape_stops_before_the_source_phase(tmp_path):
    events = []

    def executor(**kwargs):
        events.append("cell")
        raise AssertionError("source phase started on invalid arrays")

    result = run_yearprediction_confirmatory(
        PROJECT,
        gate_check=lambda root, action: _approved_decision(),
        load_arrays=lambda path, decision: {
            "X_source": np.zeros((3, 2)),
            "y_source": np.zeros(3),
            "X_holdout": np.zeros((2, 2)),
            "y_holdout": np.zeros(2),
            "labels_loaded": True,
        },
        execute_cell=executor,
        evaluate=lambda *args, **kwargs: events.append("score"),
        output_dir=tmp_path / "bad-shape",
    )
    assert events == []
    assert result["status"] == "INCOMPLETE"
    assert result["confirmatory_result"] is False
    assert result["holdout_scored"] is False
    assert result["n_source_cells"] == 0
    manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    assert manifest["confirmatory_result"] is False
    assert manifest["holdout_scored"] is False


def test_synthetic_approval_scores_only_after_the_full_source_grid(year_arrays, tmp_path):
    audit = PROJECT / "reports" / "gate_audit.jsonl"
    before = _file_sha256(audit)
    events = []

    def gate(root, action):
        events.append(("gate", action))
        return _approved_decision()

    def loader(path, decision):
        events.append(("load", Path(path).name))
        assert decision.approved is True
        assert Path(path).name == "YearPredictionMSD.txt"
        return year_arrays

    def executor(**kwargs):
        indices = np.asarray(kwargs["prefix_indices"])
        selected_x, selected_y = rows_for_prefix(
            year_arrays["X_source"], year_arrays["y_source"], indices
        )
        assert np.array_equal(selected_x[:, 0], indices.astype(np.float64))
        assert selected_x.shape[1] == 90
        events.append(("cell", int(kwargs["repetition"]), int(kwargs["sample_size"])))
        return _cell(**kwargs)

    def scorer(preprocessor, model, x_holdout, y_holdout):
        events.append(("score",))
        return {"mse": 1.0, "rmse": 1.0, "mae": 1.0, "r2": 0.0, "n_rows": int(y_holdout.shape[0])}

    result = run_yearprediction_confirmatory(
        PROJECT,
        gate_check=gate,
        load_arrays=loader,
        execute_cell=executor,
        evaluate=scorer,
        output_dir=tmp_path / "synthetic-complete",
    )
    kinds = [event[0] for event in events]
    assert kinds[0] == "gate"
    assert kinds[1] == "load"
    cell_at = [index for index, kind in enumerate(kinds) if kind == "cell"]
    score_at = [index for index, kind in enumerate(kinds) if kind == "score"]
    assert len(cell_at) == 100 * 15
    assert len(score_at) == 100 * 15 * 3
    assert max(cell_at) < min(score_at)
    assert result["status"] == "SYNTHETIC_COMPLETE"
    assert result["confirmatory_result"] is False
    assert result["synthetic"] is True
    assert result["official_data_opened_by_this_call"] is False
    protocol = load_yaml(PROJECT / "reports" / "protocol_freeze.yaml")
    protocol_path = PROJECT / "reports" / "protocol_freeze.yaml"
    assert result["protocol_status"] == "FROZEN"
    assert protocol["status"] == "FROZEN"
    assert isinstance(protocol["confirmatory_l2_authorized"], bool)
    assert result["confirmatory_l2_authorized"] is protocol["confirmatory_l2_authorized"]
    assert protocol["confirmatory_scope"] == "YEARPREDICTION_MSD_ONLY"
    assert protocol["superconductivity_confirmatory"] == "DEFERRED_MISSING_NESTED_PREFIX_SEEDS"
    identity = result["code_identity_sha256"]
    assert isinstance(identity, str) and len(identity) == 64 and identity == identity.lower()
    int(identity, 16)
    assert identity == protocol["code_identity_sha256"]
    assert identity == snapshot_sha256(PROJECT)
    assert result["protocol_file_sha256"] == hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    assert result["source_cells"][0]["sampling_seed"] == 100000037
    assert result["source_cells"][-1]["sampling_seed"] == 100000031
    assert result["uncertainty"]["pairing"] == "repetition_identifier"
    manifest_text = Path(result["manifest"]).read_text(encoding="utf-8")
    assert "synthetic-injected-gate" in manifest_text
    assert '"confirmatory_result": false' in manifest_text
    assert _file_sha256(audit) == before


def test_failed_source_phase_never_scores(year_arrays, tmp_path):
    events = []

    def executor(**kwargs):
        events.append("cell")
        raise EstimatorError("injected source failure")

    result = run_yearprediction_confirmatory(
        PROJECT,
        gate_check=lambda root, action: _approved_decision(),
        load_arrays=lambda path, decision: year_arrays,
        execute_cell=executor,
        evaluate=lambda *args, **kwargs: events.append("score"),
        output_dir=tmp_path / "failed-source",
    )
    assert events == ["cell"]
    assert result["status"] == "INCOMPLETE"
    assert result["holdout_scored"] is False
    assert result["confirmatory_result"] is False
    assert result["n_source_cells"] == 0
    assert result["source_failure"]["error"] == "EstimatorError"
    manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    assert manifest["status"] == "INCOMPLETE"
    assert "holdout_metrics" not in manifest
    assert "uncertainty" not in manifest


def test_reordered_scores_pair_by_repetition_identifier():
    rows = []
    for repetition in range(100):
        for sample_size in YEAR_SAMPLE_SIZE_LIST_ORIGINAL:
            for estimator, scale in (("min_norm", 1.0), ("ridge", 2.0), ("pcr", 5.0)):
                rows.append(
                    {
                        "repetition": repetition,
                        "sample_size": int(sample_size),
                        "estimator": estimator,
                        "mse": scale * repetition + sample_size,
                    }
                )
    order = np.random.default_rng(7).permutation(len(rows))
    shuffled = [rows[int(index)] for index in order]
    summary = uncertainty_from_scored_rows(shuffled)
    aligned = 3.0 * np.arange(100, dtype=np.float64)
    observed = summary["paired_by_sample_size"]["45"]["pcr_minus_ridge"]["sample_sd"]
    assert observed == pytest.approx(float(np.std(aligned, ddof=1)))
    positional = np.array([row["mse"] for row in shuffled if row["estimator"] == "pcr" and row["sample_size"] == 45])
    positional_ridge = np.array(
        [row["mse"] for row in shuffled if row["estimator"] == "ridge" and row["sample_size"] == 45]
    )
    assert observed != pytest.approx(float(np.std(positional - positional_ridge, ddof=1)))
    broken = list(shuffled)
    broken.append(dict(broken[0]))
    with pytest.raises(UncertaintyNotFrozen):
        uncertainty_from_scored_rows(broken)
    missing = [row for row in rows if not (row["repetition"] == 4 and row["sample_size"] == 45 and row["estimator"] == "ridge")]
    with pytest.raises(UncertaintyNotFrozen):
        uncertainty_from_scored_rows(missing)


def test_shape_validator_rejects_the_wrong_official_dimensions():
    with pytest.raises(Exception):
        validate_yearprediction_arrays(
            {
                "X_source": np.zeros((4, 90)),
                "y_source": np.zeros(4),
                "X_holdout": np.zeros((2, 90)),
                "y_holdout": np.ones(2),
            }
        )


def test_superconductivity_groups_stay_together_and_nested_prefixes_are_refused():
    base = np.array(
        [
            [0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0],
            [9.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 3.0],
        ],
        dtype=np.float64,
    )
    counts = [3, 1, 2, 1, 4, 1]
    rows = [vector.copy() for vector, count in zip(base, counts) for _ in range(count)]
    features = np.vstack(rows)
    protocol_text = load_yaml(PROJECT / "reports" / "protocol_freeze.yaml")["superconductivity_split"]
    parameters = inspect.signature(split_superconductivity_groups).parameters
    assert "protocol_text" in parameters
    assert parameters["split_seed"].default == STORED_SPLIT_SEED
    assert "y" not in parameters
    first = split_superconductivity_groups(features, protocol_text=protocol_text)
    second = split_superconductivity_groups(features, protocol_text=protocol_text)
    assert first["split_seed"] == 732111676
    assert first["n_groups"] == 6
    assert first["n_train_groups"] == 4
    assert first["n_test_groups"] == 2
    assert np.array_equal(first["train_rows"], second["train_rows"])
    assert np.array_equal(first["test_rows"], second["test_rows"])
    train = set(int(index) for index in first["train_rows"])
    test = set(int(index) for index in first["test_rows"])
    assert train.isdisjoint(test)
    assert train | test == set(range(features.shape[0]))
    for group in first["groups"]:
        members = set(int(index) for index in group)
        assert members <= train or members <= test
        assert len({features[int(index)].tobytes() for index in group}) == 1
    train_keys = {features[index].tobytes() for index in train}
    test_keys = {features[index].tobytes() for index in test}
    assert train_keys.isdisjoint(test_keys)
    assert any(group.size > 1 for group in first["groups"])
    with pytest.raises(SuperconductivityProtocolError):
        split_superconductivity_groups(features, protocol_text=protocol_text, split_seed=810000001)
    with pytest.raises(SuperconductivityProtocolError):
        split_superconductivity_groups(features, protocol_text="use a row IID split")
    with pytest.raises(SuperconductivityProtocolError):
        nested_training_prefixes([100000037, 100000041])
    module_source = inspect.getsource(split_superconductivity_groups)
    assert "17000003" not in module_source
    assert "random_seed_list" not in module_source
