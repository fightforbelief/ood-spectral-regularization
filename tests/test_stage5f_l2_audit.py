"""Stage 5F synthetic audit tests. No official data and no confirmation gate."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from p19.estimators import EstimatorError, pcr_component_count
from p19.l2_pipeline import (
    HoldoutScoringBlocked,
    RunAborted,
    declared_source_cell_keys,
    evaluate_holdout,
    fit_training_prefix,
    refit_full_prefix,
    run_declared_source_phase,
    score_completed_source_phase,
)
from p19.l2_sampling import (
    ProtocolValidationError,
    longest_prefix_folds,
    restrict_nested_folds,
    rows_for_prefix,
    source_permutation,
)
from p19.metrics import metric_bundle
from p19.protocol import (
    DRAFT_PCR_ROUNDING,
    DRAFT_ZERO_VARIANCE,
    PCR_FRACTION_GRID,
    YEAR_SAMPLE_SIZE_LIST_ORIGINAL,
    load_yaml,
)
from p19.tuning import assign_source_folds, tune_pcr, tune_ridge
from p19.uncertainty import (
    ADJACENT_SAMPLE_SIZE_PAIRS,
    APPROVED_TRAINING_SUBSET_PROCEDURE,
    UncertaintyNotFrozen,
    adjacent_sample_size_differences,
    paired_estimator_loss_differences,
    target_row_uncertainty,
    training_subset_variability,
)

PROJECT_ROOT_NAME = "p19-ood-spectral-regularization"
EXPECTED_FOLD_COUNTS = {
    45: (8, 9, 5, 7, 16),
    68: (12, 10, 10, 13, 23),
    81: (16, 13, 11, 17, 24),
    86: (17, 14, 12, 19, 24),
    88: (17, 14, 12, 21, 24),
    90: (18, 14, 12, 22, 24),
    92: (19, 15, 12, 22, 24),
    95: (19, 15, 13, 23, 25),
    99: (19, 17, 14, 24, 25),
    113: (24, 18, 16, 27, 28),
    135: (28, 22, 21, 32, 32),
    180: (33, 31, 31, 41, 44),
    270: (53, 52, 53, 54, 58),
    450: (87, 82, 94, 93, 94),
    900: (180, 180, 180, 180, 180),
}


@pytest.fixture(autouse=True)
def block_real_data_loaders(monkeypatch):
    import p19.data_loading as data_loading

    def refuse(*args, **kwargs):
        raise AssertionError("synthetic test called a real-data loader")

    for name in dir(data_loading):
        if name.startswith("load_") or name.startswith("iter_"):
            monkeypatch.setattr(data_loading, name, refuse)


def _protocol() -> dict:
    from pathlib import Path

    return load_yaml(Path(__file__).resolve().parents[1] / "reports" / "protocol_freeze.yaml")


def _seeds() -> list[int]:
    from p19.l2_sampling import sampling_seeds_from_protocol

    return sampling_seeds_from_protocol(_protocol())


def _synthetic_prefix(n: int = 45, p: int = 5, seed: int = 11):
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(n, p))
    matrix[:, 0] = 4.0
    target = 0.7 * matrix[:, 1] + 8.0 + rng.normal(scale=0.05, size=n)
    return matrix, target


def _cell(repetition: int, sample_size: int, **overrides):
    payload = {
        "repetition": repetition,
        "sample_size": sample_size,
        "preprocessor": object(),
        "models": {"min_norm": object(), "ridge": object(), "pcr": object()},
    }
    payload.update(overrides)
    return SimpleNamespace(**payload)


def _declared_cells(drop: tuple[int, int] | None = None, extra: tuple[int, int] | None = None):
    cells = []
    for repetition in range(100):
        for sample_size in YEAR_SAMPLE_SIZE_LIST_ORIGINAL:
            if drop == (repetition, sample_size):
                continue
            cells.append(_cell(repetition, sample_size))
    if extra is not None:
        cells.append(_cell(*extra))
    return cells


def _holdout_spy():
    calls = []

    def evaluate(preprocessor, model, x_holdout, y_holdout):
        calls.append((x_holdout.shape, y_holdout.shape))
        return {"mse": 0.0, "predictions": np.zeros(y_holdout.shape[0])}

    return calls, evaluate


def _assert_never_scored(source):
    calls, evaluate = _holdout_spy()
    with pytest.raises(HoldoutScoringBlocked):
        score_completed_source_phase(source, np.zeros((3, 1)), np.zeros(3), evaluate=evaluate)
    assert calls == []


def test_declared_grid_runs_once_per_cell_without_estimators_or_holdout_scoring(monkeypatch):
    import p19.l2_pipeline as pipeline

    forbidden = []

    def forbid(*args, **kwargs):
        forbidden.append("called")
        raise AssertionError("source phase called a real estimator or holdout scorer")

    for name in ("fit_min_norm", "fit_ridge", "fit_pcr", "tune_ridge", "tune_pcr", "evaluate_holdout"):
        monkeypatch.setattr(pipeline, name, forbid)
    permutation_calls = []
    fold_calls = []
    real_permutation = pipeline.source_permutation
    real_folds = pipeline.longest_prefix_folds

    def counted_permutation(sampling_seed, pool_rows=463715):
        permutation_calls.append(int(sampling_seed))
        return real_permutation(sampling_seed, pool_rows)

    def counted_folds(fold_seed=810000001):
        fold_calls.append(int(fold_seed))
        return real_folds(fold_seed)

    monkeypatch.setattr(pipeline, "source_permutation", counted_permutation)
    monkeypatch.setattr(pipeline, "longest_prefix_folds", counted_folds)

    seeds = _seeds()
    pool = np.arange(463715, dtype=np.int64)[:, None]
    records = []

    def execute_cell(**kwargs):
        indices = np.asarray(kwargs["prefix_indices"], dtype=np.int64).copy()
        assert np.array_equal(pool[indices, 0], indices)
        records.append(
            {
                "repetition": int(kwargs["repetition"]),
                "sampling_seed": int(kwargs["sampling_seed"]),
                "sample_size": int(kwargs["sample_size"]),
                "indices": indices,
                "folds": [np.asarray(fold, dtype=np.int64).copy() for fold in kwargs["folds"]],
            }
        )
        return _cell(kwargs["repetition"], kwargs["sample_size"])

    source = run_declared_source_phase(_protocol(), execute_cell)
    assert forbidden == []
    assert source.complete is True
    assert source.failure is None
    expected = [
        (repetition, int(sample_size))
        for repetition in range(100)
        for sample_size in YEAR_SAMPLE_SIZE_LIST_ORIGINAL
    ]
    observed = [(item["repetition"], item["sample_size"]) for item in records]
    assert observed == expected
    assert len(set(observed)) == 100 * 15
    assert permutation_calls == seeds
    assert fold_calls == [810000001] * 100
    folds_900 = longest_prefix_folds()
    for repetition, seed in enumerate(seeds):
        block = records[repetition * 15 : (repetition + 1) * 15]
        assert [item["sampling_seed"] for item in block] == [seed] * 15
        order = source_permutation(seed)
        assert order.shape == (463715,)
        assert int(order.max()) == 463714
        previous = None
        for item in block:
            assert np.array_equal(item["indices"], order[: item["sample_size"]])
            assert len(np.unique(item["indices"])) == item["sample_size"]
            if previous is not None:
                assert np.array_equal(previous, item["indices"][: previous.size])
            previous = item["indices"]
            restricted = restrict_nested_folds(folds_900, item["sample_size"])
            for got, expected_fold in zip(item["folds"], restricted):
                assert np.array_equal(got, expected_fold)
            assert tuple(int(fold.size) for fold in item["folds"]) == EXPECTED_FOLD_COUNTS[item["sample_size"]]
    calls, evaluate = _holdout_spy()
    scored = score_completed_source_phase(source, np.zeros((4, 1)), np.ones(4), evaluate=evaluate)
    assert len(calls) == 100 * 15 * 3
    scored_keys = [(row["repetition"], row["sample_size"], row["estimator"]) for row in scored]
    expected_scores = [
        (repetition, int(sample_size), estimator)
        for repetition, sample_size in expected
        for estimator in ("min_norm", "ridge", "pcr")
    ]
    assert scored_keys == expected_scores


def test_failure_at_the_first_intermediate_and_final_cells_never_scores():
    seeds = _seeds()

    def run_until(stop_repetition: int, stop_size: int):
        seen = []

        def execute_cell(**kwargs):
            seen.append((kwargs["repetition"], kwargs["sample_size"]))
            if kwargs["repetition"] == stop_repetition and kwargs["sample_size"] == stop_size:
                raise EstimatorError("injected source-phase failure")
            return _cell(kwargs["repetition"], kwargs["sample_size"])

        source = run_declared_source_phase(_protocol(), execute_cell)
        return seen, source

    seen, source = run_until(0, 45)
    assert source.complete is False
    assert source.cells == []
    assert source.failure["repetition"] == 0
    assert source.failure["sample_size"] == 45
    assert seen == [(0, 45)]
    _assert_never_scored(source)

    seen, source = run_until(40, 180)
    assert source.complete is False
    assert source.failure["repetition"] == 40
    assert source.failure["sample_size"] == 180
    assert (40, 180) in seen
    assert (40, 270) not in seen
    assert len(source.cells) == 40 * 15 + YEAR_SAMPLE_SIZE_LIST_ORIGINAL.index(180)
    assert all(cell.repetition < 40 or cell.sample_size < 180 for cell in source.cells)
    _assert_never_scored(source)

    seen, source = run_until(99, 900)
    assert source.complete is False
    assert len(source.cells) == 100 * 15 - 1
    assert seen[-1] == (99, 900)
    assert source.failure["sampling_seed"] == seeds[99]
    assert not any(cell.repetition == 99 and cell.sample_size == 900 for cell in source.cells)
    _assert_never_scored(source)


def test_cell_identity_mismatch_aborts_without_scoring():
    seen = []

    def execute_cell(**kwargs):
        seen.append((kwargs["repetition"], kwargs["sample_size"]))
        if len(seen) == 2:
            return _cell(kwargs["repetition"], 50)
        return _cell(kwargs["repetition"], kwargs["sample_size"])

    source = run_declared_source_phase(_protocol(), execute_cell)
    assert source.complete is False
    assert len(source.cells) == 1
    assert len(seen) == 2
    assert source.failure["message"] == "source cell identity does not match the declared grid"
    _assert_never_scored(source)


@pytest.mark.parametrize(
    "source",
    [
        None,
        SimpleNamespace(complete=True, failure=None),
        SimpleNamespace(complete=True, failure=None, cells="not-a-grid"),
        SimpleNamespace(complete=False, failure={"message": "fatal"}, cells=[]),
        SimpleNamespace(complete=True, failure={"message": "fatal"}, cells=_declared_cells()),
        SimpleNamespace(complete=True, failure=None, cells=_declared_cells(drop=(0, 45))),
        SimpleNamespace(
            complete=True,
            failure=None,
            cells=_declared_cells(drop=(99, 900), extra=(0, 45)),
        ),
        SimpleNamespace(
            complete=True,
            failure=None,
            cells=_declared_cells(drop=(99, 900), extra=(100, 900)),
        ),
        SimpleNamespace(
            complete=True,
            failure=None,
            cells=_declared_cells(drop=(3, 90), extra=(3, 50)),
        ),
        SimpleNamespace(
            complete=True,
            failure=None,
            cells=_declared_cells()[:-1] + [_cell(99, 900, models={"min_norm": object(), "ridge": object()})],
        ),
    ],
)
def test_invalid_source_phase_scores_nothing(source):
    _assert_never_scored(source)


def test_hand_built_complete_grid_scores_each_estimator_once():
    source = SimpleNamespace(complete=True, failure=None, cells=_declared_cells())
    calls, evaluate = _holdout_spy()
    scored = score_completed_source_phase(source, np.zeros((2, 1)), np.ones(2), evaluate=evaluate)
    assert len(calls) == 100 * 15 * 3
    assert {(row["repetition"], row["sample_size"], row["estimator"]) for row in scored} == {
        (repetition, int(sample_size), estimator)
        for repetition, sample_size in declared_source_cell_keys()
        for estimator in ("min_norm", "ridge", "pcr")
    }


def test_cv_preprocessor_uses_only_that_folds_training_rows(monkeypatch):
    import p19.tuning as tuning

    matrix, target = _synthetic_prefix()
    folds = longest_prefix_folds()
    restricted = restrict_nested_folds(folds, 45)
    recorded = []
    real = tuning.fit_preprocessor

    def wrapped(x_train, y_train, policy):
        recorded.append(np.asarray(x_train).copy())
        return real(x_train, y_train, policy)

    monkeypatch.setattr(tuning, "fit_preprocessor", wrapped)
    ridge = tune_ridge(matrix, target, np.array([1.0]), restricted, DRAFT_ZERO_VARIANCE)
    pcr = tune_pcr(
        matrix,
        target,
        [PCR_FRACTION_GRID[0]],
        restricted,
        DRAFT_ZERO_VARIANCE,
        DRAFT_PCR_ROUNDING,
    )
    for left, right in zip(ridge.fold_indices, pcr.fold_indices):
        assert np.array_equal(left, right)
    assert len(recorded) == 10
    for fold, train_rows in zip(restricted, recorded[:5]):
        mask = np.ones(matrix.shape[0], dtype=bool)
        mask[fold] = False
        assert train_rows.shape[0] == int(mask.sum())
        assert train_rows.shape[0] == 45 - int(fold.size)
        assert np.allclose(train_rows.mean(axis=0), matrix[mask].mean(axis=0))
        assert not np.allclose(train_rows.mean(axis=0), matrix.mean(axis=0))


def test_full_prefix_refit_shares_one_matrix_and_restores_the_target(monkeypatch):
    import hashlib

    import p19.l2_pipeline as pipeline

    matrix, target = _synthetic_prefix()
    seen = []
    real_min = pipeline.fit_min_norm
    real_ridge = pipeline.fit_ridge
    real_pcr = pipeline.fit_pcr

    def wrap(real):
        def inner(x_train, y_train, *args):
            seen.append((np.asarray(x_train).copy(), np.asarray(y_train).copy()))
            return real(x_train, y_train, *args)

        return inner

    monkeypatch.setattr(pipeline, "fit_min_norm", wrap(real_min))
    monkeypatch.setattr(pipeline, "fit_ridge", wrap(real_ridge))
    monkeypatch.setattr(pipeline, "fit_pcr", wrap(real_pcr))
    fitted = fit_training_prefix(
        matrix,
        target,
        repetition=0,
        sampling_seed=_seeds()[0],
        sample_size=45,
        prefix_indices=np.arange(45),
    )
    assert len(seen) == 3
    for transformed, centered in seen[1:]:
        assert np.array_equal(transformed, seen[0][0])
        assert np.array_equal(centered, seen[0][1])
    transformed = fitted.preprocessor.transform_X(matrix)
    digest = hashlib.sha256(np.ascontiguousarray(transformed, dtype=np.float64).tobytes()).hexdigest()
    assert fitted.transformed_training_sha256 == digest
    assert fitted.models["pcr"].numerical_rank == fitted.models["min_norm"].numerical_rank
    count = pcr_component_count(
        fitted.pcr_tune.selected,
        fitted.models["pcr"].numerical_rank,
        DRAFT_PCR_ROUNDING,
    )
    assert f"retained {count} of numerical rank {fitted.models['pcr'].numerical_rank}" in fitted.models["pcr"].warnings[0]
    holdout_x = np.random.default_rng(23).normal(size=(9, matrix.shape[1]))
    holdout_y = np.linspace(0.0, 3.0, 9)
    scored = evaluate_holdout(fitted.preprocessor, fitted.models["min_norm"], holdout_x, holdout_y)
    bundle = metric_bundle(holdout_y, scored["predictions"])
    assert scored["mse"] == pytest.approx(bundle["mse"])
    assert scored["rmse"] == pytest.approx(bundle["rmse"])
    assert scored["r2"] == pytest.approx(bundle["r2"])
    unrestored = fitted.models["min_norm"].predict(fitted.preprocessor.transform_X(holdout_x))
    assert np.allclose(scored["predictions"], unrestored + fitted.preprocessor.y_mean)


def test_distinct_pcr_fractions_recompute_count_on_the_fitted_matrix():
    rank_one = np.zeros((20, 4))
    rank_one[:, 0] = np.linspace(-1.0, 1.0, 20)
    target = np.linspace(-1.0, 1.0, 20)
    low = refit_full_prefix(rank_one, target, 1e-2, float(PCR_FRACTION_GRID[0]))
    high = refit_full_prefix(rank_one, target, 1e-2, float(PCR_FRACTION_GRID[1]))
    assert low[1]["pcr"].hyperparameter == pytest.approx(float(PCR_FRACTION_GRID[0]))
    assert high[1]["pcr"].hyperparameter == pytest.approx(float(PCR_FRACTION_GRID[1]))
    assert "retained 1 of numerical rank 1" in low[1]["pcr"].warnings[0]
    assert "retained 1 of numerical rank 1" in high[1]["pcr"].warnings[0]
    wider, wider_target = _synthetic_prefix()
    refit = refit_full_prefix(wider, wider_target, 1e-3, float(PCR_FRACTION_GRID[0]))
    full = refit_full_prefix(wider, wider_target, 1e-3, 1)
    assert refit[1]["pcr"].numerical_rank == full[1]["pcr"].numerical_rank
    assert refit[1]["pcr"].numerical_rank > 1
    small_count = pcr_component_count(PCR_FRACTION_GRID[0], refit[1]["pcr"].numerical_rank, DRAFT_PCR_ROUNDING)
    assert small_count < full[1]["pcr"].numerical_rank
    assert f"retained {small_count} of numerical rank {refit[1]['pcr'].numerical_rank}" in refit[1]["pcr"].warnings[0]
    assert f"retained {full[1]['pcr'].numerical_rank} of numerical rank {full[1]['pcr'].numerical_rank}" in full[1]["pcr"].warnings[0]


def test_one_unscorable_candidate_does_not_abort_when_another_remains(monkeypatch):
    import p19.tuning as tuning

    real = tuning.fit_pcr

    def wrapped(x_train, y_train, fraction, rule):
        if float(fraction) == float(PCR_FRACTION_GRID[0]):
            raise EstimatorError("injected unscorable fraction")
        return real(x_train, y_train, fraction, rule)

    monkeypatch.setattr(tuning, "fit_pcr", wrapped)
    matrix, target = _synthetic_prefix()
    fitted = fit_training_prefix(
        matrix,
        target,
        repetition=1,
        sampling_seed=_seeds()[1],
        sample_size=45,
        prefix_indices=np.arange(45),
    )
    assert fitted.pcr_tune.mean_scores[0] is None
    assert fitted.pcr_tune.failures[0]["error"] == "EstimatorError"
    assert fitted.pcr_tune.selected is not None
    assert fitted.pcr_tune.selected != pytest.approx(float(PCR_FRACTION_GRID[0]))
    assert fitted.ridge_tune.selected is not None
    assert set(fitted.models) == {"min_norm", "ridge", "pcr"}


def test_required_refit_failure_blocks_every_holdout_score(monkeypatch):
    import p19.l2_pipeline as pipeline

    def boom(*args, **kwargs):
        raise EstimatorError("injected full-prefix refit failure")

    monkeypatch.setattr(pipeline, "fit_min_norm", boom)
    matrix, target = _synthetic_prefix()
    with pytest.raises(RunAborted) as caught:
        fit_training_prefix(
            matrix,
            target,
            repetition=2,
            sampling_seed=_seeds()[2],
            sample_size=45,
            prefix_indices=np.arange(45),
        )
    assert caught.value.failure["error"] == "EstimatorError"
    assert "pcr_selected" not in caught.value.failure
    _assert_never_scored(SimpleNamespace(complete=False, failure=caught.value.failure, cells=[]))


def test_uncertainty_pairs_by_repetition_identifier():
    procedure = APPROVED_TRAINING_SUBSET_PROCEDURE
    values = np.arange(100, dtype=np.float64)
    summary = training_subset_variability(values, procedure)
    assert summary["sample_sd"] == pytest.approx(float(np.std(values, ddof=1)))
    assert summary["standard_error"] == pytest.approx(summary["sample_sd"] / np.sqrt(100))
    assert summary["degrees_of_freedom"] == 99
    assert summary["generalizes_beyond_fixed_holdout"] is False
    with pytest.raises(UncertaintyNotFrozen):
        training_subset_variability(values[:99], procedure)
    with pytest.raises(UncertaintyNotFrozen):
        target_row_uncertainty(values, procedure)

    rng = np.random.default_rng(12345)
    perm_mn = rng.permutation(100)
    perm_ridge = rng.permutation(100)
    perm_pcr = rng.permutation(100)
    min_norm = values ** 2
    ridge = np.where(values < 50, values, 0.0)
    pcr = values.copy()
    contrasts = paired_estimator_loss_differences(
        min_norm[perm_mn],
        ridge[perm_ridge],
        pcr[perm_pcr],
        procedure,
        repetition_ids={"min_norm": perm_mn, "ridge": perm_ridge, "pcr": perm_pcr},
    )
    aligned = {
        "pcr_minus_ridge": pcr - ridge,
        "min_norm_minus_ridge": min_norm - ridge,
        "pcr_minus_min_norm": pcr - min_norm,
    }
    positional = pcr[perm_pcr] - ridge[perm_ridge]
    assert contrasts["pcr_minus_ridge"]["sample_sd"] == pytest.approx(float(np.std(aligned["pcr_minus_ridge"], ddof=1)))
    assert contrasts["pcr_minus_ridge"]["sample_sd"] != pytest.approx(float(np.std(positional, ddof=1)))
    assert contrasts["min_norm_minus_ridge"]["mean"] == pytest.approx(float(np.mean(aligned["min_norm_minus_ridge"])))
    assert contrasts["pcr_minus_min_norm"]["mean"] == pytest.approx(float(np.mean(aligned["pcr_minus_min_norm"])))
    bad_ids = perm_pcr.copy()
    bad_ids[-1] = bad_ids[0]
    with pytest.raises(UncertaintyNotFrozen):
        paired_estimator_loss_differences(
            min_norm[perm_mn],
            ridge[perm_ridge],
            pcr[perm_pcr],
            procedure,
            repetition_ids={"min_norm": perm_mn, "ridge": perm_ridge, "pcr": bad_ids},
        )

    base_large = 10.0 * values
    risks = {size: np.zeros(100) for size in reversed(tuple(YEAR_SAMPLE_SIZE_LIST_ORIGINAL))}
    perm_large = rng.permutation(100)
    perm_small = rng.permutation(100)
    risks[68] = base_large[perm_large]
    risks[45] = values[perm_small]
    identifiers = {size: np.arange(100) for size in YEAR_SAMPLE_SIZE_LIST_ORIGINAL}
    identifiers[68] = perm_large
    identifiers[45] = perm_small
    adjacent = adjacent_sample_size_differences(risks, procedure, repetition_ids=identifiers)
    aligned_gap = base_large - values
    positional_gap = risks[68] - risks[45]
    assert list(adjacent) == [f"{larger}_minus_{smaller}" for larger, smaller in ADJACENT_SAMPLE_SIZE_PAIRS]
    assert adjacent["68_minus_45"]["sample_sd"] == pytest.approx(float(np.std(aligned_gap, ddof=1)))
    assert adjacent["68_minus_45"]["sample_sd"] != pytest.approx(float(np.std(positional_gap, ddof=1)))
    identifiers[45] = np.arange(99)
    with pytest.raises(UncertaintyNotFrozen):
        adjacent_sample_size_differences(risks, procedure, repetition_ids=identifiers)


def test_empty_restricted_fold_and_holdout_index_are_rejected():
    folds = longest_prefix_folds()
    broken = [np.asarray(fold).copy() for fold in folds]
    broken[2] = np.array([45, 46, 47], dtype=np.int64)
    with pytest.raises(ProtocolValidationError, match="empty"):
        restrict_nested_folds(broken, 45)
    pool = np.arange(463715, dtype=np.float64)[:, None]
    target = np.zeros(463715)
    with pytest.raises(ProtocolValidationError):
        rows_for_prefix(pool, target, np.array([463715], dtype=np.int64))
    assert inspect.signature(assign_source_folds).parameters["seed"].default == 0
