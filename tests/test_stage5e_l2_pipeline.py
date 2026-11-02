"""Synthetic tests for the Stage 5D YearPredictionMSD L2 candidate.

These tests do not open official datasets or target labels.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from p19.code_identity import snapshot_sha256
from p19.estimators import EstimatorError, pcr_component_count
from p19.l2_pipeline import (
    HoldoutScoringBlocked,
    RunAborted,
    evaluate_holdout,
    fit_training_prefix,
    run_declared_source_phase,
    score_completed_source_phase,
)
from p19.l2_sampling import (
    NESTED_FOLD_SEED,
    ProtocolValidationError,
    longest_prefix_folds,
    nested_folds_for_sample_size,
    rows_for_prefix,
    sampling_seed_for_repetition,
    sampling_seeds_from_protocol,
    source_permutation,
    training_prefix,
    validate_sampling_seed_list,
)
from p19.metrics import metric_bundle, mse
from p19.preprocessing import fit_preprocessor
from p19.protocol import (
    DRAFT_PCR_ROUNDING,
    DRAFT_ZERO_VARIANCE,
    PCR_FRACTION_GRID,
    YEAR_SAMPLE_SIZE_LIST_ORIGINAL,
    load_yaml,
    ridge_lambda_grid,
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

PROJECT = Path(__file__).resolve().parents[1]
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
    return load_yaml(PROJECT / "reports" / "protocol_freeze.yaml")


def _seeds() -> list[int]:
    return sampling_seeds_from_protocol(_protocol())


def _synthetic_prefix(n: int = 45, p: int = 5, seed: int = 11):
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(n, p))
    matrix[:, 0] = 4.0
    target = 0.7 * matrix[:, 1] + 8.0 + rng.normal(scale=0.05, size=n)
    return matrix, target


def test_pipeline_source_does_not_call_forbidden_operations():
    import ast

    import p19.l2_pipeline as pipeline
    import p19.l2_sampling as sampling

    for module in (pipeline, sampling):
        tree = ast.parse(inspect.getsource(module))
        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                body = getattr(node, "body", [])
                if (
                    body
                    and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)
                ):
                    docstrings.add(id(body[0].value))
        names: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                names.append(node.module or "")
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.Name):
                names.append(node.id)
            elif isinstance(node, ast.Attribute):
                names.append(node.attr)
            elif isinstance(node, ast.Constant) and id(node) not in docstrings:
                names.append(str(node.value))
        joined = " ".join(names)
        assert "fit_ridge_unnormalized" not in joined
        assert "evaluate_gate" not in joined
        assert "data_loading" not in joined
        assert "17000003" not in joined
        assert "YearPredictionMSD.txt" not in joined


def test_stored_seed_list_is_consumed_in_order_and_validated():
    document = _protocol()
    seeds = sampling_seeds_from_protocol(document)
    assert document["n_training_subset_repetitions"] == 100
    assert isinstance(document["confirmatory_l2_authorized"], bool)
    assert document["status"] == "FROZEN"
    assert document["confirmatory_scope"] == "YEARPREDICTION_MSD_ONLY"
    assert document["superconductivity_confirmatory"] == "DEFERRED_MISSING_NESTED_PREFIX_SEEDS"
    identity = document["code_identity_sha256"]
    assert isinstance(identity, str) and len(identity) == 64 and identity == identity.lower()
    int(identity, 16)
    assert identity == snapshot_sha256(PROJECT)
    assert document["original_yearprediction_sample_size_list_archived"] == list(YEAR_SAMPLE_SIZE_LIST_ORIGINAL)
    assert len(seeds) == 100
    assert len(set(seeds)) == 100
    assert seeds[0] == 100000037
    assert seeds[-1] == 100000031
    assert seeds[0] != min(seeds)
    assert sampling_seed_for_repetition(seeds, 0) == seeds[0]
    assert sampling_seed_for_repetition(seeds, 99) == seeds[-1]
    duplicate = list(seeds)
    duplicate[1] = duplicate[0]
    with pytest.raises(ProtocolValidationError):
        validate_sampling_seed_list(duplicate)
    with pytest.raises(ProtocolValidationError):
        validate_sampling_seed_list(seeds[:-1])


def test_prefixes_are_nested_without_replacement_and_exclude_holdout_rows():
    seed = _seeds()[3]
    order = source_permutation(seed)
    assert order.shape == (463715,)
    assert np.array_equal(np.sort(order), np.arange(463715))
    assert int(order.min()) == 0
    assert int(order.max()) == 463714
    previous = None
    for sample_size in YEAR_SAMPLE_SIZE_LIST_ORIGINAL:
        prefix = training_prefix(seed, sample_size)
        assert np.array_equal(prefix, order[:sample_size])
        assert len(np.unique(prefix)) == sample_size
        if previous is not None:
            assert np.array_equal(previous, prefix[: previous.size])
        previous = prefix
    other = source_permutation(_seeds()[4])
    assert not np.array_equal(order, other)


def test_prefix_selects_rows_from_an_in_memory_source_pool():
    prefix = training_prefix(_seeds()[0], 45)
    matrix = np.arange(463715, dtype=np.float64)[:, None]
    target = np.arange(463715, dtype=np.float64) + 0.25
    selected_x, selected_y = rows_for_prefix(matrix, target, prefix)
    assert selected_x.shape == (45, 1)
    assert np.array_equal(selected_x[:, 0], prefix.astype(np.float64))
    assert np.array_equal(selected_y, prefix.astype(np.float64) + 0.25)
    with pytest.raises(ProtocolValidationError):
        rows_for_prefix(matrix[:10], target[:10], prefix)


def test_nested_fold_counts_membership_and_rejection_of_seed_zero():
    folds_900 = longest_prefix_folds()
    again = assign_source_folds(900, n_folds=5, seed=NESTED_FOLD_SEED)
    for left, right in zip(folds_900, again):
        assert np.array_equal(left, right)
    with pytest.raises(ProtocolValidationError):
        longest_prefix_folds(0)
    for sample_size, expected in EXPECTED_FOLD_COUNTS.items():
        restricted = nested_folds_for_sample_size(sample_size)
        counts = tuple(int(fold.size) for fold in restricted)
        assert counts == expected
        covered = np.sort(np.concatenate(restricted))
        assert np.array_equal(covered, np.arange(sample_size))
        for fold, full in zip(restricted, folds_900):
            assert np.array_equal(fold, full[full < sample_size])
    reshuffled = assign_source_folds(45, n_folds=5, seed=NESTED_FOLD_SEED)
    restricted_45 = nested_folds_for_sample_size(45)
    assert any(not np.array_equal(left, right) for left, right in zip(restricted_45, reshuffled))


def test_ridge_and_pcr_share_restricted_folds_and_unweighted_mean():
    matrix, target = _synthetic_prefix()
    folds = nested_folds_for_sample_size(45)
    lambdas = np.array([1.0])
    ridge = tune_ridge(matrix, target, lambdas, folds, DRAFT_ZERO_VARIANCE)
    pcr = tune_pcr(
        matrix,
        target,
        [PCR_FRACTION_GRID[0], PCR_FRACTION_GRID[-1]],
        folds,
        DRAFT_ZERO_VARIANCE,
        DRAFT_PCR_ROUNDING,
    )
    for left, right in zip(ridge.fold_indices, pcr.fold_indices):
        assert np.array_equal(left, right)
    scores = []
    for fold in folds:
        mask = np.ones(matrix.shape[0], dtype=bool)
        mask[fold] = False
        pre = fit_preprocessor(matrix[mask], target[mask], DRAFT_ZERO_VARIANCE)
        from p19.estimators import fit_ridge

        model = fit_ridge(pre.transform_X(matrix[mask]), pre.transform_y(target[mask]), 1.0)
        prediction = pre.restore_y(model.predict(pre.transform_X(matrix[fold])))
        scores.append(mse(target[fold], prediction))
    unweighted = float(np.mean(scores))
    weights = np.array([fold.size for fold in folds], dtype=np.float64)
    weighted = float(np.sum(np.asarray(scores) * weights) / np.sum(weights))
    assert ridge.mean_scores[0] == pytest.approx(unweighted)
    assert unweighted != pytest.approx(weighted)


def test_exact_tie_keeps_smaller_lambda_and_failed_candidate_is_not_averaged():
    matrix = np.zeros((10, 2))
    target = np.array([1.0, -1.0] * 5)
    folds = assign_source_folds(10, seed=1)
    tied = tune_ridge(matrix, target, np.array([1e-2, 1.0, 10.0]), folds, DRAFT_ZERO_VARIANCE)
    assert tied.selected == pytest.approx(1e-2)
    assert tied.mean_scores[0] == pytest.approx(tied.mean_scores[-1])
    failed = tune_ridge(matrix, target, np.array([-1.0, 0.1]), folds, DRAFT_ZERO_VARIANCE)
    assert failed.mean_scores[0] is None
    assert failed.selected == pytest.approx(0.1)
    assert failed.failures[0]["error"] == "EstimatorError"
    assert failed.failures[0]["fold"] == 0
    assert "message" in failed.failures[0]
    assert failed.failures[0]["candidate"] == "-1.0"


def test_pcr_keeps_fraction_identity_when_component_counts_collide():
    rank = 10
    assert pcr_component_count(PCR_FRACTION_GRID[0], rank, DRAFT_PCR_ROUNDING) == 1
    assert pcr_component_count(PCR_FRACTION_GRID[1], rank, DRAFT_PCR_ROUNDING) == 1
    assert pcr_component_count(PCR_FRACTION_GRID[0], rank, DRAFT_PCR_ROUNDING) != float(PCR_FRACTION_GRID[0])
    matrix = np.zeros((45, 3))
    matrix[:, 0] = np.linspace(-1.0, 1.0, 45)
    target = np.linspace(-2.0, 2.0, 45)
    folds = nested_folds_for_sample_size(45)
    tuned = tune_pcr(
        matrix, target, list(PCR_FRACTION_GRID), folds, DRAFT_ZERO_VARIANCE, DRAFT_PCR_ROUNDING
    )
    assert len(tuned.mean_scores) == 12
    assert all(score is not None for score in tuned.mean_scores)
    assert tuned.selected == pytest.approx(float(PCR_FRACTION_GRID[0]))
    with pytest.raises(EstimatorError):
        pcr_component_count(PCR_FRACTION_GRID[0], 0, DRAFT_PCR_ROUNDING)


def test_full_prefix_refit_uses_original_scale_and_one_transformed_matrix():
    matrix, target = _synthetic_prefix()
    prefix = training_prefix(_seeds()[0], 45)
    fitted = fit_training_prefix(
        matrix,
        target,
        repetition=0,
        sampling_seed=_seeds()[0],
        sample_size=45,
        prefix_indices=prefix,
    )
    assert fitted.preprocessor.n_rows == 45
    assert fitted.models["min_norm"].n == 45
    assert fitted.models["ridge"].n == 45
    assert fitted.models["pcr"].n == 45
    assert fitted.preprocessor.scale[0] == pytest.approx(1.0)
    assert bool(fitted.preprocessor.zero_variance_columns[0])
    transformed = fitted.preprocessor.transform_X(matrix)
    assert np.allclose(transformed[:, 0], 0.0)
    assert fitted.preprocessor.y_mean == pytest.approx(float(np.mean(target)))
    for model in fitted.models.values():
        assert model.intercept == 0.0
    holdout_x = np.random.default_rng(19).normal(size=(12, matrix.shape[1]))
    holdout_x[:, 0] = 4.0
    holdout_y = 0.7 * holdout_x[:, 1] + 8.0
    scored = evaluate_holdout(fitted.preprocessor, fitted.models["ridge"], holdout_x, holdout_y)
    assert scored["n_rows"] == 12
    assert scored["primary_endpoint"] == "unweighted_holdout_mse_original_scale"
    bundle = metric_bundle(holdout_y, scored["predictions"])
    assert scored["mse"] == pytest.approx(bundle["mse"])
    assert scored["rmse"] == pytest.approx(bundle["rmse"])
    assert scored["mae"] == pytest.approx(bundle["mae"])
    assert scored["r2"] == pytest.approx(bundle["r2"])
    unrestored = fitted.models["ridge"].predict(fitted.preprocessor.transform_X(holdout_x))
    assert not np.allclose(unrestored, scored["predictions"])
    assert np.allclose(scored["predictions"], unrestored + fitted.preprocessor.y_mean)


def test_all_pcr_candidates_failing_aborts_before_a_holdout_call():
    matrix = np.zeros((45, 3))
    target = np.array([1.0, -1.0] * 22 + [0.5])
    calls = []
    with pytest.raises(RunAborted) as caught:
        fit_training_prefix(
            matrix,
            target,
            repetition=4,
            sampling_seed=_seeds()[4],
            sample_size=45,
            prefix_indices=np.arange(45),
        )
    assert caught.value.failure["pcr_selected"] is None
    assert caught.value.failure["failures"]
    with pytest.raises(HoldoutScoringBlocked):
        score_completed_source_phase(
            SimpleNamespace(complete=False, failure=caught.value.failure, cells=[]),
            matrix[:3],
            target[:3],
            evaluate=lambda *args, **kwargs: calls.append(1),
        )
    assert calls == []


def test_later_repetition_failure_blocks_every_holdout_score():
    seeds = _seeds()
    seen = []
    calls = []

    def execute_cell(**kwargs):
        seen.append((kwargs["repetition"], kwargs["sample_size"], kwargs["sampling_seed"], kwargs["prefix_indices"]))
        if kwargs["repetition"] == 2 and kwargs["sample_size"] == 45:
            raise EstimatorError("injected full-prefix failure")
        return SimpleNamespace(
            preprocessor=object(),
            models={"min_norm": object(), "ridge": object(), "pcr": object()},
            repetition=kwargs["repetition"],
            sample_size=kwargs["sample_size"],
        )

    source = run_declared_source_phase(_protocol(), execute_cell)
    assert source.complete is False
    assert source.failure["error"] == "EstimatorError"
    assert source.failure["repetition"] == 2
    assert len(source.cells) == 2 * len(YEAR_SAMPLE_SIZE_LIST_ORIGINAL)
    assert seen[0][2] == seeds[0]
    assert seen[len(YEAR_SAMPLE_SIZE_LIST_ORIGINAL)][2] == seeds[1]
    first_prefix = seen[0][3]
    assert np.array_equal(first_prefix, training_prefix(seeds[0], 45))
    with pytest.raises(HoldoutScoringBlocked):
        score_completed_source_phase(
            source,
            np.zeros((4, 2)),
            np.zeros(4),
            evaluate=lambda *args, **kwargs: calls.append(1),
        )
    assert calls == []
    assert all(item[0] < 3 for item in seen)
    assert not any(item[0] == 2 and item[1] != 45 for item in seen)


def test_completed_source_phase_scores_only_after_all_declared_cells():
    calls = []

    def execute_cell(**kwargs):
        return SimpleNamespace(
            preprocessor=object(),
            models={"min_norm": object(), "ridge": object(), "pcr": object()},
            repetition=kwargs["repetition"],
            sample_size=kwargs["sample_size"],
        )

    def evaluate(preprocessor, model, X_holdout, y_holdout):
        calls.append((X_holdout.shape, y_holdout.shape))
        return {"mse": 1.0, "predictions": np.zeros(y_holdout.shape[0])}

    source = run_declared_source_phase(_protocol(), execute_cell)
    assert source.complete is True
    assert len(source.cells) == 100 * 15
    scored = score_completed_source_phase(
        source, np.zeros((5, 2)), np.ones(5), evaluate=evaluate
    )
    assert len(calls) == 100 * 15 * 3
    assert len(scored) == len(calls)
    assert calls[0][1] == (5,)


def test_uncertainty_formulas_signs_and_incomplete_rejection():
    values = np.arange(100, dtype=np.float64)
    summary = training_subset_variability(values, APPROVED_TRAINING_SUBSET_PROCEDURE)
    assert summary["R"] == 100
    assert summary["mean"] == pytest.approx(49.5)
    assert summary["sample_sd"] == pytest.approx(float(np.std(values, ddof=1)))
    assert summary["sample_sd"] != pytest.approx(float(np.std(values, ddof=0)))
    assert summary["standard_error"] == pytest.approx(summary["sample_sd"] / np.sqrt(100))
    assert summary["degrees_of_freedom"] == 99
    assert summary["t_critical"] == pytest.approx(1.984216951586417)
    assert summary["interval"] == "pointwise"
    assert summary["generalizes_beyond_fixed_holdout"] is False
    half = summary["t_critical"] * summary["standard_error"]
    assert summary["interval_low"] == pytest.approx(summary["mean"] - half)
    assert summary["interval_high"] == pytest.approx(summary["mean"] + half)
    with pytest.raises(UncertaintyNotFrozen):
        training_subset_variability(values[:99], APPROVED_TRAINING_SUBSET_PROCEDURE)
    with pytest.raises(UncertaintyNotFrozen):
        training_subset_variability(values, None)
    with pytest.raises(UncertaintyNotFrozen):
        target_row_uncertainty(values, APPROVED_TRAINING_SUBSET_PROCEDURE)
    contrasts = paired_estimator_loss_differences(
        np.ones(100),
        np.zeros(100),
        values,
        APPROVED_TRAINING_SUBSET_PROCEDURE,
    )
    assert contrasts["pcr_minus_ridge"]["mean"] == pytest.approx(49.5)
    assert contrasts["min_norm_minus_ridge"]["mean"] == pytest.approx(1.0)
    assert contrasts["pcr_minus_min_norm"]["mean"] == pytest.approx(48.5)
    risks = {size: np.zeros(100) for size in YEAR_SAMPLE_SIZE_LIST_ORIGINAL}
    risks[45] = np.zeros(100)
    risks[68] = np.ones(100)
    adjacent = adjacent_sample_size_differences(risks, APPROVED_TRAINING_SUBSET_PROCEDURE)
    assert list(adjacent)[0] == "68_minus_45"
    assert adjacent["68_minus_45"]["mean"] == pytest.approx(1.0)
    assert len(adjacent) == len(ADJACENT_SAMPLE_SIZE_PAIRS)
    incomplete = dict(risks)
    del incomplete[900]
    with pytest.raises(UncertaintyNotFrozen):
        adjacent_sample_size_differences(incomplete, APPROVED_TRAINING_SUBSET_PROCEDURE)


def test_ridge_grid_and_pcr_grid_stay_at_the_archived_lengths():
    grid = ridge_lambda_grid()
    assert grid.shape == (25,)
    assert grid[0] == pytest.approx(1e-8)
    assert grid[-1] == pytest.approx(1e4)
    assert list(PCR_FRACTION_GRID)[0].numerator == 1
    assert list(PCR_FRACTION_GRID)[0].denominator == 20
    assert list(PCR_FRACTION_GRID)[-1] == 1
    assert len(PCR_FRACTION_GRID) == 12
