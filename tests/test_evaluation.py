from __future__ import annotations

import json

import pytest
import torch
from nhsmm import Choice, ConfigTuner, ModelConfig, ModelHealthThresholds, ValidationConfig

from adapters import (
    NHSMMTunerEvaluator,
    NHSMMTunerEvaluatorConfig,
    NHSMMTuningScoreConfig,
    WalkForwardFold,
)


def _sequence(seed: int, *, batches: int = 2, steps: int = 48) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    values = torch.empty(batches, steps, 2)
    midpoint = steps // 2
    values[:, :midpoint, 0] = 2.0
    values[:, :midpoint, 1] = -1.0
    values[:, midpoint:, 0] = -1.0
    values[:, midpoint:, 1] = 2.0
    values += 0.25 * torch.randn(values.shape, generator=generator)
    return values


def _model_config(*, max_iter: int = 8) -> ModelConfig:
    return ModelConfig(
        n_states=2,
        n_features=2,
        max_duration=16,
        causal=True,
        dropout=0.0,
        n_init=1,
        max_iter=max_iter,
        use_scheduler=False,
        convergence_stop=False,
        verbose=False,
        emission_init_mode="kmeans",
    )


def _fold(label: str, offset: int, seed: int) -> WalkForwardFold:
    return WalkForwardFold(
        label=label,
        train=_sequence(seed),
        oos=_sequence(seed + 100),
        train_end_ns=offset + 100,
        oos_start_ns=offset + 101,
    )


def test_walk_forward_fold_requires_strict_temporal_order() -> None:
    with pytest.raises(ValueError, match="train_end_ns < oos_start_ns"):
        WalkForwardFold(
            label="bad",
            train=_sequence(1),
            oos=_sequence(2),
            train_end_ns=10,
            oos_start_ns=10,
        )


def test_evaluator_produces_reproducible_fold_evidence() -> None:
    evaluator = NHSMMTunerEvaluator(
        [_fold("fold-1", 0, 11), _fold("fold-2", 1_000, 12)],
        config=NHSMMTunerEvaluatorConfig(base_seed=700),
    )
    config = _model_config()

    first = evaluator.evaluate(config)
    second = evaluator.evaluate(config)

    assert first.as_dict() == second.as_dict()
    assert len(first.folds) == 2
    assert [fold.seed for fold in first.folds] == [700, 701]
    assert all(fold.train.model_fingerprint == fold.oos.model_fingerprint for fold in first.folds)
    assert all(fold.train.data_fingerprint != fold.oos.data_fingerprint for fold in first.folds)
    assert 0.0 <= first.healthy_oos_fraction <= 1.0
    assert torch.isfinite(torch.tensor(first.score))
    json.dumps(first.as_dict(), sort_keys=True)


def test_evaluator_is_direct_config_tuner_callback() -> None:
    evaluator = NHSMMTunerEvaluator([_fold("fold-1", 0, 21)])
    tuner = ConfigTuner(
        _model_config(max_iter=4),
        {"max_iter": Choice([4, 6])},
    )
    report = tuner.run(evaluator, strategy="grid")

    assert len(report.trials) == 2
    assert report.best.evaluation.metrics["folds"] == 1.0
    assert "healthy_oos_fraction" in report.best.evaluation.metrics


def test_evaluator_rejects_validation_config_candidates() -> None:
    evaluator = NHSMMTunerEvaluator([_fold("fold-1", 0, 31)])

    with pytest.raises(TypeError, match="candidate must be ModelConfig"):
        evaluator(ValidationConfig())


def test_fold_seed_mode_rejects_candidate_seed() -> None:
    evaluator = NHSMMTunerEvaluator([_fold("fold-1", 0, 41)])

    with pytest.raises(ValueError, match="would otherwise be silently ignored"):
        evaluator(_model_config().with_overrides(seed=99))


def test_config_seed_mode_requires_explicit_candidate_seed() -> None:
    evaluator = NHSMMTunerEvaluator(
        [_fold("fold-1", 0, 42)],
        config=NHSMMTunerEvaluatorConfig(seed_mode="config"),
    )

    with pytest.raises(ValueError, match="requires candidate ModelConfig.seed"):
        evaluator(_model_config())


def test_score_policy_penalizes_unhealthy_and_drift() -> None:
    score = NHSMMTuningScoreConfig(
        oos_log_likelihood_weight=1.0,
        generalization_gap_penalty=1.0,
        occupancy_drift_penalty=1.0,
        unhealthy_oos_penalty=100.0,
    )
    evaluator = NHSMMTunerEvaluator(
        [_fold("fold-1", 0, 51)],
        config=NHSMMTunerEvaluatorConfig(
            score=score,
            invalid_oos_policy="penalize",
            health=ModelHealthThresholds(
                min_effective_states=1.0,
                max_state_occupancy=1.0,
                min_viterbi_states=1,
                max_duration_peak=1.0,
                max_transition_peak=1.0,
            ),
        ),
    )

    result = evaluator.evaluate(_model_config(max_iter=4))

    expected = (
        result.mean_oos_log_likelihood_per_timestep
        - result.mean_generalization_gap
        - result.mean_occupancy_l1_distance
        - 100.0 * (1.0 - result.healthy_oos_fraction)
    )
    assert result.score == pytest.approx(expected)


def test_evaluator_rejects_unordered_or_duplicate_folds() -> None:
    first = _fold("same", 1_000, 61)
    duplicate = _fold("same", 2_000, 62)
    with pytest.raises(ValueError, match="labels must be unique"):
        NHSMMTunerEvaluator([first, duplicate])

    later = _fold("later", 2_000, 63)
    earlier = _fold("earlier", 1_000, 64)
    with pytest.raises(ValueError, match="train_end_ns values must be strictly increasing"):
        NHSMMTunerEvaluator([later, earlier])

    first_same_oos = WalkForwardFold(
        label="first-same-oos",
        train=_sequence(65),
        oos=_sequence(66),
        train_end_ns=1_000,
        oos_start_ns=5_000,
    )
    second_same_oos = WalkForwardFold(
        label="second-same-oos",
        train=_sequence(67),
        oos=_sequence(68),
        train_end_ns=2_000,
        oos_start_ns=5_000,
    )
    with pytest.raises(ValueError, match="oos_start_ns values must be strictly increasing"):
        NHSMMTunerEvaluator([first_same_oos, second_same_oos])


def test_unhealthy_oos_is_rejected_by_default() -> None:
    evaluator = NHSMMTunerEvaluator(
        [_fold("fold-1", 0, 71)],
        config=NHSMMTunerEvaluatorConfig(
            health=ModelHealthThresholds(max_state_occupancy=0.1),
            rejected_score=-12345.0,
        ),
    )

    report = evaluator.evaluate(_model_config(max_iter=4))
    tune = report.as_tune_evaluation()

    assert report.rejected is True
    assert report.score == -12345.0
    assert report.rejection_reasons == ("fold-1: unhealthy OOS model state",)
    assert tune.metrics["rejected"] == 1.0


def test_walk_forward_context_must_match_observation_structure() -> None:
    train = _sequence(81)
    oos = _sequence(82)

    with pytest.raises(ValueError, match="does not match observation timesteps"):
        WalkForwardFold(
            label="bad-context",
            train=train,
            oos=oos,
            train_end_ns=100,
            oos_start_ns=101,
            train_context=torch.ones(train.shape[1] - 1, 3),
        )


def test_variable_length_context_lengths_must_match() -> None:
    train = [torch.ones(5, 2), torch.ones(7, 2)]
    oos = [torch.ones(6, 2), torch.ones(8, 2)]

    with pytest.raises(ValueError, match="does not match observation length"):
        WalkForwardFold(
            label="bad-variable-context",
            train=train,
            oos=oos,
            train_end_ns=100,
            oos_start_ns=101,
            train_context=[torch.ones(5, 3), torch.ones(6, 3)],
        )
