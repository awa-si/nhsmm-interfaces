from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from nhsmm import (
    ModelConfig,
    ModelHealthThresholds,
    NHSMM,
    TuneEvaluation,
    ValidationComparison,
    ValidationSnapshot,
    compare_validation_snapshots,
    evaluate_validation_snapshot,
)

SequenceInput = torch.Tensor | list[torch.Tensor]
SeedMode = Literal["fold", "config"]
InvalidOOSPolicy = Literal["reject", "penalize"]


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    """One strictly ordered train -> OOS evaluation fold.

    ``train_end_ns`` and ``oos_start_ns`` are interface-level ordering evidence.
    They may be real nanosecond timestamps or monotonically increasing ordinal
    integers, but the training boundary must precede the OOS boundary.
    """

    label: str
    train: SequenceInput
    oos: SequenceInput
    train_end_ns: int
    oos_start_ns: int
    train_context: SequenceInput | None = None
    oos_context: SequenceInput | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("label must be a non-empty string")
        for name, value in (
            ("train_end_ns", self.train_end_ns),
            ("oos_start_ns", self.oos_start_ns),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be an integer >= 0")
        if self.train_end_ns >= self.oos_start_ns:
            raise ValueError("walk-forward fold requires train_end_ns < oos_start_ns")
        train_features = _validate_sequence_input(self.train, name="train")
        oos_features = _validate_sequence_input(self.oos, name="oos")
        if train_features != oos_features:
            raise ValueError(
                "train and oos must use the same feature dimension; "
                f"got {train_features} and {oos_features}"
            )
        if self.train_context is not None:
            _validate_context_compatibility(
                self.train,
                self.train_context,
                name="train_context",
            )
        if self.oos_context is not None:
            _validate_context_compatibility(
                self.oos,
                self.oos_context,
                name="oos_context",
            )


@dataclass(frozen=True, slots=True)
class NHSMMTuningScoreConfig:
    """Domain-neutral score policy for walk-forward model selection."""

    oos_log_likelihood_weight: float = 1.0
    generalization_gap_penalty: float = 0.25
    occupancy_drift_penalty: float = 0.50
    unhealthy_oos_penalty: float = 5.0

    def __post_init__(self) -> None:
        for name in (
            "oos_log_likelihood_weight",
            "generalization_gap_penalty",
            "occupancy_drift_penalty",
            "unhealthy_oos_penalty",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number")
            if not math.isfinite(float(value)) or value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0")
        if self.oos_log_likelihood_weight == 0.0:
            raise ValueError("oos_log_likelihood_weight must be > 0")


@dataclass(frozen=True, slots=True)
class NHSMMTunerEvaluatorConfig:
    """Evaluator lifecycle and reproducibility policy."""

    device: str = "cpu"
    seed_mode: SeedMode = "fold"
    base_seed: int = 1_000
    score: NHSMMTuningScoreConfig = NHSMMTuningScoreConfig()
    health: ModelHealthThresholds | None = None
    invalid_oos_policy: InvalidOOSPolicy = "reject"
    rejected_score: float = -1.0e12

    def __post_init__(self) -> None:
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be a non-empty string")
        if self.seed_mode not in ("fold", "config"):
            raise ValueError("seed_mode must be 'fold' or 'config'")
        if isinstance(self.base_seed, bool) or not isinstance(self.base_seed, int):
            raise TypeError("base_seed must be an integer")
        if not isinstance(self.score, NHSMMTuningScoreConfig):
            raise TypeError("score must be NHSMMTuningScoreConfig")
        if self.health is not None and not isinstance(self.health, ModelHealthThresholds):
            raise TypeError("health must be ModelHealthThresholds or None")
        if self.invalid_oos_policy not in ("reject", "penalize"):
            raise ValueError("invalid_oos_policy must be 'reject' or 'penalize'")
        if isinstance(self.rejected_score, bool) or not isinstance(
            self.rejected_score, (int, float)
        ):
            raise TypeError("rejected_score must be a real number")
        if not math.isfinite(float(self.rejected_score)):
            raise ValueError("rejected_score must be finite")


@dataclass(frozen=True, slots=True)
class NHSMMFoldEvaluation:
    """One fitted fold plus its train/OOS evidence."""

    label: str
    seed: int
    train: ValidationSnapshot
    oos: ValidationSnapshot
    comparison: ValidationComparison

    def as_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "seed": self.seed,
            "train": self.train.as_dict(),
            "oos": self.oos.as_dict(),
            "comparison": self.comparison.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class NHSMMTuningEvaluationReport:
    """Walk-forward evidence used to create one tuner objective value."""

    score: float
    folds: tuple[NHSMMFoldEvaluation, ...]
    mean_oos_log_likelihood_per_timestep: float
    mean_generalization_gap: float
    mean_occupancy_l1_distance: float
    healthy_oos_fraction: float
    rejected: bool
    rejection_reasons: tuple[str, ...]

    def as_tune_evaluation(self) -> TuneEvaluation:
        return TuneEvaluation(
            score=self.score,
            metrics={
                "folds": float(len(self.folds)),
                "mean_oos_log_likelihood_per_timestep": self.mean_oos_log_likelihood_per_timestep,
                "mean_generalization_gap": self.mean_generalization_gap,
                "mean_occupancy_l1_distance": self.mean_occupancy_l1_distance,
                "healthy_oos_fraction": self.healthy_oos_fraction,
                "rejected": float(self.rejected),
            },
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "score": self.score,
            "mean_oos_log_likelihood_per_timestep": self.mean_oos_log_likelihood_per_timestep,
            "mean_generalization_gap": self.mean_generalization_gap,
            "mean_occupancy_l1_distance": self.mean_occupancy_l1_distance,
            "healthy_oos_fraction": self.healthy_oos_fraction,
            "rejected": self.rejected,
            "rejection_reasons": list(self.rejection_reasons),
            "folds": [fold.as_dict() for fold in self.folds],
        }


class NHSMMTunerEvaluator:
    """Fit/evaluate NHSMM configs over strict walk-forward folds.

    The evaluator is intentionally policy-free with respect to trading. It
    converts ``ModelConfig`` candidates into fresh
    NHSMM fits and returns a ``TuneEvaluation`` suitable for core
    ``ConfigTuner``. Market feature construction and downstream decisions stay
    outside this class.
    """

    def __init__(
        self,
        folds: list[WalkForwardFold] | tuple[WalkForwardFold, ...],
        *,
        config: NHSMMTunerEvaluatorConfig | None = None,
    ) -> None:
        self.folds = tuple(folds)
        if not self.folds:
            raise ValueError("folds must contain at least one WalkForwardFold")
        if any(not isinstance(fold, WalkForwardFold) for fold in self.folds):
            raise TypeError("folds must contain WalkForwardFold values")
        labels = [fold.label for fold in self.folds]
        if len(set(labels)) != len(labels):
            raise ValueError("walk-forward fold labels must be unique")
        train_ends = [fold.train_end_ns for fold in self.folds]
        oos_starts = [fold.oos_start_ns for fold in self.folds]
        if any(left >= right for left, right in zip(train_ends, train_ends[1:])):
            raise ValueError("walk-forward train_end_ns values must be strictly increasing")
        if any(left >= right for left, right in zip(oos_starts, oos_starts[1:])):
            raise ValueError("walk-forward oos_start_ns values must be strictly increasing")
        self.config = config or NHSMMTunerEvaluatorConfig()

    def __call__(self, candidate: ModelConfig) -> TuneEvaluation:
        return self.evaluate(candidate).as_tune_evaluation()

    def evaluate(
        self,
        candidate: ModelConfig,
    ) -> NHSMMTuningEvaluationReport:
        model_config, health = self._candidate_contract(candidate)
        completed: list[NHSMMFoldEvaluation] = []

        for index, fold in enumerate(self.folds):
            seed = self._fit_seed(model_config, index)
            fit_config = model_config.with_overrides(seed=seed)
            model = NHSMM(fit_config, device=self.config.device)
            model.initialize_distributions()
            model.optimize(fold.train, context=fold.train_context)
            model.eval()

            train = evaluate_validation_snapshot(
                model,
                fold.train,
                context=fold.train_context,
                label=f"{fold.label}:train",
                health_thresholds=health,
            )
            oos = evaluate_validation_snapshot(
                model,
                fold.oos,
                context=fold.oos_context,
                label=f"{fold.label}:oos",
                health_thresholds=health,
            )
            comparison = compare_validation_snapshots(train, oos)
            completed.append(
                NHSMMFoldEvaluation(
                    label=fold.label,
                    seed=seed,
                    train=train,
                    oos=oos,
                    comparison=comparison,
                )
            )

        return self._summarize(tuple(completed))

    def _candidate_contract(
        self,
        candidate: ModelConfig,
    ) -> tuple[ModelConfig, ModelHealthThresholds]:
        if not isinstance(candidate, ModelConfig):
            raise TypeError("candidate must be ModelConfig")
        return candidate, self.config.health or ModelHealthThresholds()

    def _fit_seed(self, model_config: ModelConfig, fold_index: int) -> int:
        if self.config.seed_mode == "fold":
            if model_config.seed is not None:
                raise ValueError(
                    "seed_mode='fold' requires candidate ModelConfig.seed=None; "
                    "candidate seeds would otherwise be silently ignored"
                )
            return self.config.base_seed + fold_index
        if model_config.seed is None:
            raise ValueError("seed_mode='config' requires candidate ModelConfig.seed")
        return model_config.seed

    def _summarize(
        self,
        folds: tuple[NHSMMFoldEvaluation, ...],
    ) -> NHSMMTuningEvaluationReport:
        count = float(len(folds))
        mean_oos_ll = sum(
            fold.oos.log_likelihood_per_timestep for fold in folds
        ) / count
        mean_gap = sum(
            max(
                0.0,
                fold.train.log_likelihood_per_timestep
                - fold.oos.log_likelihood_per_timestep,
            )
            for fold in folds
        ) / count
        mean_occupancy_l1 = sum(
            fold.comparison.occupancy_l1_distance for fold in folds
        ) / count
        healthy_fraction = sum(float(fold.oos.health.healthy) for fold in folds) / count

        rejected_folds = tuple(fold.label for fold in folds if not fold.oos.health.healthy)
        rejected = bool(rejected_folds) and self.config.invalid_oos_policy == "reject"

        policy = self.config.score
        if rejected:
            score = float(self.config.rejected_score)
            rejection_reasons = tuple(
                f"{label}: unhealthy OOS model state" for label in rejected_folds
            )
        else:
            score = (
                policy.oos_log_likelihood_weight * mean_oos_ll
                - policy.generalization_gap_penalty * mean_gap
                - policy.occupancy_drift_penalty * mean_occupancy_l1
                - policy.unhealthy_oos_penalty * (1.0 - healthy_fraction)
            )
            rejection_reasons = ()

        if not math.isfinite(score):
            raise ValueError("walk-forward tuning score is not finite")

        return NHSMMTuningEvaluationReport(
            score=float(score),
            folds=folds,
            mean_oos_log_likelihood_per_timestep=float(mean_oos_ll),
            mean_generalization_gap=float(mean_gap),
            mean_occupancy_l1_distance=float(mean_occupancy_l1),
            healthy_oos_fraction=float(healthy_fraction),
            rejected=rejected,
            rejection_reasons=rejection_reasons,
        )


def _validate_sequence_input(value: SequenceInput, *, name: str) -> int:
    if isinstance(value, torch.Tensor):
        if value.ndim not in (2, 3):
            raise ValueError(f"{name} tensor must be [T,F] or [B,T,F]")
        if value.shape[-2] < 1 or value.shape[-1] < 1:
            raise ValueError(f"{name} tensor must contain at least one timestep and feature")
        if not value.is_floating_point():
            raise TypeError(f"{name} tensor must use a floating dtype")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} tensor must contain only finite values")
        return int(value.shape[-1])

    if isinstance(value, list) and value:
        feature_dim: int | None = None
        for index, item in enumerate(value):
            if not isinstance(item, torch.Tensor) or item.ndim != 2:
                raise TypeError(f"{name}[{index}] must be a [T,F] tensor")
            if item.shape[0] < 1 or item.shape[1] < 1:
                raise ValueError(f"{name}[{index}] must not be empty")
            if not item.is_floating_point():
                raise TypeError(f"{name}[{index}] must use a floating dtype")
            if not torch.isfinite(item).all():
                raise ValueError(f"{name}[{index}] must contain only finite values")
            current = int(item.shape[1])
            if feature_dim is None:
                feature_dim = current
            elif current != feature_dim:
                raise ValueError(
                    f"{name} sequences must share one feature dimension; "
                    f"got {feature_dim} and {current}"
                )
        assert feature_dim is not None
        return feature_dim

    raise TypeError(f"{name} must be a tensor or non-empty list of tensors")


def _validate_context_compatibility(
    observations: SequenceInput,
    context: SequenceInput,
    *,
    name: str,
) -> None:
    _validate_sequence_input(context, name=name)

    if isinstance(observations, list):
        if not isinstance(context, list):
            raise TypeError(
                f"{name} must be a list when observations are variable-length sequences"
            )
        if len(context) != len(observations):
            raise ValueError(
                f"{name} list length {len(context)} does not match observation "
                f"batch size {len(observations)}"
            )
        for index, (obs, ctx) in enumerate(zip(observations, context, strict=True)):
            if ctx.shape[0] != obs.shape[0]:
                raise ValueError(
                    f"{name}[{index}] length {ctx.shape[0]} does not match "
                    f"observation length {obs.shape[0]}"
                )
        return

    if isinstance(context, list):
        raise TypeError(f"{name} must be a tensor when observations are a tensor")

    batch_size = 1 if observations.ndim == 2 else int(observations.shape[0])
    timesteps = int(observations.shape[-2])

    if context.ndim == 2 and int(context.shape[0]) != timesteps:
        raise ValueError(
            f"{name} [T,H] length {context.shape[0]} does not match "
            f"observation timesteps {timesteps}"
        )
    if context.ndim == 3:
        if int(context.shape[0]) != batch_size:
            raise ValueError(
                f"{name} batch size {context.shape[0]} does not match "
                f"observation batch size {batch_size}"
            )
        if int(context.shape[1]) not in (1, timesteps):
            raise ValueError(
                f"{name} timestep dimension must be 1 or {timesteps}, "
                f"got {context.shape[1]}"
            )
