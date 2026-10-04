# Walk-forward evaluator

`nhsmm-interfaces` owns the orchestration boundary between core NHSMM configuration/tuning and host/framework data preparation.

The core package owns model implementation, optimization semantics, configuration contracts, validation snapshots, health diagnostics, and `ConfigTuner`. This repository owns how those public primitives are composed into strict train→OOS evaluation workflows.

## Public API

```python
from adapters import (
    FoldEvaluation,
    WalkForwardEvaluator,
    WalkForwardEvaluatorConfig,
    WalkForwardReport,
    WalkForwardScoreConfig,
    TemporalFold,
)
```

## TemporalFold

A fold contains one training dataset and one strictly later OOS dataset:

```python
fold = TemporalFold(
    label="2026-Q1",
    train=train_tensor,
    oos=oos_tensor,
    train_end_ns=1_000,
    oos_start_ns=1_001,
)
```

Required temporal invariant:

```text
train_end_ns < oos_start_ns
```

The timestamps are interface-level ordering evidence. They may be real nanosecond timestamps or monotonic ordinal integers.

Supported payloads are floating tensors shaped `[T,F]` or `[B,T,F]`, or non-empty lists of `[T,F]` tensors. Optional external context follows the same sequence form.

## WalkForwardEvaluator

The evaluator accepts `ModelConfig` candidates only. `ValidationConfig` is intentionally rejected because its data/scenario/acceptance fields are not evaluator inputs and accepting it would create inert tuning paths. For every fold it:

1. creates a fresh NHSMM model;
2. initializes model distributions;
3. fits only the training side of the fold;
4. evaluates train and OOS through the public core validation-snapshot API;
5. compares train→OOS evidence using the same fitted model;
6. aggregates fold evidence into one scalar tuning score plus diagnostics.

The evaluator is directly callable by core `ConfigTuner`:

```python
from nhsmm import Choice, ConfigTuner, ModelConfig

evaluator = WalkForwardEvaluator(folds)

tuner = ConfigTuner(
    ModelConfig(
        n_states=3,
        n_features=8,
    ),
    {
        "lr": Choice([0.001, 0.003, 0.01]),
        "max_iter": Choice([40, 60, 80]),
    },
    direction="maximize",
)

report = tuner.run(evaluator, strategy="grid")
best_config = report.best.config
```

No latent labels or downstream trading outcomes are required by this interface.

## Seed policy

Default policy is `seed_mode="per_fold"`.

For fold index `i`:

```text
fit_seed = base_seed + i
```

This gives every candidate the same seed on the same fold, reducing candidate-ranking noise. In this mode candidate `ModelConfig.seed` must remain `None`; a non-null seed is rejected rather than silently ignored.

`seed_mode="candidate"` instead preserves the candidate model seed and requires `ModelConfig.seed` to be set.

## Score policy

Default score:

```text
score
  = 1.00 * mean OOS log-likelihood/timestep
  - 0.25 * positive train→OOS likelihood gap
  - 0.50 * mean occupancy L1 drift
  - 5.00 * unhealthy OOS fraction
```

All weights are explicit in `WalkForwardScoreConfig`. By default, any unhealthy OOS fold disqualifies the candidate with the finite `rejection_score`; `unhealthy_oos_policy="penalize"` is available only when soft health handling is explicitly desired. The report records `is_rejected` and concrete rejection reasons.

The score is intentionally domain-neutral. It ranks statistical OOS model quality and stability; it is not a trading-performance objective.

The returned `TuneEvaluation.metrics` includes:

- fold count;
- mean OOS log-likelihood per timestep;
- mean positive train→OOS generalization gap;
- mean state-occupancy L1 distance;
- healthy OOS fraction.

## Detailed evidence

`evaluate(...)` returns `WalkForwardReport`, retaining every fold:

```text
WalkForwardReport
├── score
├── aggregate metrics
└── folds
    ├── label
    ├── fit seed
    ├── train ValidationSnapshot
    ├── OOS ValidationSnapshot
    └── ValidationComparison
```

The report is dictionary/JSON-compatible through `as_dict()`.

## Boundaries

Belongs in this evaluator:

- strict per-fold train→OOS ordering and strictly increasing fold boundaries;
- fresh fit per fold;
- deterministic fit-seed policy;
- use of public NHSMM validation/health contracts;
- statistical OOS aggregation for configuration selection.

Does not belong here:

- feature engineering from Bars/Trades/Quotes;
- exchange/session/calendar policy;
- strategy labels;
- PnL, Sharpe, drawdown, or execution objectives;
- portfolio/risk decisions;
- order generation or execution.

The next integration layer must therefore establish the real timestamp/provenance guarantee while mapping host data into model-ready chronological folds. `TemporalFold` validates declared boundaries and tensor/context structure; it cannot infer timestamps from anonymous tensors.
