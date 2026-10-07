# NHSMM Interfaces

> **Deprecated:** this repository is no longer an active development or integration owner. It remains public as a reference implementation and historical contract record for the public [awa-si/nhsmm](https://github.com/awa-si/nhsmm) API.

The retained code documents earlier runtime-adapter, host-integration, and walk-forward patterns around NHSMM. Active integrations should be developed in their current owning repositories; for NautilusTrader, the active adapter is `awa-si/nautilus@main/adapters/nhsmm`.

This repository must not be treated as the current source of truth when a maintained owner exists elsewhere.

## Ownership

`awa-si/nhsmm` owns:

- model and optimizer implementation;
- configuration and tuning primitives;
- artifact/inference semantics;
- filtering and forecasting;
- model-health and validation snapshots.

`awa-si/nhsmm-interfaces` retains, for public reference:

- historical host-facing contracts;
- runtime-adapter and lifecycle patterns;
- framework mapping examples;
- walk-forward fit/evaluate orchestration examples over the public core API.

These retained interfaces are not automatically authoritative for current downstream integrations.

Downstream applications own:

- feature/domain policy;
- trading signals and objectives;
- portfolio/risk policy;
- execution and business logic.

## Public API

Runtime contracts:

```python
from adapters import (
    Adapter,
    Context,
    NHSMMRuntimeAdapter,
    Observation,
    StateEstimate,
    StructuredEventAdapter,
)
```

Walk-forward evaluation:

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

## Runtime path

```text
host event
  -> host/framework adapter
  -> Observation + optional Context
  -> NHSMMRuntimeAdapter
  -> nhsmm.HSMMFilterRuntime
  -> StateEstimate
  -> downstream consumer
```

`HSMMFilterRuntime` is stateful. Preserve event ordering, keep context mode consistent within a session, reset explicitly at stream boundaries, and use independent runtime state for independent streams unless batching is intentionally designed.

## Walk-forward path

```text
chronological host/model-ready data
  -> TemporalFold(s)
  -> WalkForwardEvaluator
  -> fresh NHSMM fit per fold
  -> train/OOS ValidationSnapshot
  -> ValidationComparison
  -> TuneEvaluation
  -> nhsmm.ConfigTuner
```

The evaluator is domain-neutral. It scores statistical OOS quality/stability and does not embed PnL, Sharpe, signal, risk, portfolio, or execution objectives.

## Repository layout

```text
adapters/
├── base.py          # Observation, Context, StateEstimate, Adapter
├── nhsmm.py         # NHSMMRuntimeAdapter
├── walk_forward.py  # walk-forward evaluator
├── structured.py    # StructuredEventAdapter
└── nautilus/        # NautilusTrader integration

docs/
├── adapters.md      # adapter/lifecycle contract
└── walk-forward.md   # walk-forward evaluation contract

tests/
├── test_adapter_base.py
├── test_walk_forward.py
├── test_nhsmm_adapter.py
├── test_nautilus_adapter.py
└── test_structured_adapter.py
```

## NautilusTrader

The canonical Nautilus integration has moved to `awa-si/nautilus@main/adapters/nhsmm`. The retained `adapters/nautilus/` tree is migration/reference history and is not the active contract owner.

```text
AxisObservation (CustomData)
  -> AxisObservationCollector / AxisTemporalDataActor
  -> AxisTemporalMapper
  -> TemporalObservationData (CustomData)
  -> NHSMMDataActor
  -> TemporalAdapter
  -> NHSMMRuntimeAdapter
  -> NHSMMStateData
  -> Nautilus consumer
```

Nautilus remains responsible for feature production/admission, strategy policy, portfolio/risk, and execution.

## Cross-repository contract

This repository depends only on the public `nhsmm` package API and must not import or mirror NHSMM internals.

Current core expectations:

- Python 3.12+;
- runtime construction through `nhsmm.HSMMFilterRuntime`;
- tuning through public `ModelConfig` / `ValidationConfig` / `ConfigTuner`;
- artifact/inference through public core helpers;
- validation evidence through public core snapshot/health APIs.

## Documentation

- [Adapter guide](docs/adapters.md)
- [Evaluation guide](docs/walk-forward.md)
- [NautilusTrader adapter](adapters/nautilus/README.md)
- [NautilusTrader development](adapters/nautilus/DEVELOPMENT.md)
- [NHSMM core](https://github.com/awa-si/nhsmm)

## Status

**Deprecated / reference-only.** No new integration ownership should be assigned to this repository. It remains public so NHSMM users and maintainers can inspect prior adapter contracts, implementation patterns, migration history, and walk-forward reference code.

Current NHSMM model semantics remain owned by `awa-si/nhsmm`. Active host integrations belong to their explicitly documented current owner.

## License

Apache License 2.0 © AWA.SI.
