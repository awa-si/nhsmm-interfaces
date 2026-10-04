# NHSMM Interfaces

Integration, runtime-adapter, and walk-forward evaluation layer for [awa-si/nhsmm](https://github.com/awa-si/nhsmm).

This repository owns the boundary between the public NHSMM core API and external hosts/frameworks. It does not reimplement model internals or downstream decision policy.

## Ownership

`awa-si/nhsmm` owns:

- model and optimizer implementation;
- configuration and tuning primitives;
- artifact/inference semantics;
- filtering and forecasting;
- model-health and validation snapshots.

`awa-si/nhsmm-interfaces` owns:

- canonical host-facing contracts;
- runtime adapters and lifecycle wiring;
- framework-specific mappings;
- walk-forward fit/evaluate orchestration over the public core API.

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
    NHSMMFoldEvaluation,
    NHSMMTunerEvaluator,
    NHSMMTunerEvaluatorConfig,
    NHSMMTuningEvaluationReport,
    NHSMMTuningScoreConfig,
    WalkForwardFold,
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
  -> WalkForwardFold(s)
  -> NHSMMTunerEvaluator
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
├── evaluation.py    # walk-forward tuner evaluator
├── structured.py    # StructuredEventAdapter
└── nautilus/        # NautilusTrader integration

docs/
├── adapters.md      # adapter/lifecycle contract
└── evaluation.md    # walk-forward evaluation contract

tests/
├── test_adapter_base.py
├── test_evaluation.py
├── test_nhsmm_adapter.py
├── test_nautilus_adapter.py
└── test_structured_adapter.py
```

## NautilusTrader

The canonical Nautilus integration lives under `adapters/nautilus/`.

```text
TemporalObservationData
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
- [Evaluation guide](docs/evaluation.md)
- [NautilusTrader adapter](adapters/nautilus/README.md)
- [NautilusTrader development](adapters/nautilus/DEVELOPMENT.md)
- [NHSMM core](https://github.com/awa-si/nhsmm)

## Status

The interface layer is under active development. Contracts may change until explicitly documented as stable.

## License

Apache License 2.0 © AWA.SI.
