# NHSMM Interfaces

Integration contracts and adapters for [awa-si/nhsmm](https://github.com/awa-si/nhsmm).

This repository owns host/framework integration for NHSMM. It defines canonical contracts and the concrete integration layer for supported external systems, so downstream projects configure and instantiate adapters instead of implementing their own NHSMM integrations.

## Architecture

```text
external host / domain system
        |
        +-----------------------------+
        |                             |
        v                             v
streaming adapter              walk-forward evaluator
        |                             |
        v                             v
Observation + optional Context   validated NHSMM configs
        |                             |
        v                             v
NHSMMRuntimeAdapter             fresh train -> OOS fits
        |
        v
nhsmm.HSMMFilterRuntime
        |
        v
StateEstimate
        |
        v
downstream application
```

The core model, training, filtering, forecasting, and artifact semantics remain in [awa-si/nhsmm](https://github.com/awa-si/nhsmm).

## Public adapter API

```python
from adapters import (
    Context,
    NHSMMRuntimeAdapter,
    NHSMMTunerEvaluator,
    Observation,
    StructuredEventAdapter,
    StateEstimate,
    Adapter,
)
```

### Core contracts

- `Observation` — one ordered model feature vector with optional timestamp, instrument, and metadata.
- `Context` — optional external context vector plus metadata.
- `StateEstimate` — most-probable latent state, state posterior, episode-age posterior, timestamp, and metadata.
- `Adapter` — minimal runtime-neutral event adapter pipeline.
- `NHSMMRuntimeAdapter` — adapter for the public `nhsmm.HSMMFilterRuntime`.
- `NHSMMTunerEvaluator` — domain-neutral train→OOS walk-forward evaluator for core `ConfigTuner`.
- `WalkForwardFold` — explicit temporal train/OOS split contract.
- `StructuredEventAdapter` — schema-driven adapter for structured event mappings.

## Canonical streaming path

```text
event
  -> to_observation(event)
  -> to_context(event)
  -> infer(observation, context)
  -> StateEstimate
  -> from_state(state)
```

Concrete adapters are implemented and maintained in this repository. External projects should consume these adapters through their public configuration/lifecycle surface rather than recreate framework-specific NHSMM integration. Domain decisions remain outside the adapter layer.

Examples:

- Nautilus Trader: integration implemented under `adapters/nautilus/`;
- Freqtrade: integration should be implemented under `adapters/freqtrade/`;
- structured workflows: mapped event payloads through `StructuredEventAdapter`;
- other event-driven systems: subclass `NHSMMRuntimeAdapter` or `Adapter` as appropriate.

## StructuredEventAdapter

`StructuredEventAdapter` expects upstream processing to provide explicit numerical `features` and, when external context is used, numerical `context`.

```python
adapter = StructuredEventAdapter(
    runtime,
    feature_fields=("feature_a", "feature_b"),
    context_fields=("priority",),
)

state = adapter.step({
    "timestamp": 123,
    "event_type": "processed",
    "features": {
        "feature_a": 0.7,
        "feature_b": 0.9,
    },
    "context": {
        "priority": 2.0,
    },
})
```

It validates declared feature/context fields and preserves only explicitly configured `metadata_fields`. It does not infer domain meaning or make application decisions.

## Design boundaries

Belongs here:

- concrete framework integration packages;
- framework lifecycle wiring;
- host-object or event mapping;
- deterministic feature ordering;
- optional external-context mapping;
- timestamp/instrument normalization;
- conversion to/from canonical contracts;
- runtime lifecycle integration;
- domain-neutral validation of adapter inputs.

External projects should not own duplicate NHSMM integration code. They should provide configuration, strategy/domain policy, and application composition around adapters from this repository.

Does not belong here:

- NHSMM model/optimizer implementation;
- trading-specific training objectives or model-selection policy;
- trading strategy, execution, portfolio, or risk policy;
- medical diagnosis, treatment recommendations, or autonomous eligibility decisions;
- raw-document OCR/extraction pipelines;
- application business logic.

## Runtime rules

`HSMMFilterRuntime` is stateful.

Adapters using it must:

- preserve event order;
- keep timestamp mode consistent within a runtime session;
- keep internal/external context mode consistent until `runtime.reset()`;
- use independent runtime state for independent streams unless batching is explicitly designed;
- ensure feature/context dimensions match the loaded model.

`NHSMMRuntimeAdapter` currently maps one canonical event to runtime batch size 1.

## Repository layout

Source layout follows `adapters/<adapter>/`. Wheel/distribution names are independent from source paths; for example the Nautilus adapter may later be released as the `nhsmm-nautilus` wheel while retaining the Python import `adapters.nautilus`.

```text
adapters/
├── base.py          # Observation, Context, StateEstimate, Adapter
├── nhsmm.py         # NHSMMRuntimeAdapter
├── evaluation.py    # walk-forward tuning evaluator
├── structured.py    # StructuredEventAdapter
└── nautilus/
    ├── __init__.py
    ├── actor.py
    ├── bar.py
    ├── contracts.py
    ├── temporal.py
    ├── README.md
    └── DEVELOPMENT.md

docs/
├── adapters.md      # runtime adapter usage and lifecycle rules
└── evaluation.md    # walk-forward tuning/evaluation contract

tests/
├── test_adapter_base.py
├── test_evaluation.py
├── test_nhsmm_adapter.py
├── test_nautilus_adapter.py
└── test_structured_adapter.py
```

## Cross-repository contract

`awa-si/nhsmm` is the source of truth for model, artifact, filtering, forecasting, validation, and `HSMMFilterRuntime` semantics. This repository depends only on the public `nhsmm` package API and must not import or mirror model internals.

Current core expectations:

- Python 3.12+;
- public runtime construction through `nhsmm.HSMMFilterRuntime`;
- artifact loading through public `nhsmm` artifact/inference helpers;
- context-effect validation remains core-owned and is not reimplemented here.

`awa-si/nhsmm-interfaces` is the source of truth for host/framework mapping, adapter lifecycle, canonical `Observation`/`Context`/`StateEstimate` contracts, walk-forward fit/evaluate orchestration over the public core API, and framework-specific packages such as `adapters/nautilus/`.

The core package can be installed from its released `nhsmm` wheel. This repository is currently source-deployed; future adapter wheels may package individual integrations without moving their source directories.

## Documentation

- [Adapter guide](docs/adapters.md) — architecture, contracts, lifecycle, framework patterns, and StructuredEventAdapter usage.
- [Evaluation guide](docs/evaluation.md) — walk-forward folds, tuner evaluator, scoring, and boundaries.
- [NautilusTrader adapter](adapters/nautilus/README.md) — adapter contract and scope.
- [NautilusTrader development](adapters/nautilus/DEVELOPMENT.md) — implementation, deployment, hardening, and lifecycle notes.
- [NHSMM core](https://github.com/awa-si/nhsmm) — model/runtime implementation and model-level documentation.

## Status

The interface layer is under active development. Contracts may change until explicitly documented as stable.

## License

Apache License 2.0 © AWA.SI.
