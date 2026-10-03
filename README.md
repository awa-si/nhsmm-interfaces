# NHSMM Interfaces

Integration contracts and adapters for [awa-si/nhsmm](https://github.com/awa-si/nhsmm).

This repository owns host/framework integration for NHSMM. It defines canonical contracts and the concrete integration layer for supported external systems, so downstream projects configure and instantiate adapters instead of implementing their own NHSMM bridges.

## Architecture

```text
external host / domain system
        |
        v
adapter owned by nhsmm-interfaces
        |
        v
Observation + optional Context
        |
        v
NHSMMRuntimeAdapter
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
    Observation,
    ResearchAdapter,
    StateEstimate,
    UniversalAdapter,
)
```

### Core contracts

- `Observation` — one ordered model feature vector with optional timestamp, instrument, and metadata.
- `Context` — optional external context vector plus metadata.
- `StateEstimate` — most-probable latent state, state posterior, episode-age posterior, timestamp, and metadata.
- `UniversalAdapter` — engine-neutral orchestration contract.
- `NHSMMRuntimeAdapter` — bridge to the public `nhsmm.HSMMFilterRuntime`.
- `ResearchAdapter` — schema-driven adapter for already structured research/workflow events.

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
- research/healthcare workflows: structured worker/API payloads through `ResearchAdapter`;
- other event-driven systems: subclass `NHSMMRuntimeAdapter` or `UniversalAdapter` as appropriate.

## ResearchAdapter

`ResearchAdapter` expects upstream processing to provide explicit numerical `features` and, when external context is used, numerical `context`.

```python
adapter = ResearchAdapter(
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

It validates declared feature/context fields and preserves optional workflow metadata such as `public_ref`, `event_type`, `case_state`, `document_type`, `assessment_type`, and `ai_status`.

It does not parse raw documents, perform OCR, impute data, infer domain meaning, or make application decisions. AWA Access healthcare/clinical-research workflows are one example profile using this neutral adapter.

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

- NHSMM model implementation or training;
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
├── base.py          # Observation, Context, StateEstimate, UniversalAdapter
├── nhsmm.py         # NHSMMRuntimeAdapter
├── research.py      # ResearchAdapter
└── nautilus/
    ├── __init__.py
    ├── actor.py
    ├── bar.py
    ├── contracts.py
    ├── temporal.py
    ├── README.md
    └── DEVELOPMENT.md

docs/
└── adapters.md   # detailed adapter usage and lifecycle rules

tests/
├── test_adapter_base.py
├── test_nhsmm_adapter.py
├── test_nautilus_adapter.py
└── test_research_adapter.py
```

## Documentation

- [Adapter guide](docs/adapters.md) — architecture, contracts, lifecycle, framework patterns, and ResearchAdapter usage.
- [NautilusTrader adapter](adapters/nautilus/README.md) — adapter contract and scope.
- [NautilusTrader development](adapters/nautilus/DEVELOPMENT.md) — draft design and implementation notes.
- [NHSMM core](https://github.com/awa-si/nhsmm) — model/runtime implementation and model-level documentation.

## Status

The interface layer is under active development. Contracts may change until explicitly documented as stable.

## License

Apache License 2.0 © AWA.SI.
