# NHSMM Interfaces

Integration contracts and adapters for [awa-si/nhsmm](https://github.com/awa-si/nhsmm).

This repository keeps host/framework integration outside the NHSMM core package. It defines canonical input/output contracts and thin adapters that translate external systems into the public NHSMM runtime API.

## Architecture

```text
host / domain system
        |
        v
host or domain adapter
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

Concrete adapters should translate data and lifecycle only. Domain decisions remain outside the adapter layer.

Examples:

- Nautilus Trader: market/bar/event mapping into NHSMM observations/context;
- Freqtrade: row/callback mapping into the same canonical contracts;
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

- host-object or event mapping;
- deterministic feature ordering;
- optional external-context mapping;
- timestamp/instrument normalization;
- conversion to/from canonical contracts;
- runtime lifecycle integration;
- domain-neutral validation of adapter inputs.

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

```text
adapters/
├── base.py       # Observation, Context, StateEstimate, UniversalAdapter
├── nhsmm.py      # NHSMMRuntimeAdapter
└── research.py   # ResearchAdapter

docs/
└── adapters.md   # detailed adapter usage and lifecycle rules

tests/
├── test_adapter_base.py
├── test_nhsmm_adapter.py
└── test_research_adapter.py
```

Legacy/domain work may exist elsewhere in the repository while it is migrated toward these contracts.

## Documentation

- [Adapter guide](docs/adapters.md) — architecture, contracts, lifecycle, framework patterns, and ResearchAdapter usage.
- [NHSMM core](https://github.com/awa-si/nhsmm) — model/runtime implementation and model-level documentation.

## Status

The interface layer is under active development. Contracts may change until explicitly documented as stable.

## License

Apache License 2.0 © AWA.SI.
