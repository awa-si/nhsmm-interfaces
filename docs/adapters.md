# Adapter guide

This guide defines how external systems connect to NHSMM through `nhsmm-interfaces`.

## 1. Layering

```text
host/domain event
        |
        v
adapter mapping
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
HSMMFilterState
        |
        v
StateEstimate
        |
        v
host/domain consumer
```

The adapter layer translates representation and lifecycle. It must not change NHSMM posterior semantics or embed downstream decision policy.

## 2. Public contracts

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

### Observation

```python
Observation(
    values=(...),
    timestamp=...,
    instrument="...",
    metadata={...},
)
```

- `values`: ordered feature vector expected by the model;
- `timestamp`: optional runtime timestamp;
- `instrument`: optional identifier for instrument-oriented hosts;
- `metadata`: application metadata not consumed directly by NHSMM.

The adapter must not silently reorder, pad, truncate, or synthesize feature values.

### Context

```python
Context(
    values=(...),
    metadata={...},
)
```

Use `Context` only when the runtime session uses external context. If the model uses its internal encoder context, `to_context()` should return `None`.

### StateEstimate

`StateEstimate` contains:

- `state`: argmax of the state posterior;
- `posterior`: posterior probability per latent state;
- `age_posterior`: posterior over current episode age;
- `timestamp`: observation timestamp;
- `metadata`: retained adapter metadata.

`age_posterior` is not a predicted duration. Survival/duration/transition forecasts remain separate NHSMM runtime operations.

## 3. UniversalAdapter

`UniversalAdapter` defines:

```text
event
  -> to_observation(event)
  -> to_context(event)
  -> infer(observation, context)
  -> from_state(state)
```

Required hooks:

```python
def to_observation(self, event) -> Observation: ...
def infer(self, observation, context=None) -> StateEstimate: ...
def from_state(self, state): ...
```

Optional:

```python
def to_context(self, event) -> Context | None:
    return None
```

Use `UniversalAdapter` directly only when the inference backend is not the standard NHSMM streaming runtime or when a different orchestration boundary is intentionally required.

## 4. NHSMMRuntimeAdapter

`NHSMMRuntimeAdapter` is the standard bridge to `nhsmm.HSMMFilterRuntime`.

It handles:

- canonical values -> Torch tensors;
- optional external context -> Torch tensor;
- timestamp forwarding;
- `HSMMFilterRuntime.step(...)`;
- `HSMMFilterState.state_posterior` -> `StateEstimate.posterior`;
- `HSMMFilterState.age_posterior` -> `StateEstimate.age_posterior`;
- most-probable state selection;
- observation metadata preservation.

The current bridge expects one canonical event to produce runtime batch size 1.

### Minimal host adapter

```python
from adapters import Context, NHSMMRuntimeAdapter, Observation
from nhsmm import HSMMFilterRuntime, load_artifact

model = load_artifact("model.pt")
model.eval()
runtime = HSMMFilterRuntime(model)

class HostAdapter(NHSMMRuntimeAdapter):
    def to_observation(self, event):
        return Observation(
            values=event.features,
            timestamp=event.timestamp,
        )

    def to_context(self, event):
        return Context(values=event.context_features)

adapter = HostAdapter(runtime)
state = adapter.step(event)
```

If the model uses internal context, omit `to_context()`.

## 5. Runtime lifecycle

`HSMMFilterRuntime` is stateful. Runtime lifetime must match stream lifetime.

Rules:

- preserve event ordering;
- create separate runtime state for independent streams unless batching is explicitly part of the model design;
- call `runtime.reset()` when intentionally starting a new stream/history;
- do not switch timestamp mode after the first step without reset;
- do not switch between internal and external context after the first step without reset;
- ensure feature and context dimensions match the model.

A host adapter may own lifecycle wiring, but it should not redefine runtime semantics.

## 6. Internal vs external context

### Internal context

```python
class Adapter(NHSMMRuntimeAdapter):
    def to_observation(self, event):
        return Observation(values=event.features)
```

`to_context()` remains `None`.

### External context

```python
class Adapter(NHSMMRuntimeAdapter):
    def to_observation(self, event):
        return Observation(values=event.features)

    def to_context(self, event):
        return Context(values=event.context_features)
```

Every subsequent step in that runtime session must continue using external context until reset.

## 7. ResearchAdapter

`ResearchAdapter` is a neutral, schema-driven adapter for structured research/workflow events.

It expects nested mappings:

```text
event
├── timestamp
├── optional workflow metadata
├── features
│   ├── feature_a
│   └── feature_b
└── context              # optional
    └── priority
```

Example:

```python
from adapters import ResearchAdapter
from nhsmm import HSMMFilterRuntime, load_artifact

model = load_artifact("model.pt")
model.eval()
runtime = HSMMFilterRuntime(model)

adapter = ResearchAdapter(
    runtime,
    feature_fields=("feature_a", "feature_b"),
    context_fields=("priority",),
)

state = adapter.step({
    "public_ref": "CASE-001",
    "event_type": "document_processed",
    "timestamp": 123,
    "features": {
        "feature_a": 0.7,
        "feature_b": 0.9,
    },
    "context": {
        "priority": 2.0,
    },
})
```

### Validation behavior

`ResearchAdapter`:

- requires all declared feature fields;
- requires all declared context fields when external context is configured;
- rejects booleans as numeric features/context;
- rejects non-numeric values;
- rejects NaN/Inf;
- preserves deterministic declared ordering.

When `context_fields=()`, it returns `None` from `to_context()` and therefore uses the NHSMM internal-context path.

### Workflow metadata

The current adapter recognizes these optional metadata conventions:

- `public_ref`;
- `event_type`;
- `case_state`;
- `document_type`;
- `assessment_type`;
- `ai_status`.

Their source field names are configurable through constructor arguments.

These fields are metadata only. They are not NHSMM model inputs unless an upstream pipeline explicitly places corresponding numerical values inside `features` or `context`.

### AWA Access profile

AWA Access healthcare/clinical-research is one integration profile for `ResearchAdapter`:

```text
intake / documents
    -> FastAPI / workers
    -> OCR / extraction / structuring / human review
    -> numerical features + optional context + workflow metadata
    -> ResearchAdapter
    -> NHSMM runtime
    -> StateEstimate
    -> research/navigation/coordination workflow
```

The adapter is not the Access business system of record. Raw documents, OCR, extraction, translation, normalization, privacy/consent handling, and human review remain upstream. Odoo/FastAPI/worker responsibilities remain outside `nhsmm-interfaces`.

For medical/research usage, the adapter must not be treated as a diagnostic, treatment, or autonomous study-eligibility engine.

## 8. Framework patterns

### Nautilus Trader

```text
Nautilus event/bar
    -> Nautilus-specific adapter
    -> NHSMMRuntimeAdapter
    -> StateEstimate
    -> Nautilus strategy/component
```

The adapter maps data only. Orders, portfolio logic, signals, and risk controls remain in Nautilus.

### Freqtrade

```text
Freqtrade row/callback
    -> Freqtrade-specific adapter
    -> NHSMMRuntimeAdapter
    -> StateEstimate
    -> Freqtrade strategy
```

Entry/exit rules remain in the strategy.

## 9. What belongs in adapters

Appropriate:

- host/event field extraction;
- deterministic feature ordering;
- timestamp/instrument normalization;
- optional external context construction;
- canonical contract conversion;
- host-facing representation conversion;
- runtime lifecycle integration;
- validation of adapter-level input shape/type expectations.

Not appropriate:

- model training;
- modification of NHSMM posterior semantics;
- trading decisions or execution policy;
- portfolio/risk policy;
- diagnosis/treatment logic;
- raw-document OCR or extraction;
- application-specific business decisions.

## 10. Testing

Each concrete adapter should test at minimum:

1. deterministic event -> `Observation` mapping;
2. deterministic context mapping when configured;
3. metadata/timestamp preservation;
4. invalid/missing input rejection;
5. `StateEstimate` mapping/passthrough;
6. runtime-reset boundaries where lifecycle is owned by the host adapter;
7. absence of downstream decision side effects.

Repository examples:

- `tests/test_adapter_base.py`;
- `tests/test_nhsmm_adapter.py`;
- `tests/test_research_adapter.py`.
