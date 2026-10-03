# Adapter guide

This guide defines how `nhsmm-interfaces` owns and exposes integrations between external systems and NHSMM.

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

The adapter layer translates representation and lifecycle. Framework-specific integration code belongs here, not in downstream Nautilus/Freqtrade/application repositories. Downstream projects should configure and instantiate these adapters. The adapter layer must not change NHSMM posterior semantics or embed downstream decision policy.

## 2. Public contracts

```python
from adapters import (
    Context,
    NHSMMRuntimeAdapter,
    Observation,
    StructuredEventAdapter,
    StateEstimate,
    Adapter,
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

## 3. Adapter

`Adapter` defines:

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
```

Optional hooks:

```python
def to_context(self, event) -> Context | None:
    return None

def from_state(self, state):
    return state
```

Use `Adapter` directly only when the inference backend is not the standard NHSMM streaming runtime or when a different orchestration boundary is intentionally required. New supported framework integrations should be implemented under `adapters/<framework>/`.

## 4. NHSMMRuntimeAdapter

`NHSMMRuntimeAdapter` is the standard adapter for `nhsmm.HSMMFilterRuntime`.

It handles:

- canonical values -> Torch tensors;
- optional external context -> Torch tensor;
- timestamp forwarding;
- `HSMMFilterRuntime.step(...)`;
- `HSMMFilterState.state_posterior` -> `StateEstimate.posterior`;
- `HSMMFilterState.age_posterior` -> `StateEstimate.age_posterior`;
- most-probable state selection;
- observation metadata preservation.

The current adapter expects one canonical event to produce runtime batch size 1.

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

Framework adapters in this repository own lifecycle wiring for their host framework, but must not redefine NHSMM runtime semantics.

## 6. Internal vs external context

### Internal context

```python
class ExternalContextAdapter(NHSMMRuntimeAdapter):
    def to_observation(self, event):
        return Observation(values=event.features)
```

`to_context()` remains `None`.

### External context

```python
class InternalContextAdapter(NHSMMRuntimeAdapter):
    def to_observation(self, event):
        return Observation(values=event.features)

    def to_context(self, event):
        return Context(values=event.context_features)
```

Every subsequent step in that runtime session must continue using external context until reset.

## 7. StructuredEventAdapter

`StructuredEventAdapter` maps structured event dictionaries into canonical `Observation` and optional `Context` values.

Expected shape:

```text
event
├── timestamp
├── optional configured metadata fields
├── features
│   ├── feature_a
│   └── feature_b
└── context              # optional
    └── priority
```

Example:

```python
from adapters import StructuredEventAdapter
from nhsmm import HSMMFilterRuntime, load_artifact

model = load_artifact("model.pt")
model.eval()
runtime = HSMMFilterRuntime(model)

adapter = StructuredEventAdapter(
    runtime,
    feature_fields=("feature_a", "feature_b"),
    context_fields=("priority",),
    metadata_fields=("event_type", "source_id"),
)

state = adapter.step({
    "event_type": "processed",
    "source_id": "SRC-001",
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

The adapter:

- requires all declared feature fields;
- requires all declared context fields when external context is configured;
- preserves only explicitly configured `metadata_fields`;
- rejects duplicate field declarations;
- rejects booleans, non-numeric values, NaN, and Inf in numeric vectors;
- preserves declared feature/context ordering.

When `context_fields=()`, `to_context()` returns `None` and the NHSMM runtime uses its internal-context path.

## 8. Framework patterns

### Nautilus Trader

Nautilus-specific documentation now lives with the adapter under `adapters/nautilus/`.

- [`adapters/nautilus/README.md`](../adapters/nautilus/README.md) — adapter contract and scope;
- [`adapters/nautilus/DEVELOPMENT.md`](../adapters/nautilus/DEVELOPMENT.md) — current implementation, lifecycle, deployment, and hardening notes.

The canonical flow is:

```text
TemporalObservationData (CustomData)
    -> NHSMMDataActor
    -> TemporalAdapter
    -> NHSMMRuntimeAdapter
    -> NHSMMStateData (CustomData)
    -> Nautilus strategy/component
```

The Bar mapper remains a limited fallback path, not the canonical model-facing input.

Orders, portfolio logic, signals, and risk controls remain in Nautilus.

### Freqtrade

```text
Freqtrade row/callback
    -> Freqtrade-specific adapter
    -> NHSMMRuntimeAdapter
    -> StateEstimate
    -> Freqtrade strategy
```

The Freqtrade integration should likewise live under `adapters/freqtrade/`; downstream strategies configure/consume it. Entry/exit rules remain in the strategy.

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
- `tests/test_structured_adapter.py`.
