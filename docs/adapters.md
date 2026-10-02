# Adapter guide

This document defines how host systems connect to NHSMM through `nhsmm-interfaces`.

## Architecture

The adapter layer separates framework-specific objects from NHSMM model/runtime semantics.

```text
host framework
    |
    | host event / bar / tick / sample
    v
host adapter
    |
    | Observation + optional Context
    v
NHSMMRuntimeAdapter
    |
    | torch tensors
    v
nhsmm.HSMMFilterRuntime
    |
    | HSMMFilterState
    v
StateEstimate
    |
    v
host-facing result
```

A host adapter should translate data and lifecycle only. Strategy rules, entries/exits, portfolio logic, execution policy, and risk policy remain outside this layer.

## Core contracts

The public adapter contracts are exported from `adapters`:

```python
from adapters import (
    Context,
    NHSMMRuntimeAdapter,
    Observation,
    StateEstimate,
    UniversalAdapter,
)
```

### `Observation`

One canonical model observation.

```python
Observation(
    values=(...),
    timestamp=...,
    instrument="...",
    metadata={...},
)
```

Fields:

- `values`: feature vector passed to NHSMM;
- `timestamp`: optional host timestamp forwarded to the runtime;
- `instrument`: optional instrument identifier retained in output metadata;
- `metadata`: adapter/application metadata not consumed by NHSMM directly.

The feature order and dimension must match the model used by the runtime.

### `Context`

Optional external model context for the same event.

```python
Context(
    values=(...),
    metadata={...},
)
```

`values=None` means that no external context tensor is supplied.

The context dimension must match the configured NHSMM context dimension when external context is used.

### `StateEstimate`

Canonical inference output.

It contains:

- `state`: most probable latent state (`argmax` of the state posterior);
- `posterior`: posterior probability for each latent state;
- `age_posterior`: posterior probability over the current episode age;
- `timestamp`: timestamp associated with the observation;
- `metadata`: retained adapter metadata, including `instrument` when supplied.

`age_posterior` is not a predicted duration. Duration, survival, and transition forecasts should remain separate model/runtime operations.

## `UniversalAdapter`

`UniversalAdapter` defines the common host integration lifecycle:

```text
host event
  -> to_observation(event)
  -> to_context(event)
  -> infer(observation, context)
  -> from_state(state)
  -> host-facing result
```

Required methods:

```python
def to_observation(self, event) -> Observation: ...
def infer(self, observation, context=None) -> StateEstimate: ...
def from_state(self, state): ...
```

Optional method:

```python
def to_context(self, event) -> Context | None:
    return None
```

For integrations backed by the NHSMM streaming runtime, applications normally subclass `NHSMMRuntimeAdapter` rather than implementing `infer()` themselves.

## `NHSMMRuntimeAdapter`

`NHSMMRuntimeAdapter` is the bridge to the public NHSMM streaming API.

It accepts an existing `nhsmm.HSMMFilterRuntime` and implements:

- canonical values -> Torch tensor conversion;
- optional context conversion;
- timestamp forwarding;
- `HSMMFilterRuntime.step(...)` invocation;
- `HSMMFilterState.state_posterior` -> `StateEstimate.posterior`;
- `HSMMFilterState.age_posterior` -> `StateEstimate.age_posterior`;
- most-probable state selection;
- canonical output metadata.

It expects one canonical event to produce runtime batch size `1`.

## Minimal streaming integration

```python
from adapters import Context, NHSMMRuntimeAdapter, Observation
from nhsmm import HSMMFilterRuntime, load_artifact


model = load_artifact("model.pt")
model.eval()
runtime = HSMMFilterRuntime(model)


class TradingAdapter(NHSMMRuntimeAdapter):
    def to_observation(self, bar):
        return Observation(
            values=(
                float(bar.close),
                float(bar.volume),
            ),
            timestamp=bar.timestamp,
            instrument=str(bar.instrument),
        )

    def to_context(self, bar):
        return Context(
            values=(
                float(bar.session_id),
            )
        )


adapter = TradingAdapter(runtime)
state = adapter.step(bar)

print(state.state)
print(state.posterior)
print(state.age_posterior)
```

If the NHSMM model uses its internal encoder context, do not override `to_context()`; the default returns `None`.

## Host-specific output

The default `NHSMMRuntimeAdapter.from_state()` returns the canonical `StateEstimate` unchanged.

A host adapter may translate it into a framework-specific event/value:

```python
class TradingAdapter(NHSMMRuntimeAdapter):
    def to_observation(self, bar):
        ...

    def from_state(self, state):
        return {
            "regime": state.state,
            "confidence": max(state.posterior),
            "posterior": state.posterior,
            "age_posterior": state.age_posterior,
            "timestamp": state.timestamp,
        }
```

This translation should remain representational. It should not decide whether to buy, sell, size a position, place an order, or accept risk.

## Runtime lifecycle

`HSMMFilterRuntime` is stateful. A host adapter must therefore align runtime lifetime with the host stream lifetime.

Typical rules:

- create one runtime for one independent model stream;
- preserve event ordering;
- do not share one mutable runtime across independent instruments unless the model/runtime design explicitly treats them as one batch stream;
- call `runtime.reset()` when starting a new independent stream, replacing the model context mode, or intentionally discarding filter history;
- keep timestamp usage consistent after the first event;
- keep external-context usage consistent after the first event.

The NHSMM runtime fixes timestamp mode and external-context mode for a streaming session until reset.

## Internal vs external context

Two modes are supported by the NHSMM runtime:

### Internal context

The model derives context from its configured encoder.

```python
class Adapter(NHSMMRuntimeAdapter):
    def to_observation(self, event):
        return Observation(values=event.features)
```

Do not return an external `Context` in this mode.

### External context

The host supplies context explicitly on every runtime step.

```python
class Adapter(NHSMMRuntimeAdapter):
    def to_observation(self, event):
        return Observation(values=event.features)

    def to_context(self, event):
        return Context(values=event.context_features)
```

Once a runtime session begins in external-context mode, subsequent steps must continue to provide external context until `runtime.reset()`.

## Framework adapters

Framework integrations should normally contain only the host-specific conversion layer.

### Nautilus Trader

A Nautilus adapter should map Nautilus events/bars and instrument/session information into `Observation` and optional `Context`, then expose `StateEstimate` to the strategy/component layer.

```text
Nautilus Bar/Event
    -> NautilusAdapter.to_observation()/to_context()
    -> NHSMMRuntimeAdapter
    -> StateEstimate
```

Nautilus order management, signal policy, portfolio state, and risk controls remain in Nautilus-side strategy/components.

### Freqtrade

A Freqtrade adapter should map dataframe rows/callback inputs into the same canonical contracts.

```text
Freqtrade row/callback
    -> FreqtradeAdapter.to_observation()/to_context()
    -> NHSMMRuntimeAdapter
    -> StateEstimate
```

Freqtrade entry/exit conditions remain in the strategy and consume the returned state information rather than being embedded in the adapter.

## What belongs in an adapter

Appropriate responsibilities:

- host-object field extraction;
- deterministic feature ordering;
- timestamp/instrument normalization;
- optional context construction;
- conversion to canonical contracts;
- conversion from `StateEstimate` to a host-facing representation;
- runtime lifecycle/reset integration.

Responsibilities that do not belong in the adapter:

- trading decisions;
- entry/exit thresholds;
- position sizing;
- order routing policy;
- portfolio allocation;
- stop-loss/take-profit policy;
- model training;
- modifying NHSMM posterior semantics.

## Error handling

Adapters should fail early when integration contracts do not match the model. In particular, do not silently pad, truncate, reorder, or synthesize missing model features/context dimensions.

Host adapters may validate their own event schema before constructing `Observation` or `Context`, but NHSMM dimensional and runtime invariants should remain authoritative.

## Testing an adapter

At minimum, an engine adapter should test:

1. deterministic host-event -> `Observation` mapping;
2. deterministic context mapping when used;
3. timestamp and instrument preservation;
4. `StateEstimate` passthrough/mapping;
5. runtime reset behavior at stream boundaries;
6. absence of strategy/execution side effects from adapter calls.

The repository tests under `tests/` provide examples for the universal pipeline and NHSMM runtime bridge.

## AWA Access healthcare and clinical research

`AWAAccessResearchAdapter` is the NHSMM binding for the healthcare/clinical-research workflow defined by AWA Access.

It follows the AWA Access operating boundary:

- AWA Access performs intake, information structuring, research/navigation, assessment support, and coordination;
- Odoo remains the authoritative business system of record;
- FastAPI/workers perform integration, OCR/AI processing, extraction, and orchestration;
- NHSMM receives only already structured numerical model features/context;
- the adapter does not diagnose, prescribe, recommend treatment, determine study eligibility, or replace human review.

Canonical path:

```text
AWA Access intake/documents
    -> FastAPI / workers
    -> structured numeric features + workflow metadata
    -> AWAAccessResearchAdapter
    -> NHSMMRuntimeAdapter
    -> nhsmm.HSMMFilterRuntime
    -> StateEstimate
    -> AWA Access research/coordination workflow
```

Example:

```python
from adapters import AWAAccessResearchAdapter
from nhsmm import HSMMFilterRuntime, load_artifact

model = load_artifact("model.pt")
model.eval()
runtime = HSMMFilterRuntime(model)

adapter = AWAAccessResearchAdapter(
    runtime,
    feature_fields=(
        "disease_burden",
        "document_completeness",
    ),
    context_fields=(
        "review_priority",
    ),
)

state = adapter.step({
    "public_ref": "AC-A82XK9Q4",
    "event_type": "document_processed",
    "case_state": "under_review",
    "document_type": "lab_result",
    "assessment_type": "research",
    "ai_status": "completed",
    "timestamp": 123,
    "features": {
        "disease_burden": 0.7,
        "document_completeness": 0.9,
    },
    "context": {
        "review_priority": 2.0,
    },
})
```

`features` and `context` are worker-produced structured numerical payloads. Their field order is explicitly declared by `feature_fields` and `context_fields`; the adapter never derives medical meaning from raw text or files.

The following AWA Access operational values are preserved as metadata when present:

- `public_ref`;
- `event_type`;
- `case_state`;
- `document_type`;
- `assessment_type`;
- `ai_status`.

This mirrors the Access domain model without making NHSMM or `nhsmm-interfaces` the business system of record.

Raw medical documents stay outside this adapter. OCR, extraction, translation, normalization, consent/privacy handling, and human review belong to the Access processing pipeline. Odoo should continue to store business metadata only, consistent with the AWA Access module specification.

Do not use this adapter as a diagnostic or autonomous eligibility engine. Any clinical interpretation, study eligibility decision, treatment decision, or regulated medical action remains with qualified professionals or institutions.
