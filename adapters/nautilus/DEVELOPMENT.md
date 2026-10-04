# NautilusTrader adapter development

> Status: **integration implemented**. Temporal CustomData input, DataActor runtime ownership, structured state publication, lifecycle reset, and committed framework-hook tests exist. Remaining work is production hardening rather than initial design.

## Canonical sources

- `awa-si/nautilus@main` — canonical consumer/integration reference.
- `nautechsystems/nautilus_trader` — canonical NautilusTrader framework API owner.
- `awa-si/nhsmm` — canonical NHSMM model/runtime owner.

Do not infer NHSMM model semantics from this adapter and do not place Nautilus integration code in `awa-si/nhsmm`.

## Implemented integration

```text
TemporalObservationData
        |
        v
NHSMMDataActor.on_data
        |
        v
TemporalAdapter
        |
        v
NHSMMRuntimeAdapter
        |
        v
HSMMFilterRuntime
        |
        v
NHSMMStateData
        |
        v
publish_data(...)
```

The optional Bar path remains a limited framework fallback.

## Training fold builder

`TemporalFoldBuilder` converts an already-admitted, strictly ordered `TemporalObservationData` stream into one `TemporalFold` for the generic walk-forward evaluator. It preserves the 18 transferred values exactly and derives `train_end_ns` / `oos_start_ns` from the actual selected observations.

The builder rejects non-monotonic timestamps or decision sequences, mixed instruments, trigger-provenance mismatches, and empty train/OOS selections. It does not construct market features, labels, context, or trading objectives.

The current `awa-si/nautilus@main` production/research baseline uses a separate causal 76-coordinate `raw_4tf` contract. That contract is not equivalent to `nautilus-temporal-observations-v1`; no implicit 76→18 mapping is permitted. A future Nautilus→NHSMM producer requires an explicit versioned cross-repository mapping contract.

## Input ownership

The transferred `nautilus-temporal-observations-v1` contract contains the reusable, policy-free data required at the adapter boundary:

- fixed 18-coordinate feature order;
- signed/unsigned value ranges;
- instrument identity;
- decision timestamp/sequence;
- trigger timeframe;
- optional causal timeframe provenance.

The adapter validates contract shape/ranges/causality. Nautilus consumer code owns TA/Axis construction, freshness, feature admission, and trading policy.

## DataActor contract

`NHSMMDataActorConfig` supports:

```python
NHSMMDataActorConfig(
    bar_type=None,
    feature_fields=("open", "high", "low", "close", "volume"),
    consume_temporal_observations=True,
    # native DataActorConfig fields are accepted by PyO3 __new__,
    # e.g. actor_id, log_events, log_commands
)
```

At least one input path must be enabled.

`NHSMMDataActor` receives an injected public `HSMMFilterRuntime`. The application/bootstrap layer currently owns runtime construction and artifact loading.

The actor:

- subscribes/unsubscribes temporal CustomData;
- optionally subscribes/unsubscribes Bars;
- resets the runtime on `on_reset`;
- publishes state only after successful inference;
- ignores unrelated CustomData types.

## PyO3 constructor behavior

Nautilus v2 `DataActor` and `DataActorConfig` are PyO3 types.

For `DataActor`, the native constructor must receive only config:

```python
def __new__(cls, config, runtime):
    return super().__new__(cls, config)
```

The injected runtime remains Python-owned state.

For `DataActorConfig`, native fields such as `actor_id`, `log_events`, and `log_commands` are consumed by the native `__new__` before the Python subclass `__init__` executes. The subclass therefore calls `super().__init__()` and stores only its custom fields.

This behavior is covered by `tests/test_nautilus_adapter.py`.

## Temporal CustomData timing

`TemporalObservationData` exposes:

```text
ts_event = asof_ts_ns
ts_init  = asof_ts_ns
```

This represents an already-admitted causal feature observation. Original per-timeframe market timestamps remain available in `TimeframeProvenance`.

## Output

`NHSMMStateData` is the single common state output type.

Temporal input preserves:

- instrument;
- state posterior;
- age posterior;
- observation contract;
- decision sequence;
- trigger timeframe;
- event/init timestamp.

Bar fallback output additionally preserves `bar_type`.

State IDs remain opaque.

## Artifact and forecast boundaries

`NHSMMArtifactIdentity` is optional adapter compatibility metadata only:

- `artifact_id`;
- `n_features`;
- `n_states`;
- `max_duration`;
- observation contract.

It does not load or define artifacts.

`NHSMMForecastData` is an optional future adapter payload for:

- `next_state_prior`;
- episode-end probability;
- state-change probability;
- explicit horizons;
- survival probability;
- end-within probability.

It is intentionally not published by the current actor until the public NHSMM forecast API and Nautilus consumption contract are wired and tested together.

## Ordering and concurrency

A runtime is mutable and ordered.

Required invariants:

- one serialized callback sequence per runtime;
- no concurrent `step()` calls on one runtime;
- no sharing one runtime across independent streams unless explicitly designed;
- publish only the state produced by the current successful event;
- never publish stale output after an inference exception.

## External context

NHSMM external context remains optional and is not silently inferred from Nautilus account, position, PnL, order, or strategy state.

Any context mapping must be explicit, deterministic, and part of the model/interface contract.

## Deployment / consumer integration

The adapter is currently deployed from source; this repository does not yet define a standalone Python package manifest for the Nautilus wheel. A consumer such as `awa-si/nautilus` should make the repository root available on `PYTHONPATH` during source-based development and install the compatible public `nhsmm` package (Python 3.12+), PyTorch, and the latest available NautilusTrader v2 pre-release in that environment.

This source/PYTHONPATH deployment is transitional. The source remains under `adapters/nautilus/`; a later release workflow can package that source as the `nhsmm-nautilus` wheel without changing the repository layout or Python import path.

### 1. Install runtime dependencies

Use the consumer environment's normal dependency mechanism. For a direct development checkout:

```bash
python -m pip install -U --pre nautilus_trader
python -m pip install -U torch
# install awa-si/nhsmm by the consumer's normal source/package workflow
```

Before deployment, verify which NautilusTrader pre-release actually resolves:

```bash
python - <<'PY'
import nautilus_trader
print(nautilus_trader.__version__)
PY
```

Re-run the adapter tests whenever that resolved pre-release changes.

### 2. Expose the interface checkout

Until packaging is added, the consumer must expose this repository directly:

```bash
export PYTHONPATH="/path/to/nhsmm-interfaces:$PYTHONPATH"
```

Do not copy the Nautilus adapter implementation into the consumer repository.

### 3. Build/load the NHSMM runtime in bootstrap code

Artifact loading remains outside this adapter. The application/bootstrap layer must construct a public `HSMMFilterRuntime` using the canonical `awa-si/nhsmm` API, then inject that runtime into the actor:

```python
from adapters.nautilus import NHSMMDataActor, NHSMMDataActorConfig

runtime = ...  # construct/load through public awa-si/nhsmm APIs

config = NHSMMDataActorConfig(
    consume_temporal_observations=True,
)

actor = NHSMMDataActor(config, runtime)
```

Do not duplicate NHSMM artifact parsing or model semantics inside the Nautilus consumer.

### 4. Register the actor with the Nautilus application

Register the constructed `NHSMMDataActor` through the consumer's current Nautilus component/bootstrap registration path before replay/live execution starts.

The exact registration call is Nautilus-version-sensitive and must be verified against the resolved pre-release and the current `awa-si/nautilus` bootstrap. The adapter contract itself does not own engine construction.

Required ordering:

```text
construct/load NHSMM runtime
        ->
construct NHSMMDataActor
        ->
register actor with Nautilus application
        ->
start engine/replay/live node
```

### 5. Publish admitted temporal observations

The Nautilus consumer produces `TemporalObservationData` only after its own feature freshness/admission checks pass:

```python
from adapters.nautilus import (
    TEMPORAL_OBSERVATION_DATA_TYPE,
    TemporalObservationData,
    TimeframeProvenance,
)
from nautilus_trader.model import CustomData

payload = TemporalObservationData(
    instrument_id="BTCUSDT.BINANCE",
    values=temporal_values,  # exactly 18 values in canonical order
    asof_ts_ns=decision_ts_ns,
    decision_sequence=decision_sequence,
    trigger_timeframe="5m",
    provenance=(
        TimeframeProvenance(
            timeframe="5m",
            source_ts_event_ns=source_ts_event_ns,
            source_ts_init_ns=source_ts_init_ns,
            processed_sequence=decision_sequence,
        ),
    ),
)

custom = CustomData(TEMPORAL_OBSERVATION_DATA_TYPE, payload)
```

Publish/inject that `CustomData` through the consumer's existing Nautilus data path. The actor subscribes to `TEMPORAL_OBSERVATION_DATA_TYPE` on `on_start`.

### 6. Consume NHSMM state

Strategies or other actors subscribe to `NHSMM_STATE_DATA_TYPE` and consume `NHSMMStateData`.

Consumer code may use the posterior/age/decision metadata, but must keep trading/risk interpretation outside the adapter.

### 7. Deployment checks

Run at minimum:

```bash
PYTHONPATH=. python -m pytest -q tests/test_nautilus_adapter.py
python -m py_compile adapters/nautilus/*.py
```

Before promoting a consumer deployment, additionally verify:

- the resolved NautilusTrader pre-release and PyO3 constructor behavior;
- actor registration succeeds in the actual backtest/live bootstrap;
- one ordered runtime is used per independent stream;
- no stale state is emitted after inference failure;
- replay/live timestamps remain causal;
- the consumer's 18-value feature order exactly matches `TEMPORAL_OBSERVATION_NAMES`.


## Tests

Committed Nautilus adapter tests currently cover:

- CustomData timestamp exposure;
- native DataActorConfig field preservation;
- TemporalObservationData -> NHSMMStateData publication;
- runtime reset delegation;
- temporal subscribe/unsubscribe lifecycle;
- unrelated CustomData isolation;
- wrong temporal payload rejection;
- fail-closed inference errors with no stale publication;
- future-provenance rejection;
- invalid no-input actor configuration;
- temporal training-fold construction and causal boundary validation.

These are framework-hook tests. End-to-end tests with a real NHSMM artifact/runtime and Nautilus backtest/live engine remain production-hardening work.

## Remaining work

1. Re-verify against the latest available NautilusTrader v2 pre-release whenever the resolved pre-release changes.
2. Define warm-up/history behavior.
3. Wire artifact identity/bootstrap without duplicating NHSMM artifact semantics.
4. Decide whether forecast publication is required by consumers.
5. Add strategy-consumer and backtest lifecycle integration tests.
6. Add multi-stream runtime ownership if multiple instruments/streams are required.
7. Add CustomData persistence/catalog serialization only if a consumer needs it.

Do not add trading decisions to the adapter or actor. Do not recreate Nautilus↔NHSMM integration code in downstream trading repositories.
