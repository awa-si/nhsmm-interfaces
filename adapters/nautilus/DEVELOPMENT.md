# NautilusTrader adapter development

> Status: **bridge prototype implemented**. Temporal CustomData input, DataActor runtime ownership, structured state publication, lifecycle reset, and committed framework-hook tests exist. Remaining work is production hardening rather than initial design.

## Canonical sources

- `awa-si/nautilus@main` — canonical consumer/integration reference.
- `nautechsystems/nautilus_trader` — canonical NautilusTrader framework API owner.
- `awa-si/nhsmm` — canonical NHSMM model/runtime owner.

Do not infer NHSMM model semantics from this adapter and do not place Nautilus integration code in `awa-si/nhsmm`.

## Implemented bridge

```text
TemporalObservationData
        |
        v
NHSMMDataActor.on_data
        |
        v
NautilusTemporalAdapter
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

## Input ownership

The transferred `nautilus-temporal-observations-v1` contract contains the reusable, policy-free data required at the bridge boundary:

- fixed 18-coordinate feature order;
- signed/unsigned value ranges;
- instrument identity;
- decision timestamp/sequence;
- trigger timeframe;
- optional causal timeframe provenance.

The bridge validates contract shape/ranges/causality. Nautilus consumer code owns TA/Axis construction, freshness, feature admission, and trading policy.

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

`NHSMMArtifactIdentity` is optional bridge compatibility metadata only:

- `artifact_id`;
- `n_features`;
- `n_states`;
- `max_duration`;
- observation contract.

It does not load or define artifacts.

`NHSMMForecastData` is an optional future bridge payload for:

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

## Tests

Committed Nautilus bridge tests currently cover:

- CustomData timestamp exposure;
- native DataActorConfig field preservation;
- TemporalObservationData -> NHSMMStateData publication;
- runtime reset delegation;
- temporal subscribe/unsubscribe lifecycle;
- unrelated CustomData isolation;
- wrong temporal payload rejection;
- fail-closed inference errors with no stale publication;
- future-provenance rejection;
- invalid no-input actor configuration.

These are framework-hook tests. End-to-end tests with a real NHSMM artifact/runtime and Nautilus backtest/live engine remain production-hardening work.

## Remaining work

1. Re-verify against the latest available NautilusTrader v2 pre-release whenever the resolved pre-release changes.
2. Define warm-up/history behavior.
3. Wire artifact identity/bootstrap without duplicating NHSMM artifact semantics.
4. Decide whether forecast publication is required by consumers.
5. Add strategy-consumer and backtest lifecycle integration tests.
6. Add multi-stream runtime ownership if multiple instruments/streams are required.
7. Add CustomData persistence/catalog serialization only if a consumer needs it.

Do not add trading decisions to the adapter or actor. Do not recreate Nautilus↔NHSMM bridge code in downstream trading repositories.
