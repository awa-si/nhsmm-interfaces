# NautilusTrader adapter development

> Status: **prototype**. The Bar/DataActor framework hook and transferred temporal input contract exist; artifact loading, warm-up, temporal-input actor wiring, persistence, and live integration remain unstabilized. The public-facing adapter overview lives in [README.md](README.md).

Development references:

- `awa-si/nautilus@main` — canonical consumer/integration repository. Use its current lifecycle, configuration, CustomData, replay, and strategy-consumption patterns when designing and validating this adapter.
- `nautechsystems/nautilus_trader` — canonical framework API owner. Use the upstream version/API surface to verify `DataActor`, `DataActorConfig`, `DataType`, `CustomData`, subscriptions, lifecycle callbacks, and publication semantics.

Previously reviewed upstream revision: `nautechsystems/nautilus_trader` `develop` at `51d37c2ce809897dfe09f7d019ff6d278407ae3b`. Re-verify against the currently targeted NautilusTrader version before implementation changes.

## Goal

Implement the reusable NautilusTrader↔NHSMM integration inside `nhsmm-interfaces`, so Nautilus application repositories only configure and consume it without maintaining parallel bridge code. `awa-si/nautilus@main` is the primary consumer target and integration-validation reference.

Recommended direction:

```text
Nautilus market data
        |
        v
NHSMM DataActor
        |
        v
NHSMMRuntimeAdapter
        |
        v
nhsmm.HSMMFilterRuntime
        |
        v
NHSMMStateData (CustomData)
        |
        v
Nautilus Strategy / Actor consumers
```

The NHSMM component should produce state information. Trading decisions remain in Nautilus strategies or other downstream components.

## Why DataActor-first

NautilusTrader separates data actors from strategies:

- a `DataActor` receives/subscribes to market and custom data and owns component state;
- a `Strategy` adds order-management capabilities;
- actors can publish structured custom data and signals.

NHSMM filtering is stateful inference, not order management. Therefore the initial interface should target a dedicated `DataActor` rather than embed NHSMM directly in `Strategy`.

A strategy should not own a separate NHSMM adapter implementation. It may instantiate/configure the adapter-owned actor/component for experiments, but reusable integration code remains in this repository.

## Transferred consumer contract

The first reusable data contract has been extracted from `awa-si/nautilus@main` rather than reimplemented from memory. `adapters/nautilus/contracts.py` carries the fixed `nautilus-temporal-observations-v1` 18-coordinate schema, signed/unsigned ranges, and causal provenance fields required to transport an admitted temporal observation into NHSMM.

The transfer intentionally excludes Nautilus-owned freshness thresholds, TA/Axis feature construction, RSM hazard policy, H1-H4 semantics, trading policy, and the frozen local NHSMM research implementation. Those are not adapter responsibilities.

## Nautilus inputs

Initial draft priority:

1. `Bar`;
2. `TradeTick`;
3. `QuoteTick`;
4. custom structured data where an application supplies derived features.

The first implementation should probably support `Bar` only and generalize after the lifecycle and timestamp behavior is verified.

### Bar mapping

A Nautilus `Bar` provides:

- `bar_type`;
- `open`;
- `high`;
- `low`;
- `close`;
- `volume`;
- `ts_event`;
- `ts_init`.

The adapter must not assume a universal NHSMM feature vector. Feature extraction should be explicit/configured.

Example draft configuration:

```python
feature_fields = (
    "open",
    "high",
    "low",
    "close",
    "volume",
)
```

Derived indicators or transformed values should be produced explicitly by the Nautilus-side component/feature layer before they enter the NHSMM contract.

## Timestamp contract

Nautilus distinguishes:

- `ts_event`: when the market event occurred;
- `ts_init`: when Nautilus initialized the object.

Draft rule:

- use `ts_event` as the NHSMM runtime timestamp;
- preserve `ts_init` as metadata;
- keep one NHSMM runtime per ordered stream.

This is important because `HSMMFilterRuntime` requires strictly increasing timestamps once timestamp mode is enabled.

A stream key must therefore prevent unrelated instruments or bar streams from being mixed into the same runtime.

Candidate stream identity:

```text
(instrument_id, bar_type)
```

For tick inputs:

```text
(instrument_id, input_kind)
```

Multi-input or synchronized-feature models need a separate explicit design and should not be inferred automatically.

## Draft adapter layers

### 1. Nautilus event mapper

Implemented and maintained in this repository. Responsible only for converting Nautilus objects into canonical interface contracts.

Conceptually:

```python
class NautilusBarAdapter(NHSMMRuntimeAdapter):
    def to_observation(self, bar: Bar) -> Observation:
        ...
```

This mapper should preserve:

- instrument identity;
- bar type;
- `ts_event`;
- `ts_init`;
- configured feature order.

### 2. NHSMM actor

Implemented and maintained in this repository. A higher-level Nautilus `DataActor` should own:

- market-data subscriptions;
- NHSMM runtime instance(s);
- adapter instance(s);
- lifecycle/reset handling;
- output publication.

Draft shape:

```python
class NHSMMDataActor(DataActor):
    def on_start(self) -> None:
        self.subscribe_bars(self.config.bar_type)

    def on_bar(self, bar: Bar) -> None:
        state = self.adapter.step(bar)
        self.publish_data(...)

    def on_reset(self) -> None:
        self.runtime.reset()
```

This is architectural pseudocode, not a committed API.

## Output: structured CustomData

Nautilus custom data is the preferred output transport because NHSMM state is structured.

Signals are not suitable as the primary interface because Nautilus signals carry a single value and are string-oriented.

Draft payload:

```python
@dataclass(frozen=True)
class NHSMMStateData:
    instrument_id: str
    stream_id: str
    state: int | None
    posterior: tuple[float, ...]
    age_posterior: tuple[float, ...] | None
    ts_event: int
    ts_init: int
```

Optional future fields should be added only when they are stable model outputs, for example separately computed survival or transition forecasts.

Do not collapse posterior information into a trading label such as `BULL`, `BEAR`, `BUY`, or `SELL` at this layer.

### DataType

Draft routing identity should use Nautilus `DataType` and structured `CustomData`.

Possible naming:

```text
DataType("NHSMMStateData", metadata={...})
```

Metadata may identify model/artifact/stream, but the exact routing schema remains draft.

Persistence would require registering a serializable custom-data class according to Nautilus custom-data requirements.

## Consumer pattern

A Nautilus strategy should subscribe to the NHSMM custom data and consume the state estimate.

```text
NHSMMDataActor
    |
publish_data(NHSMMStateData)
    |
MessageBus
    |
Strategy.on_data(...)
    |
strategy policy / orders / risk
```

The strategy remains responsible for:

- entry/exit logic;
- confidence thresholds;
- position sizing;
- execution;
- portfolio constraints;
- risk policy.

## Lifecycle mapping

Nautilus actors expose `on_start`, `on_stop`, `on_resume`, `on_reset`, `on_degrade`, `on_fault`, and `on_dispose`.

Draft NHSMM mapping:

| Nautilus lifecycle | NHSMM behavior |
| --- | --- |
| `on_start` | subscribe/input setup; runtime starts empty unless restored intentionally |
| `on_stop` | stop subscriptions/resources; do not implicitly destroy model state unless host policy requires it |
| `on_resume` | resume subscriptions; retain runtime state when continuation semantics are intended |
| `on_reset` | call `runtime.reset()`; required between independent backtest/session histories |
| `on_degrade` | stop/limit input processing according to host policy; no model-semantic change |
| `on_fault` | fail closed; do not publish stale new estimates |
| `on_dispose` | release actor-owned resources |

Exact stop/resume behavior must be tested against Nautilus backtest/live lifecycle before stabilization.

## Backtest/live parity

Nautilus routes normalized data through the same data-engine/message-bus concepts across backtest, sandbox, and live environments.

The NHSMM interface should therefore avoid environment-specific inference paths.

Draft requirement:

```text
same normalized event mapping
+ same feature ordering
+ same NHSMM runtime semantics
= same adapter contract
```

Any environment-specific handling should be limited to lifecycle, subscription, persistence, and infrastructure concerns.

## Ordering and concurrency

NHSMM runtime state is mutable and ordered.

Draft constraints:

- one callback sequence must update a given runtime serially;
- do not share one runtime across independent instruments;
- do not concurrently call `step()` on the same runtime;
- publish the state produced by the current event only after `step()` succeeds;
- never publish a stale state after an inference exception.

For multiple instruments, use one runtime per stream key or an explicitly designed fixed-batch runtime. The first implementation should prefer one runtime per stream for clarity.

## Context mapping

NHSMM external context should remain optional.

Possible Nautilus context sources include:

- explicitly computed session/calendar features;
- instrument/static features;
- explicitly derived market-state features;
- custom upstream data.

Context construction must be configured and deterministic.

Do not silently use account, position, PnL, or order state as NHSMM context. If such inputs are desired, they must be an explicit model/interface decision because doing so couples model inference to strategy/account state.

## Prototype API

The current prototype implements:

```python
NautilusAdapterConfig
NautilusBarAdapter
NHSMMStateData
NHSMMDataActorConfig
NHSMMDataActor
```

Potential config fields:

```python
NautilusAdapterConfig(
    bar_type=...,
    feature_fields=(...),
    context_fields=(...),
    model_artifact=...,
    publish_data_type=...,
)
```

This configuration needs further review against Nautilus config serialization/import patterns before code is committed.

## Dependency boundary

Draft recommendation:

- keep NautilusTrader an optional integration dependency;
- ship the concrete Nautilus integration from this repository;
- downstream Nautilus projects import/configure it instead of reimplementing it;
- do not import Nautilus from `adapters/base.py` or `adapters/nhsmm.py`;
- place any eventual implementation in a dedicated module/package, e.g. `adapters/nautilus/`;
- importing the generic adapter package should not require NautilusTrader to be installed.

## Open questions before implementation

1. Exact `DataType` metadata schema for per-model/per-instrument routing.
2. Whether output should be one common `NHSMMStateData` type or separate model-specific data types.
3. Custom-data serialization and catalog persistence requirements.
4. Artifact loading ownership: actor config vs factory/bootstrap layer.
5. Warm-up/history policy before live subscription.
6. Whether initial version accepts only `Bar` or also ticks.
7. Multi-instrument runtime ownership and resource limits.
8. Model version/artifact identity carried in output metadata.
9. Failure policy: log/drop, actor degrade, or actor fault.
10. Exact actor config pattern and importable config compatibility.

## Initial recommendation

Implement in this order after the draft is accepted:

```text
Bar mapper
    -> one-stream actor
    -> NHSMMStateData custom output
    -> strategy consumer test
    -> backtest lifecycle test
    -> reset/replay determinism test
    -> multi-instrument ownership
    -> optional persistence/live validation
```

Do not add trading decisions to the adapter or actor. Do not push Nautilus↔NHSMM bridge ownership into downstream trading repositories.

Bridge-level compatibility data also includes opaque NHSMM state semantics, optional policy-free forecast channels (`next_state_prior`, episode/state-change probabilities, survival/end-within by explicit horizons), and artifact compatibility identity (`artifact_id`, observation contract, feature/state/duration dimensions). These describe integration data only and do not duplicate model or evaluation logic.
