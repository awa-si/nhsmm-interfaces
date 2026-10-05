# NautilusTrader adapter

> Status: **integration implemented; production hardening remains**.

This directory is the canonical implementation owner for the NautilusTrader↔NHSMM integration. Adapter development uses `awa-si/nautilus@main` as the canonical consumer/integration reference for lifecycle, configuration, CustomData, replay, and strategy-consumption patterns.

## Ownership

- `nhsmm-interfaces/adapters/nautilus` owns Nautilus↔NHSMM integration.
- `awa-si/nhsmm` owns NHSMM model/runtime semantics and has no Nautilus dependency.
- `awa-si/nautilus` owns configuration, feature production, strategy policy, execution, and risk.
- Downstream Nautilus projects must consume this adapter rather than reimplement the integration locally.

## Architecture

```text
Nautilus AxisObservation (CustomData)
        |
        v
AxisObservationCollector / AxisTemporalDataActor
        |
        v
AxisTemporalMapper
        |
        v
TemporalObservationData (CustomData)
        |
        v
NHSMMDataActor
        |
        v
TemporalAdapter
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

The design is DataActor-first because NHSMM filtering is stateful data processing, not order management.

## Replay input boundary

Current `awa-si/nautilus@main` publishes neutral causal `AxisObservation` values as Nautilus `CustomData` using schema `axis-observation-v2`. `AxisObservationCollector` subscribes to that DataType directly and preserves selected source payloads unchanged for replay/training orchestration. It defaults to the 5m decision boundary and enforces strictly increasing `asof_ts_ns` and `decision_sequence` per instrument.

The collector does not import Nautilus consumer model classes and does not depend on the ML `FeatureSnapshot` path. The explicit `AxisTemporalMapper` contract `axis-observation-v2-to-nautilus-temporal-observations-v1` maps stable `AxisState` primitives into the existing 18-coordinate temporal model contract. The mapping identity is carried on every produced `TemporalObservationData` value and walk-forward construction rejects mixed mapping identities.

A Nautilus replay can attach the collector through the runner's existing actor boundary:

```python
from adapters.nautilus import AxisObservationCollector
from backtest import BacktestRunner

collector = AxisObservationCollector(trigger_timeframes=("5m",))
runner = BacktestRunner(..., actors=(collector,))
runner.run()
observations = collector.observations
```

No hook/callback infrastructure is required in the Nautilus consumer repository.

## Axis→temporal mapping

The current explicit mapping is fixed by `AXIS_TEMPORAL_MAPPING_FIELDS` and uses only stable Axis primitives:

```text
1h:  direction, efficiency, persistence, volatility, compression, activity
15m: direction, efficiency, persistence, volatility, compression, activity, participation, flow
5m:  direction, volatility_accel, activity_accel
1m:  shock
```

`AxisTemporalDataActor` can subscribe directly to `AxisObservation` CustomData and republish mapped `TemporalObservationData` CustomData. This bridge contains no ML admission, prediction, signal, risk, or execution logic.

## Model input contract

The current model-facing adapter input remains `TemporalObservationData`, schema `nautilus-temporal-observations-v1`.

It carries:

- 18 coordinates in deterministic order;
- `instrument_id`;
- `asof_ts_ns`;
- `decision_sequence`;
- `trigger_timeframe`;
- optional per-timeframe causal provenance.

The feature names and signed/unsigned ranges are defined in `contracts.py`.

`TemporalObservationData v1` is a frozen older interface contract. Current `awa-si/nautilus@main` does not produce it directly. The current integration boundary is `AxisObservation CustomData`; any future Axis→TemporalObservation mapping must be explicit and versioned in this repository.

For Nautilus CustomData timing, the admitted observation exposes:

```text
ts_event = asof_ts_ns
ts_init  = asof_ts_ns
```

The NHSMM runtime receives `asof_ts_ns` as its timestamp.

## Training input

`TemporalFoldBuilder` converts a strictly ordered sequence of already-admitted `TemporalObservationData` values into a generic `TemporalFold` for walk-forward training/evaluation. It preserves the transferred values exactly, derives fold boundaries from the actual selected observation identities, and rejects mixed instruments, non-monotonic decision identity, trigger-provenance mismatch, and empty train/OOS selections.

The builder does not construct Bars/Trades/Quotes features, labels, context, or trading objectives. Current `awa-si/nautilus@main` `AxisObservation v2` is mapped explicitly by `AxisTemporalMapper`; the separate supervised `raw_4tf` 76-coordinate ML contract is not used or reinterpreted by the NHSMM path.

## Output contract

The actor publishes `NHSMMStateData` using schema `nautilus-nhsmm-state-v1`.

Current payload fields:

```python
@dataclass(frozen=True, slots=True)
class NHSMMStateData:
    instrument_id: str
    state: int | None
    posterior: tuple[float, ...]
    age_posterior: tuple[float, ...] | None
    ts_event: int
    ts_init: int
    observation_contract: str | None = None
    decision_sequence: int | None = None
    trigger_timeframe: str | None = None
    bar_type: str | None = None
```

Latent state IDs are opaque. No `BUY`, `SELL`, `BULL`, `BEAR`, sizing, execution, or risk policy belongs in this payload.

## Optional compatibility contracts

`NHSMMArtifactIdentity` is an adapter-level compatibility record only. It is not an artifact loader and does not redefine the NHSMM artifact format.

`NHSMMForecastData` describes optional policy-free forecast channels retained from the former Nautilus research boundary. The current DataActor does not publish forecast data yet.

## Bar fallback

`BarAdapter` remains available as a limited framework fallback for one ordered Bar stream and explicit OHLCV field selection. It is not the canonical model-facing input path.

## Lifecycle

Current actor behavior:

- `on_start`: subscribe to temporal CustomData and optional Bar input;
- `on_stop`: unsubscribe;
- `on_reset`: call `runtime.reset()`;
- `on_data`: consume `TemporalObservationData` and publish `NHSMMStateData`;
- `on_bar`: optional fallback path.

One mutable NHSMM runtime must receive one ordered callback sequence. Do not call `step()` concurrently on the same runtime or share one runtime across independent streams without an explicit batching design.

## Dependency boundary

NautilusTrader is an integration dependency only for this package. Generic imports remain Nautilus-independent:

- `adapters/base.py`
- `adapters/nhsmm.py`
- `adapters/structured.py`

Importing `adapters.nautilus` requires NautilusTrader.

## Development sources

- Consumer/integration reference: `awa-si/nautilus@main`
- Framework API authority: `nautechsystems/nautilus_trader`
- NHSMM model/runtime authority: `awa-si/nhsmm`

Target and verify against the latest available NautilusTrader v2 pre-release. Re-verify the concrete DataActor/PyO3 API whenever that pre-release changes.

## Remaining hardening

Production stabilization still needs:

- continued verification against the latest available NautilusTrader v2 pre-release whenever it changes;
- warm-up/history policy;
- artifact bootstrap/identity wiring;
- optional forecast publication;
- persistence/catalog serialization if required;
- multi-stream runtime ownership;
- live lifecycle integration tests.

Native BacktestEngine/DataBus replay is verified on `nautilus_trader==2.0.0rc5` against the mounted canonical BTCUSDT catalog (`2025-01-01` through `2025-01-05` test window): `AxisTemporalDataActor` consumed real `AxisObservation` CustomData and published 492 ordered `TemporalObservationData` events with the 18-coordinate mapping contract preserved.
