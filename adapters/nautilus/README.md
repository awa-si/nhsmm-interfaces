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
Nautilus feature/data producer
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

## Primary input contract

The primary adapter input is `TemporalObservationData`, schema `nautilus-temporal-observations-v1`.

It carries:

- 18 coordinates in deterministic order;
- `instrument_id`;
- `asof_ts_ns`;
- `decision_sequence`;
- `trigger_timeframe`;
- optional per-timeframe causal provenance.

The feature names and signed/unsigned ranges are defined in `contracts.py`.

`awa-si/nautilus` remains responsible for producing/admitting those features from its TA/Axis pipeline, including freshness and trading-side admission policy. Those policies are not duplicated here.

For Nautilus CustomData timing, the admitted observation exposes:

```text
ts_event = asof_ts_ns
ts_init  = asof_ts_ns
```

The NHSMM runtime receives `asof_ts_ns` as its timestamp.

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
- `adapters/research.py`

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
- backtest/live lifecycle integration tests.
