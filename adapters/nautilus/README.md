# NautilusTrader adapter

> Status: **draft / not implemented**.

This directory is the canonical home for the NautilusTrader integration.

## Purpose

The Nautilus adapter connects NautilusTrader market/data components to the public NHSMM runtime without embedding trading decisions in the adapter layer.

Target architecture:

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

The current design is **DataActor-first**:

- the actor subscribes to market/custom data;
- the actor owns NHSMM runtime state;
- NHSMM state is published as structured custom data;
- strategies consume the state output and retain all order, execution, portfolio, and risk policy.

## Initial scope

The first implementation target is a single ordered `Bar` stream.

Later extensions may cover:

- `TradeTick`;
- `QuoteTick`;
- custom feature data;
- multiple independent streams;
- optional custom-data persistence.

Feature extraction remains explicit and model-specific. The adapter must not assume that OHLCV is always the model feature vector.

## Timestamp mapping

Draft mapping:

- Nautilus `ts_event` -> NHSMM runtime timestamp;
- Nautilus `ts_init` -> metadata;
- one independent NHSMM runtime per ordered stream.

A candidate bar-stream identity is:

```text
(instrument_id, bar_type)
```

## Output contract

The intended Nautilus-facing output is structured `CustomData`, not a trading signal.

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

No `BUY`, `SELL`, `BULL`, or `BEAR` policy belongs in this payload.

## Dependency boundary

NautilusTrader should remain an optional integration dependency.

Generic imports must continue to work without NautilusTrader installed:

- `adapters/base.py`
- `adapters/nhsmm.py`
- `adapters/research.py`

Any Nautilus implementation belongs under this directory.

## Development

See [DEVELOPMENT.md](DEVELOPMENT.md) for the reviewed Nautilus API surface, draft lifecycle mapping, open questions, and implementation sequence.
