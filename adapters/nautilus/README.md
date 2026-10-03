# NautilusTrader adapter

> Status: **prototype implemented; integration not yet production-stable**.

This directory is the canonical and implementation-owning home for the NautilusTrader integration. Adapter development uses `awa-si/nautilus@main` as the canonical consumer/integration reference for real repository lifecycle, configuration, data-flow, replay, and strategy-consumption patterns.

## Purpose

The Nautilus adapter connects NautilusTrader market/data components to the public NHSMM runtime without embedding trading decisions in the adapter layer. The integration is implemented here; Nautilus application repositories should not need their own NHSMM bridge classes.

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

Ownership rule:

- `nhsmm-interfaces/adapters/nautilus` owns the Nautilus↔NHSMM integration;
- downstream Nautilus projects own only configuration, composition, strategy policy, execution, and risk;
- adapter fixes/versioning happen here once and are reused by all Nautilus consumers.

The current design is **DataActor-first**:

- the actor subscribes to market/custom data;
- the actor owns NHSMM runtime state;
- NHSMM state is published as structured custom data;
- strategies consume the state output and retain all order, execution, portfolio, and risk policy.

## Canonical temporal input contract

The reusable adapter carries forward the useful, policy-free parts of the existing `awa-si/nautilus` NHSMM research input contract. The canonical transferred input schema is `nautilus-temporal-observations-v1`, with 18 fixed coordinates in deterministic order plus causal decision/provenance metadata.

The adapter owns the reusable representation in `contracts.py`; `awa-si/nautilus@main` remains responsible for producing these values from its TA/Axis pipeline and for freshness/admission policy. Training gates, RSM policy, trading labels, risk and execution data are deliberately not copied into the adapter.

## Initial scope

The current prototype implements a single ordered `Bar` stream for framework-hook validation. The transferred 18-coordinate temporal contract is the intended model-facing input boundary for the real integration; wiring that contract into the actor is the next implementation step.

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

All Nautilus-specific NHSMM integration implementation belongs under this directory. Downstream projects should import the public adapter/actor/config objects rather than subclassing or duplicating the bridge unless an explicitly unsupported extension requires it.

## Development reference

Development and integration verification are anchored to [`awa-si/nautilus@main`](https://github.com/awa-si/nautilus). That repository is the canonical consumer reference for how the adapter must plug into the AWA Nautilus stack. The upstream [`nautechsystems/nautilus_trader`](https://github.com/nautechsystems/nautilus_trader) project remains the authority for NautilusTrader framework API semantics and version compatibility.

See [DEVELOPMENT.md](DEVELOPMENT.md) for the reviewed Nautilus API surface, draft lifecycle mapping, open questions, and implementation sequence.

Bridge-level compatibility data also includes opaque NHSMM state semantics, optional policy-free forecast channels (`next_state_prior`, episode/state-change probabilities, survival/end-within by explicit horizons), and artifact compatibility identity (`artifact_id`, observation contract, feature/state/duration dimensions). These describe integration data only and do not duplicate model or evaluation logic.
