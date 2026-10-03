from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


# Copied from the frozen Nautilus research input contract at
# awa-si/nautilus@3947be02c6b98198813c613aee1c5b2ea27aaf31.
# This adapter module now owns the reusable host-facing representation; the
# consumer repository remains responsible for producing the values causally.
TEMPORAL_OBSERVATION_CONTRACT = "nautilus-temporal-observations-v1"
TEMPORAL_OBSERVATION_NAMES = (
    "1h_direction",
    "1h_efficiency",
    "1h_persistence",
    "1h_volatility",
    "1h_compression",
    "1h_activity",
    "15m_direction",
    "15m_efficiency",
    "15m_persistence",
    "15m_volatility",
    "15m_compression",
    "15m_activity",
    "15m_participation",
    "15m_flow",
    "5m_direction",
    "5m_volatility_accel",
    "5m_activity_accel",
    "1m_shock",
)

SIGNED_TEMPORAL_OBSERVATIONS = frozenset(
    {
        "1h_direction",
        "15m_direction",
        "15m_flow",
        "5m_direction",
        "5m_volatility_accel",
        "5m_activity_accel",
    }
)


@dataclass(frozen=True, slots=True)
class TimeframeProvenance:
    """Causal provenance for one timeframe contributing to a temporal input."""

    timeframe: str
    source_ts_event_ns: int
    source_ts_init_ns: int
    processed_sequence: int
    timeframe_bar_count: int | None = None
    age_ns: int | None = None

    def __post_init__(self) -> None:
        if not self.timeframe:
            raise ValueError("timeframe must be non-empty")
        if min(self.source_ts_event_ns, self.source_ts_init_ns, self.processed_sequence) < 0:
            raise ValueError("timeframe provenance must be non-negative")
        if self.timeframe_bar_count is not None and self.timeframe_bar_count < 0:
            raise ValueError("timeframe_bar_count must be non-negative")
        if self.age_ns is not None and self.age_ns < 0:
            raise ValueError("age_ns must be non-negative")

    @property
    def available_ts_ns(self) -> int:
        return max(self.source_ts_event_ns, self.source_ts_init_ns)


@dataclass(frozen=True, slots=True)
class TemporalObservationData:
    """Nautilus-facing NHSMM input built from the fixed 18-coordinate contract.

    The adapter validates shape, finite/range constraints and causal provenance.
    It intentionally does not own TA/Axis construction, freshness policy,
    strategy admission, or trading semantics.
    """

    instrument_id: str
    values: tuple[float, ...]
    asof_ts_ns: int
    decision_sequence: int
    trigger_timeframe: str
    provenance: tuple[TimeframeProvenance, ...] = ()
    contract: str = TEMPORAL_OBSERVATION_CONTRACT

    def __post_init__(self) -> None:
        if not self.instrument_id:
            raise ValueError("instrument_id must be non-empty")
        if self.contract != TEMPORAL_OBSERVATION_CONTRACT:
            raise ValueError("unsupported temporal observation contract")
        if self.asof_ts_ns < 0 or self.decision_sequence < 0:
            raise ValueError("decision identity must be non-negative")
        if not self.trigger_timeframe:
            raise ValueError("trigger_timeframe must be non-empty")
        if len(self.values) != len(TEMPORAL_OBSERVATION_NAMES):
            raise ValueError(
                f"temporal observation must contain {len(TEMPORAL_OBSERVATION_NAMES)} values"
            )

        normalized = tuple(float(value) for value in self.values)
        for name, value in zip(TEMPORAL_OBSERVATION_NAMES, normalized, strict=True):
            if not isfinite(value):
                raise ValueError(f"temporal observation {name!r} must be finite")
            low = -1.0 if name in SIGNED_TEMPORAL_OBSERVATIONS else 0.0
            if value < low or value > 1.0:
                raise ValueError(
                    f"temporal observation {name!r} must be within [{low}, 1.0]"
                )
        object.__setattr__(self, "values", normalized)

        seen = set()
        for item in self.provenance:
            if item.timeframe in seen:
                raise ValueError(f"duplicate timeframe provenance: {item.timeframe}")
            seen.add(item.timeframe)
            if item.available_ts_ns > self.asof_ts_ns:
                raise ValueError(f"future timeframe provenance: {item.timeframe}")
            if item.processed_sequence > self.decision_sequence:
                raise ValueError(f"future processed sequence: {item.timeframe}")
