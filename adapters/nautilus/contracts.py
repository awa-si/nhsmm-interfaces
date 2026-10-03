from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


# Canonical Nautilus<->NHSMM bridge schemas.
TEMPORAL_OBSERVATION_CONTRACT = "nautilus-temporal-observations-v1"
TEMPORAL_OBSERVATION_DATA_TYPE_NAME = "NHSMMTemporalObservationData"
NHSMM_STATE_DATA_SCHEMA = "nautilus-nhsmm-state-v1"

# Copied from the frozen Nautilus research input contract at
# awa-si/nautilus@3947be02c6b98198813c613aee1c5b2ea27aaf31.
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
    """Primary Nautilus-facing NHSMM input contract.

    The consumer produces the already-admitted 18-coordinate observation.
    The bridge validates shape/ranges and causal provenance only; TA/Axis
    construction, freshness policy and trading admission remain outside.
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

    @property
    def ts_event(self) -> int:
        return self.asof_ts_ns

    @property
    def ts_init(self) -> int:
        return self.asof_ts_ns


@dataclass(frozen=True, slots=True)
class NHSMMArtifactIdentity:
    """Optional bridge compatibility identity for a loaded NHSMM artifact.

    This is not an artifact loader or artifact-format contract. It records only
    the dimensions/observation identity the Nautilus bridge may need to reject
    incompatible composition.
    """

    artifact_id: str
    n_features: int
    n_states: int
    max_duration: int
    observation_contract: str = TEMPORAL_OBSERVATION_CONTRACT

    def __post_init__(self) -> None:
        if not self.artifact_id:
            raise ValueError("artifact_id must be non-empty")
        if self.observation_contract != TEMPORAL_OBSERVATION_CONTRACT:
            raise ValueError("unsupported NHSMM observation contract")
        if self.n_features != len(TEMPORAL_OBSERVATION_NAMES):
            raise ValueError("NHSMM artifact must match the 18-coordinate observation contract")
        if self.n_states < 1 or self.max_duration < 1:
            raise ValueError("NHSMM artifact dimensions must be positive")


@dataclass(frozen=True, slots=True)
class NHSMMForecastData:
    """Optional future bridge payload for policy-free NHSMM forecast channels.

    The current DataActor does not publish this payload yet.
    """

    next_state_prior: tuple[float, ...]
    episode_end_probability: float
    state_change_probability: float
    horizons: tuple[int, ...] = ()
    survival_probability: tuple[float, ...] = ()
    end_within_probability: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        prior = tuple(float(value) for value in self.next_state_prior)
        if not prior or any(not isfinite(v) or v < 0.0 or v > 1.0 for v in prior):
            raise ValueError("next_state_prior must be a non-empty probability vector")
        if abs(sum(prior) - 1.0) > 1e-6:
            raise ValueError("next_state_prior must sum to one")
        object.__setattr__(self, "next_state_prior", prior)

        for name in ("episode_end_probability", "state_change_probability"):
            value = float(getattr(self, name))
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")
            object.__setattr__(self, name, value)

        horizons = tuple(int(value) for value in self.horizons)
        if any(value <= 0 for value in horizons) or tuple(sorted(set(horizons))) != horizons:
            raise ValueError("horizons must be strictly increasing positive integers")
        object.__setattr__(self, "horizons", horizons)

        survival = tuple(float(value) for value in self.survival_probability)
        end_within = tuple(float(value) for value in self.end_within_probability)
        if len(survival) != len(horizons) or len(end_within) != len(horizons):
            raise ValueError("forecast vectors must match horizons")
        if any(not isfinite(v) or not 0.0 <= v <= 1.0 for v in survival + end_within):
            raise ValueError("forecast probabilities must be within [0, 1]")
        object.__setattr__(self, "survival_probability", survival)
        object.__setattr__(self, "end_within_probability", end_within)
