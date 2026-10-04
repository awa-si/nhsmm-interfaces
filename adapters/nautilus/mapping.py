from __future__ import annotations

from dataclasses import dataclass

from nautilus_trader.common import DataActor
from nautilus_trader.model import CustomData

from .actor import TEMPORAL_OBSERVATION_DATA_TYPE
from .axis import AXIS_OBSERVATION_DATA_TYPE, AxisObservationCollector
from .contracts import TemporalObservationData, TimeframeProvenance


AXIS_TEMPORAL_MAPPING_CONTRACT = "axis-observation-v2-to-nautilus-temporal-observations-v1"
AXIS_TEMPORAL_MAPPING_FIELDS = (
    ("1h", "direction"),
    ("1h", "efficiency"),
    ("1h", "persistence"),
    ("1h", "volatility"),
    ("1h", "compression"),
    ("1h", "activity"),
    ("15m", "direction"),
    ("15m", "efficiency"),
    ("15m", "persistence"),
    ("15m", "volatility"),
    ("15m", "compression"),
    ("15m", "activity"),
    ("15m", "participation"),
    ("15m", "flow"),
    ("5m", "direction"),
    ("5m", "volatility_accel"),
    ("5m", "activity_accel"),
    ("1m", "shock"),
)


@dataclass(frozen=True, slots=True)
class AxisTemporalMapper:
    """Explicit versioned mapping from AxisObservation v2 to temporal v1."""

    mapping_contract: str = AXIS_TEMPORAL_MAPPING_CONTRACT

    def to_temporal(self, observation: object) -> TemporalObservationData:
        if getattr(observation, "trigger_timeframe", None) != "5m":
            raise ValueError("Axis temporal mapping requires a 5m trigger observation")
        AxisObservationCollector._validate_observation(observation)
        if self.mapping_contract != AXIS_TEMPORAL_MAPPING_CONTRACT:
            raise ValueError("unsupported Axis temporal mapping contract")

        values = []
        for timeframe, field in AXIS_TEMPORAL_MAPPING_FIELDS:
            state = observation.axis_by_tf[timeframe]
            if not hasattr(state, field):
                raise TypeError(
                    f"Axis state {timeframe!r} is missing mapped field {field!r}"
                )
            values.append(float(getattr(state, field)))

        provenance = tuple(
            self._provenance(timeframe, observation)
            for timeframe in ("1h", "15m", "5m", "1m")
        )
        return TemporalObservationData(
            instrument_id=str(observation.instrument_id),
            values=tuple(values),
            asof_ts_ns=int(observation.asof_ts_ns),
            decision_sequence=int(observation.decision_sequence),
            trigger_timeframe="5m",
            provenance=provenance,
            mapping_contract=self.mapping_contract,
        )

    @staticmethod
    def _provenance(timeframe: str, observation: object) -> TimeframeProvenance:
        source = observation.provenance_by_tf[timeframe]
        available = max(int(source.source_ts_event_ns), int(source.source_ts_init_ns))
        return TimeframeProvenance(
            timeframe=timeframe,
            source_ts_event_ns=int(source.source_ts_event_ns),
            source_ts_init_ns=int(source.source_ts_init_ns),
            processed_sequence=int(source.processed_sequence),
            timeframe_bar_count=(
                None
                if not hasattr(source, "timeframe_bar_count")
                else source.timeframe_bar_count
            ),
            age_ns=int(observation.asof_ts_ns) - available,
        )


class AxisTemporalDataActor(DataActor):
    """Bridge canonical Nautilus AxisObservation CustomData to temporal input."""

    def __new__(cls, mapper: AxisTemporalMapper | None = None):
        return super().__new__(cls)

    def __init__(self, mapper: AxisTemporalMapper | None = None) -> None:
        super().__init__()
        self.mapper = mapper or AxisTemporalMapper()
        self.latest: TemporalObservationData | None = None

    def on_start(self) -> None:
        self.subscribe_data(AXIS_OBSERVATION_DATA_TYPE)

    def on_stop(self) -> None:
        self.unsubscribe_data(AXIS_OBSERVATION_DATA_TYPE)

    def on_reset(self) -> None:
        self.latest = None

    def on_data(self, data: CustomData) -> None:
        if data.data_type != AXIS_OBSERVATION_DATA_TYPE:
            return
        observation = data.data
        if getattr(observation, "trigger_timeframe", None) != "5m":
            return
        temporal = self.mapper.to_temporal(observation)
        self.latest = temporal
        self.publish_data(
            TEMPORAL_OBSERVATION_DATA_TYPE,
            CustomData(TEMPORAL_OBSERVATION_DATA_TYPE, temporal),
        )
