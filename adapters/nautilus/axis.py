from __future__ import annotations

from collections.abc import Iterable

from nautilus_trader.common import DataActor
from nautilus_trader.model import CustomData, DataType


AXIS_OBSERVATION_SCHEMA = "axis-observation-v2"
AXIS_OBSERVATION_DATA_TYPE_NAME = "AxisObservation"
AXIS_OBSERVATION_TFS = ("1m", "5m", "15m", "1h")
AXIS_OBSERVATION_DATA_TYPE = DataType(
    AXIS_OBSERVATION_DATA_TYPE_NAME,
    metadata={"schema": AXIS_OBSERVATION_SCHEMA},
)


class AxisObservationCollector(DataActor):
    """Collect causal Nautilus AxisObservation CustomData for replay/training.

    The collector intentionally preserves the producer payload unchanged. It
    does not derive NHSMM features or depend on the Nautilus consumer package's
    Python classes; the versioned CustomData schema is the cross-repository
    boundary.
    """

    def __new__(
        cls,
        *,
        trigger_timeframes: Iterable[str] = ("5m",),
        instrument_id: str | None = None,
    ):
        return super().__new__(cls)

    def __init__(
        self,
        *,
        trigger_timeframes: Iterable[str] = ("5m",),
        instrument_id: str | None = None,
    ) -> None:
        super().__init__()
        timeframes = tuple(str(value) for value in trigger_timeframes)
        if not timeframes or any(not value for value in timeframes):
            raise ValueError("trigger_timeframes must contain non-empty values")
        if len(set(timeframes)) != len(timeframes):
            raise ValueError("trigger_timeframes must be unique")
        if instrument_id is not None and not str(instrument_id):
            raise ValueError("instrument_id must be non-empty when provided")
        self.trigger_timeframes = timeframes
        self.instrument_id = None if instrument_id is None else str(instrument_id)
        self._observations: list[object] = []
        self._last_identity_by_instrument: dict[str, tuple[int, int]] = {}

    @property
    def observations(self) -> tuple[object, ...]:
        return tuple(self._observations)

    def on_start(self) -> None:
        self.subscribe_data(AXIS_OBSERVATION_DATA_TYPE)

    def on_stop(self) -> None:
        self.unsubscribe_data(AXIS_OBSERVATION_DATA_TYPE)

    def on_reset(self) -> None:
        self._observations.clear()
        self._last_identity_by_instrument.clear()

    def on_data(self, data: CustomData) -> None:
        if data.data_type != AXIS_OBSERVATION_DATA_TYPE:
            return
        observation = data.data
        identity = self._validate_observation(observation)
        if observation.trigger_timeframe not in self.trigger_timeframes:
            return
        if self.instrument_id is not None and observation.instrument_id != self.instrument_id:
            return

        previous = self._last_identity_by_instrument.get(observation.instrument_id)
        if previous is not None and (
            identity[0] <= previous[0] or identity[1] <= previous[1]
        ):
            raise ValueError("axis observations must be strictly ordered per instrument")
        self._last_identity_by_instrument[observation.instrument_id] = identity
        self._observations.append(observation)

    @staticmethod
    def _validate_observation(observation: object) -> tuple[int, int]:
        required = (
            "instrument_id",
            "ts_event",
            "ts_init",
            "asof_ts_ns",
            "decision_sequence",
            "trigger_timeframe",
            "axis_by_tf",
            "provenance_by_tf",
            "schema_version",
        )
        missing = tuple(name for name in required if not hasattr(observation, name))
        if missing:
            raise TypeError(
                "axis CustomData payload is missing required fields: "
                + ", ".join(missing)
            )
        if observation.schema_version != AXIS_OBSERVATION_SCHEMA:
            raise ValueError("unsupported axis observation schema")
        if not str(observation.instrument_id):
            raise ValueError("axis observation instrument_id must be non-empty")

        ts_event = int(observation.ts_event)
        ts_init = int(observation.ts_init)
        asof = int(observation.asof_ts_ns)
        sequence = int(observation.decision_sequence)
        if min(ts_event, ts_init, asof, sequence) < 0:
            raise ValueError("axis observation identity must be non-negative")
        if ts_event > ts_init or ts_init > asof:
            raise ValueError("axis observation timestamps are not causally ordered")
        trigger_timeframe = str(observation.trigger_timeframe)
        if trigger_timeframe not in AXIS_OBSERVATION_TFS:
            raise ValueError("invalid axis observation trigger_timeframe")

        if tuple(observation.axis_by_tf) != AXIS_OBSERVATION_TFS:
            raise ValueError("axis observation requires canonical timeframe ordering")
        if tuple(observation.provenance_by_tf) != AXIS_OBSERVATION_TFS:
            raise ValueError("axis observation provenance requires canonical timeframe ordering")

        for timeframe in AXIS_OBSERVATION_TFS:
            source = observation.provenance_by_tf[timeframe]
            required_source = (
                "source_ts_event_ns",
                "source_ts_init_ns",
                "processed_sequence",
            )
            if any(not hasattr(source, name) for name in required_source):
                raise TypeError(f"invalid axis observation provenance: {timeframe}")
            available = max(int(source.source_ts_event_ns), int(source.source_ts_init_ns))
            processed = int(source.processed_sequence)
            if available > asof or processed > sequence:
                raise ValueError(f"axis observation contains unavailable state: {timeframe}")

        trigger = observation.provenance_by_tf[trigger_timeframe]
        if (
            int(trigger.source_ts_event_ns) != ts_event
            or int(trigger.source_ts_init_ns) != ts_init
            or int(trigger.processed_sequence) != sequence
        ):
            raise ValueError("axis observation trigger provenance mismatch")
        return asof, sequence
