from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from nautilus_trader.common import DataActor
from nautilus_trader.config import DataActorConfig
from nautilus_trader.model import Bar, BarType, CustomData, DataType
from nhsmm import HSMMFilterRuntime

from ..base import StateEstimate
from .bar import NautilusBarAdapter, PROTOTYPE_BAR_FIELDS, state_data_fields
from .contracts import (
    NHSMM_STATE_DATA_SCHEMA,
    TEMPORAL_OBSERVATION_CONTRACT,
    TEMPORAL_OBSERVATION_DATA_TYPE_NAME,
    TemporalObservationData,
)
from .temporal import NautilusTemporalAdapter, temporal_state_fields


TEMPORAL_OBSERVATION_DATA_TYPE = DataType(
    TEMPORAL_OBSERVATION_DATA_TYPE_NAME,
    metadata={"schema": TEMPORAL_OBSERVATION_CONTRACT},
)

NHSMM_STATE_DATA_TYPE = DataType(
    "NHSMMStateData",
    metadata={"schema": NHSMM_STATE_DATA_SCHEMA},
)


@dataclass(frozen=True, slots=True)
class NHSMMStateData:
    """Structured, policy-free NHSMM state published into Nautilus."""

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


class NHSMMDataActorConfig(DataActorConfig):
    """Config for one NHSMM runtime owned by one Nautilus DataActor."""

    def __init__(
        self,
        *,
        bar_type: BarType | None = None,
        feature_fields: Iterable[str] = PROTOTYPE_BAR_FIELDS,
        consume_temporal_observations: bool = True,
        **_kwargs,
    ) -> None:
        # DataActorConfig is a PyO3 type. Native fields such as actor_id,
        # log_events and log_commands are consumed by __new__ before this runs.
        super().__init__()
        self.bar_type = bar_type
        self.feature_fields = tuple(feature_fields)
        self.consume_temporal_observations = bool(consume_temporal_observations)
        if self.bar_type is not None and not self.feature_fields:
            raise ValueError("feature_fields must not be empty when bar input is enabled")
        if self.bar_type is None and not self.consume_temporal_observations:
            raise ValueError("actor must enable temporal input or a prototype bar input")


class NHSMMDataActor(DataActor):
    """Own one NHSMM streaming runtime behind Nautilus data contracts.

    The primary bridge consumes TemporalObservationData CustomData. The
    optional Bar path remains a prototype/framework fallback only.
    """

    def __new__(
        cls,
        config: NHSMMDataActorConfig,
        runtime: HSMMFilterRuntime,
    ):
        return super().__new__(cls, config)

    def __init__(
        self,
        config: NHSMMDataActorConfig,
        runtime: HSMMFilterRuntime,
    ) -> None:
        self.runtime = runtime
        self.temporal_adapter = NautilusTemporalAdapter(runtime)
        self.bar_adapter = (
            None
            if config.bar_type is None
            else NautilusBarAdapter(runtime, feature_fields=config.feature_fields)
        )

    def on_start(self) -> None:
        if self.config.consume_temporal_observations:
            self.subscribe_data(TEMPORAL_OBSERVATION_DATA_TYPE)
        if self.config.bar_type is not None:
            self.subscribe_bars(self.config.bar_type)

    def on_stop(self) -> None:
        if self.config.consume_temporal_observations:
            self.unsubscribe_data(TEMPORAL_OBSERVATION_DATA_TYPE)
        if self.config.bar_type is not None:
            self.unsubscribe_bars(self.config.bar_type)

    def on_reset(self) -> None:
        self.runtime.reset()

    def on_data(self, data: CustomData) -> None:
        if data.data_type != TEMPORAL_OBSERVATION_DATA_TYPE:
            return
        payload = data.data
        if not isinstance(payload, TemporalObservationData):
            raise TypeError("temporal CustomData payload must be TemporalObservationData")
        estimate = self.temporal_adapter.step(payload)
        if not isinstance(estimate, StateEstimate):
            raise TypeError("NautilusTemporalAdapter must return StateEstimate")
        self._publish_state(self._temporal_state_data(estimate))

    def on_bar(self, bar: Bar) -> None:
        if self.config.bar_type is None or bar.bar_type != self.config.bar_type:
            return
        if self.bar_adapter is None:
            raise RuntimeError("bar adapter is not configured")
        estimate = self.bar_adapter.step(bar)
        if not isinstance(estimate, StateEstimate):
            raise TypeError("NautilusBarAdapter must return StateEstimate")
        self._publish_state(self._bar_state_data(estimate))

    def _publish_state(self, payload: NHSMMStateData) -> None:
        custom = CustomData(NHSMM_STATE_DATA_TYPE, payload)
        self.publish_data(NHSMM_STATE_DATA_TYPE, custom)

    @staticmethod
    def _temporal_state_data(state: StateEstimate) -> NHSMMStateData:
        (
            instrument_id,
            ts_event,
            decision_sequence,
            trigger_timeframe,
            observation_contract,
        ) = temporal_state_fields(state)
        return NHSMMStateData(
            instrument_id=instrument_id,
            state=state.state,
            posterior=tuple(float(value) for value in state.posterior),
            age_posterior=(
                None
                if state.age_posterior is None
                else tuple(float(value) for value in state.age_posterior)
            ),
            ts_event=ts_event,
            ts_init=ts_event,
            observation_contract=observation_contract,
            decision_sequence=decision_sequence,
            trigger_timeframe=trigger_timeframe,
        )

    @staticmethod
    def _bar_state_data(state: StateEstimate) -> NHSMMStateData:
        instrument_id, bar_type, ts_event, ts_init = state_data_fields(state)
        return NHSMMStateData(
            instrument_id=instrument_id,
            bar_type=bar_type,
            state=state.state,
            posterior=tuple(float(value) for value in state.posterior),
            age_posterior=(
                None
                if state.age_posterior is None
                else tuple(float(value) for value in state.age_posterior)
            ),
            ts_event=ts_event,
            ts_init=ts_init,
        )
