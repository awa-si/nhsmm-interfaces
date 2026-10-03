from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from nautilus_trader.common import DataActor
from nautilus_trader.config import DataActorConfig
from nautilus_trader.model import Bar, BarType, CustomData, DataType
from nhsmm import HSMMFilterRuntime

from ..base import StateEstimate
from .bar import NautilusBarAdapter, PROTOTYPE_BAR_FIELDS, state_data_fields


NHSMM_STATE_DATA_TYPE = DataType(
    "NHSMMStateData",
    metadata={"schema": "prototype-v1"},
)


@dataclass(frozen=True, slots=True)
class NHSMMStateData:
    """Structured, policy-free NHSMM state published into Nautilus."""

    instrument_id: str
    bar_type: str
    state: int | None
    posterior: tuple[float, ...]
    age_posterior: tuple[float, ...] | None
    ts_event: int
    ts_init: int


class NHSMMDataActorConfig(DataActorConfig):
    """Prototype config for one ordered Nautilus bar stream."""

    def __init__(
        self,
        *,
        bar_type: BarType,
        feature_fields: Iterable[str] = PROTOTYPE_BAR_FIELDS,
        **_kwargs,
    ) -> None:
        super().__init__()
        self.bar_type = bar_type
        self.feature_fields = tuple(feature_fields)
        if not self.feature_fields:
            raise ValueError("feature_fields must not be empty")


class NHSMMDataActor(DataActor):
    """Own one NHSMM streaming runtime for one ordered Nautilus bar stream.

    Prototype boundary: the runtime is injected by the application bootstrap.
    Artifact loading and warm-up/history requests are intentionally deferred.
    """

    def __new__(
        cls,
        config: NHSMMDataActorConfig,
        runtime: HSMMFilterRuntime,
    ):
        # DataActor is a PyO3 type in NautilusTrader v2. Its native constructor
        # must receive only the actor config; runtime remains Python-owned state.
        return super().__new__(cls, config)

    def __init__(
        self,
        config: NHSMMDataActorConfig,
        runtime: HSMMFilterRuntime,
    ) -> None:
        self.runtime = runtime
        self.adapter = NautilusBarAdapter(
            runtime,
            feature_fields=config.feature_fields,
        )

    def on_start(self) -> None:
        self.subscribe_bars(self.config.bar_type)

    def on_stop(self) -> None:
        self.unsubscribe_bars(self.config.bar_type)

    def on_reset(self) -> None:
        self.runtime.reset()

    def on_bar(self, bar: Bar) -> None:
        if bar.bar_type != self.config.bar_type:
            return
        estimate = self.adapter.step(bar)
        if not isinstance(estimate, StateEstimate):
            raise TypeError("NautilusBarAdapter must return StateEstimate")
        payload = self._to_state_data(estimate)
        custom = CustomData(NHSMM_STATE_DATA_TYPE, payload)
        self.publish_data(NHSMM_STATE_DATA_TYPE, custom)

    @staticmethod
    def _to_state_data(state: StateEstimate) -> NHSMMStateData:
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
