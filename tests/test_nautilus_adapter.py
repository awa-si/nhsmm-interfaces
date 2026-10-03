import sys
import types

import pytest


def _ensure_model_stubs():
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        torch = types.ModuleType("torch")
        torch.float32 = object()
        torch.as_tensor = lambda *args, **kwargs: None
        torch.argmax = lambda *args, **kwargs: None
        sys.modules["torch"] = torch

    try:
        import nhsmm  # noqa: F401
    except ModuleNotFoundError:
        nhsmm = types.ModuleType("nhsmm")

        class HSMMFilterRuntime:
            def __init__(self):
                self.reset_calls = 0

            def reset(self):
                self.reset_calls += 1

        nhsmm.HSMMFilterRuntime = HSMMFilterRuntime
        sys.modules["nhsmm"] = nhsmm


_ensure_model_stubs()

from adapters.base import StateEstimate
from adapters.nautilus import (
    NHSMMDataActor,
    NHSMMDataActorConfig,
    NHSMMStateData,
    NHSMM_STATE_DATA_TYPE,
    TEMPORAL_OBSERVATION_CONTRACT,
    TEMPORAL_OBSERVATION_DATA_TYPE,
    TEMPORAL_OBSERVATION_NAMES,
    TemporalObservationData,
    TimeframeProvenance,
)
from nhsmm import HSMMFilterRuntime
from nautilus_trader.model import ActorId, CustomData


def _temporal_values():
    signed = {
        "1h_direction",
        "15m_direction",
        "15m_flow",
        "5m_direction",
        "5m_volatility_accel",
        "5m_activity_accel",
    }
    return tuple(-0.25 if name in signed else 0.5 for name in TEMPORAL_OBSERVATION_NAMES)


def test_temporal_observation_exposes_nautilus_timestamps():
    payload = TemporalObservationData(
        instrument_id="BTCUSDT.BINANCE",
        values=_temporal_values(),
        asof_ts_ns=123,
        decision_sequence=7,
        trigger_timeframe="5m",
        provenance=(TimeframeProvenance("5m", 120, 123, 7),),
    )
    assert payload.ts_event == 123
    assert payload.ts_init == 123
    wrapped = CustomData(TEMPORAL_OBSERVATION_DATA_TYPE, payload)
    assert wrapped.ts_event == 123
    assert wrapped.ts_init == 123


def test_actor_config_preserves_native_data_actor_fields():
    config = NHSMMDataActorConfig(
        actor_id=ActorId("NHSMM-001"),
        log_events=False,
    )
    assert str(config.actor_id) == "NHSMM-001"
    assert config.log_events is False
    assert config.consume_temporal_observations is True


def test_actor_consumes_temporal_custom_data_and_publishes_state():
    runtime = HSMMFilterRuntime()
    actor = NHSMMDataActor(NHSMMDataActorConfig(), runtime)

    payload = TemporalObservationData(
        instrument_id="BTCUSDT.BINANCE",
        values=_temporal_values(),
        asof_ts_ns=123,
        decision_sequence=7,
        trigger_timeframe="5m",
    )
    estimate = StateEstimate(
        state=2,
        posterior=(0.1, 0.2, 0.7),
        age_posterior=(0.4, 0.6),
        timestamp=123,
        metadata={
            "instrument": payload.instrument_id,
            "observation_contract": TEMPORAL_OBSERVATION_CONTRACT,
            "decision_sequence": payload.decision_sequence,
            "trigger_timeframe": payload.trigger_timeframe,
        },
    )
    actor.temporal_adapter.step = lambda event: estimate
    published = []
    actor.publish_data = lambda data_type, data: published.append((data_type, data))

    actor.on_data(CustomData(TEMPORAL_OBSERVATION_DATA_TYPE, payload))

    assert len(published) == 1
    data_type, wrapped = published[0]
    assert data_type == NHSMM_STATE_DATA_TYPE
    assert wrapped.data_type == NHSMM_STATE_DATA_TYPE
    assert isinstance(wrapped.data, NHSMMStateData)
    assert wrapped.data.instrument_id == payload.instrument_id
    assert wrapped.data.state == 2
    assert wrapped.data.posterior == pytest.approx((0.1, 0.2, 0.7))
    assert wrapped.data.observation_contract == TEMPORAL_OBSERVATION_CONTRACT
    assert wrapped.data.decision_sequence == 7
    assert wrapped.data.trigger_timeframe == "5m"
    assert wrapped.data.ts_event == 123
    assert wrapped.data.ts_init == 123


def test_actor_reset_delegates_to_runtime():
    runtime = HSMMFilterRuntime()
    actor = NHSMMDataActor(NHSMMDataActorConfig(), runtime)
    actor.on_reset()
    assert runtime.reset_calls == 1
