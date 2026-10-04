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
    BAR_FIELDS,
    BarAdapter,
    NHSMMDataActor,
    NHSMMDataActorConfig,
    NHSMMStateData,
    NHSMM_STATE_DATA_TYPE,
    TEMPORAL_OBSERVATION_CONTRACT,
    TEMPORAL_OBSERVATION_DATA_TYPE,
    TEMPORAL_OBSERVATION_NAMES,
    TemporalAdapter,
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


def test_actor_lifecycle_subscribes_and_unsubscribes_temporal_data():
    runtime = HSMMFilterRuntime()
    actor = NHSMMDataActor(NHSMMDataActorConfig(), runtime)
    calls = []
    actor.subscribe_data = lambda data_type: calls.append(("subscribe", data_type))
    actor.unsubscribe_data = lambda data_type: calls.append(("unsubscribe", data_type))

    actor.on_start()
    actor.on_stop()

    assert calls == [
        ("subscribe", TEMPORAL_OBSERVATION_DATA_TYPE),
        ("unsubscribe", TEMPORAL_OBSERVATION_DATA_TYPE),
    ]


def test_actor_ignores_unrelated_custom_data():
    from nautilus_trader.model import DataType

    runtime = HSMMFilterRuntime()
    actor = NHSMMDataActor(NHSMMDataActorConfig(), runtime)
    called = []
    actor.temporal_adapter.step = lambda event: called.append(event)

    payload = TemporalObservationData(
        instrument_id="BTCUSDT.BINANCE",
        values=_temporal_values(),
        asof_ts_ns=123,
        decision_sequence=7,
        trigger_timeframe="5m",
    )
    other_type = DataType("OtherData", metadata={"schema": "other-v1"})
    actor.on_data(CustomData(other_type, payload))

    assert called == []


def test_actor_rejects_wrong_temporal_payload_type():
    runtime = HSMMFilterRuntime()
    actor = NHSMMDataActor(NHSMMDataActorConfig(), runtime)

    class WrongPayload:
        ts_event = 123
        ts_init = 123

    with pytest.raises(TypeError, match="TemporalObservationData"):
        actor.on_data(CustomData(TEMPORAL_OBSERVATION_DATA_TYPE, WrongPayload()))


def test_actor_does_not_publish_after_inference_failure():
    runtime = HSMMFilterRuntime()
    actor = NHSMMDataActor(NHSMMDataActorConfig(), runtime)
    payload = TemporalObservationData(
        instrument_id="BTCUSDT.BINANCE",
        values=_temporal_values(),
        asof_ts_ns=123,
        decision_sequence=7,
        trigger_timeframe="5m",
    )
    actor.temporal_adapter.step = lambda event: (_ for _ in ()).throw(RuntimeError("boom"))
    published = []
    actor.publish_data = lambda data_type, data: published.append((data_type, data))

    with pytest.raises(RuntimeError, match="boom"):
        actor.on_data(CustomData(TEMPORAL_OBSERVATION_DATA_TYPE, payload))

    assert published == []


def test_temporal_observation_rejects_future_provenance():
    with pytest.raises(ValueError, match="future timeframe provenance"):
        TemporalObservationData(
            instrument_id="BTCUSDT.BINANCE",
            values=_temporal_values(),
            asof_ts_ns=123,
            decision_sequence=7,
            trigger_timeframe="5m",
            provenance=(TimeframeProvenance("5m", 124, 124, 7),),
        )


def test_actor_config_rejects_no_input_path():
    with pytest.raises(ValueError, match="must enable temporal input"):
        NHSMMDataActorConfig(
            consume_temporal_observations=False,
            bar_type=None,
        )


def _training_observation(ts: int, sequence: int, *, instrument: str = "BTCUSDT.BINANCE"):
    return TemporalObservationData(
        instrument_id=instrument,
        values=_temporal_values(),
        asof_ts_ns=ts,
        decision_sequence=sequence,
        trigger_timeframe="5m",
        provenance=(TimeframeProvenance("5m", ts, ts, sequence),),
    )


def test_temporal_fold_builder_preserves_values_and_real_boundaries():
    import torch

    from adapters.nautilus import TemporalFoldBuilder

    observations = [
        _training_observation(100, 1),
        _training_observation(200, 2),
        _training_observation(300, 3),
        _training_observation(400, 4),
    ]

    fold = TemporalFoldBuilder().build_fold(
        observations,
        label="fold-1",
        train_through_ns=250,
        oos_through_ns=400,
    )

    assert fold.train_end_ns == 200
    assert fold.oos_start_ns == 300
    assert fold.train.shape == (2, len(TEMPORAL_OBSERVATION_NAMES))
    assert fold.oos.shape == (2, len(TEMPORAL_OBSERVATION_NAMES))
    assert fold.train.dtype == torch.float32
    assert tuple(fold.train[0].tolist()) == pytest.approx(observations[0].values)
    assert tuple(fold.oos[-1].tolist()) == pytest.approx(observations[-1].values)


def test_temporal_fold_builder_rejects_non_monotonic_identity():
    from adapters.nautilus import TemporalFoldBuilder

    observations = [
        _training_observation(200, 1),
        _training_observation(100, 2),
    ]

    with pytest.raises(ValueError, match="asof_ts_ns must be strictly increasing"):
        TemporalFoldBuilder().build_fold(
            observations,
            label="bad",
            train_through_ns=150,
        )


def test_temporal_fold_builder_rejects_cross_instrument_stream():
    from adapters.nautilus import TemporalFoldBuilder

    observations = [
        _training_observation(100, 1),
        _training_observation(200, 2, instrument="ETHUSDT.BINANCE"),
    ]

    with pytest.raises(ValueError, match="one instrument"):
        TemporalFoldBuilder().build_fold(
            observations,
            label="bad",
            train_through_ns=100,
        )


def test_temporal_fold_builder_rejects_trigger_provenance_mismatch():
    from adapters.nautilus import TemporalFoldBuilder

    bad = TemporalObservationData(
        instrument_id="BTCUSDT.BINANCE",
        values=_temporal_values(),
        asof_ts_ns=200,
        decision_sequence=2,
        trigger_timeframe="5m",
        provenance=(TimeframeProvenance("5m", 199, 199, 1),),
    )
    observations = [_training_observation(100, 1), bad]

    with pytest.raises(ValueError, match="trigger provenance must match"):
        TemporalFoldBuilder().build_fold(
            observations,
            label="bad",
            train_through_ns=100,
        )


def test_temporal_fold_builder_rejects_empty_train_or_oos_selection():
    from adapters.nautilus import TemporalFoldBuilder

    observations = [_training_observation(100, 1), _training_observation(200, 2)]
    builder = TemporalFoldBuilder()

    with pytest.raises(ValueError, match="training selection"):
        builder.build_fold(observations, label="bad-train", train_through_ns=50)

    with pytest.raises(ValueError, match="OOS selection"):
        builder.build_fold(observations, label="bad-oos", train_through_ns=200)


class _AxisPayload:
    def __init__(
        self,
        *,
        instrument_id="BTCUSDT-PERP.BINANCE",
        asof_ts_ns=200,
        decision_sequence=2,
        trigger_timeframe="5m",
        schema_version="axis-observation-v2",
    ):
        self.instrument_id = instrument_id
        self.ts_event = asof_ts_ns
        self.ts_init = asof_ts_ns
        self.asof_ts_ns = asof_ts_ns
        self.decision_sequence = decision_sequence
        self.trigger_timeframe = trigger_timeframe
        self.axis_by_tf = {}
        self.provenance_by_tf = {}
        self.schema_version = schema_version


def test_axis_observation_collector_subscribes_to_custom_data_contract():
    from adapters.nautilus import AXIS_OBSERVATION_DATA_TYPE, AxisObservationCollector

    actor = AxisObservationCollector()
    calls = []
    actor.subscribe_data = lambda data_type: calls.append(("subscribe", data_type))
    actor.unsubscribe_data = lambda data_type: calls.append(("unsubscribe", data_type))

    actor.on_start()
    actor.on_stop()

    assert calls == [
        ("subscribe", AXIS_OBSERVATION_DATA_TYPE),
        ("unsubscribe", AXIS_OBSERVATION_DATA_TYPE),
    ]


def test_axis_observation_collector_preserves_payload_without_mapping():
    from adapters.nautilus import AXIS_OBSERVATION_DATA_TYPE, AxisObservationCollector

    actor = AxisObservationCollector()
    payload = _AxisPayload()

    actor.on_data(CustomData(AXIS_OBSERVATION_DATA_TYPE, payload))

    assert actor.observations == (payload,)


def test_axis_observation_collector_filters_trigger_timeframe_and_instrument():
    from adapters.nautilus import AXIS_OBSERVATION_DATA_TYPE, AxisObservationCollector

    actor = AxisObservationCollector(instrument_id="BTCUSDT-PERP.BINANCE")
    actor.on_data(
        CustomData(
            AXIS_OBSERVATION_DATA_TYPE,
            _AxisPayload(trigger_timeframe="1m", asof_ts_ns=100, decision_sequence=1),
        )
    )
    actor.on_data(
        CustomData(
            AXIS_OBSERVATION_DATA_TYPE,
            _AxisPayload(instrument_id="ETHUSDT-PERP.BINANCE", asof_ts_ns=150, decision_sequence=2),
        )
    )
    selected = _AxisPayload(asof_ts_ns=200, decision_sequence=3)
    actor.on_data(CustomData(AXIS_OBSERVATION_DATA_TYPE, selected))

    assert actor.observations == (selected,)


def test_axis_observation_collector_rejects_non_monotonic_delivery():
    from adapters.nautilus import AXIS_OBSERVATION_DATA_TYPE, AxisObservationCollector

    actor = AxisObservationCollector()
    actor.on_data(
        CustomData(
            AXIS_OBSERVATION_DATA_TYPE,
            _AxisPayload(asof_ts_ns=200, decision_sequence=2),
        )
    )

    with pytest.raises(ValueError, match="strictly ordered"):
        actor.on_data(
            CustomData(
                AXIS_OBSERVATION_DATA_TYPE,
                _AxisPayload(asof_ts_ns=200, decision_sequence=3),
            )
        )


def test_axis_observation_collector_reset_clears_replay_state():
    from adapters.nautilus import AXIS_OBSERVATION_DATA_TYPE, AxisObservationCollector

    actor = AxisObservationCollector()
    payload = _AxisPayload()
    actor.on_data(CustomData(AXIS_OBSERVATION_DATA_TYPE, payload))

    actor.on_reset()

    assert actor.observations == ()
