import torch

from adapters import NHSMMRuntimeAdapter, Observation
from nhsmm import HSMMFilterRuntime, HSMMFilterState


class DemoNHSMMAdapter(NHSMMRuntimeAdapter):
    def to_observation(self, event):
        return Observation(
            values=event["values"],
            timestamp=event.get("timestamp"),
            instrument=event.get("instrument"),
        )


def test_nhsmm_runtime_adapter_maps_public_filter_state(monkeypatch):
    runtime = HSMMFilterRuntime(model=None)

    joint = torch.tensor(
        [[[0.10, 0.20, 0.00], [0.05, 0.15, 0.50]]],
        dtype=torch.float32,
    )
    filter_state = HSMMFilterState(joint.log())

    def fake_step(observation, *, context=None, timestamp=None):
        assert tuple(observation.tolist()) == (1.0, 2.0)
        assert context is None
        assert timestamp == 123
        return filter_state

    monkeypatch.setattr(runtime, "step", fake_step)

    adapter = DemoNHSMMAdapter(runtime)
    result = adapter.step(
        {
            "values": (1.0, 2.0),
            "timestamp": 123,
            "instrument": "BTCUSDT",
        }
    )

    assert result.state == 1
    assert result.posterior == (0.3, 0.7)
    assert result.age_posterior == (0.15, 0.35, 0.5)
    assert result.timestamp == 123
    assert result.metadata["instrument"] == "BTCUSDT"
