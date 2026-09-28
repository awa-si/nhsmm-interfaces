from adapters import Context, Observation, StateEstimate, UniversalAdapter


class DemoAdapter(UniversalAdapter):
    def to_observation(self, event):
        return Observation(
            values=(event["close"], event["volume"]),
            timestamp=event["timestamp"],
            instrument=event["instrument"],
        )

    def to_context(self, event):
        return Context(values=(event["session"],))

    def infer(self, observation, context=None):
        assert tuple(observation.values) == (101.5, 7.0)
        assert tuple(context.values) == (1.0,)
        return StateEstimate(
            state=2,
            posterior=(0.1, 0.2, 0.7),
            age_posterior=(0.05, 0.15, 0.3, 0.5),
            timestamp=observation.timestamp,
        )

    def from_state(self, state):
        return {
            "state": state.state,
            "confidence": max(state.posterior),
            "age_mode": max(
                range(len(state.age_posterior)),
                key=state.age_posterior.__getitem__,
            )
            + 1,
        }


def test_universal_adapter_step_pipeline():
    adapter = DemoAdapter()
    result = adapter.step(
        {
            "close": 101.5,
            "volume": 7.0,
            "timestamp": 123,
            "instrument": "BTCUSDT",
            "session": 1.0,
        }
    )

    assert result == {"state": 2, "confidence": 0.7, "age_mode": 4}
