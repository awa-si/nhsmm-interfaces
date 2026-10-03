import pytest

from adapters import StructuredEventAdapter
from nhsmm import HSMMFilterRuntime


def _adapter(**kwargs):
    return StructuredEventAdapter(
        HSMMFilterRuntime(model=None),
        feature_fields=("feature_a", "feature_b"),
        **kwargs,
    )


def test_structured_adapter_maps_event():
    adapter = _adapter(
        context_fields=("priority",),
        metadata_fields=("event_type", "source_id"),
    )

    event = {
        "event_type": "processed",
        "source_id": "SRC-001",
        "timestamp": 123,
        "features": {
            "feature_a": 0.7,
            "feature_b": 0.9,
        },
        "context": {
            "priority": 2.0,
        },
    }

    observation = adapter.to_observation(event)
    context = adapter.to_context(event)

    assert observation.values == (0.7, 0.9)
    assert observation.timestamp == 123
    assert observation.metadata == {
        "event_type": "processed",
        "source_id": "SRC-001",
    }
    assert context is not None
    assert context.values == (2.0,)
    assert context.metadata == observation.metadata


def test_structured_adapter_without_external_context():
    adapter = _adapter()

    event = {
        "features": {
            "feature_a": 0.2,
            "feature_b": 0.8,
        }
    }

    assert adapter.to_context(event) is None


@pytest.mark.parametrize(
    ("features", "error"),
    [
        ({"feature_a": 0.2}, KeyError),
        ({"feature_a": float("nan"), "feature_b": 0.8}, ValueError),
        ({"feature_a": True, "feature_b": 0.8}, TypeError),
    ],
)
def test_structured_adapter_rejects_invalid_features(features, error):
    adapter = _adapter()

    with pytest.raises(error):
        adapter.to_observation({"features": features})


def test_structured_adapter_requires_feature_payload():
    adapter = _adapter()

    with pytest.raises(KeyError, match="features"):
        adapter.to_observation({"event_type": "processed"})


def test_structured_adapter_rejects_duplicate_metadata_fields():
    with pytest.raises(ValueError, match="metadata_fields"):
        _adapter(metadata_fields=("source_id", "source_id"))
