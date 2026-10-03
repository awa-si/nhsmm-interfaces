import pytest

from adapters import StructuredEventAdapter
from nhsmm import HSMMFilterRuntime


def _adapter(**kwargs):
    return StructuredEventAdapter(
        HSMMFilterRuntime(model=None),
        feature_fields=("disease_burden", "document_completeness"),
        **kwargs,
    )


def test_research_adapter_maps_structured_worker_output():
    adapter = _adapter(context_fields=("review_priority",))

    event = {
        "public_ref": "AC-A82XK9Q4",
        "event_type": "document_processed",
        "case_state": "under_review",
        "document_type": "lab_result",
        "assessment_type": "research",
        "ai_status": "completed",
        "timestamp": 123,
        "features": {
            "disease_burden": 0.7,
            "document_completeness": 0.9,
        },
        "context": {
            "review_priority": 2.0,
        },
    }

    observation = adapter.to_observation(event)
    context = adapter.to_context(event)

    assert observation.values == (0.7, 0.9)
    assert observation.timestamp == 123
    assert observation.metadata == {
        "public_ref": "AC-A82XK9Q4",
        "event_type": "document_processed",
        "case_state": "under_review",
        "document_type": "lab_result",
        "assessment_type": "research",
        "ai_status": "completed",
    }
    assert context is not None
    assert context.values == (2.0,)


def test_research_adapter_internal_context_mode():
    adapter = _adapter()

    event = {
        "features": {
            "disease_burden": 0.2,
            "document_completeness": 0.8,
        }
    }

    assert adapter.to_context(event) is None


@pytest.mark.parametrize(
    ("features", "error"),
    [
        ({"disease_burden": 0.2}, KeyError),
        ({"disease_burden": float("nan"), "document_completeness": 0.8}, ValueError),
        ({"disease_burden": True, "document_completeness": 0.8}, TypeError),
    ],
)
def test_research_adapter_rejects_invalid_structured_features(features, error):
    adapter = _adapter()

    with pytest.raises(error):
        adapter.to_observation({"features": features})


def test_research_adapter_requires_structured_feature_payload():
    adapter = _adapter()

    with pytest.raises(KeyError, match="features"):
        adapter.to_observation({"public_ref": "AC-A82XK9Q4"})
