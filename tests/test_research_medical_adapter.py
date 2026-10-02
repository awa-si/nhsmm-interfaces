import pytest

from adapters import ResearchMedicalAdapter
from nhsmm import HSMMFilterRuntime


def _adapter(**kwargs):
    return ResearchMedicalAdapter(
        HSMMFilterRuntime(model=None),
        feature_fields=("heart_rate", "spo2"),
        **kwargs,
    )


def test_research_medical_adapter_maps_explicit_schema():
    adapter = _adapter(
        context_fields=("activity_level",),
        metadata_fields=("site",),
    )

    event = {
        "subject_id": "subject-7",
        "sample_id": "sample-11",
        "timestamp": 123,
        "heart_rate": 72,
        "spo2": 98.5,
        "activity_level": 2,
        "site": "study-a",
    }

    observation = adapter.to_observation(event)
    context = adapter.to_context(event)

    assert observation.values == (72.0, 98.5)
    assert observation.timestamp == 123
    assert observation.metadata == {
        "subject_id": "subject-7",
        "sample_id": "sample-11",
        "site": "study-a",
    }
    assert context is not None
    assert context.values == (2.0,)


def test_research_medical_adapter_without_context_uses_internal_context():
    adapter = _adapter()

    assert adapter.to_context(
        {
            "heart_rate": 72,
            "spo2": 98,
        }
    ) is None


@pytest.mark.parametrize(
    ("event", "error"),
    [
        ({"heart_rate": 72}, KeyError),
        ({"heart_rate": 72, "spo2": float("nan")}, ValueError),
        ({"heart_rate": 72, "spo2": True}, TypeError),
        ({"heart_rate": 72, "spo2": "missing"}, TypeError),
    ],
)
def test_research_medical_adapter_rejects_invalid_measurements(event, error):
    adapter = _adapter()

    with pytest.raises(error):
        adapter.to_observation(event)


def test_research_medical_adapter_requires_declared_metadata():
    adapter = _adapter(metadata_fields=("site",))

    with pytest.raises(KeyError, match="site"):
        adapter.to_observation(
            {
                "heart_rate": 72,
                "spo2": 98,
            }
        )
