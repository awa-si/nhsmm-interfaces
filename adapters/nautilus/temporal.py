from __future__ import annotations

from ..base import Observation, StateEstimate
from ..nhsmm import NHSMMRuntimeAdapter
from .contracts import TEMPORAL_OBSERVATION_CONTRACT, TemporalObservationData


class TemporalAdapter(NHSMMRuntimeAdapter):
    """Map admitted Nautilus temporal observations into the NHSMM runtime."""

    def to_observation(self, event: TemporalObservationData) -> Observation:
        if not isinstance(event, TemporalObservationData):
            raise TypeError("event must be TemporalObservationData")
        return Observation(
            values=event.values,
            timestamp=event.asof_ts_ns,
            instrument=event.instrument_id,
            metadata={
                "observation_contract": event.contract,
                "decision_sequence": event.decision_sequence,
                "trigger_timeframe": event.trigger_timeframe,
                "provenance": event.provenance,
            },
        )


def temporal_state_fields(
    state: StateEstimate,
) -> tuple[str, int, int, str, str]:
    metadata = state.metadata
    try:
        instrument_id = str(metadata["instrument"])
        ts_event = int(state.timestamp)
        decision_sequence = int(metadata["decision_sequence"])
        trigger_timeframe = str(metadata["trigger_timeframe"])
        observation_contract = str(metadata["observation_contract"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("state estimate is missing temporal observation provenance") from exc
    if observation_contract != TEMPORAL_OBSERVATION_CONTRACT:
        raise ValueError("state estimate has incompatible temporal observation contract")
    return instrument_id, ts_event, decision_sequence, trigger_timeframe, observation_contract


# Compatibility alias for pre-rename consumers.
NautilusTemporalAdapter = TemporalAdapter
