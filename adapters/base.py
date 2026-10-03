from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class Observation:
    """Canonical one-step observation passed toward an NHSMM runtime."""

    values: Sequence[float]
    timestamp: Any | None = None
    instrument: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Context:
    """Canonical one-step context accompanying an observation."""

    values: Sequence[float] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StateEstimate:
    """Engine-neutral representation of one NHSMM filtering result."""

    state: int | None
    posterior: Sequence[float]
    age_posterior: Sequence[float] | None = None
    timestamp: Any | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class UniversalAdapter:
    """Minimal engine-neutral one-event adapter pipeline."""

    def to_observation(self, event: Any) -> Observation:
        raise NotImplementedError

    def to_context(self, event: Any) -> Context | None:
        return None

    def infer(
        self,
        observation: Observation,
        context: Context | None = None,
    ) -> StateEstimate:
        raise NotImplementedError

    def from_state(self, state: StateEstimate) -> Any:
        return state

    def step(self, event: Any) -> Any:
        observation = self.to_observation(event)
        state = self.infer(observation, self.to_context(event))
        return self.from_state(state)
