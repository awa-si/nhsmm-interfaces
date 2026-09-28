from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class Observation:
    """Canonical one-step observation passed toward an NHSMM runtime."""

    values: Sequence[float]
    timestamp: Any | None = None
    instrument: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Context:
    """Canonical one-step context accompanying an observation."""

    values: Sequence[float] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StateEstimate:
    """Engine-neutral representation of one NHSMM filtering result."""

    state: int | None
    posterior: Sequence[float]
    age_posterior: Sequence[float] | None = None
    timestamp: Any | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class UniversalAdapter(ABC):
    """Engine-neutral boundary between host systems and NHSMM inference.

    Concrete adapters translate host-specific input objects into canonical
    observations/context and translate an NHSMM result back into a host-facing
    value. They must not contain strategy, signal, portfolio, execution, or
    risk policy.
    """

    @abstractmethod
    def to_observation(self, event: Any) -> Observation:
        """Translate one host event/bar/tick/sample into canonical features."""

    def to_context(self, event: Any) -> Context | None:
        """Translate host metadata into optional model context."""
        return None

    @abstractmethod
    def infer(
        self,
        observation: Observation,
        context: Context | None = None,
    ) -> StateEstimate:
        """Invoke the configured NHSMM batch/runtime boundary."""

    @abstractmethod
    def from_state(self, state: StateEstimate) -> Any:
        """Translate a canonical state estimate into a host-facing value."""

    def step(self, event: Any) -> Any:
        """Canonical one-event adapter pipeline."""
        observation = self.to_observation(event)
        context = self.to_context(event)
        state = self.infer(observation, context)
        return self.from_state(state)
