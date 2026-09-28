from __future__ import annotations

from typing import Any

import torch
from nhsmm import HSMMFilterRuntime

from .base import Context, Observation, StateEstimate, UniversalAdapter


class NHSMMRuntimeAdapter(UniversalAdapter):
    """UniversalAdapter bridge backed by the public NHSMM streaming runtime.

    Host-specific adapters only need to implement ``to_observation`` and may
    override ``to_context`` / ``from_state``. This bridge owns the conversion
    between canonical interface objects and ``HSMMFilterRuntime``.
    """

    def __init__(self, runtime: HSMMFilterRuntime) -> None:
        if not isinstance(runtime, HSMMFilterRuntime):
            raise TypeError("runtime must be an nhsmm.HSMMFilterRuntime")
        self.runtime = runtime

    def infer(
        self,
        observation: Observation,
        context: Context | None = None,
    ) -> StateEstimate:
        obs_tensor = torch.as_tensor(observation.values, dtype=torch.float32)

        context_tensor = None
        if context is not None and context.values is not None:
            context_tensor = torch.as_tensor(context.values, dtype=torch.float32)

        result = self.runtime.step(
            obs_tensor,
            context=context_tensor,
            timestamp=observation.timestamp,
        )

        posterior = result.state_posterior
        age_posterior = result.age_posterior
        if posterior.shape[0] != 1 or age_posterior.shape[0] != 1:
            raise ValueError(
                "NHSMMRuntimeAdapter expects one canonical event to produce batch size 1"
            )

        posterior_1 = posterior[0]
        age_posterior_1 = age_posterior[0]
        state = int(torch.argmax(posterior_1).item())

        metadata: dict[str, Any] = dict(observation.metadata)
        if observation.instrument is not None:
            metadata.setdefault("instrument", observation.instrument)

        return StateEstimate(
            state=state,
            posterior=tuple(float(v) for v in posterior_1.detach().cpu().tolist()),
            age_posterior=tuple(
                float(v) for v in age_posterior_1.detach().cpu().tolist()
            ),
            timestamp=observation.timestamp,
            metadata=metadata,
        )

    def from_state(self, state: StateEstimate) -> StateEstimate:
        """Default host-facing representation is the canonical state itself."""
        return state
