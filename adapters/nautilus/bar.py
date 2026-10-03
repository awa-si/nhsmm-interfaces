from __future__ import annotations

from math import isfinite
from typing import Any, Iterable

from nautilus_trader.model import Bar

from ..base import Observation, StateEstimate
from ..nhsmm import NHSMMRuntimeAdapter


PROTOTYPE_BAR_FIELDS = ("open", "high", "low", "close", "volume")


class NautilusBarAdapter(NHSMMRuntimeAdapter):
    """Map one Nautilus ``Bar`` stream into the canonical NHSMM runtime bridge."""

    def __init__(
        self,
        runtime,
        *,
        feature_fields: Iterable[str] = PROTOTYPE_BAR_FIELDS,
    ) -> None:
        super().__init__(runtime)
        fields = tuple(feature_fields)
        if not fields:
            raise ValueError("feature_fields must not be empty")
        if len(set(fields)) != len(fields):
            raise ValueError("feature_fields must be unique")
        unsupported = tuple(name for name in fields if name not in PROTOTYPE_BAR_FIELDS)
        if unsupported:
            raise ValueError(
                "prototype NautilusBarAdapter supports only bar fields: "
                + ", ".join(PROTOTYPE_BAR_FIELDS)
            )
        self.feature_fields = fields

    def to_observation(self, bar: Bar) -> Observation:
        values = tuple(self._finite_bar_value(bar, name) for name in self.feature_fields)
        return Observation(
            values=values,
            timestamp=int(bar.ts_event),
            instrument=str(bar.bar_type.instrument_id),
            metadata={
                "bar_type": str(bar.bar_type),
                "ts_init": int(bar.ts_init),
                "feature_fields": self.feature_fields,
            },
        )

    @staticmethod
    def _finite_bar_value(bar: Any, name: str) -> float:
        value = float(getattr(bar, name))
        if not isfinite(value):
            raise ValueError(f"bar field {name!r} must be finite")
        return value


def state_data_fields(state: StateEstimate) -> tuple[str, str, int, int]:
    """Extract the Nautilus routing/provenance fields retained by the bar mapper."""
    metadata = state.metadata
    try:
        instrument_id = str(metadata["instrument"])
        bar_type = str(metadata["bar_type"])
        ts_init = int(metadata["ts_init"])
        ts_event = int(state.timestamp)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("state estimate is missing Nautilus bar provenance") from exc
    return instrument_id, bar_type, ts_event, ts_init
