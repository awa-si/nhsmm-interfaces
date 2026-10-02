from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from nhsmm import HSMMFilterRuntime

from .base import Context, Observation
from .nhsmm import NHSMMRuntimeAdapter


class ResearchMedicalAdapter(NHSMMRuntimeAdapter):
    """Schema-driven adapter for normalized research and medical measurements.

    Input events are mapping-like records whose numerical feature order is
    declared explicitly at construction time. The adapter does not perform
    clinical interpretation, diagnosis, imputation, unit conversion, or
    decision support.
    """

    def __init__(
        self,
        runtime: HSMMFilterRuntime,
        *,
        feature_fields: Sequence[str],
        context_fields: Sequence[str] = (),
        timestamp_field: str | None = "timestamp",
        subject_id_field: str | None = "subject_id",
        sample_id_field: str | None = "sample_id",
        metadata_fields: Sequence[str] = (),
    ) -> None:
        super().__init__(runtime)

        self.feature_fields = self._validate_field_list(
            feature_fields,
            name="feature_fields",
            allow_empty=False,
        )
        self.context_fields = self._validate_field_list(
            context_fields,
            name="context_fields",
            allow_empty=True,
        )
        self.metadata_fields = self._validate_field_list(
            metadata_fields,
            name="metadata_fields",
            allow_empty=True,
        )
        self.timestamp_field = self._validate_optional_field(
            timestamp_field,
            name="timestamp_field",
        )
        self.subject_id_field = self._validate_optional_field(
            subject_id_field,
            name="subject_id_field",
        )
        self.sample_id_field = self._validate_optional_field(
            sample_id_field,
            name="sample_id_field",
        )

    @staticmethod
    def _validate_field_list(
        fields: Sequence[str],
        *,
        name: str,
        allow_empty: bool,
    ) -> tuple[str, ...]:
        values = tuple(fields)
        if not allow_empty and not values:
            raise ValueError(f"{name} must contain at least one field")
        if any(not isinstance(field, str) or not field for field in values):
            raise TypeError(f"{name} must contain non-empty strings")
        if len(set(values)) != len(values):
            raise ValueError(f"{name} must not contain duplicate fields")
        return values

    @staticmethod
    def _validate_optional_field(field: str | None, *, name: str) -> str | None:
        if field is not None and (not isinstance(field, str) or not field):
            raise TypeError(f"{name} must be a non-empty string or None")
        return field

    @staticmethod
    def _require_mapping(event: Any) -> Mapping[str, Any]:
        if not isinstance(event, Mapping):
            raise TypeError("research/medical event must be a mapping")
        return event

    @staticmethod
    def _numeric_values(
        event: Mapping[str, Any],
        fields: Sequence[str],
        *,
        kind: str,
    ) -> tuple[float, ...]:
        values: list[float] = []
        for field in fields:
            if field not in event:
                raise KeyError(f"missing {kind} field: {field}")
            raw = event[field]
            if isinstance(raw, bool):
                raise TypeError(f"{kind} field {field!r} must be numeric, not bool")
            try:
                value = float(raw)
            except (TypeError, ValueError) as exc:
                raise TypeError(f"{kind} field {field!r} must be numeric") from exc
            if not math.isfinite(value):
                raise ValueError(f"{kind} field {field!r} must be finite")
            values.append(value)
        return tuple(values)

    def _metadata(self, event: Mapping[str, Any]) -> dict[str, Any]:
        metadata: dict[str, Any] = {}

        if self.subject_id_field is not None and self.subject_id_field in event:
            metadata["subject_id"] = event[self.subject_id_field]

        if self.sample_id_field is not None and self.sample_id_field in event:
            metadata["sample_id"] = event[self.sample_id_field]

        for field in self.metadata_fields:
            if field not in event:
                raise KeyError(f"missing metadata field: {field}")
            metadata[field] = event[field]

        return metadata

    def to_observation(self, event: Any) -> Observation:
        record = self._require_mapping(event)
        timestamp = (
            record.get(self.timestamp_field)
            if self.timestamp_field is not None
            else None
        )

        return Observation(
            values=self._numeric_values(
                record,
                self.feature_fields,
                kind="feature",
            ),
            timestamp=timestamp,
            metadata=self._metadata(record),
        )

    def to_context(self, event: Any) -> Context | None:
        if not self.context_fields:
            return None

        record = self._require_mapping(event)
        return Context(
            values=self._numeric_values(
                record,
                self.context_fields,
                kind="context",
            )
        )
