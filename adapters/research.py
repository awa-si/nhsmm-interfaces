from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from nhsmm import HSMMFilterRuntime

from .base import Context, Observation
from .nhsmm import NHSMMRuntimeAdapter


class ResearchAdapter(NHSMMRuntimeAdapter):
    """Bridge structured research workflow events to NHSMM.

    The adapter consumes already structured numeric model features produced by
    an upstream research/data-processing layer. Optional workflow metadata can
    be preserved, but raw-document parsing, OCR, diagnosis, eligibility logic,
    and domain decisions remain outside this adapter.
    """

    def __init__(
        self,
        runtime: HSMMFilterRuntime,
        *,
        feature_fields: Sequence[str],
        context_fields: Sequence[str] = (),
        features_field: str = "features",
        context_field: str = "context",
        timestamp_field: str = "timestamp",
        public_ref_field: str = "public_ref",
        event_type_field: str = "event_type",
        case_state_field: str = "case_state",
        document_type_field: str = "document_type",
        assessment_type_field: str = "assessment_type",
        ai_status_field: str = "ai_status",
    ) -> None:
        super().__init__(runtime)
        self.feature_fields = self._validate_fields(
            feature_fields, name="feature_fields", allow_empty=False
        )
        self.context_fields = self._validate_fields(
            context_fields, name="context_fields", allow_empty=True
        )
        self.features_field = self._validate_name(features_field, "features_field")
        self.context_field = self._validate_name(context_field, "context_field")
        self.timestamp_field = self._validate_name(timestamp_field, "timestamp_field")
        self.public_ref_field = self._validate_name(public_ref_field, "public_ref_field")
        self.event_type_field = self._validate_name(event_type_field, "event_type_field")
        self.case_state_field = self._validate_name(case_state_field, "case_state_field")
        self.document_type_field = self._validate_name(
            document_type_field, "document_type_field"
        )
        self.assessment_type_field = self._validate_name(
            assessment_type_field, "assessment_type_field"
        )
        self.ai_status_field = self._validate_name(ai_status_field, "ai_status_field")

    @staticmethod
    def _validate_name(value: str, name: str) -> str:
        if not isinstance(value, str) or not value:
            raise TypeError(f"{name} must be a non-empty string")
        return value

    @staticmethod
    def _validate_fields(
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
    def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            raise TypeError(f"{name} must be a mapping")
        return value

    @classmethod
    def _vector(
        cls,
        source: Mapping[str, Any],
        fields: Sequence[str],
        *,
        kind: str,
    ) -> tuple[float, ...]:
        values: list[float] = []
        for field in fields:
            if field not in source:
                raise KeyError(f"missing {kind} field: {field}")
            raw = source[field]
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
        pairs = (
            ("public_ref", self.public_ref_field),
            ("event_type", self.event_type_field),
            ("case_state", self.case_state_field),
            ("document_type", self.document_type_field),
            ("assessment_type", self.assessment_type_field),
            ("ai_status", self.ai_status_field),
        )
        return {
            canonical: event[field]
            for canonical, field in pairs
            if field in event and event[field] is not None
        }

    def to_observation(self, event: Any) -> Observation:
        record = self._mapping(event, name="research event")
        if self.features_field not in record:
            raise KeyError(f"missing structured features: {self.features_field}")

        features = self._mapping(
            record[self.features_field],
            name=self.features_field,
        )
        timestamp = record.get(self.timestamp_field)

        return Observation(
            values=self._vector(features, self.feature_fields, kind="feature"),
            timestamp=timestamp,
            metadata=self._metadata(record),
        )

    def to_context(self, event: Any) -> Context | None:
        if not self.context_fields:
            return None

        record = self._mapping(event, name="research event")
        if self.context_field not in record:
            raise KeyError(f"missing structured context: {self.context_field}")

        context = self._mapping(record[self.context_field], name=self.context_field)
        return Context(
            values=self._vector(context, self.context_fields, kind="context"),
            metadata=self._metadata(record),
        )
