from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from .contracts import TEMPORAL_OBSERVATION_CONTRACT, TemporalObservationData

if TYPE_CHECKING:
    from adapters.walk_forward import TemporalFold


class TemporalFoldBuilder:
    """Build causal NHSMM walk-forward input from admitted temporal observations.

    The builder preserves the transferred observation vector exactly. It does
    not construct or reinterpret Nautilus market features.
    """

    def __init__(self, *, dtype=None) -> None:
        self._dtype = torch.float32 if dtype is None else dtype

    def build_fold(
        self,
        observations: Sequence[TemporalObservationData],
        *,
        label: str,
        train_through_ns: int,
        oos_through_ns: int | None = None,
    ) -> TemporalFold:
        """Build one train-to-OOS fold from one ordered instrument stream."""
        if isinstance(train_through_ns, bool) or not isinstance(train_through_ns, int):
            raise TypeError("train_through_ns must be an integer")
        if train_through_ns < 0:
            raise ValueError("train_through_ns must be >= 0")
        if oos_through_ns is not None:
            if isinstance(oos_through_ns, bool) or not isinstance(oos_through_ns, int):
                raise TypeError("oos_through_ns must be an integer or None")
            if oos_through_ns <= train_through_ns:
                raise ValueError("oos_through_ns must be > train_through_ns")

        ordered = tuple(observations)
        self._validate_observations(ordered)

        train_items = tuple(item for item in ordered if item.asof_ts_ns <= train_through_ns)
        oos_items = tuple(
            item
            for item in ordered
            if item.asof_ts_ns > train_through_ns
            and (oos_through_ns is None or item.asof_ts_ns <= oos_through_ns)
        )
        if not train_items:
            raise ValueError("training selection must contain at least one observation")
        if not oos_items:
            raise ValueError("OOS selection must contain at least one observation")

        from adapters.walk_forward import TemporalFold

        return TemporalFold(
            label=label,
            train=self._tensor(train_items),
            oos=self._tensor(oos_items),
            train_end_ns=train_items[-1].asof_ts_ns,
            oos_start_ns=oos_items[0].asof_ts_ns,
        )

    def _tensor(self, observations: tuple[TemporalObservationData, ...]):
        values = [item.values for item in observations]
        return torch.tensor(values, dtype=self._dtype)

    @staticmethod
    def _validate_observations(observations: tuple[TemporalObservationData, ...]) -> None:
        if len(observations) < 2:
            raise ValueError("at least two temporal observations are required")
        if any(not isinstance(item, TemporalObservationData) for item in observations):
            raise TypeError("observations must contain TemporalObservationData values")

        first = observations[0]
        if first.contract != TEMPORAL_OBSERVATION_CONTRACT:
            raise ValueError("unsupported temporal observation contract")
        instrument_id = first.instrument_id
        contract = first.contract
        mapping_contract = first.mapping_contract

        previous_ts = -1
        previous_sequence = -1
        seen_identity: set[tuple[int, int]] = set()

        for item in observations:
            if item.contract != contract:
                raise ValueError("observations must use one temporal observation contract")
            if item.mapping_contract != mapping_contract:
                raise ValueError("observations must use one temporal mapping contract")
            if item.instrument_id != instrument_id:
                raise ValueError("observations must belong to one instrument")

            identity = (item.asof_ts_ns, item.decision_sequence)
            if identity in seen_identity:
                raise ValueError("duplicate temporal decision identity")
            seen_identity.add(identity)

            if item.asof_ts_ns <= previous_ts:
                raise ValueError("asof_ts_ns must be strictly increasing")
            if item.decision_sequence <= previous_sequence:
                raise ValueError("decision_sequence must be strictly increasing")
            previous_ts = item.asof_ts_ns
            previous_sequence = item.decision_sequence

            trigger = next(
                (
                    source
                    for source in item.provenance
                    if source.timeframe == item.trigger_timeframe
                ),
                None,
            )
            if trigger is not None and (
                trigger.available_ts_ns != item.asof_ts_ns
                or trigger.processed_sequence != item.decision_sequence
            ):
                raise ValueError("trigger provenance must match temporal decision identity")
