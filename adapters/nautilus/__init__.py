"""NautilusTrader integration for NHSMM.

Importing this package requires the optional NautilusTrader dependency. Generic
``adapters`` imports remain independent of NautilusTrader.
"""

from .actor import (
    NHSMMDataActor,
    NHSMMDataActorConfig,
    NHSMMStateData,
    NHSMM_STATE_DATA_TYPE,
)
from .bar import NautilusBarAdapter, PROTOTYPE_BAR_FIELDS
from .contracts import (
    SIGNED_TEMPORAL_OBSERVATIONS,
    TEMPORAL_OBSERVATION_CONTRACT,
    TEMPORAL_OBSERVATION_NAMES,
    TemporalObservationData,
    TimeframeProvenance,
)

__all__ = [
    "NHSMMDataActor",
    "NHSMMDataActorConfig",
    "NHSMMStateData",
    "NHSMM_STATE_DATA_TYPE",
    "NautilusBarAdapter",
    "PROTOTYPE_BAR_FIELDS",
    "SIGNED_TEMPORAL_OBSERVATIONS",
    "TEMPORAL_OBSERVATION_CONTRACT",
    "TEMPORAL_OBSERVATION_NAMES",
    "TemporalObservationData",
    "TimeframeProvenance",
]
