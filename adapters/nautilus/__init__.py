"""NautilusTrader integration for NHSMM.

Importing this package requires the optional NautilusTrader dependency. Generic
``adapters`` imports remain independent of NautilusTrader.
"""

from .actor import (
    NHSMMDataActor,
    NHSMMDataActorConfig,
    NHSMMStateData,
    NHSMM_STATE_DATA_TYPE,
    TEMPORAL_OBSERVATION_DATA_TYPE,
)
from .bar import BAR_FIELDS, BarAdapter, NautilusBarAdapter, PROTOTYPE_BAR_FIELDS
from .temporal import NautilusTemporalAdapter, TemporalAdapter
from .contracts import (
    SIGNED_TEMPORAL_OBSERVATIONS,
    TEMPORAL_OBSERVATION_CONTRACT,
    TEMPORAL_OBSERVATION_DATA_TYPE_NAME,
    NHSMM_STATE_DATA_SCHEMA,
    TEMPORAL_OBSERVATION_NAMES,
    TemporalObservationData,
    TimeframeProvenance,
    NHSMMArtifactIdentity,
    NHSMMForecastData,
)

__all__ = [
    "NHSMMDataActor",
    "NHSMMDataActorConfig",
    "NHSMMStateData",
    "NHSMM_STATE_DATA_TYPE",
    "TEMPORAL_OBSERVATION_DATA_TYPE",
    "BarAdapter",
    "NautilusBarAdapter",
    "TemporalAdapter",
    "NautilusTemporalAdapter",
    "BAR_FIELDS",
    "PROTOTYPE_BAR_FIELDS",
    "SIGNED_TEMPORAL_OBSERVATIONS",
    "TEMPORAL_OBSERVATION_CONTRACT",
    "TEMPORAL_OBSERVATION_DATA_TYPE_NAME",
    "NHSMM_STATE_DATA_SCHEMA",
    "TEMPORAL_OBSERVATION_NAMES",
    "TemporalObservationData",
    "TimeframeProvenance",
    "NHSMMArtifactIdentity",
    "NHSMMForecastData",
]
