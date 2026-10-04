"""NautilusTrader integration for NHSMM.

Importing this package requires the optional NautilusTrader dependency. Generic
``adapters`` imports remain independent of NautilusTrader.
"""

from .axis import (
    AXIS_OBSERVATION_DATA_TYPE,
    AXIS_OBSERVATION_DATA_TYPE_NAME,
    AXIS_OBSERVATION_SCHEMA,
    AxisObservationCollector,
)
from .actor import (
    NHSMMDataActor,
    NHSMMDataActorConfig,
    NHSMMStateData,
    NHSMM_STATE_DATA_TYPE,
    TEMPORAL_OBSERVATION_DATA_TYPE,
)
from .bar import BAR_FIELDS, BarAdapter
from .mapping import (
    AXIS_TEMPORAL_MAPPING_CONTRACT,
    AXIS_TEMPORAL_MAPPING_FIELDS,
    AxisTemporalDataActor,
    AxisTemporalMapper,
)
from .temporal import TemporalAdapter
from .training import TemporalFoldBuilder
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
    "AXIS_OBSERVATION_DATA_TYPE",
    "AXIS_OBSERVATION_DATA_TYPE_NAME",
    "AXIS_OBSERVATION_SCHEMA",
    "AxisObservationCollector",
    "NHSMMDataActor",
    "NHSMMDataActorConfig",
    "NHSMMStateData",
    "NHSMM_STATE_DATA_TYPE",
    "TEMPORAL_OBSERVATION_DATA_TYPE",
    "BarAdapter",
    "AXIS_TEMPORAL_MAPPING_CONTRACT",
    "AXIS_TEMPORAL_MAPPING_FIELDS",
    "AxisTemporalDataActor",
    "AxisTemporalMapper",
    "TemporalAdapter",
    "TemporalFoldBuilder",
    "BAR_FIELDS",
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
