from .base import Adapter, Context, Observation, StateEstimate

__all__ = [
    "Adapter",
    "Context",
    "Observation",
    "StateEstimate",
    "NHSMMRuntimeAdapter",
    "WalkForwardFold",
    "NHSMMTuningScoreConfig",
    "NHSMMTuningEvaluationReport",
    "NHSMMTunerEvaluatorConfig",
    "NHSMMTunerEvaluator",
    "NHSMMFoldEvaluation",
    "StructuredEventAdapter",
]


def __getattr__(name: str):
    if name in {
        "NHSMMFoldEvaluation",
        "NHSMMTunerEvaluator",
        "NHSMMTunerEvaluatorConfig",
        "NHSMMTuningEvaluationReport",
        "NHSMMTuningScoreConfig",
        "WalkForwardFold",
    }:
        from .evaluation import (
            NHSMMFoldEvaluation,
            NHSMMTunerEvaluator,
            NHSMMTunerEvaluatorConfig,
            NHSMMTuningEvaluationReport,
            NHSMMTuningScoreConfig,
            WalkForwardFold,
        )

        return {
            "NHSMMFoldEvaluation": NHSMMFoldEvaluation,
            "NHSMMTunerEvaluator": NHSMMTunerEvaluator,
            "NHSMMTunerEvaluatorConfig": NHSMMTunerEvaluatorConfig,
            "NHSMMTuningEvaluationReport": NHSMMTuningEvaluationReport,
            "NHSMMTuningScoreConfig": NHSMMTuningScoreConfig,
            "WalkForwardFold": WalkForwardFold,
        }[name]
    if name == "NHSMMRuntimeAdapter":
        from .nhsmm import NHSMMRuntimeAdapter
        return NHSMMRuntimeAdapter
    if name == "StructuredEventAdapter":
        from .structured import StructuredEventAdapter
        return StructuredEventAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
