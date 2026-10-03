from .base import Adapter, Context, Observation, StateEstimate

__all__ = [
    "Adapter",
    "Context",
    "Observation",
    "StateEstimate",
    "NHSMMRuntimeAdapter",
    "StructuredEventAdapter",
]


def __getattr__(name: str):
    if name == "NHSMMRuntimeAdapter":
        from .nhsmm import NHSMMRuntimeAdapter
        return NHSMMRuntimeAdapter
    if name == "StructuredEventAdapter":
        from .structured import StructuredEventAdapter
        return StructuredEventAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
