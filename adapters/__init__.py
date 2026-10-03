from .base import Adapter, Context, Observation, StateEstimate, UniversalAdapter

__all__ = [
    "Adapter",
    "Context",
    "Observation",
    "StateEstimate",
    "UniversalAdapter",
    "NHSMMRuntimeAdapter",
    "StructuredEventAdapter",
    "StructuredAdapter",
    "ResearchAdapter",
]


def __getattr__(name: str):
    if name == "NHSMMRuntimeAdapter":
        from .nhsmm import NHSMMRuntimeAdapter

        return NHSMMRuntimeAdapter
    if name in {"StructuredEventAdapter", "StructuredAdapter", "ResearchAdapter"}:
        from .research import ResearchAdapter, StructuredAdapter, StructuredEventAdapter

        return {"StructuredEventAdapter": StructuredEventAdapter, "StructuredAdapter": StructuredAdapter, "ResearchAdapter": ResearchAdapter}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
