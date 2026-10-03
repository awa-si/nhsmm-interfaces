from .base import Context, Observation, StateEstimate, UniversalAdapter

__all__ = [
    "Context",
    "Observation",
    "StateEstimate",
    "UniversalAdapter",
    "NHSMMRuntimeAdapter",
    "ResearchAdapter",
]


def __getattr__(name: str):
    if name == "NHSMMRuntimeAdapter":
        from .nhsmm import NHSMMRuntimeAdapter

        return NHSMMRuntimeAdapter
    if name == "ResearchAdapter":
        from .research import ResearchAdapter

        return ResearchAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
