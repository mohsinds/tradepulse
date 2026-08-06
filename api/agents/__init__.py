from typing import Any

from .base import Agent

__all__ = ["Agent", "RankerAgent"]


def __getattr__(name: str) -> Any:
    """Lazily expose optional agents so missing extras never break imports."""
    if name == "RankerAgent":
        try:
            from .ranker import RankerAgent
        except Exception as exc:  # noqa: BLE001 - langgraph/langchain optional
            raise ImportError(
                "RankerAgent requires the langgraph/langchain extras: "
                f"{exc}"
            ) from exc
        return RankerAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
