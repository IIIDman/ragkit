"""Guards that screen retrieved text before it reaches the LLM."""

from .injection import InjectionGuard

__all__ = [
    "InjectionGuard",
]
