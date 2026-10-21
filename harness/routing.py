"""Replaceable decision interface. This pilot implements validation selection only."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .common import MODEL_ORDER


@dataclass(frozen=True)
class DecisionContext:
    dataset: str
    split: int
    problem_type: str
    validation_losses: dict[str, float]
    # Add fit summaries, budget or frozen ToolCards here for a future LLM router.
    # Test metrics/targets deliberately have no field in this interface.


class Router(Protocol):
    def select(self, context: DecisionContext) -> str: ...


class ValidationSelector:
    def select(self, context):
        if set(context.validation_losses) != set(MODEL_ORDER):
            raise ValueError("ValSelect requires all three valid candidates; no silent smaller pool")
        return min(MODEL_ORDER, key=lambda m: (context.validation_losses[m], MODEL_ORDER.index(m)))
