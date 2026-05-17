"""
nyayaeval.core — Pure Domain Layer
===================================

This package contains the foundational domain primitives for NyayaEval.
Everything here is intentionally free of external I/O dependencies (no database
drivers, no HTTP clients, no framework imports). This isolation guarantees:

    1. **Testability** — domain logic can be unit-tested without mocks.
    2. **Portability** — swapping Neo4j for another graph DB or changing the
       LLM provider never touches this layer.
    3. **Clarity** — new contributors can read core/ to understand the data
       model without navigating infrastructure concerns.

Modules:
    models  : Pydantic domain entities (Case, Statute, Judge, etc.)
    state   : LangGraph TypedDict pipeline state with annotated reducers
    schemas : Evaluation result schemas and correction feedback types
"""

from nyayaeval.core.models import (
    Case,
    Judge,
    LegalEntity,
    Precedent,
    Section,
    Statute,
    TokenBoundary,
)
from nyayaeval.core.schemas import CorrectionFeedback, EvaluationResult, EvaluationScores
from nyayaeval.core.state import NyayaEvalState

__all__ = [
    # Domain entities
    "Case",
    "Section",
    "Statute",
    "Judge",
    "Precedent",
    "LegalEntity",
    "TokenBoundary",
    # Evaluation
    "EvaluationScores",
    "EvaluationResult",
    "CorrectionFeedback",
    # Pipeline state
    "NyayaEvalState",
]
