"""
nyayaeval.core.schemas — Evaluation & Feedback Schemas
=======================================================

Pydantic models for evaluation results and correction feedback. These live in
the core layer (not in agents/) because they are part of the domain vocabulary:
multiple agents and the routing logic depend on these types.

Separation from ``models.py``:
    ``models.py`` contains entities that map to Neo4j nodes (Case, Judge, etc.).
    ``schemas.py`` contains pipeline-internal data structures that never persist
    to the knowledge graph — they exist to carry evaluation signals between
    the evaluator, router, and corrector agents.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class EvaluationVerdict(StrEnum):
    """Overall verdict from the evaluation agent."""

    PASS = "pass"
    FAIL = "fail"
    MARGINAL = "marginal"


class EvaluationScores(BaseModel):
    """
    Composite evaluation scores computed by the RAGAS evaluation agent.

    Each score is a float in [0.0, 1.0]. The evaluator agent computes these
    by running RAGAS metrics against the adapted text, source text, and
    knowledge graph context.

    The routing function compares these against configured thresholds
    (from ``config.settings``) to set ``needs_correction``.
    """

    faithfulness: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "RAGAS faithfulness score — proportion of claims in the adapted text "
            "that are grounded in the source document. Primary hallucination detector."
        ),
    )
    context_recall: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "RAGAS context recall — proportion of source-document information "
            "preserved in the adapted text. Catches omissions."
        ),
    )
    legal_consistency: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Custom metric — measures whether legal references (sections, statutes, "
            "precedents) in the adapted text are internally consistent and correctly "
            "attributed. Computed via KG validation queries."
        ),
    )
    answer_relevancy: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "RAGAS answer relevancy — measures whether the adapted text addresses "
            "the semantic intent of the original document."
        ),
    )

    def passes_thresholds(
        self,
        faithfulness_min: float = 0.85,
        context_recall_min: float = 0.80,
        legal_consistency_min: float = 0.80,
    ) -> bool:
        """Check if all scores meet minimum thresholds."""
        return (
            self.faithfulness >= faithfulness_min
            and self.context_recall >= context_recall_min
            and self.legal_consistency >= legal_consistency_min
        )

    @property
    def verdict(self) -> EvaluationVerdict:
        """Derive a human-readable verdict from scores."""
        if self.passes_thresholds():
            return EvaluationVerdict.PASS
        # Marginal: close but not quite (within 0.05 of all thresholds)
        if self.passes_thresholds(
            faithfulness_min=0.80,
            context_recall_min=0.75,
            legal_consistency_min=0.75,
        ):
            return EvaluationVerdict.MARGINAL
        return EvaluationVerdict.FAIL


class EvaluationResult(BaseModel):
    """
    Full evaluation output from a single evaluation cycle.

    Includes scores, verdict, per-metric diagnostics, and timing.
    This is what the evaluator agent returns as part of its state update.
    """

    scores: EvaluationScores
    verdict: EvaluationVerdict
    evaluated_at: datetime = Field(default_factory=datetime.utcnow)
    diagnostics: dict[str, str] = Field(
        default_factory=dict,
        description="Per-metric diagnostic messages (e.g., which claims failed faithfulness)",
    )
    evaluation_model: str = Field(
        default="", description="LLM model used for evaluation"
    )
    duration_seconds: float = Field(
        default=0.0, ge=0.0, description="Wall-clock time for evaluation"
    )


class CorrectionFeedback(BaseModel):
    """
    Structured feedback from the evaluator to the corrector agent.

    When evaluation fails, the evaluator generates targeted correction
    instructions. The corrector agent consumes this to know *what* to fix
    and *why*, rather than blindly re-running the entire pipeline.
    """

    attempt_number: int = Field(..., ge=1, description="Which correction cycle this is")
    failed_metrics: list[str] = Field(
        ...,
        description="Names of metrics that fell below threshold (e.g., ['faithfulness', 'legal_consistency'])",
    )
    specific_issues: list[str] = Field(
        default_factory=list,
        description=(
            "Concrete issues identified by the evaluator. "
            "E.g., 'Section 302 IPC incorrectly translated as Section 302 CPC'"
        ),
    )
    correction_instructions: str = Field(
        ...,
        description="LLM-generated natural language instructions for the corrector agent",
    )
    scores_before: EvaluationScores = Field(
        ..., description="Scores from the evaluation that triggered correction"
    )
