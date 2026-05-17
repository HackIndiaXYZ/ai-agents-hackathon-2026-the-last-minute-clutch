"""
nyayaeval.core.state — LangGraph Pipeline State
=================================================

This module defines the central ``NyayaEvalState`` TypedDict — the single
source of truth that flows through every node in the LangGraph evaluation
pipeline.

Design rationale — TypedDict vs Pydantic BaseModel:
    We use ``TypedDict`` (not Pydantic) for the graph state because:
    1. LangGraph nodes return *partial* state updates. TypedDict allows
       sparse updates natively; BaseModel would require explicit
       ``model_copy(update=...)`` everywhere.
    2. No runtime validation overhead on every node transition — validation
       happens at the *boundaries* (API ingestion, Neo4j writes) via
       Pydantic models in ``core.models`` and ``core.schemas``.
    3. Annotated reducers (e.g., ``operator.add`` for logs) compose cleanly
       with TypedDict fields.

State lifecycle:
    1. Initialized by the API layer with ``raw_text`` and document metadata.
    2. Each agent node reads relevant fields, processes, and returns a
       partial dict with only the fields it mutated.
    3. The evaluator node writes ``evaluation_scores`` and sets routing
       flags (``needs_correction``, ``is_verified``).
    4. The routing function inspects flags to decide: loop back to
       corrector, or proceed to export.
    5. The export node reads the verified state and serializes it.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from nyayaeval.core.models import LegalEntity, TokenBoundary
from nyayaeval.core.schemas import EvaluationScores


# ─── Reducer-annotated list type ──────────────────────────────────────────────
# Using operator.add as the reducer means LangGraph *appends* new items to the
# existing list rather than overwriting. This is critical for execution_logs:
# each node appends its own log entries without clobbering prior nodes' logs.


class NyayaEvalState(TypedDict, total=False):
    """
    Central pipeline state flowing through the LangGraph evaluation graph.

    All fields use ``total=False`` so that agent nodes can return partial
    updates containing only the fields they modified. LangGraph merges
    these partial dicts into the accumulated state automatically.

    Fields are grouped by pipeline phase for readability.
    """

    # ── Document Identity ─────────────────────────────────────────────────
    document_id: str
    """Unique identifier for the source document being processed."""

    source_language: str
    """ISO 639-1 code of the document's original language (e.g., 'hi', 'ta')."""

    document_metadata: dict[str, Any]
    """Arbitrary metadata about the source document (filename, page count, etc.)."""

    # ── Ingestion Phase ───────────────────────────────────────────────────
    raw_text: str
    """Full extracted text from the source PDF/document."""

    raw_pages: list[str]
    """Per-page extracted text, preserving page boundaries."""

    token_boundaries: list[TokenBoundary]
    """Positional annotations marking entity spans in raw_text."""

    # ── Entity Extraction ─────────────────────────────────────────────────
    entities: list[LegalEntity]
    """All legal entities extracted from the document (cases, statutes, etc.)."""

    entity_extraction_metadata: dict[str, Any]
    """Diagnostics from the extraction process (model used, token count, etc.)."""

    # ── Adaptation / Translation Phase ────────────────────────────────────
    adapted_text: str
    """Standardized English translation of the raw text."""

    adapted_segments: list[dict[str, str]]
    """
    Per-segment translation mapping.
    Each dict contains: {"source": ..., "target": ..., "language": ...}
    Preserved for segment-level evaluation and correction.
    """

    adaptation_metadata: dict[str, Any]
    """Adaption API response metadata (model version, detected language, etc.)."""

    # ── Knowledge Graph Phase ─────────────────────────────────────────────
    graph_context: dict[str, Any]
    """
    Knowledge graph query results and relationship data.
    Populated by the graph-builder agent after writing to Neo4j.
    Contains: node IDs, relationship counts, subgraph summaries.
    """

    graph_node_ids: list[str]
    """Neo4j internal node IDs created for this document's entities."""

    # ── Evaluation Phase ──────────────────────────────────────────────────
    evaluation_scores: EvaluationScores
    """
    RAGAS + custom metric scores from the evaluation agent.
    Drives the routing decision: correct or export.
    """

    evaluation_details: dict[str, Any]
    """Per-metric breakdown and diagnostic information from evaluation."""

    # ── Correction / Self-Healing ─────────────────────────────────────────
    correction_feedback: str
    """
    LLM-generated correction instructions when evaluation fails.
    Consumed by the corrector agent to fix translation or extraction errors.
    """

    correction_history: Annotated[list[dict[str, Any]], operator.add]
    """
    Append-only record of all correction attempts.
    Each entry: {"attempt": int, "feedback": str, "scores_before": ..., "scores_after": ...}
    Uses operator.add reducer so each correction cycle appends without overwriting.
    """

    # ── Routing & Control Flow ────────────────────────────────────────────
    current_phase: str
    """
    Name of the current pipeline phase.
    One of: 'ingestion', 'adaptation', 'graph_building', 'evaluation',
    'correction', 'export', 'completed', 'failed'.
    """

    needs_correction: bool
    """
    Deterministic routing flag set by the evaluator.
    True → route to corrector node. False → route to export node.
    """

    retry_count: int
    """
    Number of correction cycles completed for this document.
    Compared against max_retries to prevent infinite loops.
    """

    is_verified: bool
    """
    Terminal flag indicating the document passed all evaluation thresholds.
    Only verified documents proceed to export.
    """

    error: str | None
    """
    Error message if any phase failed unrecoverably.
    Presence of this field triggers routing to the 'failed' terminal node.
    """

    # ── Observability ─────────────────────────────────────────────────────
    execution_logs: Annotated[list[str], operator.add]
    """
    Append-only execution trace.

    Each agent node appends structured log entries describing what it did,
    how long it took, and any warnings encountered. The ``operator.add``
    reducer ensures entries from different nodes accumulate without loss.

    Example entries:
        "[2024-01-15T10:30:00] ingestion: Extracted 5 pages, 12,340 tokens"
        "[2024-01-15T10:30:05] adaptation: Translated hi→en, 98.2% confidence"
    """
