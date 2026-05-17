"""
nyayaeval.api.routes — FastAPI Route Handlers
================================================

Defines the HTTP endpoints for the NyayaEval API:
    - POST /pipeline/run       : Submit a document for processing
    - GET  /pipeline/status     : Pipeline info
    - GET  /health              : Service health check (Neo4j + Redis)
    - GET  /health/detailed     : Detailed infrastructure status
"""

from __future__ import annotations

import uuid
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

router = APIRouter()


# ─── Request / Response Models ────────────────────────────────────────────────


class PipelineRequest(BaseModel):
    """Request body for pipeline execution."""

    document_id: str | None = Field(
        default=None,
        description="Unique document identifier. Auto-generated if not provided.",
    )
    raw_text: str = Field(..., min_length=1, description="Extracted document text")
    source_language: str = Field(
        default="hi", description="ISO 639-1 source language code"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Optional document metadata"
    )


class PipelineResponse(BaseModel):
    """Response body for pipeline execution."""

    document_id: str
    status: str
    is_verified: bool = False
    evaluation_scores: dict[str, float] | None = None
    execution_logs: list[str] = Field(default_factory=list)
    error: str | None = None


class HealthResponse(BaseModel):
    """Response body for health check."""

    status: str
    services: dict[str, Any] = Field(default_factory=dict)


# ─── Endpoints ────────────────────────────────────────────────────────────────


@router.post("/pipeline/run", response_model=PipelineResponse)
async def run_pipeline(request: PipelineRequest, req: Request) -> PipelineResponse:
    """
    Submit a document for synchronous pipeline processing.

    The document text is run through the full pipeline:
    ingestion → adaptation → graph_builder → evaluator → [corrector ↺] → export

    Returns the final pipeline state including evaluation scores and verdict.
    """
    doc_id = request.document_id or f"doc-{uuid.uuid4().hex[:12]}"
    logger.info(
        "api.pipeline.run",
        document_id=doc_id,
        source_language=request.source_language,
        text_length=len(request.raw_text),
    )

    # Get the compiled pipeline from app state
    pipeline = getattr(req.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    # Build initial state
    initial_state: dict[str, Any] = {
        "document_id": doc_id,
        "raw_text": request.raw_text,
        "source_language": request.source_language,
        "document_metadata": request.metadata,
        "retry_count": 0,
        "needs_correction": False,
        "is_verified": False,
        "execution_logs": [],
        "correction_history": [],
    }

    try:
        # Invoke the pipeline synchronously
        final_state = await pipeline.ainvoke(initial_state)

        # Extract results
        scores = final_state.get("evaluation_scores")
        scores_dict = None
        if scores:
            scores_dict = {
                "faithfulness": scores.faithfulness,
                "context_recall": scores.context_recall,
                "legal_consistency": scores.legal_consistency,
                "answer_relevancy": scores.answer_relevancy,
            }

        return PipelineResponse(
            document_id=doc_id,
            status=final_state.get("current_phase", "unknown"),
            is_verified=final_state.get("is_verified", False),
            evaluation_scores=scores_dict,
            execution_logs=final_state.get("execution_logs", []),
            error=final_state.get("error"),
        )

    except Exception as exc:
        logger.error("api.pipeline.failed", document_id=doc_id, error=str(exc))
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline execution failed: {exc}",
        ) from exc


@router.get("/health", response_model=HealthResponse)
async def health_check(req: Request) -> HealthResponse:
    """
    Quick health check — verifies core services are reachable.
    """
    services: dict[str, Any] = {}

    # Neo4j
    neo4j = getattr(req.app.state, "neo4j", None)
    if neo4j:
        services["neo4j"] = await neo4j.health_check()
    else:
        services["neo4j"] = {"status": "not_configured"}

    # Redis
    redis = getattr(req.app.state, "redis", None)
    if redis:
        services["redis"] = await redis.health_check()
    else:
        services["redis"] = {"status": "not_configured"}

    # Determine overall status
    all_healthy = all(
        s.get("status") in ("healthy", "not_configured")
        for s in services.values()
    )

    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        services=services,
    )


@router.get("/health/detailed", response_model=HealthResponse)
async def health_detailed(req: Request) -> HealthResponse:
    """
    Detailed health check with pipeline info.
    """
    base = await health_check(req)

    pipeline = getattr(req.app.state, "pipeline", None)
    base.services["pipeline"] = {
        "status": "compiled" if pipeline else "not_initialized",
    }

    return base
