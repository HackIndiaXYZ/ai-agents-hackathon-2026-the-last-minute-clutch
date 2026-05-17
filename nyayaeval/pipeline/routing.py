"""
nyayaeval.pipeline.routing — Conditional Edge Routing Functions
================================================================

Pure functions that inspect the pipeline state and return the name of
the next node to execute. Used by LangGraph's ``add_conditional_edges``.

These functions encapsulate the self-correction loop logic:
    - If evaluation passes → route to export
    - If evaluation fails and retries remain → route to corrector
    - If evaluation fails and retries exhausted → route to failed terminal
"""

from __future__ import annotations

import structlog

from nyayaeval.config.settings import get_settings
from nyayaeval.core.state import NyayaEvalState

logger = structlog.get_logger(__name__)


def evaluation_router(state: NyayaEvalState) -> str:
    """
    Route after the evaluator node based on scores and retry count.

    Decision logic:
        1. If ``is_verified`` is True → "export"
        2. If ``retry_count`` >= ``max_retries`` → "failed"
        3. Otherwise → "corrector" (re-enter correction loop)

    Args:
        state: Current pipeline state with evaluation results.

    Returns:
        Next node name: "export", "corrector", or "failed".
    """
    is_verified = state.get("is_verified", False)
    retry_count = state.get("retry_count", 0)
    document_id = state.get("document_id", "unknown")

    settings = get_settings()
    max_retries = settings.max_retries

    if is_verified:
        logger.info("routing.evaluation_passed", document_id=document_id)
        return "export"

    if retry_count >= max_retries:
        logger.warning(
            "routing.max_retries_exhausted",
            document_id=document_id,
            retry_count=retry_count,
            max_retries=max_retries,
        )
        return "failed"

    logger.info(
        "routing.needs_correction",
        document_id=document_id,
        retry_count=retry_count,
        remaining=max_retries - retry_count,
    )
    return "corrector"
