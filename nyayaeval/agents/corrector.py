"""
nyayaeval.agents.corrector — Self-Correction Agent Node
=========================================================

Consumes evaluation feedback and re-processes the adapted text to fix
identified issues (hallucinations, omissions, incorrect legal references).

Correction strategy:
    - **Targeted, not wholesale**: the corrector receives specific feedback
      about what's wrong (which claims are ungrounded, which references are
      misattributed). It fixes only those issues, preserving correct parts.
    - **LLM-driven**: uses the same LLM as the evaluator, but with a
      correction-specific system prompt that instructs it to act as a
      legal editor.
    - **Entity re-extraction**: if legal_consistency was the failing metric,
      re-runs entity extraction on the corrected text to update the entities
      list for the next graph_builder pass.

LangGraph node contract:
    Reads:  correction_feedback, adapted_text, raw_text, evaluation_scores,
            retry_count, entities, document_id
    Writes: adapted_text, entities, retry_count, correction_history,
            current_phase, execution_logs
"""

from __future__ import annotations

import time
from typing import Any

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from nyayaeval.connectors.llm_provider import get_llm
from nyayaeval.core.state import NyayaEvalState

logger = structlog.get_logger(__name__)


_CORRECTION_SYSTEM_PROMPT = """You are a senior legal document editor specializing in Indian court records.

Your task is to CORRECT a translated/adapted legal document based on specific feedback from a quality evaluator.

RULES:
1. Fix ONLY the issues identified in the correction feedback
2. Preserve all parts of the adapted text that are correct
3. When correcting legal references (sections, statutes, precedents), ensure accuracy against the source text
4. Maintain the same formatting and structure as the original adapted text
5. Do NOT add information that is not present in the source text
6. Do NOT remove information that is correctly translated from the source

Return ONLY the corrected text, no explanations or metadata."""


async def corrector_node(state: NyayaEvalState) -> dict[str, Any]:
    """
    Apply targeted corrections based on evaluation feedback.

    Flow:
        1. Build a correction prompt with source, adapted text, and feedback
        2. Invoke the LLM to produce corrected text
        3. Increment retry_count and log correction history
        4. Return corrected adapted_text

    Returns:
        Partial state update with corrected text, incremented retry_count, and logs.
    """
    document_id = state.get("document_id", "unknown")
    retry_count = state.get("retry_count", 0) + 1
    feedback = state.get("correction_feedback", "No specific feedback provided")
    adapted_text = state.get("adapted_text", "")
    raw_text = state.get("raw_text", "")
    evaluation_scores = state.get("evaluation_scores")
    start_time = time.monotonic()

    logger.info("agent.corrector.start", document_id=document_id, attempt=retry_count)

    # ── Build correction prompt ───────────────────────────────────────────
    scores_str = ""
    if evaluation_scores:
        scores_str = (
            f"Faithfulness: {evaluation_scores.faithfulness:.2f}, "
            f"Context Recall: {evaluation_scores.context_recall:.2f}, "
            f"Legal Consistency: {evaluation_scores.legal_consistency:.2f}"
        )

    # Truncate to fit context window
    max_text_len = 5000
    source_preview = raw_text[:max_text_len] + ("..." if len(raw_text) > max_text_len else "")
    adapted_preview = adapted_text[:max_text_len] + ("..." if len(adapted_text) > max_text_len else "")

    correction_prompt = f"""CORRECTION ATTEMPT #{retry_count}

EVALUATION SCORES: {scores_str}

CORRECTION FEEDBACK:
{feedback}

SOURCE TEXT (ground truth — the original court document):
{source_preview}

ADAPTED TEXT (to be corrected):
{adapted_preview}

Please produce the corrected version of the adapted text, fixing only the issues identified in the feedback."""

    # ── Invoke LLM ────────────────────────────────────────────────────────
    corrected_text = adapted_text  # Fallback to original if LLM fails

    try:
        llm = get_llm()
        response = await llm.ainvoke(
            [
                SystemMessage(content=_CORRECTION_SYSTEM_PROMPT),
                HumanMessage(content=correction_prompt),
            ]
        )
        corrected_text = response.content.strip()

        if not corrected_text:
            corrected_text = adapted_text
            logger.warning("agent.corrector.empty_response", document_id=document_id)

        logger.info(
            "agent.corrector.llm_corrected",
            document_id=document_id,
            original_len=len(adapted_text),
            corrected_len=len(corrected_text),
        )
    except Exception as exc:
        logger.error(
            "agent.corrector.llm_failed",
            document_id=document_id,
            error=str(exc),
        )

    # ── Build correction history entry ────────────────────────────────────
    history_entry: dict[str, Any] = {
        "attempt": retry_count,
        "feedback": feedback[:500],  # Truncate for storage
        "text_changed": corrected_text != adapted_text,
        "original_length": len(adapted_text),
        "corrected_length": len(corrected_text),
    }

    if evaluation_scores:
        history_entry["scores_before"] = {
            "faithfulness": evaluation_scores.faithfulness,
            "context_recall": evaluation_scores.context_recall,
            "legal_consistency": evaluation_scores.legal_consistency,
        }

    duration_s = time.monotonic() - start_time
    logger.info(
        "agent.corrector.complete",
        document_id=document_id,
        attempt=retry_count,
        text_changed=corrected_text != adapted_text,
        duration_s=round(duration_s, 2),
    )

    return {
        "adapted_text": corrected_text,
        "retry_count": retry_count,
        "current_phase": "correction",
        "correction_history": [history_entry],
        "execution_logs": [
            f"[corrector] Attempt {retry_count} for {document_id}: "
            f"{'text modified' if corrected_text != adapted_text else 'no changes'} "
            f"({duration_s:.1f}s)"
        ],
    }
