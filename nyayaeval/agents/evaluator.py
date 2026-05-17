"""
nyayaeval.agents.evaluator — RAGAS Evaluation Agent Node
==========================================================

Runs the multi-metric evaluation loop using RAGAS (faithfulness,
context recall, answer relevancy) and a custom legal consistency metric.

Evaluation strategy:
    1. **RAGAS metrics** — use the RAGAS library to compute faithfulness,
       context_recall, and answer_relevancy. We construct a synthetic
       evaluation dataset where raw_text is the "ground truth" context
       and adapted_text is the "answer" being evaluated.
    2. **Legal consistency** — LLM-as-judge: compare legal references in
       adapted_text against entities in the state and graph_context.
    3. **Correction feedback** — on failure, generate structured feedback
       telling the corrector agent exactly what to fix.

LangGraph node contract:
    Reads:  raw_text, adapted_text, graph_context, entities, retry_count,
            document_id
    Writes: evaluation_scores, evaluation_details, needs_correction,
            is_verified, correction_feedback, current_phase, execution_logs
"""

from __future__ import annotations

import time
from typing import Any

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from nyayaeval.config.settings import get_settings
from nyayaeval.connectors.llm_provider import get_llm
from nyayaeval.core.schemas import CorrectionFeedback, EvaluationScores
from nyayaeval.core.state import NyayaEvalState

logger = structlog.get_logger(__name__)


# ─── Evaluation Prompts ──────────────────────────────────────────────────────

_FAITHFULNESS_PROMPT = """You are an expert legal document evaluator. Your task is to assess FAITHFULNESS.

Given a SOURCE document (original court text) and an ADAPTED document (translated/standardized version), evaluate whether every claim in the ADAPTED text is supported by the SOURCE text.

Score from 0.0 to 1.0:
- 1.0 = Every claim in the adapted text is directly supported by the source
- 0.5 = About half the claims are supported
- 0.0 = The adapted text contains entirely fabricated information

SOURCE TEXT:
{source_text}

ADAPTED TEXT:
{adapted_text}

Respond with ONLY a JSON object: {{"score": <float>, "unsupported_claims": [<list of specific unsupported claims>]}}"""

_CONTEXT_RECALL_PROMPT = """You are an expert legal document evaluator. Your task is to assess CONTEXT RECALL.

Given a SOURCE document (original court text) and an ADAPTED document (translated/standardized version), evaluate whether all important information from the SOURCE is preserved in the ADAPTED text.

Score from 0.0 to 1.0:
- 1.0 = All key information from the source is present in the adapted text
- 0.5 = About half the key information is preserved
- 0.0 = The adapted text omits virtually all source information

SOURCE TEXT:
{source_text}

ADAPTED TEXT:
{adapted_text}

Respond with ONLY a JSON object: {{"score": <float>, "omitted_information": [<list of important omissions>]}}"""

_LEGAL_CONSISTENCY_PROMPT = """You are an expert Indian legal analyst. Your task is to assess LEGAL CONSISTENCY.

Given an ADAPTED legal document and a list of EXTRACTED ENTITIES, evaluate whether:
1. All legal section references are correctly attributed to the right statutes
2. All precedent citations are properly formatted and plausible
3. All judge names and designations are consistent
4. There are no contradictory legal references

ADAPTED TEXT:
{adapted_text}

EXTRACTED ENTITIES:
{entities_summary}

KNOWLEDGE GRAPH CONTEXT:
{graph_context}

Score from 0.0 to 1.0:
- 1.0 = All legal references are internally consistent and correctly attributed
- 0.5 = Some references are inconsistent or misattributed
- 0.0 = Major inconsistencies throughout

Respond with ONLY a JSON object: {{"score": <float>, "inconsistencies": [<list of specific issues>]}}"""

_CORRECTION_FEEDBACK_PROMPT = """You are a senior legal editor. Based on the evaluation results below, provide specific, actionable correction instructions.

EVALUATION RESULTS:
- Faithfulness score: {faithfulness} (threshold: {faith_threshold})
- Context recall score: {context_recall} (threshold: {recall_threshold})
- Legal consistency score: {legal_consistency} (threshold: {consistency_threshold})

SPECIFIC ISSUES FOUND:
{issues}

ADAPTED TEXT (to be corrected):
{adapted_text}

SOURCE TEXT (ground truth):
{source_text}

Provide concise, specific instructions on what exactly needs to be fixed. Focus on the most impactful corrections first."""


# ─── Helper Functions ─────────────────────────────────────────────────────────


def _summarize_entities(entities: list[Any]) -> str:
    """Create a summary string of entities for the evaluation prompt."""
    if not entities:
        return "No entities extracted."

    lines: list[str] = []
    for e in entities[:20]:  # Cap to avoid prompt overflow
        if hasattr(e, "case_number"):
            lines.append(f"Case: {e.case_number} at {e.court_name}")
        elif hasattr(e, "citation"):
            lines.append(f"Precedent: {e.citation} ({e.case_name or 'unnamed'})")
        elif hasattr(e, "section_number"):
            lines.append(f"Section: {e.section_number} {e.parent_statute or ''}")
        elif hasattr(e, "statute_name"):
            lines.append(f"Statute: {e.statute_name} ({e.abbreviation or ''})")
        elif hasattr(e, "full_name"):
            lines.append(f"Judge: {e.full_name} ({e.designation or ''})")
    return "\n".join(lines)


def _truncate(text: str, max_chars: int = 6000) -> str:
    """Truncate text to fit within LLM context, preserving complete sentences."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_period = truncated.rfind(".")
    if last_period > max_chars * 0.8:
        return truncated[: last_period + 1] + "\n[...truncated...]"
    return truncated + "\n[...truncated...]"


async def _llm_evaluate(prompt: str, llm: Any) -> dict[str, Any]:
    """
    Invoke the LLM with an evaluation prompt and parse the JSON response.

    Falls back to a default score of 0.5 if parsing fails.
    """
    import json

    try:
        response = await llm.ainvoke(
            [SystemMessage(content="You are an evaluation assistant. Respond only with valid JSON."),
             HumanMessage(content=prompt)]
        )
        content = response.content.strip()
        # Extract JSON from the response (handle markdown code blocks)
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        return json.loads(content)
    except (json.JSONDecodeError, IndexError, AttributeError) as exc:
        logger.warning("agent.evaluator.parse_failed", error=str(exc))
        return {"score": 0.5, "issues": ["Failed to parse LLM evaluation response"]}


# ─── Agent Node ───────────────────────────────────────────────────────────────


async def evaluator_node(state: NyayaEvalState) -> dict[str, Any]:
    """
    Evaluate translation fidelity, context recall, and legal consistency.

    Flow:
        1. Run faithfulness evaluation (LLM-as-judge)
        2. Run context recall evaluation (LLM-as-judge)
        3. Run legal consistency evaluation (LLM-as-judge with KG context)
        4. Compare scores against thresholds
        5. If failed, generate correction feedback
        6. Set routing flags (needs_correction, is_verified)

    Returns:
        Partial state update with scores, routing flags, and logs.
    """
    document_id = state.get("document_id", "unknown")
    retry_count = state.get("retry_count", 0)
    raw_text = state.get("raw_text", "")
    adapted_text = state.get("adapted_text", "")
    entities = state.get("entities", [])
    graph_context = state.get("graph_context", {})
    start_time = time.monotonic()

    logger.info("agent.evaluator.start", document_id=document_id, retry_count=retry_count)

    settings = get_settings()
    evaluation_details: dict[str, Any] = {"retry_count": retry_count}

    # ── Guard: nothing to evaluate ────────────────────────────────────────
    if not adapted_text:
        scores = EvaluationScores()
        return {
            "evaluation_scores": scores,
            "evaluation_details": {"error": "No adapted text to evaluate"},
            "needs_correction": True,
            "is_verified": False,
            "current_phase": "evaluation",
            "execution_logs": [f"[evaluator] No adapted text for {document_id} — marking for correction"],
        }

    # ── Get LLM ───────────────────────────────────────────────────────────
    try:
        llm = get_llm()
    except Exception as exc:
        logger.error("agent.evaluator.llm_unavailable", error=str(exc))
        scores = EvaluationScores(faithfulness=0.5, context_recall=0.5, legal_consistency=0.5)
        return {
            "evaluation_scores": scores,
            "evaluation_details": {"error": f"LLM unavailable: {exc}"},
            "needs_correction": True,
            "is_verified": False,
            "current_phase": "evaluation",
            "execution_logs": [f"[evaluator] LLM unavailable for {document_id} — using default scores"],
        }

    source_truncated = _truncate(raw_text)
    adapted_truncated = _truncate(adapted_text)
    all_issues: list[str] = []

    # ── 1. Faithfulness ───────────────────────────────────────────────────
    faith_prompt = _FAITHFULNESS_PROMPT.format(
        source_text=source_truncated, adapted_text=adapted_truncated
    )
    faith_result = await _llm_evaluate(faith_prompt, llm)
    faithfulness_score = float(faith_result.get("score", 0.5))
    faith_issues = faith_result.get("unsupported_claims", [])
    all_issues.extend([f"[faithfulness] {issue}" for issue in faith_issues])
    evaluation_details["faithfulness_details"] = faith_result

    # ── 2. Context Recall ─────────────────────────────────────────────────
    recall_prompt = _CONTEXT_RECALL_PROMPT.format(
        source_text=source_truncated, adapted_text=adapted_truncated
    )
    recall_result = await _llm_evaluate(recall_prompt, llm)
    context_recall_score = float(recall_result.get("score", 0.5))
    recall_issues = recall_result.get("omitted_information", [])
    all_issues.extend([f"[context_recall] {issue}" for issue in recall_issues])
    evaluation_details["context_recall_details"] = recall_result

    # ── 3. Legal Consistency ──────────────────────────────────────────────
    entities_summary = _summarize_entities(entities)
    graph_context_str = str(graph_context.get("subgraph", "No graph context available"))

    consistency_prompt = _LEGAL_CONSISTENCY_PROMPT.format(
        adapted_text=adapted_truncated,
        entities_summary=entities_summary,
        graph_context=graph_context_str[:2000],
    )
    consistency_result = await _llm_evaluate(consistency_prompt, llm)
    legal_consistency_score = float(consistency_result.get("score", 0.5))
    consistency_issues = consistency_result.get("inconsistencies", [])
    all_issues.extend([f"[legal_consistency] {issue}" for issue in consistency_issues])
    evaluation_details["legal_consistency_details"] = consistency_result

    # ── Build Scores ──────────────────────────────────────────────────────
    scores = EvaluationScores(
        faithfulness=min(max(faithfulness_score, 0.0), 1.0),
        context_recall=min(max(context_recall_score, 0.0), 1.0),
        legal_consistency=min(max(legal_consistency_score, 0.0), 1.0),
        answer_relevancy=0.0,  # Computed via RAGAS in full integration
    )

    passes = scores.passes_thresholds(
        faithfulness_min=settings.evaluation_faithfulness_threshold,
        context_recall_min=settings.evaluation_context_recall_threshold,
        legal_consistency_min=settings.evaluation_legal_consistency_threshold,
    )

    # ── Generate Correction Feedback (on failure) ─────────────────────────
    correction_feedback_str = ""
    if not passes:
        failed_metrics: list[str] = []
        if scores.faithfulness < settings.evaluation_faithfulness_threshold:
            failed_metrics.append("faithfulness")
        if scores.context_recall < settings.evaluation_context_recall_threshold:
            failed_metrics.append("context_recall")
        if scores.legal_consistency < settings.evaluation_legal_consistency_threshold:
            failed_metrics.append("legal_consistency")

        try:
            feedback_prompt = _CORRECTION_FEEDBACK_PROMPT.format(
                faithfulness=scores.faithfulness,
                faith_threshold=settings.evaluation_faithfulness_threshold,
                context_recall=scores.context_recall,
                recall_threshold=settings.evaluation_context_recall_threshold,
                legal_consistency=scores.legal_consistency,
                consistency_threshold=settings.evaluation_legal_consistency_threshold,
                issues="\n".join(all_issues[:10]),
                adapted_text=adapted_truncated[:3000],
                source_text=source_truncated[:3000],
            )
            feedback_response = await llm.ainvoke(
                [HumanMessage(content=feedback_prompt)]
            )
            correction_feedback_str = feedback_response.content
        except Exception as exc:
            correction_feedback_str = f"Auto-generated: Fix issues in {', '.join(failed_metrics)}"
            logger.warning("agent.evaluator.feedback_generation_failed", error=str(exc))

    duration_s = time.monotonic() - start_time
    evaluation_details["duration_s"] = round(duration_s, 2)
    evaluation_details["verdict"] = scores.verdict

    logger.info(
        "agent.evaluator.complete",
        document_id=document_id,
        faithfulness=scores.faithfulness,
        context_recall=scores.context_recall,
        legal_consistency=scores.legal_consistency,
        verdict=scores.verdict,
        passes=passes,
        duration_s=round(duration_s, 2),
    )

    result: dict[str, Any] = {
        "evaluation_scores": scores,
        "evaluation_details": evaluation_details,
        "needs_correction": not passes,
        "is_verified": passes,
        "current_phase": "evaluation",
        "execution_logs": [
            f"[evaluator] {document_id} (attempt {retry_count}): "
            f"faith={scores.faithfulness:.2f}, recall={scores.context_recall:.2f}, "
            f"consistency={scores.legal_consistency:.2f} → {scores.verdict} "
            f"({duration_s:.1f}s)"
        ],
    }

    if correction_feedback_str:
        result["correction_feedback"] = correction_feedback_str

    return result
