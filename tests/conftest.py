"""
Shared test fixtures for the NyayaEval test suite.

Provides pre-configured instances of domain models, states, and mock
clients for use across unit and integration tests.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from nyayaeval.core.models import Case, CaseStatus, EntityType
from nyayaeval.core.schemas import EvaluationScores
from nyayaeval.core.state import NyayaEvalState


@pytest.fixture
def sample_case() -> Case:
    """A minimal Case entity for testing."""
    return Case(
        case_number="CRL.A.123/2024",
        court_name="District Court, Varanasi",
        case_title="State of UP vs. John Doe",
        status=CaseStatus.PENDING,
        jurisdiction="Varanasi",
        original_language="hi",
    )


@pytest.fixture
def sample_scores_passing() -> EvaluationScores:
    """Evaluation scores that pass all default thresholds."""
    return EvaluationScores(
        faithfulness=0.92,
        context_recall=0.88,
        legal_consistency=0.90,
        answer_relevancy=0.85,
    )


@pytest.fixture
def sample_scores_failing() -> EvaluationScores:
    """Evaluation scores that fail faithfulness threshold."""
    return EvaluationScores(
        faithfulness=0.60,
        context_recall=0.88,
        legal_consistency=0.90,
        answer_relevancy=0.85,
    )


@pytest.fixture
def sample_pipeline_state() -> dict[str, Any]:
    """A minimal pipeline state dict for testing agent nodes."""
    return {
        "document_id": "test-doc-001",
        "source_language": "hi",
        "document_metadata": {"filename": "test.pdf", "pages": 5},
        "raw_text": "यह एक परीक्षण दस्तावेज़ है।",
        "raw_pages": ["यह एक परीक्षण दस्तावेज़ है।"],
        "token_boundaries": [],
        "entities": [],
        "adapted_text": "",
        "adapted_segments": [],
        "adaptation_metadata": {},
        "graph_context": {},
        "graph_node_ids": [],
        "evaluation_scores": EvaluationScores(),
        "evaluation_details": {},
        "needs_correction": False,
        "retry_count": 0,
        "is_verified": False,
        "error": None,
        "current_phase": "ingestion",
        "execution_logs": [],
        "correction_history": [],
    }


@pytest.fixture(autouse=True)
def _set_test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set required environment variables for Settings in test mode."""
    monkeypatch.setenv("NEO4J_PASSWORD", "test_password")
    monkeypatch.setenv("ADAPTION_API_KEY", "test_api_key")
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test_gemini_key")
    monkeypatch.setenv("OPENAI_API_KEY", "test_openai_key")
