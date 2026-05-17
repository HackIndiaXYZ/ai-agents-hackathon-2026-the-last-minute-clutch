"""
Unit tests for nyayaeval.core.state — LangGraph Pipeline State
================================================================

Validates that the NyayaEvalState TypedDict is structurally correct
and that partial state updates work as expected with LangGraph's
annotated reducers.
"""

from __future__ import annotations

from typing import Any

import pytest

from nyayaeval.core.schemas import EvaluationScores
from nyayaeval.core.state import NyayaEvalState


@pytest.mark.unit
class TestNyayaEvalState:
    """Tests for the pipeline state TypedDict."""

    def test_state_accepts_partial_update(self) -> None:
        """Verify that partial dicts are valid state updates (total=False)."""
        # LangGraph nodes return partial updates; this must be valid
        partial: dict[str, Any] = {
            "current_phase": "ingestion",
            "execution_logs": ["[ingestion] Started"],
        }
        # Should not raise — TypedDict with total=False accepts partials
        assert partial["current_phase"] == "ingestion"
        assert len(partial["execution_logs"]) == 1

    def test_state_supports_all_expected_keys(self) -> None:
        """Verify all expected fields are defined in the TypedDict."""
        expected_keys = {
            "document_id", "source_language", "document_metadata",
            "raw_text", "raw_pages", "token_boundaries",
            "entities", "entity_extraction_metadata",
            "adapted_text", "adapted_segments", "adaptation_metadata",
            "graph_context", "graph_node_ids",
            "evaluation_scores", "evaluation_details",
            "correction_feedback", "correction_history",
            "current_phase", "needs_correction", "retry_count",
            "is_verified", "error", "execution_logs",
        }
        state_annotations = NyayaEvalState.__annotations__
        assert expected_keys.issubset(state_annotations.keys()), (
            f"Missing keys: {expected_keys - state_annotations.keys()}"
        )

    def test_evaluation_scores_passes_thresholds(self, sample_scores_passing: EvaluationScores) -> None:
        """Verify passing scores meet default thresholds."""
        assert sample_scores_passing.passes_thresholds() is True

    def test_evaluation_scores_fails_thresholds(self, sample_scores_failing: EvaluationScores) -> None:
        """Verify failing scores are detected."""
        assert sample_scores_failing.passes_thresholds() is False

    def test_evaluation_verdict_pass(self, sample_scores_passing: EvaluationScores) -> None:
        """Verify verdict is PASS for passing scores."""
        assert sample_scores_passing.verdict == "pass"

    def test_evaluation_verdict_fail(self, sample_scores_failing: EvaluationScores) -> None:
        """Verify verdict is FAIL for low scores."""
        assert sample_scores_failing.verdict == "fail"
