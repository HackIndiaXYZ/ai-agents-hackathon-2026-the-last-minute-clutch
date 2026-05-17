"""
Unit tests for nyayaeval.agents — Agent Node Functions
=======================================================

Tests each agent node's state contract: correct input fields read,
correct output fields returned, and proper error handling.

These tests use mock connectors via the registry, so no running
infrastructure is required.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nyayaeval.core.models import (
    Case,
    CaseStatus,
    Judge,
    Precedent,
    Section,
    Statute,
)
from nyayaeval.core.schemas import EvaluationScores


@pytest.mark.unit
class TestIngestionNode:
    """Tests for the ingestion agent node."""

    @pytest.mark.asyncio
    async def test_ingestion_with_raw_text(self, sample_pipeline_state: dict[str, Any]) -> None:
        """Ingestion should process pre-populated raw_text."""
        from nyayaeval.agents.ingestion import ingestion_node

        state = {**sample_pipeline_state, "raw_text": "This is a test court document."}

        with patch("nyayaeval.agents.ingestion.get_llm") as mock_llm:
            # Mock the LLM to return empty entities
            mock_structured = AsyncMock()
            mock_structured.ainvoke = AsyncMock(return_value=MagicMock(
                cases=[], sections=[], statutes=[], judges=[], precedents=[]
            ))
            mock_llm.return_value.with_structured_output.return_value = mock_structured

            result = await ingestion_node(state)

        assert "raw_text" in result
        assert "raw_pages" in result
        assert "entities" in result
        assert "current_phase" in result
        assert result["current_phase"] == "ingestion"
        assert len(result["execution_logs"]) > 0

    @pytest.mark.asyncio
    async def test_ingestion_empty_text_fails(self) -> None:
        """Ingestion with no text and no file should return error."""
        from nyayaeval.agents.ingestion import ingestion_node

        state: dict[str, Any] = {
            "document_id": "test-empty",
            "raw_text": "",
            "document_metadata": {},
        }
        result = await ingestion_node(state)

        assert result.get("error") is not None
        assert result["current_phase"] == "failed"


@pytest.mark.unit
class TestAdaptationNode:
    """Tests for the adaptation agent node."""

    @pytest.mark.asyncio
    async def test_english_passthrough(self, sample_pipeline_state: dict[str, Any]) -> None:
        """English documents should pass through without API calls."""
        from nyayaeval.agents.adaptation import adaptation_node

        state = {
            **sample_pipeline_state,
            "source_language": "en",
            "raw_text": "This is English text.",
            "raw_pages": ["This is English text."],
        }

        result = await adaptation_node(state)

        assert result["adapted_text"] == "This is English text."
        assert result["current_phase"] == "adaptation"
        assert result["adaptation_metadata"]["method"] == "passthrough"

    @pytest.mark.asyncio
    async def test_adaptation_without_clients(self, sample_pipeline_state: dict[str, Any]) -> None:
        """Should fall back to passthrough when no Adaption client is configured."""
        from nyayaeval.agents.adaptation import adaptation_node
        from nyayaeval.connectors import registry

        registry.reset()

        state = {
            **sample_pipeline_state,
            "source_language": "hi",
            "raw_text": "Hindi text here",
            "raw_pages": ["Hindi text here"],
        }

        result = await adaptation_node(state)

        assert result["adapted_text"] == "Hindi text here"
        assert "passthrough" in result["adaptation_metadata"]["method"]


@pytest.mark.unit
class TestGraphBuilderNode:
    """Tests for the graph builder agent node."""

    @pytest.mark.asyncio
    async def test_graph_builder_no_entities(self, sample_pipeline_state: dict[str, Any]) -> None:
        """Graph builder with empty entities should return empty context."""
        from nyayaeval.agents.graph_builder import graph_builder_node

        state = {**sample_pipeline_state, "entities": []}
        result = await graph_builder_node(state)

        assert result["graph_node_ids"] == []
        assert result["current_phase"] == "graph_building"

    @pytest.mark.asyncio
    async def test_graph_builder_no_neo4j(self, sample_pipeline_state: dict[str, Any]) -> None:
        """Graph builder without Neo4j should warn and continue."""
        from nyayaeval.agents.graph_builder import graph_builder_node
        from nyayaeval.connectors import registry

        registry.reset()

        case = Case(
            case_number="TEST.001/2024",
            court_name="Test Court",
            source_document_id="test-doc",
        )
        state = {**sample_pipeline_state, "entities": [case]}
        result = await graph_builder_node(state)

        assert "warning" in result["graph_context"] or result["graph_node_ids"] == []
        assert result["current_phase"] == "graph_building"


@pytest.mark.unit
class TestEvaluatorNode:
    """Tests for the evaluator agent node."""

    @pytest.mark.asyncio
    async def test_evaluator_no_adapted_text(self, sample_pipeline_state: dict[str, Any]) -> None:
        """Evaluator with no adapted text should mark for correction."""
        from nyayaeval.agents.evaluator import evaluator_node

        state = {**sample_pipeline_state, "adapted_text": ""}
        result = await evaluator_node(state)

        assert result["needs_correction"] is True
        assert result["is_verified"] is False
        assert result["current_phase"] == "evaluation"

    @pytest.mark.asyncio
    async def test_evaluator_returns_scores(self, sample_pipeline_state: dict[str, Any]) -> None:
        """Evaluator should return EvaluationScores in output."""
        from nyayaeval.agents.evaluator import evaluator_node

        state = {
            **sample_pipeline_state,
            "adapted_text": "Translated legal document text",
            "raw_text": "Original legal document text",
        }

        with patch("nyayaeval.agents.evaluator.get_llm") as mock_llm:
            mock_model = AsyncMock()
            mock_model.ainvoke = AsyncMock(return_value=MagicMock(
                content='{"score": 0.9, "unsupported_claims": []}'
            ))
            mock_llm.return_value = mock_model

            result = await evaluator_node(state)

        assert "evaluation_scores" in result
        assert isinstance(result["evaluation_scores"], EvaluationScores)
        assert result["current_phase"] == "evaluation"


@pytest.mark.unit
class TestCorrectorNode:
    """Tests for the corrector agent node."""

    @pytest.mark.asyncio
    async def test_corrector_increments_retry(self, sample_pipeline_state: dict[str, Any]) -> None:
        """Corrector should increment retry_count."""
        from nyayaeval.agents.corrector import corrector_node

        state = {
            **sample_pipeline_state,
            "retry_count": 1,
            "adapted_text": "Text to correct",
            "raw_text": "Original text",
            "correction_feedback": "Fix section references",
        }

        with patch("nyayaeval.agents.corrector.get_llm") as mock_llm:
            mock_model = AsyncMock()
            mock_model.ainvoke = AsyncMock(return_value=MagicMock(
                content="Corrected text with fixed references"
            ))
            mock_llm.return_value = mock_model

            result = await corrector_node(state)

        assert result["retry_count"] == 2
        assert result["current_phase"] == "correction"
        assert len(result["correction_history"]) == 1
        assert result["correction_history"][0]["attempt"] == 2

    @pytest.mark.asyncio
    async def test_corrector_appends_history(self, sample_pipeline_state: dict[str, Any]) -> None:
        """Corrector should append to correction_history (uses operator.add reducer)."""
        from nyayaeval.agents.corrector import corrector_node

        state = {
            **sample_pipeline_state,
            "retry_count": 0,
            "adapted_text": "Original adapted",
            "correction_feedback": "Fix it",
        }

        with patch("nyayaeval.agents.corrector.get_llm") as mock_llm:
            mock_model = AsyncMock()
            mock_model.ainvoke = AsyncMock(return_value=MagicMock(content="Fixed text"))
            mock_llm.return_value = mock_model

            result = await corrector_node(state)

        assert isinstance(result["correction_history"], list)
        assert len(result["correction_history"]) == 1


@pytest.mark.unit
class TestExportNode:
    """Tests for the export agent node."""

    @pytest.mark.asyncio
    async def test_export_node_returns_completed(self, sample_pipeline_state: dict[str, Any]) -> None:
        """Export node should set current_phase to completed."""
        from nyayaeval.agents.export_node import export_node

        state = {
            **sample_pipeline_state,
            "adapted_text": "Verified text",
            "is_verified": True,
        }

        result = await export_node(state)

        assert "completed" in result["current_phase"]
        assert len(result["execution_logs"]) > 0
