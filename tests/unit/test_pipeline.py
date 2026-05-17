"""
Unit tests for nyayaeval.pipeline — Graph Construction & Routing
=================================================================

Tests that the pipeline graph builds correctly, compiles without errors,
and the routing function returns the right next-node for all branches.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest


@pytest.mark.unit
class TestPipelineGraph:
    """Tests for pipeline graph construction."""

    def test_build_pipeline_creates_all_nodes(self) -> None:
        """The graph should contain all 6 agent nodes."""
        from nyayaeval.pipeline.graph import build_pipeline

        graph = build_pipeline()
        expected_nodes = {"ingestion", "adaptation", "graph_builder", "evaluator", "corrector", "export"}
        assert expected_nodes.issubset(set(graph.nodes.keys()))

    def test_compile_pipeline_succeeds(self) -> None:
        """The pipeline should compile without errors."""
        from nyayaeval.pipeline.graph import compile_pipeline

        compiled = compile_pipeline(checkpointer=None)
        assert compiled is not None


@pytest.mark.unit
class TestEvaluationRouter:
    """Tests for the evaluation routing function."""

    def test_router_passes_to_export(self) -> None:
        """Verified state should route to export."""
        from nyayaeval.pipeline.routing import evaluation_router

        state: dict[str, Any] = {
            "is_verified": True,
            "retry_count": 0,
            "document_id": "test-doc",
        }
        assert evaluation_router(state) == "export"

    def test_router_fails_to_corrector(self) -> None:
        """Failed state with retries remaining should route to corrector."""
        from nyayaeval.pipeline.routing import evaluation_router

        state: dict[str, Any] = {
            "is_verified": False,
            "retry_count": 0,
            "document_id": "test-doc",
        }
        assert evaluation_router(state) == "corrector"

    def test_router_exhausted_retries_to_failed(self) -> None:
        """Exhausted retries should route to failed."""
        from nyayaeval.pipeline.routing import evaluation_router

        state: dict[str, Any] = {
            "is_verified": False,
            "retry_count": 3,  # Default max_retries is 3
            "document_id": "test-doc",
        }
        assert evaluation_router(state) == "failed"

    def test_router_partial_retries_to_corrector(self) -> None:
        """Still has retries → should route to corrector."""
        from nyayaeval.pipeline.routing import evaluation_router

        state: dict[str, Any] = {
            "is_verified": False,
            "retry_count": 1,
            "document_id": "test-doc",
        }
        assert evaluation_router(state) == "corrector"
