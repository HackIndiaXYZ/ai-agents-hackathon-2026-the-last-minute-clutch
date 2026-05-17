"""
nyayaeval.pipeline.graph — LangGraph StateGraph Construction
==============================================================

Builds and compiles the NyayaEval evaluation pipeline as a LangGraph
StateGraph. The graph topology implements a cyclic self-correction loop:

    ingestion → adaptation → graph_builder → evaluator
                                                ↓
                                        [passes?] ──yes──→ export → END
                                            │
                                           no
                                            │
                                        corrector → evaluator (retry)
                                            │
                                    [max retries?] ──yes──→ END (failed)

Design: The graph is compiled once at import/startup and reused for all
pipeline invocations. Thread-safe via LangGraph's internal state isolation.
"""

from __future__ import annotations

import structlog
from langgraph.graph import END, StateGraph

from nyayaeval.agents.adaptation import adaptation_node
from nyayaeval.agents.corrector import corrector_node
from nyayaeval.agents.evaluator import evaluator_node
from nyayaeval.agents.export_node import export_node
from nyayaeval.agents.graph_builder import graph_builder_node
from nyayaeval.agents.ingestion import ingestion_node
from nyayaeval.core.state import NyayaEvalState
from nyayaeval.pipeline.routing import evaluation_router

logger = structlog.get_logger(__name__)


def build_pipeline() -> StateGraph:
    """
    Construct the NyayaEval StateGraph.

    Returns an uncompiled StateGraph. Call ``.compile()`` with an optional
    checkpointer to get a runnable graph.

    Returns:
        Configured StateGraph with all nodes and edges.
    """
    graph = StateGraph(NyayaEvalState)

    # ── Register nodes ────────────────────────────────────────────────────
    graph.add_node("ingestion", ingestion_node)
    graph.add_node("adaptation", adaptation_node)
    graph.add_node("graph_builder", graph_builder_node)
    graph.add_node("evaluator", evaluator_node)
    graph.add_node("corrector", corrector_node)
    graph.add_node("export", export_node)

    # ── Define edges ──────────────────────────────────────────────────────
    # Linear flow: ingestion → adaptation → graph_builder → evaluator
    graph.set_entry_point("ingestion")
    graph.add_edge("ingestion", "adaptation")
    graph.add_edge("adaptation", "graph_builder")
    graph.add_edge("graph_builder", "evaluator")

    # Conditional routing after evaluation
    graph.add_conditional_edges(
        "evaluator",
        evaluation_router,
        {
            "corrector": "corrector",
            "export": "export",
            "failed": END,
        },
    )

    # Corrector always loops back to evaluator
    graph.add_edge("corrector", "evaluator")

    # Export is the terminal success node
    graph.add_edge("export", END)

    logger.info("pipeline.graph_built", nodes=list(graph.nodes.keys()))
    return graph


def compile_pipeline(checkpointer: object | None = None) -> object:
    """
    Build and compile the pipeline into a runnable graph.

    Args:
        checkpointer: Optional LangGraph checkpointer for state persistence.

    Returns:
        Compiled, runnable LangGraph graph.
    """
    graph = build_pipeline()
    compiled = graph.compile(checkpointer=checkpointer)
    logger.info("pipeline.compiled")
    return compiled
