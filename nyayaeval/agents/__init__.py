"""
nyayaeval.agents — LangGraph Pipeline Node Functions
=====================================================

Each module in this package implements a single LangGraph node — a pure
function that takes the current ``NyayaEvalState``, performs its stage-
specific logic, and returns a *partial* state update dict.

Node contract:
    Input:  NyayaEvalState (full accumulated state)
    Output: dict (partial state update — only mutated fields)

Agents NEVER import drivers or clients directly. They receive connector
instances through the pipeline context or dependency injection, keeping
the agent logic testable in isolation.

Pipeline flow:
    ingestion → adaptation → graph_builder → evaluator → [corrector ↺] → export
"""
