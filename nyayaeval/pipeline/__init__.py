"""
nyayaeval.pipeline — LangGraph Orchestration Layer
====================================================

Constructs the LangGraph StateGraph, defines conditional routing edges,
and configures checkpoint persistence. This is the "wiring" layer that
connects agents into a coherent evaluation pipeline.

Modules:
    graph        : StateGraph construction and compilation
    routing      : Conditional edge functions for evaluation loop control
    checkpointer : LangGraph persistence backend configuration
"""
