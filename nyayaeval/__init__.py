"""
NyayaEval — High-throughput multilingual legal document evaluation pipeline.

This is the top-level package for the NyayaEval system. It orchestrates the
ingestion, adaptation, knowledge-graph construction, evaluation, and export
of Indian district court documents.

Architecture follows a hexagonal (ports & adapters) pattern:
    - core/        : Pure domain models and state (zero external deps)
    - agents/      : LangGraph node functions (pipeline stages)
    - connectors/  : External I/O adapters (Neo4j, Redis, APIs)
    - pipeline/    : LangGraph graph construction and routing
    - export/      : Output serialization (JSONL, CSV)
    - api/         : FastAPI HTTP interface
    - config/      : Environment-based configuration
"""

__version__ = "0.1.0"
