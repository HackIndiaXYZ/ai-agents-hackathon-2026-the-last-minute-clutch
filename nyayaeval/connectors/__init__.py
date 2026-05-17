"""
nyayaeval.connectors — External I/O Adapters
==============================================

This package isolates all external system interactions behind clean async
interfaces. The hexagonal architecture ensures that:

    1. **Agent nodes never import drivers directly** — they access connectors
       via the registry module (see ``registry.py``).
    2. **Swapping infrastructure is localized** — replacing Neo4j with
       another graph DB means changing only the connector, not every agent.
    3. **Testing is trivial** — mock the connector interface, test agent
       logic in isolation.

Connectors:
    neo4j_client    : Async, connection-pooled Neo4j driver wrapper
    adaption_client : Official Adaption SDK wrapper (Adaptive Data platform)
    redis_client    : Async Redis adapter for caching and checkpointing
    llm_provider    : Multi-provider LLM abstraction (OpenAI, Gemini, Groq)
    registry        : Module-level DI registry for connector access from agents
"""

from nyayaeval.connectors.neo4j_client import NyayaNeo4jClient
from nyayaeval.connectors.redis_client import NyayaRedisClient
from nyayaeval.connectors.adaption_client import AdaptiveDataClient

__all__ = [
    "NyayaNeo4jClient",
    "NyayaRedisClient",
    "AdaptiveDataClient",
]
