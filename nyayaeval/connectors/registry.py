"""
nyayaeval.connectors.registry — Connector Dependency Registry
================================================================

Solves the "how do LangGraph nodes access connectors?" problem.

LangGraph nodes are pure functions that receive only ``state: NyayaEvalState``
as input. They cannot accept injected constructor arguments. But agents need
Neo4j clients, Redis caches, Adaption API clients, and LLMs.

This module provides a thread-safe, module-level registry that the FastAPI
lifespan populates at startup and agents import at execution time. This is
a deliberate compromise:

    - We CANNOT put connectors in the state dict (they're non-serializable).
    - We CANNOT use constructor injection (LangGraph nodes are bare functions).
    - We CAN use a module-level registry that is initialized once and accessed
      by all nodes running in the same process.

For testing, call ``reset()`` then ``register_*()`` with mocks before invoking
agent nodes.
"""

from __future__ import annotations


import structlog

from nyayaeval.connectors.adaption_client import AdaptiveDataClient
from nyayaeval.connectors.neo4j_client import NyayaNeo4jClient
from nyayaeval.connectors.redis_client import NyayaRedisClient

logger = structlog.get_logger(__name__)

# ─── Module-level singleton references ────────────────────────────────────────
_neo4j_client: NyayaNeo4jClient | None = None
_redis_client: NyayaRedisClient | None = None
_adaption_client: AdaptiveDataClient | None = None


def register_neo4j(client: NyayaNeo4jClient) -> None:
    """Register the Neo4j client singleton."""
    global _neo4j_client
    _neo4j_client = client
    logger.info("registry.neo4j_registered")


def register_redis(client: NyayaRedisClient) -> None:
    """Register the Redis client singleton."""
    global _redis_client
    _redis_client = client
    logger.info("registry.redis_registered")


def register_adaption(client: AdaptiveDataClient) -> None:
    """Register the Adaptive Data client singleton."""
    global _adaption_client
    _adaption_client = client
    logger.info("registry.adaption_registered")


def get_neo4j() -> NyayaNeo4jClient:
    """Retrieve the registered Neo4j client. Raises if not registered."""
    if _neo4j_client is None:
        raise RuntimeError(
            "Neo4j client not registered. Call register_neo4j() during app startup."
        )
    return _neo4j_client


def get_redis() -> NyayaRedisClient:
    """Retrieve the registered Redis client. Raises if not registered."""
    if _redis_client is None:
        raise RuntimeError(
            "Redis client not registered. Call register_redis() during app startup."
        )
    return _redis_client


def get_adaption() -> AdaptiveDataClient:
    """Retrieve the registered Adaptive Data client. Raises if not registered."""
    if _adaption_client is None:
        raise RuntimeError(
            "Adaption client not registered. Call register_adaption() during app startup."
        )
    return _adaption_client


def reset() -> None:
    """Clear all registered connectors. Used in tests."""
    global _neo4j_client, _redis_client, _adaption_client
    _neo4j_client = None
    _redis_client = None
    _adaption_client = None
    logger.info("registry.reset")
