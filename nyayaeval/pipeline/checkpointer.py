"""
nyayaeval.pipeline.checkpointer — LangGraph Persistence Configuration
=======================================================================

Configures the LangGraph checkpointer for durable state persistence.
Uses Redis as the backend so that pipeline executions survive process
restarts and can be inspected/debugged via time-travel.

TODO (Phase 2):
    - Implement Redis-backed checkpointer once LangGraph's Redis
      checkpoint adapter is finalized
    - Add PostgresSaver as an alternative for production deployments
"""

from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)


def get_checkpointer() -> object | None:
    """
    Create and return a LangGraph checkpointer.

    Currently returns None (in-memory execution). Phase 2 will integrate
    a Redis or Postgres-backed checkpointer for durable persistence.

    Returns:
        A LangGraph-compatible checkpointer, or None for in-memory mode.
    """
    # TODO: Implement Redis checkpointer
    # from langgraph.checkpoint.redis import RedisSaver
    # from nyayaeval.config import get_settings
    # settings = get_settings()
    # return RedisSaver(url=settings.redis_url)

    logger.info("checkpointer.using_in_memory", reason="Phase 1 scaffold")
    return None
