"""
nyayaeval.connectors.redis_client — Async Redis Adapter
=========================================================

Provides a thin async wrapper around the ``redis`` library for:
    1. Caching translated segments (avoid re-translating identical text)
    2. LangGraph checkpoint persistence (state durability across restarts)

Uses redis[hiredis] for optimized C-level parsing performance.
"""

from __future__ import annotations

from typing import Any

import redis.asyncio as aioredis
import structlog

logger = structlog.get_logger(__name__)


class NyayaRedisClient:
    """Async Redis client with connection pooling via redis-py."""

    def __init__(self, url: str = "redis://localhost:6379/0") -> None:
        self._url = url
        self._client: aioredis.Redis | None = None

    async def connect(self) -> None:
        """Initialize the Redis connection pool."""
        self._client = aioredis.from_url(self._url, decode_responses=True)
        await self._client.ping()
        logger.info("redis_client.connected", url=self._url)

    async def close(self) -> None:
        """Close the Redis connection pool."""
        if self._client:
            await self._client.aclose()
            self._client = None
            logger.info("redis_client.closed")

    async def __aenter__(self) -> NyayaRedisClient:
        await self.connect()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    async def get(self, key: str) -> str | None:
        """Get a value by key. Returns None if not found."""
        if not self._client:
            raise RuntimeError("Redis client not connected.")
        return await self._client.get(key)

    async def set(self, key: str, value: str, ttl: int | None = None) -> None:
        """Set a key-value pair with optional TTL in seconds."""
        if not self._client:
            raise RuntimeError("Redis client not connected.")
        await self._client.set(key, value, ex=ttl)

    async def delete(self, key: str) -> None:
        """Delete a key."""
        if not self._client:
            raise RuntimeError("Redis client not connected.")
        await self._client.delete(key)

    async def health_check(self) -> dict[str, Any]:
        """Verify Redis connectivity."""
        if not self._client:
            return {"status": "disconnected", "url": self._url}
        try:
            await self._client.ping()
            return {"status": "healthy", "url": self._url}
        except Exception as exc:
            return {"status": "unhealthy", "url": self._url, "error": str(exc)}
