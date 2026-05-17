"""
nyayaeval.connectors.neo4j_client — Async Neo4j Database Interface
===================================================================

Production-grade async Neo4j client with connection pooling, retry logic,
and context management. See module docstrings for design rationale.

Usage:
    async with NyayaNeo4jClient(uri, user, password) as client:
        records = await client.execute_read("MATCH (c:Case) RETURN c LIMIT 10")
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

import structlog
from neo4j import AsyncGraphDatabase, AsyncManagedTransaction, AsyncSession
from neo4j.exceptions import AuthError, Neo4jError, ServiceUnavailable, SessionExpired
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = structlog.get_logger(__name__)
_RETRYABLE = (ServiceUnavailable, SessionExpired)


class Neo4jConnectionError(Exception):
    """Cannot establish or maintain a Neo4j connection."""


class Neo4jQueryError(Exception):
    """Query failed after exhausting retries."""


class NyayaNeo4jClient:
    """Async, connection-pooled Neo4j client with retry logic."""

    def __init__(self, uri: str, user: str, password: str,
                 database: str = "neo4j", max_connection_pool_size: int = 50) -> None:
        self._uri = uri
        self._user = user
        self._password = password
        self._database = database
        self._max_pool_size = max_connection_pool_size
        self._driver: Any = None
        self._is_connected = False

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Initialize the async driver and verify connectivity."""
        if self._is_connected and self._driver is not None:
            return
        try:
            self._driver = AsyncGraphDatabase.driver(
                self._uri, auth=(self._user, self._password),
                max_connection_pool_size=self._max_pool_size,
            )
            await self._driver.verify_connectivity()
            self._is_connected = True
            logger.info("neo4j_client.connected", uri=self._uri, database=self._database)
        except AuthError as exc:
            raise Neo4jConnectionError(f"Auth failed for {self._uri}: {exc}") from exc
        except ServiceUnavailable as exc:
            raise Neo4jConnectionError(f"Neo4j unavailable at {self._uri}: {exc}") from exc
        except Exception as exc:
            raise Neo4jConnectionError(f"Connection failed to {self._uri}: {exc}") from exc

    async def close(self) -> None:
        """Gracefully close the driver. Safe to call multiple times."""
        if self._driver is not None:
            await self._driver.close()
            self._driver = None
            self._is_connected = False
            logger.info("neo4j_client.closed", uri=self._uri)

    async def __aenter__(self) -> NyayaNeo4jClient:
        await self.connect()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    # ── Health Check ──────────────────────────────────────────────────────

    async def health_check(self) -> dict[str, Any]:
        """Verify connectivity and return diagnostic info."""
        if self._driver is None:
            return {"status": "disconnected", "uri": self._uri}
        start = time.monotonic()
        try:
            await self._driver.verify_connectivity()
            latency_ms = (time.monotonic() - start) * 1000
            server_info = await self._driver.get_server_info()
            return {"status": "healthy", "uri": self._uri, "database": self._database,
                    "server_agent": str(server_info.agent) if server_info else "unknown",
                    "latency_ms": round(latency_ms, 2)}
        except Exception as exc:
            return {"status": "unhealthy", "uri": self._uri, "error": str(exc)}

    # ── Internal Helpers ──────────────────────────────────────────────────

    def _ensure_connected(self) -> None:
        if self._driver is None or not self._is_connected:
            raise Neo4jConnectionError("Not connected. Call connect() or use 'async with'.")

    def _get_session(self, **kwargs: Any) -> AsyncSession:
        self._ensure_connected()
        return self._driver.session(database=self._database, **kwargs)

    # ── Query Execution ───────────────────────────────────────────────────

    @retry(retry=retry_if_exception_type(_RETRYABLE), stop=stop_after_attempt(3),
           wait=wait_exponential(multiplier=1, min=1, max=10), reraise=True)
    async def execute_read(self, query: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Execute a read query with automatic retry on transient failures."""
        self._ensure_connected()
        qhash = hashlib.sha256(query.encode()).hexdigest()[:12]
        start = time.monotonic()

        async def _tx(tx: AsyncManagedTransaction) -> list[dict[str, Any]]:
            result = await tx.run(query, parameters or {})
            return [record.data() async for record in result]

        try:
            async with self._get_session() as session:
                records = await session.execute_read(_tx)
                logger.debug("neo4j_client.read_ok", query_hash=qhash, count=len(records),
                             ms=round((time.monotonic() - start) * 1000, 2))
                return records
        except _RETRYABLE:
            raise
        except Neo4jError as exc:
            raise Neo4jQueryError(f"Read failed (hash={qhash}): {exc}") from exc

    @retry(retry=retry_if_exception_type(_RETRYABLE), stop=stop_after_attempt(3),
           wait=wait_exponential(multiplier=1, min=1, max=10), reraise=True)
    async def execute_write(self, query: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Execute a write query with automatic retry on transient failures."""
        self._ensure_connected()
        qhash = hashlib.sha256(query.encode()).hexdigest()[:12]
        start = time.monotonic()

        async def _tx(tx: AsyncManagedTransaction) -> list[dict[str, Any]]:
            result = await tx.run(query, parameters or {})
            return [record.data() async for record in result]

        try:
            async with self._get_session() as session:
                records = await session.execute_write(_tx)
                logger.info("neo4j_client.write_ok", query_hash=qhash, count=len(records),
                            ms=round((time.monotonic() - start) * 1000, 2))
                return records
        except _RETRYABLE:
            raise
        except Neo4jError as exc:
            raise Neo4jQueryError(f"Write failed (hash={qhash}): {exc}") from exc

    async def execute_write_batch(self, query: str, batch_data: list[dict[str, Any]],
                                   batch_param_name: str = "batch") -> list[dict[str, Any]]:
        """Batched write using UNWIND for high-throughput ingestion."""
        return await self.execute_write(query, parameters={batch_param_name: batch_data})

    # ── Schema Operations ─────────────────────────────────────────────────

    async def ensure_constraints(self) -> None:
        """Create uniqueness constraints for the legal KG schema. Idempotent."""
        constraints = [
            ("constraint_case_number",
             "CREATE CONSTRAINT constraint_case_number IF NOT EXISTS FOR (c:Case) REQUIRE c.case_number IS UNIQUE"),
            ("constraint_statute_name",
             "CREATE CONSTRAINT constraint_statute_name IF NOT EXISTS FOR (s:Statute) REQUIRE s.statute_name IS UNIQUE"),
            ("constraint_judge_name",
             "CREATE CONSTRAINT constraint_judge_name IF NOT EXISTS FOR (j:Judge) REQUIRE j.full_name IS UNIQUE"),
            ("constraint_precedent_citation",
             "CREATE CONSTRAINT constraint_precedent_citation IF NOT EXISTS FOR (p:Precedent) REQUIRE p.citation IS UNIQUE"),
        ]
        for name, query in constraints:
            try:
                await self.execute_write(query)
                logger.info("neo4j_client.constraint_ensured", constraint=name)
            except Neo4jQueryError:
                logger.warning("neo4j_client.constraint_failed", constraint=name)
