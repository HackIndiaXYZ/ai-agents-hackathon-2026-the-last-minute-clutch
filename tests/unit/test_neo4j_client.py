"""
Unit tests for nyayaeval.connectors.neo4j_client — Neo4j Client
=================================================================

Tests the Neo4j client's lifecycle, error handling, and guard clauses
WITHOUT requiring a running Neo4j instance.
"""

from __future__ import annotations

import pytest

from nyayaeval.connectors.neo4j_client import Neo4jConnectionError, NyayaNeo4jClient


@pytest.mark.unit
class TestNyayaNeo4jClient:
    """Unit tests for the async Neo4j client."""

    def test_client_initializes_disconnected(self) -> None:
        """Client should start in disconnected state."""
        client = NyayaNeo4jClient(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="test",
        )
        assert client._is_connected is False
        assert client._driver is None

    def test_client_default_pool_size(self) -> None:
        """Client should use default pool size of 50."""
        client = NyayaNeo4jClient(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="test",
        )
        assert client._max_pool_size == 50

    def test_client_custom_pool_size(self) -> None:
        """Client should accept custom pool size."""
        client = NyayaNeo4jClient(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="test",
            max_connection_pool_size=100,
        )
        assert client._max_pool_size == 100

    @pytest.mark.asyncio
    async def test_execute_read_raises_when_disconnected(self) -> None:
        """Queries on a disconnected client should raise Neo4jConnectionError."""
        client = NyayaNeo4jClient(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="test",
        )
        with pytest.raises(Neo4jConnectionError, match="Not connected"):
            await client.execute_read("MATCH (n) RETURN n LIMIT 1")

    @pytest.mark.asyncio
    async def test_execute_write_raises_when_disconnected(self) -> None:
        """Write queries on a disconnected client should raise Neo4jConnectionError."""
        client = NyayaNeo4jClient(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="test",
        )
        with pytest.raises(Neo4jConnectionError, match="Not connected"):
            await client.execute_write("CREATE (n:Test) RETURN n")

    @pytest.mark.asyncio
    async def test_health_check_disconnected(self) -> None:
        """Health check should report disconnected status."""
        client = NyayaNeo4jClient(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="test",
        )
        result = await client.health_check()
        assert result["status"] == "disconnected"

    @pytest.mark.asyncio
    async def test_close_is_idempotent(self) -> None:
        """Calling close() on an already-closed client should not raise."""
        client = NyayaNeo4jClient(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="test",
        )
        # Should not raise even though never connected
        await client.close()
        await client.close()
