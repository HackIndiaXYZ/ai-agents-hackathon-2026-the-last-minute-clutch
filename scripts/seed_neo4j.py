"""
scripts/seed_neo4j.py — Neo4j Schema & Sample Data Seeder
============================================================

Standalone script to initialize the Neo4j database with:
    1. Uniqueness constraints for all legal entity labels
    2. Optional sample data for development and testing

Usage:
    python -m scripts.seed_neo4j

Requires:
    - Running Neo4j instance (docker compose up -d neo4j)
    - .env file with NEO4J_* variables configured
"""

from __future__ import annotations

import asyncio

import structlog

from nyayaeval.config import get_settings
from nyayaeval.connectors.neo4j_client import NyayaNeo4jClient

logger = structlog.get_logger(__name__)


async def seed() -> None:
    """Initialize Neo4j schema constraints."""
    settings = get_settings()
    client = NyayaNeo4jClient(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
        max_connection_pool_size=5,
    )

    async with client:
        logger.info("seed.ensuring_constraints")
        await client.ensure_constraints()
        logger.info("seed.complete")


if __name__ == "__main__":
    asyncio.run(seed())
