"""
nyayaeval.main — FastAPI Application Entrypoint
=================================================

Creates and configures the FastAPI application with:
    - Lifespan management (connect/disconnect infrastructure on startup/shutdown)
    - Route registration
    - Middleware setup
    - Structured logging configuration

Run with:
    uvicorn nyayaeval.main:app --reload
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from nyayaeval.api.middleware import setup_middleware
from nyayaeval.api.routes import router
from nyayaeval.config.logging import configure_logging
from nyayaeval.config.settings import get_settings
from nyayaeval.connectors.adaption_client import AdaptiveDataClient
from nyayaeval.connectors.neo4j_client import NyayaNeo4jClient
from nyayaeval.connectors.redis_client import NyayaRedisClient
from nyayaeval.connectors.registry import (
    register_adaption,
    register_neo4j,
    register_redis,
    reset,
)
from nyayaeval.pipeline.checkpointer import get_checkpointer
from nyayaeval.pipeline.graph import compile_pipeline

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Application lifespan manager.

    Initializes all infrastructure connections on startup, compiles the
    LangGraph pipeline, and tears everything down on shutdown.

    Connectors are registered in the module-level registry so that
    LangGraph agent nodes can access them without constructor injection.
    """
    settings = get_settings()
    configure_logging(log_level=settings.log_level, log_format=settings.log_format)

    logger.info("app.startup", version="0.1.0")

    # ── Neo4j ─────────────────────────────────────────────────────────────
    neo4j_client = NyayaNeo4jClient(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
        max_connection_pool_size=settings.neo4j_max_connection_pool_size,
    )
    try:
        await neo4j_client.connect()
        await neo4j_client.ensure_constraints()
        register_neo4j(neo4j_client)
        logger.info("app.neo4j_ready")
    except Exception as exc:
        logger.warning("app.neo4j_failed", error=str(exc), detail="Pipeline will run without KG")

    # ── Redis ─────────────────────────────────────────────────────────────
    redis_client = NyayaRedisClient(url=settings.redis_url)
    try:
        await redis_client.connect()
        register_redis(redis_client)
        logger.info("app.redis_ready")
    except Exception as exc:
        logger.warning("app.redis_failed", error=str(exc), detail="Caching disabled")

    # ── Adaption (Adaptive Data Platform) ────────────────────────────────
    adaption_client = AdaptiveDataClient(
        api_key=settings.adaption_api_key,
        timeout=settings.adaption_api_timeout,
    )
    try:
        await adaption_client.connect()
        register_adaption(adaption_client)
        logger.info("app.adaption_ready")
    except Exception as exc:
        logger.warning("app.adaption_failed", error=str(exc), detail="Adaptation via passthrough")

    # ── Compile Pipeline ──────────────────────────────────────────────────
    checkpointer = get_checkpointer()
    pipeline = compile_pipeline(checkpointer=checkpointer)
    app.state.pipeline = pipeline
    logger.info("app.pipeline_compiled")

    # ── Store clients on app.state for route access ───────────────────────
    app.state.neo4j = neo4j_client
    app.state.redis = redis_client
    app.state.adaption = adaption_client

    yield

    # ── Teardown ──────────────────────────────────────────────────────────
    logger.info("app.shutdown.start")
    await adaption_client.close()
    await redis_client.close()
    await neo4j_client.close()
    reset()
    logger.info("app.shutdown.complete")


# ── Application Factory ──────────────────────────────────────────────────────

app = FastAPI(
    title="NyayaEval",
    description=(
        "High-throughput multilingual legal document evaluation pipeline "
        "for Indian district court records."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# Register middleware
setup_middleware(app)

# Register routes
app.include_router(router, tags=["pipeline"])
