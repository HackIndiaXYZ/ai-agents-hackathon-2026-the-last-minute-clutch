"""
nyayaeval.config.settings — Pydantic BaseSettings Configuration
================================================================

All runtime configuration is loaded from environment variables (or a ``.env``
file). This module provides a single ``Settings`` class and a ``get_settings()``
factory that caches the instance.

Design decisions:
    - ``pydantic-settings`` (not raw ``os.environ``) gives us:
        • Type coercion (str → int, str → float, str → bool automatically)
        • Validation on startup (fail fast if a required key is missing)
        • IDE autocompletion and type safety
        • Centralized documentation of every config key
    - We use a functional cache (``@lru_cache``) rather than a module-level
      global because it makes testing trivial: call ``get_settings.cache_clear()``
      before injecting test overrides.
    - Field grouping mirrors the ``.env.example`` structure.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application-wide configuration.

    Loaded from environment variables or a ``.env`` file in the project root.
    All fields have sensible development defaults where safe to do so;
    secrets (API keys, passwords) intentionally have no defaults to force
    explicit configuration.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Don't fail on unexpected env vars
    )

    # ── Neo4j Knowledge Graph ─────────────────────────────────────────────
    neo4j_uri: str = Field(
        default="bolt://localhost:7687",
        description="Bolt URI for the Neo4j instance",
    )
    neo4j_user: str = Field(default="neo4j", description="Neo4j username")
    neo4j_password: str = Field(
        ..., description="Neo4j password (required, no default for security)"
    )
    neo4j_database: str = Field(
        default="neo4j", description="Target Neo4j database name"
    )
    neo4j_max_connection_pool_size: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Maximum connections in the Neo4j driver pool",
    )

    # ── Redis ─────────────────────────────────────────────────────────────
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL (used for caching and LangGraph checkpoints)",
    )

    # ── Adaption (Adaptive Data Platform) ────────────────────────────────
    adaption_api_key: str = Field(
        default="", description="Adaption API key (get from adaptionlabs.ai → Settings → API Keys)"
    )
    adaption_api_timeout: int = Field(
        default=120,
        ge=5,
        le=600,
        description="Timeout in seconds for Adaption SDK operations",
    )

    # ── LLM Provider ─────────────────────────────────────────────────────
    llm_provider: str = Field(
        default="gemini",
        description="LLM provider: 'openai', 'gemini', or 'groq'",
    )
    openai_api_key: str = Field(
        default="", description="OpenAI API key (required if llm_provider='openai')"
    )
    gemini_api_key: str = Field(
        default="", description="Google Gemini API key (required if llm_provider='gemini')"
    )
    groq_api_key: str = Field(
        default="", description="Groq API key (required if llm_provider='groq')"
    )
    llm_model_name: str = Field(
        default="gpt-4o",
        description="LLM model identifier — auto-resolved per provider if left as default",
    )
    llm_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="LLM temperature (0.0 for deterministic evaluation)",
    )

    # ── Pipeline Thresholds ───────────────────────────────────────────────
    max_retries: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum correction cycles before marking a document as failed",
    )
    evaluation_faithfulness_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Minimum faithfulness score to pass evaluation",
    )
    evaluation_context_recall_threshold: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Minimum context recall score to pass evaluation",
    )
    evaluation_legal_consistency_threshold: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Minimum legal consistency score to pass evaluation",
    )

    # ── Logging ───────────────────────────────────────────────────────────
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    log_format: str = Field(
        default="json",
        description="Log output format ('json' for structured, 'console' for human-readable)",
    )

    # ── FastAPI ───────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0", description="API server bind address")
    api_port: int = Field(default=8000, ge=1, le=65535, description="API server port")
    api_reload: bool = Field(
        default=True, description="Enable auto-reload in development"
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Cached settings factory.

    Returns the same ``Settings`` instance on every call. Use
    ``get_settings.cache_clear()`` in tests to force re-initialization
    with different environment variables.
    """
    return Settings()  # type: ignore[call-arg]
