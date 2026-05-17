"""
nyayaeval.connectors.llm_provider — Multi-Provider LLM Abstraction
=====================================================================

Wraps LangChain chat model initialization behind a factory function.
Supports three providers via the ``LLM_PROVIDER`` environment variable:

    - ``openai``  : ChatOpenAI (GPT-4o, GPT-4o-mini)
    - ``gemini``  : ChatGoogleGenerativeAI (Gemini 2.0 Flash, Pro)
    - ``groq``    : ChatGroq (Llama 3.3 70B, Mixtral)

Agents never know which provider they're using — they receive a
``BaseChatModel`` instance and call ``.ainvoke()`` / ``.with_structured_output()``.
"""

from __future__ import annotations

from functools import lru_cache

import structlog
from langchain_core.language_models import BaseChatModel

from nyayaeval.config.settings import get_settings

logger = structlog.get_logger(__name__)

# ─── Provider defaults ────────────────────────────────────────────────────────
_PROVIDER_DEFAULTS: dict[str, str] = {
    "openai": "gpt-4o",
    "gemini": "gemini-2.0-flash",
    "groq": "llama-3.3-70b-versatile",
}


def _build_openai(model: str, temperature: float, api_key: str) -> BaseChatModel:
    """Construct a ChatOpenAI instance."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model, temperature=temperature, api_key=api_key)


def _build_gemini(model: str, temperature: float, api_key: str) -> BaseChatModel:
    """Construct a ChatGoogleGenerativeAI instance."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        google_api_key=api_key,
    )


def _build_groq(model: str, temperature: float, api_key: str) -> BaseChatModel:
    """Construct a ChatGroq instance."""
    from langchain_groq import ChatGroq

    return ChatGroq(model=model, temperature=temperature, api_key=api_key)


# ─── Provider registry ───────────────────────────────────────────────────────
_BUILDERS = {
    "openai": _build_openai,
    "gemini": _build_gemini,
    "groq": _build_groq,
}


@lru_cache(maxsize=1)
def get_llm(
    model_name: str | None = None, temperature: float | None = None
) -> BaseChatModel:
    """
    Factory for the evaluation/correction LLM.

    Reads ``LLM_PROVIDER`` from settings to select the backend.
    Falls back to provider-specific defaults for model names.

    Args:
        model_name: Override model identifier. Defaults per provider.
        temperature: Sampling temperature. Defaults to settings value.

    Returns:
        A LangChain BaseChatModel instance.

    Raises:
        ValueError: If the provider is not supported.
    """
    settings = get_settings()
    provider = settings.llm_provider.lower()
    _temp = temperature if temperature is not None else settings.llm_temperature

    # Resolve model name: explicit arg → settings → provider default
    _model = model_name or settings.llm_model_name
    if _model in ("gpt-4o", ""):
        # If the user hasn't changed from the OpenAI default but switched
        # providers, use the provider's default model instead
        _model = _PROVIDER_DEFAULTS.get(provider, _model)

    # Resolve API key based on provider
    api_key_map = {
        "openai": settings.openai_api_key,
        "gemini": settings.gemini_api_key,
        "groq": settings.groq_api_key,
    }
    api_key = api_key_map.get(provider)
    if not api_key:
        raise ValueError(
            f"No API key configured for provider '{provider}'. "
            f"Set the corresponding env var (OPENAI_API_KEY / GEMINI_API_KEY / GROQ_API_KEY)."
        )

    builder = _BUILDERS.get(provider)
    if builder is None:
        supported = ", ".join(_BUILDERS.keys())
        raise ValueError(f"Unsupported LLM provider: '{provider}'. Supported: {supported}")

    llm = builder(_model, _temp, api_key)
    logger.info(
        "llm_provider.initialized",
        provider=provider,
        model=_model,
        temperature=_temp,
    )
    return llm
