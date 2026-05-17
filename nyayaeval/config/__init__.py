"""
nyayaeval.config — Configuration Management
=============================================

Centralizes all environment-based configuration using Pydantic Settings.
Provides a cached singleton accessor so the settings object is loaded once
and reused across the application.
"""

from nyayaeval.config.settings import Settings, get_settings

__all__ = ["Settings", "get_settings"]
