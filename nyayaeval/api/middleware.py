"""
nyayaeval.api.middleware — FastAPI Middleware
===============================================

Custom middleware for request logging, error handling, and CORS.
Applied to the FastAPI app in main.py.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

logger = structlog.get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Logs every incoming request with a unique request ID, method, path,
    status code, and response time.

    Adds the ``X-Request-ID`` header to responses for traceability.
    """

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        request_id = str(uuid.uuid4())[:8]
        start = time.monotonic()

        logger.info(
            "http.request.start",
            request_id=request_id,
            method=request.method,
            path=str(request.url.path),
        )

        response: Response = await call_next(request)

        duration_ms = (time.monotonic() - start) * 1000
        logger.info(
            "http.request.complete",
            request_id=request_id,
            status_code=response.status_code,
            duration_ms=round(duration_ms, 2),
        )

        response.headers["X-Request-ID"] = request_id
        return response


def setup_middleware(app: FastAPI) -> None:
    """
    Register all middleware on the FastAPI application.

    Called from main.py during app initialization.
    """
    # CORS — permissive for development, restrict in production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request logging
    app.add_middleware(RequestLoggingMiddleware)
