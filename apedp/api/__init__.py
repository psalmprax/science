"""APEDP API — REST server for the Autonomous Physics & Engineering Discovery Platform.

Trust boundary, stated plainly rather than implied:

* **Authentication** is required by default. Set ``APEDP_API_KEY`` to a non-empty
  value to enable authentication. Without it the server will log a critical
  warning and refuse to start in production environments. For local single-user
  development you may set ``APEDP_ALLOW_UNAUTHENTICATED=1`` to explicitly opt
  in to unauthenticated mode — this is NOT recommended for any network-exposed
  deployment.
* **Rate limiting** defaults to 120 req/min per client
  (``DEFAULT_RATE_LIMIT_PER_MINUTE``). Set ``APEDP_RATE_LIMIT_PER_MINUTE=0`` to
  disable or a higher value for batch work; without it a caller can drive
  unbounded symbolic-regression work, which is expensive.
* **Request body size** is limited to 1 MB by default (``MAX_REQUEST_BODY_SIZE``).
  Increase via ``APEDP_MAX_REQUEST_BODY_SIZE`` if larger payloads are needed.
"""
from __future__ import annotations

import hmac
import logging
import os
import threading
import time
from collections import deque

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from apedp.api.routes import discovery, epistemic, health, metrology, pipeline, repair

logger = logging.getLogger(__name__)

#: Routes that never require the API key: liveness and documentation. A health
#: check that needs a credential cannot be used by the orchestrator that is
#: trying to decide whether to send traffic here.
AUTH_EXEMPT_PATHS = frozenset({"/health", "/docs", "/redoc", "/openapi.json"})

#: Sliding window for the limiter, in seconds.
_RATE_WINDOW_SECONDS = 60.0

#: Maximum request body size in bytes (default 1 MB)
MAX_REQUEST_BODY_SIZE = int(os.environ.get("APEDP_MAX_REQUEST_BODY_SIZE", "1048576"))

#: Allow unauthenticated mode only for explicit local development
def _allow_unauthenticated() -> bool:
    return os.environ.get("APEDP_ALLOW_UNAUTHENTICATED", "0") == "1"


def api_key() -> str | None:
    """The configured API key, or ``None`` when the server is open."""
    key = os.environ.get("APEDP_API_KEY", "")
    return key or None


#: Default budget when ``APEDP_RATE_LIMIT_PER_MINUTE`` is unset: throttled but
#: usable for single-user local work. Set to ``0`` explicitly to disable.
DEFAULT_RATE_LIMIT_PER_MINUTE = 120


def rate_limit_per_minute() -> int:
    """Allowed requests per client per minute; ``0`` disables the limiter."""
    raw = os.environ.get("APEDP_RATE_LIMIT_PER_MINUTE", str(DEFAULT_RATE_LIMIT_PER_MINUTE))
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning(
            "Ignoring non-numeric APEDP_RATE_LIMIT_PER_MINUTE=%r, using default %d",
            raw, DEFAULT_RATE_LIMIT_PER_MINUTE,
        )
        return DEFAULT_RATE_LIMIT_PER_MINUTE


class _ClientRateLimiter:
    """Sliding-window limiter, keyed by client address.

    Deliberately in-process and in-memory: it bounds a single worker's work, and
    it is not a substitute for a limiter at the edge once more than one process
    serves traffic. Documented rather than discovered.
    """

    def __init__(self, limit: int):
        self.limit = limit
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, client: str, now: float) -> float | None:
        """Return ``None`` when allowed, else seconds until the next slot."""
        with self._lock:
            window = self._hits.setdefault(client, deque())
            while window and now - window[0] >= _RATE_WINDOW_SECONDS:
                window.popleft()
            if len(window) >= self.limit:
                return max(0.0, _RATE_WINDOW_SECONDS - (now - window[0]))
            window.append(now)
            return None


def cors_origins() -> list[str]:
    """Browser origins allowed to call the API, from ``APEDP_CORS_ORIGINS``.

    Comma-separated; empty by default, which leaves the API same-origin.
    Wildcard (``*``) is explicitly DISALLOWED with credentials. If you need
    cross-origin access, list specific origins.
    """
    raw = os.environ.get("APEDP_CORS_ORIGINS", "")
    origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
    if "*" in origins:
        logger.warning("CORS wildcard '*' is not allowed; ignoring. Specify explicit origins.")
        origins = [o for o in origins if o != "*"]
    return origins


def create_app() -> FastAPI:
    app = FastAPI(
        title="APEDP",
        description="Autonomous Physics & Engineering Discovery Platform",
        version="1.0.0",
    )
    key = api_key()
    limit = rate_limit_per_minute()
    limiter = _ClientRateLimiter(limit) if limit else None

    # Enforce authentication unless explicitly opted out for local development
    if key is None:
        if _allow_unauthenticated():
            logger.warning(
                "APEDP_API_KEY is not set and APEDP_ALLOW_UNAUTHENTICATED=1: "
                "the API is UNAUTHENTICATED. This is ONLY for local development. "
                "Set APEDP_API_KEY to enable authentication."
            )
        else:
            logger.critical(
                "APEDP_API_KEY is not set and APEDP_ALLOW_UNAUTHENTICATED is not enabled. "
                "Refusing to start without authentication. Set APEDP_API_KEY or "
                "APEDP_ALLOW_UNAUTHENTICATED=1 for local development only."
            )
            raise RuntimeError("Authentication required: set APEDP_API_KEY")

    @app.middleware("http")
    async def _trust_boundary(request: Request, call_next):
        """Authenticate and rate-limit before any route does work."""
        # Enforce request body size limit. ``Content-Length`` is attacker-
        # controlled, so it is parsed defensively: a malformed value is a 400,
        # not a 500. A chunked body carries no ``Content-Length`` at all, so the
        # header check cannot bound it; reject chunked/streamed transfer outright
        # rather than accept an unbounded body while advertising a 1 MB cap.
        transfer_encoding = request.headers.get("transfer-encoding", "").lower()
        if transfer_encoding and transfer_encoding != "identity":
            return JSONResponse(
                {"detail": "Chunked/streamed request bodies are not supported; send Content-Length"},
                status_code=411,
            )
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                declared_length = int(content_length)
            except ValueError:
                return JSONResponse(
                    {"detail": "Malformed Content-Length header"},
                    status_code=400,
                )
            if declared_length > MAX_REQUEST_BODY_SIZE:
                return JSONResponse(
                    {"detail": f"Request body exceeds maximum size of {MAX_REQUEST_BODY_SIZE} bytes"},
                    status_code=413,
                )

        client = request.client.host if request.client else "unknown"
        if limiter is not None:
            retry_after = limiter.check(client, time.monotonic())
            if retry_after is not None:
                return JSONResponse(
                    {"detail": "Rate limit exceeded"},
                    status_code=429,
                    headers={"Retry-After": str(int(retry_after) + 1)},
                )
        if key is not None and request.url.path not in AUTH_EXEMPT_PATHS:
            presented = request.headers.get("X-API-Key", "")
            # ``compare_digest`` keeps the comparison constant-time; a plain
            # ``!=`` leaks the shared prefix length to a timing attacker.
            if not hmac.compare_digest(presented, key):
                return JSONResponse({"detail": "Invalid or missing API key"}, status_code=401)
        return await call_next(request)

    origins = cors_origins()
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        logger.info("CORS enabled for origins: %s", ", ".join(origins))
    app.include_router(health.router, tags=["health"])
    app.include_router(discovery.router, tags=["discovery"])
    app.include_router(pipeline.router, tags=["pipeline"])
    app.include_router(metrology.router, tags=["metrology"])
    app.include_router(repair.router, tags=["repair"])
    app.include_router(epistemic.router, tags=["epistemic"])
    return app