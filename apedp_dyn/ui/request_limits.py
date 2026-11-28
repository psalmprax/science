"""What a request may ask of the page's server, checked before any route does work.

Status: production (installed by :func:`apedp_dyn.ui.route_endpoint.mount` on the UI app).

An outside audit of 27 November 2026 found that the page's server had no limit on the size of a request's body, no
limit on how many requests it would work on, one request field that set the length of a loop with no upper bound
(``n_seeds`` of the stability audit), an error reply that carried the exception's own text, and scratch directories
that were never removed. The routes themselves live in ``server.py``, which the v1.18 freeze record pins, so the
limits are enforced here, in front of them:

* a body larger than :data:`MAX_BODY_BYTES` is refused with a 413, whether its length is declared or streamed;
* a field listed in :data:`FIELD_LIMITS` outside its limits is refused with a 422 that names it;
* no more than :func:`max_requests_in_flight` requests that do work (POST) run at once; the next gets a 429;
* a caller that is not on this machine gets at most :func:`rate_limit_per_minute` requests a minute;
* a 5xx reply never carries the text of the exception that caused it (the audit log keeps it);
* scratch directories of the external-benchmark route older than a day are removed.

These are limits on work, not scientific thresholds. A deployment behind a proxy on the same machine sees every
caller as local, so the per-caller limit does not apply there and belongs at the proxy.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import time
from collections import deque
from collections.abc import Awaitable, Callable, MutableMapping
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

_LOG = logging.getLogger("apedp_dyn.ui.server")

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]

# The largest table the route endpoint takes (20,000 rows of 12 columns and a target, as JSON numbers) is about
# 5 MB; three times that leaves room for long decimals.
MAX_BODY_BYTES = 16 * 1024 * 1024
# path -> field -> (lowest, highest). ``n_seeds`` is the length of a loop of whole discovery runs; ``max_windows``
# is cut to the windows that exist by the sweep itself and is listed so that a negative count is refused.
FIELD_LIMITS: dict[str, dict[str, tuple[int, int]]] = {
    "/api/stability/audit": {"n_seeds": (1, 50)},
    "/api/market/sweep": {"max_windows": (1, 10_000)},
}
RATE_WINDOW_SECONDS = 60.0
SCRATCH_PREFIX, SCRATCH_MAX_AGE_SECONDS, SCRATCH_SWEEP_EVERY_SECONDS = "ext60_", 24 * 3600.0, 3600.0
LOCAL_PEERS = frozenset({"127.0.0.1", "::1", "localhost", "testclient"})
REASON_KEPT = "(the reason is in the server's audit log)"


def _setting(name: str, default: int) -> int:
    try:
        return max(0, int(os.environ.get(name, default)))
    except ValueError:
        return default


def max_requests_in_flight() -> int:
    """How many requests that do work may run at once (``APEDP_UI_MAX_IN_FLIGHT``; 0 turns the limit off)."""
    return _setting("APEDP_UI_MAX_IN_FLIGHT", 8)


def rate_limit_per_minute() -> int:
    """Requests a minute for one caller that is not on this machine (``APEDP_UI_RATE_LIMIT_PER_MINUTE``; 0 = off)."""
    return _setting("APEDP_UI_RATE_LIMIT_PER_MINUTE", 120)


def field_objection(path: str, body: bytes) -> str | None:
    """Why this body's fields are outside the limits for ``path``, or None. A body that is not JSON is the route's
    own business (it answers 422 itself)."""
    limits = FIELD_LIMITS.get(path.rstrip("/") or path)
    if not limits or not body:
        return None
    try:
        sent = json.loads(body)
    except (ValueError, UnicodeDecodeError, RecursionError):
        return None
    if not isinstance(sent, dict):
        return None
    for name, (lowest, highest) in limits.items():
        if name not in sent:
            continue
        value = sent[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value or value != int(value):
            return f"{name} must be a whole number"
        if not lowest <= int(value) <= highest:
            return f"{name} must be between {lowest} and {highest}"
    return None


def remove_stale_scratch(directory: Path | None = None, *, now: float | None = None) -> list[str]:
    """Remove scratch directories of the external-benchmark route that are more than a day old; returns their names.

    Only a directory this server could have made is touched: its name starts with ``ext60_`` and it holds nothing but
    the ``apedp_<block>.json`` files that route writes.
    """
    base = Path(tempfile.gettempdir()) if directory is None else directory
    moment = time.time() if now is None else now
    removed: list[str] = []
    try:
        candidates = sorted(base.glob(f"{SCRATCH_PREFIX}*"))
    except OSError:
        return removed
    for path in candidates:
        try:
            if path.is_symlink() or not path.is_dir() or moment - path.stat().st_mtime < SCRATCH_MAX_AGE_SECONDS:
                continue
            inside = list(path.iterdir())
            if any(item.is_dir() or not (item.name.startswith("apedp_") and item.suffix == ".json") for item in inside):
                continue
            shutil.rmtree(path)
            removed.append(path.name)
        except OSError as problem:                      # somebody else's file, or it went away meanwhile
            _LOG.debug("scratch directory %s left in place: %s", path, problem)
    return removed


class RequestLimits:
    """The limits above as one ASGI layer in front of every ``/api/`` route."""

    def __init__(self, app: Callable[[Scope, Receive, Send], Awaitable[None]]) -> None:
        self.app = app
        self.in_flight = 0
        self.seen: dict[str, deque[float]] = {}
        self.swept_at = 0.0

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http" or not str(scope.get("path", "")).startswith("/api/"):
            await self.app(scope, receive, send)
            return
        path, method = str(scope["path"]), str(scope.get("method", "GET")).upper()
        refusal = self._refused_before_reading(scope)
        if refusal is not None:
            await refusal(scope, receive, send)
            return
        works = method in ("POST", "PUT", "PATCH", "DELETE")
        limit = max_requests_in_flight()
        if works and limit and self.in_flight >= limit:
            await JSONResponse({"detail": "The server is busy with other requests; try again shortly."}, status_code=429,
                               headers={"Retry-After": "5"})(scope, receive, send)
            return
        # Counted from here, before the body is read. It used to be counted after: a request's body arrives over time,
        # so any number of slow requests could pass the check above together and all go on to do work (an outside
        # review of 28 November 2026 found the order).
        self.in_flight += 1 if works else 0
        try:
            body, refusal = await self._body(receive) if works else (b"", None)
            if refusal is None and works:
                objection = field_objection(path, body)
                if objection is not None:
                    refusal = JSONResponse({"detail": objection}, status_code=422)
            if refusal is not None:
                await refusal(scope, receive, send)
                return
            replayed = not works

            async def replay() -> Message:
                nonlocal replayed
                if not replayed:
                    replayed = True
                    return {"type": "http.request", "body": body, "more_body": False}
                return await receive()

            self._sweep_scratch()
            await self.app(scope, replay, send)
        finally:
            self.in_flight -= 1 if works else 0

    def _refused_before_reading(self, scope: Scope) -> Response | None:
        headers = {key.decode("latin-1").lower(): value.decode("latin-1") for key, value in scope.get("headers", [])}
        declared = headers.get("content-length")
        if declared is not None:
            if not declared.strip().isdigit():
                return JSONResponse({"detail": "Malformed Content-Length header"}, status_code=400)
            if int(declared) > MAX_BODY_BYTES:
                return JSONResponse({"detail": f"Request body exceeds {MAX_BODY_BYTES} bytes"}, status_code=413)
        client = scope.get("client")
        peer = str(client[0]) if client else ""
        per_minute = rate_limit_per_minute()
        if per_minute and peer not in LOCAL_PEERS:
            now = time.monotonic()
            window = self.seen.setdefault(peer, deque())
            while window and now - window[0] >= RATE_WINDOW_SECONDS:
                window.popleft()
            if len(window) >= per_minute:
                wait = int(RATE_WINDOW_SECONDS - (now - window[0])) + 1
                return JSONResponse({"detail": "Rate limit exceeded"}, status_code=429, headers={"Retry-After": str(wait)})
            window.append(now)
            if len(self.seen) > 10_000:                 # callers long gone must not pile up
                self.seen = {name: times for name, times in self.seen.items() if times and now - times[-1] < RATE_WINDOW_SECONDS}
        return None

    @staticmethod
    async def _body(receive: Receive) -> tuple[bytes, Response | None]:
        parts: list[bytes] = []
        size = 0
        while True:
            message = await receive()
            if message.get("type") != "http.request":
                return b"", JSONResponse({"detail": "The request ended before its body was read"}, status_code=400)
            chunk = message.get("body", b"") or b""
            size += len(chunk)
            if size > MAX_BODY_BYTES:
                return b"", JSONResponse({"detail": f"Request body exceeds {MAX_BODY_BYTES} bytes"}, status_code=413)
            parts.append(chunk)
            if not message.get("more_body", False):
                return b"".join(parts), None

    def _sweep_scratch(self) -> None:
        now = time.monotonic()
        if now - self.swept_at >= SCRATCH_SWEEP_EVERY_SECONDS or self.swept_at == 0.0:
            self.swept_at = now
            removed = remove_stale_scratch()
            if removed:
                _LOG.info("removed %d stale scratch directories", len(removed))


async def server_error_without_the_cause(request: Request, exc: Exception) -> Response:
    """A 5xx reply keeps its own words and loses the text of the exception that caused it.

    ``raise HTTPException(500, f"... {e}") from e`` hands the caller whatever ``e`` says: paths, values, internals.
    The cause is known here (``__cause__``), so its text is cut out of the reply and written to the server's log.
    """
    assert isinstance(exc, StarletteHTTPException)
    cause = exc.__cause__ or exc.__context__
    text = str(cause) if cause is not None else ""
    if exc.status_code >= 500 and isinstance(exc.detail, str) and text and text in exc.detail:
        _LOG.error("%s %s answered %d: %s", request.method, request.url.path, exc.status_code, exc.detail)
        exc = StarletteHTTPException(status_code=exc.status_code, detail=exc.detail.replace(text, REASON_KEPT),
                                     headers=getattr(exc, "headers", None))
    return await http_exception_handler(request, exc)


def install(app: FastAPI) -> None:
    """Put the limits in front of ``app``'s routes."""
    app.add_middleware(RequestLimits)
    app.add_exception_handler(StarletteHTTPException, server_error_without_the_cause)
