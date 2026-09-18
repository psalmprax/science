"""Health check endpoints."""
from __future__ import annotations

from fastapi import APIRouter

from apedp import __version__

router = APIRouter()


@router.get("/health", tags=["health"])
async def health() -> dict:
    return {"status": "ok", "version": __version__}


@router.get("/", tags=["health"])
async def root() -> dict:
    return {"service": "APEDP", "version": __version__}