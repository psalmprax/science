"""Repository path resolution (ARCHITECTURE.md §4.2).

Single source of truth for resolving repository and artifact directories.
No module may hardcode absolute user-home paths; all resolution goes through
the environment variables below with repo-relative fallbacks.
"""
from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """Repository root; ``APEDP_REPO_ROOT`` overrides the derived location."""
    env = os.environ.get("APEDP_REPO_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def lean_root() -> Path:
    """Lean toolchain/output root; ``APEDP_LEAN_ROOT`` overrides the in-repo default."""
    env = os.environ.get("APEDP_LEAN_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return repo_root() / "apedp_dyn" / "lean_formal"


def artifact_dir() -> Path:
    """Scratch/artifact output directory; ``APEDP_ARTIFACT_DIR`` overrides the in-repo default."""
    env = os.environ.get("APEDP_ARTIFACT_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return repo_root() / "artifacts"
