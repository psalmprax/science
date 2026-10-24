"""Environment and Runtime Storage Configuration Interface for APEDP-X / CPMD.

Provides unified, cross-platform helpers for accessing temporary storage directories,
persistent holdout ledger directories, runtime platform detection, and environment settings.
"""
from __future__ import annotations

import os
from pathlib import Path

from apedp_dyn.notebook_adapter import (
    NotebookRuntimeConfig,
    PlatformType,
    StorageTier,
    configure_storage,
    detect_notebook,
    detect_platform,
    ensure_kernel,
    get_recommended_worker_count,
    init_environment,
    is_colab,
    is_databricks,
    is_kaggle,
    mount_google_drive,
    temporary_storage_scope,
)


def get_temp_storage_dir() -> Path:
    """Return the currently configured temporary storage directory.

    If not already configured, initializes with default settings.
    """
    env_dir = os.environ.get("APEDP_TEMP_STORAGE")
    if env_dir:
        p = Path(env_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p
    t_dir, _, _, _ = configure_storage()
    return t_dir


def get_persistent_storage_dir() -> Path:
    """Return the currently configured persistent storage directory."""
    env_dir = os.environ.get("APEDP_STORAGE_DIR")
    if env_dir:
        p = Path(env_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p
    _, p_dir, _, _ = configure_storage()
    return p_dir


def get_ledger_path(filename: str = "experiment_ledger.jsonl") -> Path:
    """Return the absolute path to an experiment or trial ledger file."""
    env_path = os.environ.get("APEDP_LEDGER_PATH")
    if env_path and (Path(env_path).name == filename or filename == "experiment_ledger.jsonl"):
        p = Path(env_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    return get_persistent_storage_dir() / filename


def get_runtime_config(quiet: bool = True) -> NotebookRuntimeConfig:
    """Obtain or initialize the runtime configuration object."""
    return init_environment(quiet=quiet)


__all__ = [
    "NotebookRuntimeConfig",
    "PlatformType",
    "StorageTier",
    "configure_storage",
    "detect_notebook",
    "detect_platform",
    "ensure_kernel",
    "get_ledger_path",
    "get_persistent_storage_dir",
    "get_recommended_worker_count",
    "get_runtime_config",
    "get_temp_storage_dir",
    "init_environment",
    "is_colab",
    "is_databricks",
    "is_kaggle",
    "mount_google_drive",
    "temporary_storage_scope",
]
