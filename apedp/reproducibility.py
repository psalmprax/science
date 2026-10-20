"""Seed controls and machine-readable provenance for scientific runs."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def _validate_seed(seed: int) -> None:
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python/NumPy and available Torch backends.

    JAX uses explicit PRNG keys; use :func:`jax_key` to construct one. Accelerator
    kernels can still vary across devices, drivers, and library versions.
    """
    _validate_seed(seed)

    random.seed(seed)
    np.random.seed(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


def jax_key(seed: int) -> Any:
    """Return a JAX PRNG key derived from ``seed`` without global RNG state."""
    _validate_seed(seed)
    import jax

    return jax.random.PRNGKey(seed)


def dataset_sha256(dataset: Any) -> str:
    """Hash arrays and nested dataset structures including shape and dtype."""
    digest = hashlib.sha256()

    def update(value: Any) -> None:
        if callable(getattr(value, "to_csv", None)):
            digest.update(b"table:")
            digest.update(value.to_csv(index=True).encode())
        elif isinstance(value, np.ndarray) or (
            not isinstance(value, (str, bytes, Mapping, tuple, list))
            and hasattr(value, "shape")
            and hasattr(value, "dtype")
        ):
            array = np.ascontiguousarray(np.asarray(value))
            digest.update(b"ndarray:")
            digest.update(str(array.dtype).encode())
            digest.update(json.dumps(array.shape).encode())
            if array.dtype.hasobject:
                digest.update(
                    json.dumps(
                        _json_safe(array.tolist()),
                        sort_keys=True,
                        allow_nan=False,
                    ).encode()
                )
            else:
                digest.update(array.tobytes())
        elif isinstance(value, Mapping):
            digest.update(b"mapping:")
            for key in sorted(value, key=str):
                update(str(key))
                update(value[key])
        elif isinstance(value, (tuple, list)):
            digest.update(b"sequence:")
            for item in value:
                update(item)
        elif isinstance(value, bytes):
            digest.update(b"bytes:")
            digest.update(value)
        else:
            digest.update(b"json:")
            digest.update(json.dumps(value, sort_keys=True, default=str).encode())

    update(dataset)
    return digest.hexdigest()


def _config_snapshot(config: Any) -> dict[str, Any]:
    if is_dataclass(config) and not isinstance(config, type):
        config = asdict(config)
    if not isinstance(config, Mapping):
        raise TypeError("config must be a dataclass or mapping")
    return dict(config)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        return str(value)
    return value


def write_provenance_artifact(
    path: str | Path,
    *,
    dataset: Any,
    config: Any,
    seed: int,
    deterministic: bool,
    metrics: Mapping[str, Any] | None = None,
    results: Any = None,
) -> dict[str, Any]:
    """Write a versioned JSON provenance record and return its payload."""
    _validate_seed(seed)
    config_snapshot = _config_snapshot(config)
    if "seed" in config_snapshot and config_snapshot["seed"] != seed:
        raise ValueError("provenance seed must match config seed")
    if "deterministic" in config_snapshot and config_snapshot["deterministic"] != deterministic:
        raise ValueError("provenance deterministic flag must match config")
    safe_config = _json_safe(config_snapshot)
    config_json = json.dumps(safe_config, sort_keys=True, allow_nan=False)
    git = shutil.which("git")
    try:
        if git is None:
            raise FileNotFoundError("git executable unavailable")
        commit = subprocess.check_output(  # noqa: S603 - fixed argv; no user-controlled input
            [git, "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            stderr=subprocess.DEVNULL,
            timeout=2,
            text=True,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        commit = None

    payload = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "dataset_sha256": dataset_sha256(dataset),
        "config_sha256": hashlib.sha256(config_json.encode()).hexdigest(),
        "config": safe_config,
        "seed": seed,
        "deterministic": deterministic,
        "metrics": _json_safe(metrics or {}),
        "results": _json_safe(results),
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(destination)
    return payload
