"""Model persistence — save and load discovered symbolic regression models."""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sympy as sp

from apedp.safety import safe_parse_expr

logger = logging.getLogger(__name__)

# Regex to detect path traversal attempts
_PATH_TRAVERSAL_RE = re.compile(r'(\.\.|//|\\\\|%2e%2e|%2f|%5c)', re.IGNORECASE)


def _sanitize_path_component(name: str) -> str:
    """Sanitize a path component to prevent directory traversal."""
    # Remove any path separators and traversal sequences
    sanitized = re.sub(r'[\\/]+', '_', name)
    sanitized = _PATH_TRAVERSAL_RE.sub('', sanitized)
    # Limit length
    if len(sanitized) > 200:
        sanitized = sanitized[:200]
    if not sanitized:
        sanitized = "unnamed"
    return sanitized


def _atomic_write_text(path: Path, text: str) -> None:
    """Replace ``path`` with ``text`` without ever exposing a partial file.

    The previous implementation called ``json.dump`` straight onto the target,
    so an interrupted write (crash, full disk, kill) left a *truncated* file in
    place of a valid one: the store then contained a model that ``load_model``
    could not parse, and the previous contents were already gone. Writing to a
    sibling temp file, flushing it to disk, and then ``os.replace``-ing makes the
    swap atomic on POSIX and on same-volume Windows renames — a reader sees
    either the old file or the complete new one, never a half-written one.
    """
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


class ModelStore:
    """Persistent store for discovered models with versioning."""

    def __init__(self, base_dir: str = "apedp_models"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._registry: dict[str, dict[str, Any]] = {}

    def save_model(
        self,
        model_name: str,
        expression: sp.Expr,
        variables: list[str],
        r_squared: float,
        complexity: int,
        metadata: dict | None = None,
    ) -> str:
        """Save a discovered model to disk."""
        safe_name = _sanitize_path_component(model_name)
        model_entry = {
            "model_name": safe_name,
            "expression": str(expression),
            "variables": variables,
            "r_squared": r_squared,
            "complexity": complexity,
            "metadata": metadata or {},
            "sympy_expr": sp.latex(expression),
        }
        model_id = f"{safe_name}_{self._timestamp()}"
        filepath = self.base_dir / f"{model_id}.json"
        _atomic_write_text(filepath, json.dumps(model_entry, indent=2, default=str))
        self._registry[model_id] = model_entry
        logger.info("Saved model '%s' to %s", safe_name, filepath)
        return model_id

    def load_model(self, model_id: str) -> dict | None:
        """Load a model *record* (metadata plus the expression as a string).

        Note the return type: this is the stored JSON record, not a symbolic
        expression. Use :meth:`load_model_expr` when you want something
        symbolically manipulable.
        """
        safe_id = _sanitize_path_component(model_id)
        if safe_id in self._registry:
            return self._registry[safe_id]
        filepath = self.base_dir / f"{safe_id}.json"
        if not filepath.exists():
            logger.warning("Model '%s' not found", safe_id)
            return None
        with open(filepath, encoding="utf-8") as f:
            model: dict[str, Any] = json.load(f)
        self._registry[safe_id] = model
        return model

    def load_model_expr(self, model_id: str) -> sp.Expr | None:
        """Load a saved model as a real sympy expression, or ``None``.

        ``load_model`` hands back a dict whose ``sympy_expr`` field is LaTeX:
        display-ready, but not an expression, so a caller that wanted to compute
        with a stored model had to re-parse it themselves. This parses the stored
        plain-text ``expression`` through the sandboxed parser
        (:func:`apedp.safety.safe_parse_expr`) and returns an ``Expr``, or
        ``None`` for a missing/corrupt record rather than raising.
        """
        record = self.load_model(model_id)
        if record is None:
            return None
        expression = record.get("expression")
        if not isinstance(expression, str) or not expression.strip():
            logger.warning("Model '%s' has no usable expression string", model_id)
            return None
        try:
            return safe_parse_expr(expression)
        except (TypeError, ValueError, sp.SympifyError) as e:
            # A record on disk may predate the current parser or have been
            # hand-edited; report it rather than failing the caller's read path.
            logger.warning("Model '%s' could not be re-parsed: %s", model_id, e)
            return None

    def list_models(self) -> list[str]:
        """List every model ID on disk, plus anything loaded this session.

        Previously this returned the in-memory registry only, so a fresh process
        reported an empty store while the files were there -- the caller could not
        discover a saved model without already knowing its ID.
        """
        on_disk = sorted(path.stem for path in self.base_dir.glob("*.json") if not path.name.startswith("."))
        known = set(on_disk) | set(self._registry)
        return sorted(known)

    def delete_model(self, model_id: str) -> bool:
        """Delete a model by ID."""
        safe_id = _sanitize_path_component(model_id)
        filepath = self.base_dir / f"{safe_id}.json"
        if filepath.exists():
            filepath.unlink()
        self._registry.pop(safe_id, None)
        logger.info("Deleted model '%s'", safe_id)
        return True

    def _timestamp(self) -> str:
        """A UTC, microsecond-resolution, collision-free model-ID suffix.

        Second resolution collided on rapid saves: two models saved in the same
        second produced the same ID and the second overwrote the first.
        """
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        suffix = ""
        counter = 0
        while (self.base_dir / f"{stamp}{suffix}.json").exists():
            counter += 1
            suffix = f"_{counter}"
        return f"{stamp}{suffix}"


class ExperimentLog:
    """Log of experiments with parameters, results, and metadata.

    Bounded: keeps at most ``max_entries`` in memory and rotates the JSONL
    file once it exceeds ``max_bytes`` (keeps ``*.1`` backup). Previously
    the log grew unboundedly.
    """

    def __init__(
        self,
        log_path: str = "apedp_experiments.jsonl",
        max_entries: int = 10000,
        max_bytes: int = 10_000_000,
    ):
        self.log_path = Path(log_path)
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._entries: list[dict] = []

    def log(self, experiment_type: str, parameters: dict, results: dict, status: str = "success") -> None:
        entry = {
            "experiment_type": experiment_type,
            "parameters": parameters,
            "results": results,
            "status": status,
            # UTC with an explicit offset: a naive local timestamp cannot be
            # compared across machines, and this log is the experiment record.
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._entries.append(entry)
        if len(self._entries) > self.max_entries:
            self._entries = self._entries[-self.max_entries :]
        try:
            if self.log_path.exists() and self.log_path.stat().st_size > self.max_bytes:
                backup = self.log_path.with_suffix(self.log_path.suffix + ".1")
                # ``os.replace`` rotates atomically; the previous read-then-
                # ``write_text("")`` pair truncated the live log in place, so a
                # crash in between lost the whole file instead of keeping it as
                # the ``.1`` backup.
                os.replace(self.log_path, backup)
        except OSError:
            logger.warning("Experiment log rotation failed", exc_info=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
            # Flush and fsync: an appended line that is still in the OS cache is
            # not an experiment record, and this log is the evidence trail.
            f.flush()
            os.fsync(f.fileno())
        logger.info("Logged experiment: %s", experiment_type)

    def get_experiments(self, experiment_type: str | None = None) -> list[dict]:
        """Retrieve logged experiments, optionally filtered by type."""
        if experiment_type:
            return [e for e in self._entries if e["experiment_type"] == experiment_type]
        return list(self._entries)