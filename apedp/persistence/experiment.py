"""Experiment logging for APEDP pipeline."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


class ExperimentLog:
    """Log of experiments with parameters, results, and metadata (bounded, rotating)."""

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

    def log(
        self,
        experiment_type: str,
        parameters: dict,
        results: dict,
        status: str = "success",
    ) -> None:
        entry = {
            "experiment_type": experiment_type,
            "parameters": parameters,
            "results": results,
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._entries.append(entry)
        if len(self._entries) > self.max_entries:
            self._entries = self._entries[-self.max_entries :]
        try:
            if self.log_path.exists() and self.log_path.stat().st_size > self.max_bytes:
                backup = self.log_path.with_suffix(self.log_path.suffix + ".1")
                backup.write_bytes(self.log_path.read_bytes())
                self.log_path.write_text("")
        except OSError:
            logger.warning("Experiment log rotation failed", exc_info=True)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        logger.info("Logged experiment: %s", experiment_type)

    def get_experiments(self, experiment_type: str | None = None) -> list[dict]:
        """Retrieve logged experiments, optionally filtered by type."""
        if experiment_type:
            return [e for e in self._entries if e["experiment_type"] == experiment_type]
        return list(self._entries)

    def to_dataframe(self):
        """Return experiments as a pandas DataFrame (if pandas is available)."""
        try:
            import pandas as pd
            return pd.DataFrame(self._entries)
        except ImportError:
            logger.warning("pandas not available, returning list of dicts")
            return self._entries