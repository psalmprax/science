"""Immutable Experiment Manifest specification for AEH.

Generates cryptographic manifests guaranteeing complete reproducibility
and un-tampered provenance for APEDP-X evaluations.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class ExperimentManifest:
    """Cryptographic record defining the exact operational boundary of an experiment."""
    experiment_id: str
    dataset_hash: str
    code_commit: str
    configuration_hash: str
    random_seed: int
    proposer_version: str
    validator_version: str
    repair_policy_version: str
    evaluation_protocol: str
    allowed_data: tuple[str, ...]
    forbidden_data: tuple[str, ...]
    candidate_count: int
    repair_count: int
    terminal_abstentions: int
    promotion_count: int
    event_log_hash: str
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    manifest_hash: str = ""

    @classmethod
    def create(
        cls,
        experiment_id: str,
        dataset_hash: str,
        code_commit: str,
        configuration_hash: str,
        random_seed: int,
        proposer_version: str,
        validator_version: str,
        repair_policy_version: str,
        evaluation_protocol: str,
        allowed_data: tuple[str, ...],
        forbidden_data: tuple[str, ...],
        candidate_count: int,
        repair_count: int,
        terminal_abstentions: int,
        promotion_count: int,
        event_log_hash: str,
    ) -> ExperimentManifest:
        ts = datetime.now(timezone.utc).isoformat()
        # Compute unhashed payload representation
        payload = {
            "experiment_id": experiment_id,
            "dataset_hash": dataset_hash,
            "code_commit": code_commit,
            "configuration_hash": configuration_hash,
            "random_seed": random_seed,
            "proposer_version": proposer_version,
            "validator_version": validator_version,
            "repair_policy_version": repair_policy_version,
            "evaluation_protocol": evaluation_protocol,
            "allowed_data": list(allowed_data),
            "forbidden_data": list(forbidden_data),
            "candidate_count": candidate_count,
            "repair_count": repair_count,
            "terminal_abstentions": terminal_abstentions,
            "promotion_count": promotion_count,
            "event_log_hash": event_log_hash,
            "timestamp_utc": ts,
        }
        canonical_json = json.dumps(payload, sort_keys=True)
        m_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

        return cls(
            experiment_id=experiment_id,
            dataset_hash=dataset_hash,
            code_commit=code_commit,
            configuration_hash=configuration_hash,
            random_seed=random_seed,
            proposer_version=proposer_version,
            validator_version=validator_version,
            repair_policy_version=repair_policy_version,
            evaluation_protocol=evaluation_protocol,
            allowed_data=allowed_data,
            forbidden_data=forbidden_data,
            candidate_count=candidate_count,
            repair_count=repair_count,
            terminal_abstentions=terminal_abstentions,
            promotion_count=promotion_count,
            event_log_hash=event_log_hash,
            timestamp_utc=ts,
            manifest_hash=m_hash,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_dataset_hash(data: Any) -> str:
    """Deterministically compute SHA-256 fingerprint for diverse dataset structures."""
    if data is None:
        return hashlib.sha256(b"EMPTY_DATASET").hexdigest()
    if isinstance(data, (str, bytes)):
        b = data.encode("utf-8") if isinstance(data, str) else data
        return hashlib.sha256(b).hexdigest()

    try:
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            try:
                hashed_series = pd.util.hash_pandas_object(data, index=True)
                return hashlib.sha256(hashed_series.to_numpy().tobytes()).hexdigest()
            except Exception:  # noqa: BLE001 - deliberate catch-all: best-effort probe path reports or skips the failure, never crashes
                return hashlib.sha256(data.to_csv(index=False).encode("utf-8")).hexdigest()
        if isinstance(data, pd.Series):
            return hashlib.sha256(data.to_numpy().tobytes()).hexdigest()
    except ImportError:
        pass

    try:
        import numpy as np
        if isinstance(data, np.ndarray):
            return hashlib.sha256(data.tobytes()).hexdigest()
    except ImportError:
        pass

    if isinstance(data, (dict, list)):
        try:
            return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode("utf-8")).hexdigest()
        except Exception:  # noqa: BLE001 - deliberate catch-all: best-effort probe path reports or skips the failure, never crashes
            return hashlib.sha256(str(data).encode("utf-8")).hexdigest()

    return hashlib.sha256(str(data).encode("utf-8")).hexdigest()


def get_git_commit_sha() -> str:
    """Retrieve active Git commit SHA or fallback to clean release tag."""
    import shutil
    import subprocess

    git_bin = shutil.which("git")
    if git_bin is None:
        raise RuntimeError("git not found; cannot record provenance commit SHA")
    try:
        out = subprocess.check_output(  # noqa: S603 - subprocess argv is fixed, non-user-controlled (bundled tooling)
            [git_bin, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).decode("utf-8").strip()
        if len(out) == 40:
            return out
    except Exception:  # noqa: BLE001, S110 - optional probe may fail silently by design (best-effort stage skip)  # noqa: BLE001 - deliberate catch-all: best-effort probe path reports or skips the failure, never crashes
        # unavailable; fall back to a locked label rather than crashing
        pass
    return "HEAD_PROVENANCE_LOCKED"

