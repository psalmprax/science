"""APEDP-X v1.0-FROZEN: Cryptographic Specification & Benchmark Provenance.

Freezes all core referee kernels, claim-evidence ladder logic, domain adapters,
threshold configurations, and grammar primitive spaces to guarantee reproducibility
and prevent post-hoc overfitting during independent blinded evaluations.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from apedp_dyn.sandbox.grammar_space import GRAMMAR_FINGERPRINT


def compute_file_sha256(path: Path) -> str:
    """Compute SHA256 hex digest of a source code file."""
    if not path.is_file():
        return "NOT_FOUND"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


BASE_DIR = Path(__file__).resolve().parents[2]

# Core frozen module paths
ADJUDICATE_PATH = BASE_DIR / "apedp_dyn" / "epistemic" / "adjudicate.py"
CLAIM_LADDER_PATH = BASE_DIR / "apedp_dyn" / "epistemic" / "claim_ladder.py"
GRAMMAR_SPACE_PATH = BASE_DIR / "apedp_dyn" / "sandbox" / "grammar_space.py"

# Cryptographic Pinned Registry for All Frozen Records (drift fails loudly)
PINNED_FROZEN_REGISTRY: dict[str, str] = {
    "apedp_dyn/epistemic/adjudicate.py": "da604e6a21f8687e5834362989f370f3b386649e89fbced2ec624030ffd4ff73",
    "apedp_dyn/epistemic/claim_ladder.py": "aba9fd0045abc137a36dab90e1316b8934597beb02c553cf35d15c3029c0886c",
    "apedp_dyn/sandbox/grammar_space.py": "6f4da69fb15b47462e0f802f5e20213ee8e9d225a5818c8ce9d414cf6403263a",
    "outputs/trading_np93_freeze.json": "462655b914271f06ccc095362d30a87935f4977527170e7080b53b72961cca0e",
    "outputs/nano_ds10_freeze.json": "d37d0f1c46286df22cdb065ca002fca67e7c1221f54e261722ae7af7f3b07bbd",
    "outputs/dyn_epistemic_freeze.json": "d3b8665230bce749a645111c08b8c53d0a8e5897d42c4678dc9c7a34cad10556",
    "outputs/d5_conf_freeze.json": "d9e1a17b46765f5096f5e2c63beee8d9e6bd3c9df82ea8d67c44bfb0ee0ab453",
    "outputs/d5_conf_v2_freeze.json": "26b2a1914db9a01f6987beda3463c3f079294f1896493e57059ec67bedb03ac6",
    "benchmarks/l2_control_freeze.json": "ec117721d1cd9da0b09950bb9b5894110c2411475dfec073b684477c33e96872",
}


class FreezeDriftError(RuntimeError):
    """Raised when a frozen specification file or benchmark record has drifted."""
    pass


class MissingProvenanceError(RuntimeError):
    """Raised when a published ledger table lacks a valid provenance/make_record.py artifact."""
    pass


# Canonical frozen thresholds
FROZEN_THRESHOLDS: dict[str, Any] = {
    "dynamical": {
        "tau_epi_default": 0.0200,
        "bootstrap_n_boot": 500,
        "sole_survivor_rule": True,
        "extrapolation_ladder_depth": 5,
    },
    "causal": {
        "fdr_alpha": 0.05,
        "faithfulness_tolerance": 0.05,
        "confounder_sensitivity_bound": 0.10,
    },
    "geometry_optimality": {
        "tau_opt": 0.0010,
        "feasibility_overlap_tolerance": 1e-6,
        "isostatic_coordination_min_ratio": 2.0,  # Z >= 2D
    },
    "physics_pde": {
        "tau_pde": 0.010,
        "energy_leakage_tolerance": 0.05,
        "entropy_violation_tolerance": 1e-4,
    },
    "volatility_empirical": {
        "tau_epi_market": 0.001488,
        "cross_asset_min_assets": 15,
    },
}

FROZEN_SPEC: dict[str, Any] = {
    "version": "APEDP-X v1.0-FROZEN",
    "architecture_freeze_tag": "2026.10-BLINDED-INTERNAL-EVALUATION",
    "grammar_sha256": GRAMMAR_FINGERPRINT,
    "referee_kernel_sha256": compute_file_sha256(ADJUDICATE_PATH),
    "claim_ladder_sha256": compute_file_sha256(CLAIM_LADDER_PATH),
    "thresholds": FROZEN_THRESHOLDS,
    "pinned_registry": PINNED_FROZEN_REGISTRY,
}


def get_frozen_spec() -> dict[str, Any]:
    """Retrieve the immutable APEDP-X v1.0-FROZEN specification dictionary."""
    return dict(FROZEN_SPEC)


def verify_all_frozen_records_or_raise() -> dict[str, str]:
    """Verify that every registered frozen record matches its pinned cryptographic SHA256 digest.

    Fails loudly with FreezeDriftError if any drift is detected.
    """
    results: dict[str, str] = {}
    drifts: list[str] = []
    for rel_path, expected_hash in PINNED_FROZEN_REGISTRY.items():
        file_path = BASE_DIR / rel_path
        if not file_path.is_file():
            drifts.append(f"MISSING: {rel_path} does not exist")
            continue
        actual_hash = compute_file_sha256(file_path)
        results[rel_path] = actual_hash
        if actual_hash != expected_hash:
            drifts.append(
                f"DRIFT in {rel_path}: expected {expected_hash}, got {actual_hash}"
            )
    if drifts:
        raise FreezeDriftError(
            "CRITICAL FREEZE DRIFT DETECTED across frozen specification records:\n  - "
            + "\n  - ".join(drifts)
            + "\nModifications to frozen records block publication."
        )
    return results


def require_table_provenance_record(table_id: str, record_dir: Path | str) -> dict[str, Any]:
    """Validate that a table entering the ledger has a verified provenance/make_record.py artifact.

    Fails loudly with MissingProvenanceError if manifest is missing or returncode is non-zero.
    """
    import json
    rec_path = Path(record_dir)
    manifest_path = rec_path / "manifest.json"
    if not manifest_path.is_file():
        raise MissingProvenanceError(
            f"Table '{table_id}' cannot enter ledger: missing '{manifest_path}'. "
            f"All published tables require provenance/make_record.py output."
        )
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("returncode") != 0:
        raise MissingProvenanceError(
            f"Table '{table_id}' provenance record failed with non-zero returncode {manifest.get('returncode')}."
        )
    if not manifest.get("script_hashes"):
        raise MissingProvenanceError(
            f"Table '{table_id}' provenance manifest lacks script_hashes binding."
        )
    return manifest


def format_frozen_spec_banner() -> str:
    """Format human-readable provenance banner for benchmark CLI outputs."""
    spec = get_frozen_spec()
    return (
        f"ARCHITECTURAL SPECIFICATION: {spec['version']} [{spec['architecture_freeze_tag']}]\n"
        f"  • Grammar Fingerprint   (SHA256): {spec['grammar_sha256'][:16]}...\n"
        f"  • Referee Kernel Hash   (SHA256): {spec['referee_kernel_sha256'][:16]}...\n"
        f"  • Claim Ladder Hash     (SHA256): {spec['claim_ladder_sha256'][:16]}..."
    )

