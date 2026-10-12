"""Documentation & Configuration Integrity Test Suite.

Automates the verification of:
  1. Cryptographic SHA-256 integrity of canonical preregistered specifications.
  2. Runtime code consumption of frozen constants with zero drift.
  3. Consistency between documentation text, canonical spec, and runtime implementations.
  4. Detection of broken file links in documentation.
  5. Detection of orphaned source modules missing from DOCUMENTATION.md index.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import re
import pytest

from apedp_dyn.conf.canonical_spec import (
    SPEC_PATH,
    PINNED_SPEC_SHA256,
    TACTICAL_ACTIVE_MIN,
    TACTICAL_ACTIVE_MAX,
    CONTINUOUS_FACTOR_ACTIVE_MIN,
    CONTINUOUS_FACTOR_ACTIVE_MAX,
    TAIL_RISK_MIN_EVENTS,
    DSR_THRESHOLD,
    DIEBOLD_MARIANO_P_VALUE_THRESHOLD,
    MAX_LITERATURE_R2,
    REFEREE_MAX_FPR,
    REFEREE_MIN_CAR,
    REFEREE_MIN_VRR,
    STOCK_SPLIT_INVARIANCE_TOLERANCE,
    load_and_verify_canonical_spec,
    SpecificationIntegrityError,
)
import apedp_dyn.sandbox.canonical_ledger as cl
import apedp_dyn.epistemic.novelty_adjudication as na
import apedp_dyn.epistemic.promotion_lifecycle as pl
import apedp_dyn.epistemic.trial_ledger as tl
import apedp_dyn.sandbox.calibrated_benchmark_suite as cbs


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_canonical_spec_hash_integrity():
    """Verify that canonical spec file matches its pinned cryptographic SHA-256 hash."""
    assert SPEC_PATH.exists(), f"Canonical specification missing at {SPEC_PATH}"
    content = SPEC_PATH.read_bytes()
    computed_hash = hashlib.sha256(content).hexdigest()
    assert computed_hash == PINNED_SPEC_SHA256, (
        f"Canonical spec has been mutated without updating pinned hash!\n"
        f"Expected: {PINNED_SPEC_SHA256}\nComputed: {computed_hash}"
    )

    # Calling loader with enforce_hash=True must succeed
    spec_data = load_and_verify_canonical_spec(enforce_hash=True)
    assert spec_data["spec_id"] == "APEDPX_PREREGISTERED_CANONICAL_V1"
    assert spec_data["thresholds"]["tactical_active_fraction_max"] == 0.90


def test_runtime_code_matches_canonical_spec():
    """Ensure runtime functions consume canonical thresholds with zero drift."""
    # 1. canonical_ledger: assert_strategy_class_compliance defaults
    sig_comp = inspect.signature(cl.assert_strategy_class_compliance)
    params_comp = sig_comp.parameters
    assert params_comp["min_active_tactical"].default == TACTICAL_ACTIVE_MIN
    assert params_comp["max_active_tactical"].default == TACTICAL_ACTIVE_MAX
    assert params_comp["min_events"].default == TAIL_RISK_MIN_EVENTS
    assert params_comp["min_active_continuous"].default == CONTINUOUS_FACTOR_ACTIVE_MIN

    # 2. canonical_ledger: assert_stock_split_invariance default atol
    sig_split = inspect.signature(cl.assert_stock_split_invariance)
    assert sig_split.parameters["atol"].default == STOCK_SPLIT_INVARIANCE_TOLERANCE

    # 3. novelty_adjudication: defaults
    sig_dm = inspect.signature(na.evaluate_predictive_incrementality)
    assert sig_dm.parameters["alpha"].default == DIEBOLD_MARIANO_P_VALUE_THRESHOLD

    sig_novelty = inspect.signature(na.evaluate_structural_novelty)
    assert sig_novelty.parameters["novelty_threshold_r2"].default == MAX_LITERATURE_R2

    # 4. promotion_lifecycle: stage defaults
    tracker = pl.PromotionTracker(candidate_id="test_candidate")
    sig_s3 = inspect.signature(tracker.verify_stage_3)
    assert sig_s3.parameters["dm_alpha"].default == DIEBOLD_MARIANO_P_VALUE_THRESHOLD
    assert sig_s3.parameters["max_r2_ceiling"].default == MAX_LITERATURE_R2

    sig_s5 = inspect.signature(tracker.verify_stage_5)
    assert sig_s5.parameters["min_dsr_hurdle"].default == DSR_THRESHOLD

    # 5. calibrated_benchmark_suite: referee contract constants
    assert cbs.REFEREE_MAX_FPR == REFEREE_MAX_FPR == 0.05
    assert cbs.REFEREE_MIN_CAR == REFEREE_MIN_CAR == 0.70
    assert cbs.REFEREE_MIN_VRR == REFEREE_MIN_VRR == 0.75


def test_documentation_constants_match_runtime_code():
    """Check that documentation and orientation files describe the correct tactical window [5%, 90%]."""
    doc_path = REPO_ROOT / "docs" / "DOCUMENTATION.md"
    readme_path = REPO_ROOT / "README.md"
    arch_path = REPO_ROOT / "ARCHITECTURE.md"

    for file_path in [doc_path, readme_path, arch_path]:
        assert file_path.exists(), f"File {file_path} must exist"
        text = file_path.read_text(encoding="utf-8")
        # Ensure the outdated 95% threshold is not present in tactical window context
        assert "active in [5%, 95%]" not in text, f"Outdated 95% tactical threshold found in {file_path}"
        assert "[5%, 95%]" not in text, f"Outdated [5%, 95%] tactical threshold found in {file_path}"
        # Ensure the corrected 90% threshold is mentioned
        assert ("90%" in text or "90\\%" in text), f"Expected 90% tactical bound not mentioned in {file_path}"


def test_documentation_links_exist_on_disk():
    """Validate that every relative file link or file:// link in DOCUMENTATION.md points to a real file."""
    doc_path = REPO_ROOT / "docs" / "DOCUMENTATION.md"
    content = doc_path.read_text(encoding="utf-8")

    # Match file:///home/psalmprax/ALL_PROJECTS/science/...
    file_uris = re.findall(r"file:///home/psalmprax/ALL_PROJECTS/science/([a-zA-Z0-9_\-\./]+)", content)
    for rel_path in file_uris:
        full_path = REPO_ROOT / rel_path
        assert full_path.exists(), f"Broken file link in DOCUMENTATION.md: {rel_path} does not exist"

    # Match relative markdown links like [Foo](SYSTEM_CAPABILITIES_AND_PHILOSOPHY.md)
    rel_md_links = re.findall(r"\]\(([a-zA-Z0-9_\-\./]+\.md)\)", content)
    for link in rel_md_links:
        if link.startswith("http"):
            continue
        target = (doc_path.parent / link).resolve()
        assert target.exists(), f"Broken relative doc link in DOCUMENTATION.md: {link} does not exist at {target}"


def test_orphan_module_detection():
    """Detect active production modules in apedp_dyn that are missing from DOCUMENTATION.md."""
    doc_path = REPO_ROOT / "docs" / "DOCUMENTATION.md"
    doc_text = doc_path.read_text(encoding="utf-8")

    # Active core packages that MUST be indexed in DOCUMENTATION.md
    core_dirs = [
        REPO_ROOT / "apedp_dyn" / "conf",
        REPO_ROOT / "apedp_dyn" / "epistemic",
        REPO_ROOT / "apedp_dyn" / "grammar",
        REPO_ROOT / "apedp_dyn" / "proposers",
    ]

    orphans = []
    for d in core_dirs:
        for py_file in d.glob("*.py"):
            if py_file.name.startswith("__"):
                continue
            rel_str = str(py_file.relative_to(REPO_ROOT))
            if rel_str not in doc_text:
                orphans.append(rel_str)

    assert not orphans, (
        f"Orphan module detection failed! The following active modules are missing from DOCUMENTATION.md:\n"
        + "\n".join(f" - {o}" for o in orphans)
    )
