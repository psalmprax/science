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
import re
from pathlib import Path

import apedp_dyn.epistemic.novelty_adjudication as na
import apedp_dyn.epistemic.promotion_lifecycle as pl
import apedp_dyn.sandbox.calibrated_benchmark_suite as cbs
import apedp_dyn.sandbox.canonical_ledger as cl
from apedp_dyn.conf.canonical_spec import (
    CONTINUOUS_FACTOR_ACTIVE_MIN,
    DIEBOLD_MARIANO_P_VALUE_THRESHOLD,
    DSR_THRESHOLD,
    MAX_LITERATURE_R2,
    PINNED_SPEC_SHA256,
    REFEREE_MAX_FPR,
    REFEREE_MIN_CAR,
    REFEREE_MIN_VRR,
    SPEC_PATH,
    STOCK_SPLIT_INVARIANCE_TOLERANCE,
    TACTICAL_ACTIVE_MAX,
    TACTICAL_ACTIVE_MIN,
    TAIL_RISK_MIN_EVENTS,
    load_and_verify_canonical_spec,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


# Pinned documentation-debt ledger (2026-10-19). Debt was reduced from 125 to
# 67 by indexing the core packages and the benchmark problem catalog, and to ZERO
# by indexing the remaining historical sandbox harnesses. The ledger is now empty:
# any new undocumented module fails this test, and there is no pin to hide behind.
PINNED_UNINDEXED_COUNT = 0
PINNED_UNINDEXED_MODULES = frozenset(
    """
""".split()
)


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
    """Validate every repository-relative link in DOCUMENTATION.md.

    Links are resolved from ``docs/`` (ARCHITECTURE.md §4.2). Absolute
    ``file:///home/<user>/...`` links are rejected outright: they are not
    portable and this test used to honor only that non-portable form.
    """
    doc_path = REPO_ROOT / "docs" / "DOCUMENTATION.md"
    content = doc_path.read_text(encoding="utf-8")

    assert "file:///" not in content, (
        "Absolute file:// links are forbidden in the live documentation index "
        "(use repository-relative links; see ARCHITECTURE.md §4.2)"
    )

    links = re.findall(r"\]\(([^)\s]+)\)", content)
    checked = 0
    for link in links:
        if link.startswith(("http://", "https://", "mailto:", "#")):
            continue
        target = (doc_path.parent / link.split("#", 1)[0]).resolve()
        assert target.exists(), (
            f"Broken relative link in DOCUMENTATION.md: {link} does not exist at {target}"
        )
        checked += 1
    assert checked >= 200, f"documentation link check covered too few links ({checked})"


def test_orphan_module_detection():
    """Detect production modules in apedp_dyn missing from DOCUMENTATION.md.

    The scan is recursive over the whole ``apedp_dyn`` tree (the original
    four-directory, non-recursive scan missed 143/250 modules, including the
    entire ``epistemic/active_disambiguation`` and most of ``discovery_l2``,
    which are now indexed). The pinned debt ledger was fully repaid on
    2026-10-19 (125 -> 67 -> 0): every ``apedp_dyn`` module is now indexed.
    Adding a module without indexing it fails this test; there is no pin left
    to mask a regression.
    """
    doc_path = REPO_ROOT / "docs" / "DOCUMENTATION.md"
    doc_text = doc_path.read_text(encoding="utf-8")

    all_modules = sorted(
        py_file
        for py_file in (REPO_ROOT / "apedp_dyn").rglob("*.py")
        if "__pycache__" not in py_file.parts and not py_file.name.startswith("__")
    )
    unindexed = {
        str(py_file.relative_to(REPO_ROOT))
        for py_file in all_modules
        if str(py_file.relative_to(REPO_ROOT)) not in doc_text
    }

    added = sorted(unindexed - PINNED_UNINDEXED_MODULES)
    repaid = sorted(PINNED_UNINDEXED_MODULES - unindexed)
    assert not added, (
        "New undocumented modules detected (index them in DOCUMENTATION.md "
        "or the pin may not grow):\n" + "\n".join(f" - {a}" for a in added)
    )
    assert not repaid, (
        "Documentation debt was paid down; shrink PINNED_UNINDEXED_MODULES "
        "to match (debt must never mask a regression):\n"
        + "\n".join(f" - {r}" for r in repaid)
    )
    assert len(PINNED_UNINDEXED_MODULES) == PINNED_UNINDEXED_COUNT


def test_registry_status_tags_match_code_status_labels():
    """Registry Status tags and module docstring `Status:` lines must agree.

    AGENTS.md section 3 defines exactly four canonical status tags. The
    DOCUMENTATION.md module registry must use only those tags, and wherever a
    linked module carries an explicit `Status:` line in its docstring, it must
    match the registry tag: documentation cannot certify a stability level
    the code does not declare.
    """
    doc_path = REPO_ROOT / "docs" / "DOCUMENTATION.md"
    content = doc_path.read_text(encoding="utf-8")

    valid_tags = {"production", "experimental", "audit-only", "deprecated"}
    row_re = re.compile(
        r"\|\s*\[`([^`]+)`\]\((\.\./[^)]+)\)[^|]*\|[^|]*\|\s*`([a-z-]+)`\s*\|"
    )
    status_re = re.compile(r"Status:\s*(production|experimental|audit-only|deprecated)\b")

    checked = 0
    for _display, rel, tag in row_re.findall(content):
        assert tag in valid_tags, f"Non-canonical status tag `{tag}` for {rel} in DOCUMENTATION.md"
        target = (doc_path.parent / rel).resolve()
        if not target.exists() or target.suffix != ".py":
            continue
        head = target.read_text(encoding="utf-8")[:4000]
        m = status_re.search(head)
        if m is None:
            continue
        assert m.group(1) == tag, (
            f"Status mismatch for {rel}: DOCUMENTATION.md says `{tag}`, "
            f"module docstring says `{m.group(1)}`"
        )
        checked += 1

    assert checked >= 10, f"registry status consistency check covered too few modules ({checked})"
