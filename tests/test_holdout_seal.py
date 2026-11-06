"""The burn log is sealed, and the run validates before it spends a slice.

Two separate defects are pinned here.

1. **The registry had no integrity check.** `FullLineageExperimentLedger` rebuilds
   its burned set by reading a JSONL file; a file with a burn deleted reads as a
   file that never had one, and nothing noticed. `holdout_seal` folds every line
   into a SHA-256 chain and fails closed on a mismatch. The tests below break the
   registry in the three ways that matter -- truncation, an edited burn, and a
   rewind -- and require each to raise.

2. **Validation ran after the irreversible step.** Tranche 5's first attempt
   decided twenty holdouts and was then refused by a series check, burning twenty
   slices with nothing to show for them. The producer now evaluates every
   refusable condition in a pre-flight block before the ledger is opened; the
   ordering is asserted on the source, because the cost of regressing it is paid
   in slices that cannot be recovered.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from apedp_dyn.epistemic.holdout_seal import (
    SEAL_VERSION,
    HoldoutSealError,
    compute_seal,
    read_seal,
    seal_path,
    verify_seal,
    write_seal,
)

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "benchmarks" / "quantum_compilation" / "quantum_signal_holdouts.jsonl"
PRODUCER = ROOT / "scripts" / "measure_quantum_signal_ledger.py"

#: Pinned state of the registry as committed. A rewind of the registry, or an edit
#: to a burn, changes the chain and fails here even if the sidecar was rewritten to
#: agree with it -- which is the one thing the seal alone cannot catch.
PINNED_LINES = 282
PINNED_BYTE_LENGTH = 198027
PINNED_CHAIN_SHA256 = "ac4227742cd05fd0f98f911a6db1f37ba35f5f5c5c6053c25da8ec1f9c444b96"


def test_the_seal_matches_the_registry_on_disk():
    assert REGISTRY.is_file(), f"burned-holdout registry missing at {REGISTRY}"
    seal = verify_seal(REGISTRY)
    assert seal.lines == PINNED_LINES
    assert seal.byte_length == PINNED_BYTE_LENGTH
    assert seal.chain_sha256 == PINNED_CHAIN_SHA256
    assert read_seal(REGISTRY).to_dict()["seal_version"] == SEAL_VERSION


def test_the_pinned_chain_is_what_the_file_actually_hashes_to():
    """The seal file and the registry cannot drift apart silently."""
    assert compute_seal(REGISTRY).chain_sha256 == PINNED_CHAIN_SHA256
    assert seal_path(REGISTRY).is_file(), "the sidecar must be committed beside the log"


def test_truncation_is_detected(tmp_path):
    copy = tmp_path / "holdouts.jsonl"
    lines = REGISTRY.read_text(encoding="utf-8").splitlines(keepends=True)
    copy.write_text("".join(lines[:-1]), encoding="utf-8")
    write_seal(copy)  # seal the truncated file, so only the chain can object
    copy.write_text("".join(lines[:-2]), encoding="utf-8")
    with pytest.raises(HoldoutSealError, match="fewer lines|does not match"):
        verify_seal(copy)


def test_an_edited_burn_is_detected(tmp_path):
    copy = tmp_path / "holdouts.jsonl"
    lines = REGISTRY.read_text(encoding="utf-8").splitlines(keepends=True)
    copy.write_text("".join(lines), encoding="utf-8")
    write_seal(copy)
    # Flip one burn's recorded outcome in place; every later line is unchanged and
    # the line count is identical, so only the chain can see it.
    payload = json.loads(lines[0])
    mutated = json.dumps({**payload, "injected": True}) + "\n"
    copy.write_text(mutated + "".join(lines[1:]), encoding="utf-8")
    with pytest.raises(HoldoutSealError, match="does not match|bytes"):
        verify_seal(copy)


def test_a_rewind_is_detected(tmp_path):
    """Truncating while keeping the original seal is the git-checkout case."""
    copy = tmp_path / "holdouts.jsonl"
    copy.write_text("".join(REGISTRY.read_text(encoding="utf-8").splitlines(True)[:100]),
                    encoding="utf-8")
    seal_path(copy).write_text(
        json.dumps({"seal_version": SEAL_VERSION, "artifact": str(REGISTRY),
                    "lines": PINNED_LINES, "byte_length": PINNED_BYTE_LENGTH,
                    "chain_sha256": PINNED_CHAIN_SHA256}, indent=2) + "\n",
        encoding="utf-8")
    with pytest.raises(HoldoutSealError, match="fewer lines|bytes|does not match"):
        verify_seal(copy)


def test_a_missing_seal_on_an_existing_registry_fails_closed(tmp_path):
    copy = tmp_path / "holdouts.jsonl"
    copy.write_text("{}\n", encoding="utf-8")
    with pytest.raises(HoldoutSealError, match="no seal"):
        verify_seal(copy)


def test_a_fresh_registry_is_not_a_tampered_one(tmp_path):
    """A registry that does not exist yet has a defined genesis, not an error."""
    seal = verify_seal(tmp_path / "not-yet.jsonl")
    assert seal.lines == 0 and seal.byte_length == 0
    assert len(seal.chain_sha256) == 64


def test_write_then_verify_round_trips(tmp_path):
    copy = tmp_path / "holdouts.jsonl"
    copy.write_text("".join(REGISTRY.read_text(encoding="utf-8").splitlines(True)[:5]),
                    encoding="utf-8")
    written = write_seal(copy)
    assert verify_seal(copy) == written


def test_the_producer_verifies_the_seal_before_it_opens_the_ledger():
    """Ordering is the fix: the seal and every refusable check precede the burn.

    Read as source rather than executed, because the assertion is about statement
    order and a run would consume slices to prove it.
    """
    source = PRODUCER.read_text(encoding="utf-8")
    seal_at = source.index("verify_seal(holdout_ledger_path)")
    ledger_at = source.index("FullLineageExperimentLedger(persistence_path=holdout_ledger_path)")
    preflight_at = source.index("failed_preflight = [name for name, ok in preflight.items()")
    series_check_at = source.index("cost = cost_sensitivity(rows, baselines, series=series)")
    assert seal_at < ledger_at, "the burn log must be verified before the ledger is loaded"
    assert series_check_at < ledger_at, (
        "the series check must run before the ledger is opened, or a refusal costs slices")
    assert preflight_at < ledger_at, (
        "a run that will be refused must not consume a holdout slice")


def test_the_producer_reseals_after_it_writes():
    source = PRODUCER.read_text(encoding="utf-8")
    assert re.search(r"holdout_seal = write_seal\(holdout_ledger_path\)", source)
    assert "the holdout seal did not verify immediately after writing" in source
