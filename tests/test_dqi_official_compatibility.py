"""Pins for Experiment D: what the pinned competition binary can compile.

`docs/DQI_RESOURCE_RECONCILIATION.md` §2 blamed the missing `MR` opcode alone.
Experiment D measures the whole surface of both builds. These tests pin the
findings: no `mr`, no `h` (so `hmr` cannot be composed into one), no
`gf_product` policy, an AND clear that is already free, the arm building on one
build and not the other, and the one candidate the pinned binary *can* compile.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RECORD = ROOT / "benchmarks" / "quantum_compilation" / "dqi_official_compatibility.json"
SCRIPT = ROOT / "scripts" / "measure_dqi_official_compatibility.py"


@pytest.fixture(scope="module")
def record() -> dict:
    return json.loads(RECORD.read_text())


def test_the_pinned_binary_reproduces_the_reference_row(record):
    reference = record["builds"]["pinned"]["reference"]
    assert reference["qubits"] == 1893
    assert reference["mean_toffoli_like"] == pytest.approx(5_725_071.609375, abs=1e-6)


def test_the_two_binaries_count_the_reference_identically(record):
    assert record["comparison"]["per_shot_arrays_identical"] is True
    assert record["builds"]["pinned"]["reference"]["toffoli_per_shot"] == \
        record["builds"]["local"]["reference"]["toffoli_per_shot"]


def test_the_pinned_surface_is_measured(record):
    surface = record["builds"]["pinned"]["surface"]
    assert surface["mr"] is False
    assert surface["h"] is False
    assert surface["hmr"] is True
    assert surface["gf_product"] is False
    assert surface["push_condition"] is True and surface["bit_store"] is True


def test_the_hmr_substitution_is_blocked_by_the_missing_hadamard(record):
    pinned = record["builds"]["pinned"]
    assert pinned["mr_call"]["works"] is False
    assert pinned["h_call"]["works"] is False
    assert "h" in record["blockers"]["mr"]["why_it_fails"]


def test_the_product_policy_is_a_second_blocker(record):
    blocker = record["blockers"]["gf_product"]
    assert blocker["present_on_pinned"] is False
    assert blocker["documented_before_this_record"] is False
    assert blocker["worth_per_shot_reference"] > 500_000


def test_the_arm_builds_on_one_build_only(record):
    assert record["builds"]["pinned"]["arm"]["built"] is False
    assert "mr" in record["builds"]["pinned"]["arm"]["error"]
    assert record["builds"]["local"]["arm"]["built"] is True
    assert record["builds"]["local"]["arm"]["qubits"] == 1913


def test_clearing_an_and_is_already_free_on_the_pinned_binary(record):
    pinned = record["builds"]["pinned"]["and_clear"]
    assert pinned["init_and"] == 1
    assert pinned["del_and"] == 0


def test_one_candidate_is_compilable_and_measured(record):
    candidate = record["builds"]["pinned"]["candidate"]
    assert candidate["built"] is True
    assert candidate["syndromes_restored"] is True
    assert candidate["qubits"] > record["builds"]["pinned"]["reference"]["qubits"]
    assert "Gf54MulMixin" in candidate["call"]
    assert candidate["mean_toffoli_like"] < record["builds"]["pinned"]["reference"][
        "mean_toffoli_like"]


def test_the_record_does_not_claim_a_receipt(record):
    assert "no submission" in record["scope"]["not_measured"]
    assert "no officially compilable candidate" not in record["verdict"]


def test_the_record_pins_the_script(record):
    assert record["reproducibility"]["script_sha256"] == hashlib.sha256(
        SCRIPT.read_bytes()).hexdigest()
    assert record["failed_checks"] == []
