"""The GF(2^12) tensor search as a proposer: exact screens, no correctness claims.

Status: audit-only

The numbers here are measured from the engine, not chosen: the composed 54 is the
best decomposition that reconstructs, the published 51-Toffoli incumbent ships with
all-zero ``U``/``V``/``W`` and cannot be screened as a decomposition at all, and the
engine's flip-graph move changes the factors without re-solving ``W``, so every
variant it produces fails the exact check. Those three facts are what the target
"fewer than 51 Toffolis" has to be read against.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from apedp_dyn.quantum.field_tensor_search import verify_tensor_decomposition
from apedp_dyn.quantum_compilation.candidates import CandidateStatus
from apedp_dyn.quantum_compilation.gf2_mul_proposer import (
    PUBLISHED_INCUMBENT_TOFFOLI,
    TARGET_TOFFOLI,
    Gf2MultiplicationProposer,
    screen_report,
)
from apedp_dyn.quantum_compilation.ledger import QuantumCandidateLedger


@pytest.fixture(scope="module")
def proposer() -> Gf2MultiplicationProposer:
    return Gf2MultiplicationProposer()


@pytest.fixture(scope="module")
def candidates(proposer, tmp_path_factory):
    return proposer.propose(tmp_path_factory.mktemp("gf2mul"))


def _by_name(candidates, name: str):
    matches = [c for c in candidates if c.parameters["name"] == name]
    assert matches, f"no candidate named {name}"
    return matches


def test_the_exact_screen_accepts_the_pinned_hierarchy(candidates):
    for name, rank in (
        ("Schoolbook-144", 144),
        ("Karatsuba-63", 63),
        ("Karatsuba-54-BaseComposed", 54),
    ):
        (candidate,) = _by_name(candidates, name)
        assert candidate.screen_metrics["valid"] is True, name
        assert candidate.screen_metrics["rank"] == rank
        # The engine's uncontrolled model prices one Toffoli per bilinear product.
        assert candidate.screen_metrics["toffoli"] == rank
        assert candidate.screen_metrics["meets_target"] is False


def test_the_target_is_still_met_by_nothing_that_reconstructs(candidates):
    report = screen_report(candidates)
    assert report["best_exact_name"] == "Karatsuba-54-BaseComposed"
    assert report["best_exact_toffoli"] == 54
    assert report["target_met"] is False
    assert report["exact_candidates"] == 3
    assert report["target_toffoli_exclusive"] == PUBLISHED_INCUMBENT_TOFFOLI == TARGET_TOFFOLI


def test_the_published_incumbent_is_a_count_without_a_decomposition(candidates, proposer):
    (published,) = _by_name(candidates, "Published-51-Incumbent")
    assert published.screen_metrics["valid"] is False
    assert published.screen_metrics["reason"].startswith("decomposition_not_exact")
    assert published.screen_metrics["toffoli"] == PUBLISHED_INCUMBENT_TOFFOLI
    # Its zero factors reconstruct the zero tensor, so it differs from the real one
    # exactly where the real one is nonzero -- not everywhere.
    assert published.screen_metrics["tensor_mismatches"] == int(proposer.tensor.sum())


def test_the_flip_graph_move_breaks_the_tensor_it_started_from(candidates):
    variants = [c for c in candidates if c.parameters["name"].startswith("FlipGraph-")]
    assert len(variants) == 3
    for variant in variants:
        assert variant.screen_metrics["valid"] is False
        assert variant.screen_metrics["tensor_mismatches"] > 0


def test_a_screen_can_only_propose_and_never_claims_correctness(tmp_path, proposer):
    ledger = QuantumCandidateLedger(tmp_path / "ledger.jsonl")
    recorded = proposer.propose(tmp_path / "artifacts", ledger=ledger)

    assert {c.status for c in recorded} == {
        CandidateStatus.PROPOSED,
        CandidateStatus.SCREEN_REJECTED,
    }
    assert all(c.verifier_receipt is None for c in recorded)
    summary = ledger.verdict_summary()
    assert summary["candidates"] == len(recorded) == 7
    assert summary["by_status"] == {"PROPOSED": 3, "SCREEN_REJECTED": 4}
    # No verifier ran, so no status qualifies as a measured outcome and nothing is
    # an incumbent -- but the four refusals are real refusals, not capability gaps.
    assert summary["unverifiable"] == 0
    assert summary["eligible"] == 7
    assert ledger.incumbent("gf2mul") == (None, None)


def test_the_artifact_reproduces_the_tensor_it_was_screened_on(tmp_path, proposer):
    (candidate,) = proposer.propose(
        tmp_path / "artifacts",
        decompositions=[proposer.seeds()[2]],
    )
    payload = json.loads(open(candidate.artifact_path, encoding="utf-8").read())
    U = np.asarray(payload["U"], dtype=np.uint8)
    V = np.asarray(payload["V"], dtype=np.uint8)
    W = np.asarray(payload["W"], dtype=np.uint8)
    exact, mismatches = verify_tensor_decomposition(
        proposer.tensor,
        type(proposer.seeds()[2])(rank=int(payload["rank"]), U=U, V=V, W=W),
    )
    assert exact, mismatches
    assert payload["spec_sha256"] == candidate.contract_sha256 == proposer.spec_sha256


def test_a_candidate_id_is_pinned_to_the_field_and_the_target():
    baseline = Gf2MultiplicationProposer(flip_graph_steps=10)
    identical = Gf2MultiplicationProposer(flip_graph_steps=10)
    assert baseline.spec_sha256 == identical.spec_sha256

    permuted = Gf2MultiplicationProposer(tensor=baseline.tensor[::-1].copy())
    assert permuted.spec_sha256 != baseline.spec_sha256
    first = baseline.candidate(baseline.seeds()[2], "unused")
    second = permuted.candidate(permuted.seeds()[2], "unused")
    assert first.candidate_id != second.candidate_id
    assert first.contract_sha256 == baseline.spec_sha256
