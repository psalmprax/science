"""The candidate payload protocol: a proposal reference is scored, never grammar.

Status: audit-only

``gate4_provenance`` scores a free-form ``CandidateProposal``, which has no term set.
``CandidatePayload`` is what lets that payload travel through ``ScoredCandidate``
without being called a grammar candidate: it carries an identity and provenance, and
its term set is recorded as empty. These tests pin that boundary -- including that the
reference is deliberately *not* a ``CandidateDynamics``, so no downstream consumer can
read terms the proposal never had.
"""
from __future__ import annotations

from apedp_dyn.epistemic.gate4_provenance import _ProposalReference
from apedp_dyn.grammar.candidate_ir import (
    CandidateDynamics,
    CandidatePayload,
    ScoredCandidate,
    SystemCandidate,
    Term,
)


def _reference() -> _ProposalReference:
    return _ProposalReference(candidate_id="free-form-1", formula="x + y")


def test_reference_satisfies_the_payload_protocol() -> None:
    payload: CandidatePayload = _reference()
    assert payload.candidate_id == "free-form-1"
    assert payload.grammar_version == ""
    assert payload.generation_seed == 0
    assert payload.parent_candidates == ()
    assert payload.parameter_names() == ()
    assert isinstance(payload.metadata, dict)


def test_scored_payload_records_an_empty_term_set() -> None:
    provenance = ScoredCandidate(candidate=_reference()).base_provenance()
    assert provenance["term_set"] == []
    assert provenance["parameter_bounds"] == {}
    assert provenance["parameter_names"] == []
    assert provenance["candidate_id"] == "free-form-1"


def test_grammar_candidates_still_record_their_terms() -> None:
    dynamics = CandidateDynamics(
        lhs="x", terms=(Term(name="x", parameter="k"),), candidate_id="grammar-1")
    provenance = ScoredCandidate(candidate=dynamics).base_provenance()
    assert provenance["term_set"] == ["x"]
    assert provenance["parameter_names"] == ["k"]


def test_a_payload_reference_cannot_be_read_as_a_grammar_candidate() -> None:
    reference = _reference()
    assert isinstance(reference, CandidatePayload)
    assert not isinstance(reference, CandidateDynamics), (
        "a free-form proposal has no term set; treating it as a grammar candidate "
        "would let a consumer rank a grammar the proposal never came from")
    assert not isinstance(reference, SystemCandidate)
