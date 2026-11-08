"""Pins for the patent prior-art index: provenance, search and scope."""
from __future__ import annotations

import json

import pytest

from apedp.patent_prior_art import (
    KNOWN_IN_PATENT_CORPUS,
    NO_SEARCHABLE_CLAIMS,
    NOT_FOUND_IN_PATENT_CORPUS,
    RELATED_STRUCTURE_IN_PATENT_CORPUS,
    PatentPriorArtIndex,
    PatentProvenanceError,
    PatentRecord,
    load_patent_corpus,
)


def _record(**overrides) -> dict:
    base = {
        "patent_id": "US-TEST-0001-B2",
        "jurisdiction": "US",
        "title": "Saturation response model for a sensor",
        "claim": "A response y obeying y = a*x/(1+b*x) for an input x.",
        "expression": "a*x/(1+b*x)",
        "source": "",
        "retrieved": "",
        "synthetic": True,
    }
    base.update(overrides)
    return base


def _write(tmp_path, rows) -> str:
    path = tmp_path / "corpus.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return str(path)


def test_unattributed_patent_record_is_refused():
    """A record claiming a real patent without a source cannot enter the corpus."""
    with pytest.raises(PatentProvenanceError, match="source URL"):
        PatentRecord.from_dict(_record(synthetic=False))
    with pytest.raises(PatentProvenanceError, match="retrieval date"):
        PatentRecord.from_dict(_record(synthetic=False, source="https://example.org/patent"))


def test_attributed_record_loads(tmp_path):
    path = _write(tmp_path, [_record(
        synthetic=False, source="https://example.org/patent/US-TEST-0001-B2",
        retrieved="2026-10-29")])
    records = load_patent_corpus(path)
    assert len(records) == 1
    assert records[0].synthetic is False


def test_duplicate_patent_ids_are_refused(tmp_path):
    path = _write(tmp_path, [_record(), _record(title="a second copy")])
    with pytest.raises(PatentProvenanceError, match="duplicate patent_id"):
        load_patent_corpus(path)


def test_unparseable_claim_expression_is_refused():
    with pytest.raises(PatentProvenanceError, match="does not parse"):
        PatentRecord.from_dict(_record(expression="a*x/("))


def test_exact_formula_match_reports_the_patent_claim(tmp_path):
    index = PatentPriorArtIndex.from_jsonl(_write(tmp_path, [_record()]))
    result = index.audit("b*x/(1+c*x)")
    assert result.status == KNOWN_IN_PATENT_CORPUS
    assert result.closest_patent_id == "US-TEST-0001-B2"
    assert result.closest_claim.startswith("A response y obeying")
    assert result.searched_patents == 1
    assert result.searchable_claims == 1


def test_same_structure_different_parameters_is_related_not_known(tmp_path):
    index = PatentPriorArtIndex.from_jsonl(_write(
        tmp_path, [_record(expression="x/(1+2*x)")]))
    result = index.audit("x/(1+3*x)")
    assert result.status == RELATED_STRUCTURE_IN_PATENT_CORPUS
    assert result.isomorphism.startswith("STRUCTURAL_SIGNATURE_MATCH")
    assert result.closest_patent_id == "US-TEST-0001-B2"


def test_absent_form_is_reported_as_absent_not_novel(tmp_path):
    index = PatentPriorArtIndex.from_jsonl(_write(
        tmp_path, [_record(expression="a*x/(1+b*x)")]))
    result = index.audit("a*exp(-b*x)")
    assert result.status == NOT_FOUND_IN_PATENT_CORPUS
    assert "not a novelty opinion" in result.epistemic_disclaimer


def test_claims_without_a_formula_are_counted_not_ignored(tmp_path):
    index = PatentPriorArtIndex.from_jsonl(_write(tmp_path, [
        _record(expression=None, claim="A sensor comprising a membrane."),
        _record(patent_id="US-TEST-0002-B2", expression=None,
                claim="A method of calibrating the sensor of claim 1."),
    ]))
    result = index.audit("a*x/(1+b*x)")
    assert result.status == NO_SEARCHABLE_CLAIMS
    assert result.searched_patents == 2
    assert result.searchable_claims == 0
    assert "No record carried a normalised claim formula" in result.epistemic_disclaimer


def test_synthetic_records_are_disclosed_in_every_verdict(tmp_path):
    index = PatentPriorArtIndex.from_jsonl(_write(tmp_path, [_record()]))
    for candidate in ("a*x/(1+b*x)", "a*exp(-b*x)"):
        result = index.audit(candidate)
        assert result.synthetic_records == 1
        assert "1 synthetic" in result.epistemic_disclaimer
