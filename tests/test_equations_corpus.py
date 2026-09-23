"""Offline tests for the equations corpus pipeline (no network)."""
from apedp.equations.canonicalize import group_representations
from apedp.equations.normalize import clean_latex
from apedp.equations.pipeline import run_pipeline
from apedp.equations.schema import EquationRecord


def fake_fetch(page):
    return ["F = m*a", "a = F/m", "x/x"]


def test_clean_latex_frac():
    assert clean_latex(r"\frac{a}{b}") == "(a)/(b)"


def test_canonical_groups_collapse_rearrangements():
    groups = group_representations(["F-m*a", "a-F/m", "m*a-F"])
    assert len(groups) == 1
    assert len(groups[0]["members"]) == 3


def test_pipeline_offline_end_to_end():
    seeds = [{"page": "Test", "domain": "mechanics",
              "license": "CC-BY-SA-4.0", "name": "Newton Test"}]
    out = run_pipeline(seeds=seeds, fetch_fn=fake_fetch)
    assert out["metrics"]["fetched"] == 3
    recs = out["records"]
    # F=ma family collapses; x/x is its own identity group
    assert len(recs) == 2
    kinds = {r.kind for r in recs}
    assert "IDENTITY" in kinds or "CANDIDATE" in kinds
    # license + provenance carried
    assert all(r.license == "CC-BY-SA-4.0" for r in recs)
    assert all(r.provenance.get("source") == "wikipedia-math" for r in recs)


def test_schema_roundtrip():
    r = EquationRecord(equation_id="a.b", canonical="F-m*a")
    d = r.to_dict()
    r2 = EquationRecord.from_dict(d)
    assert r2.equation_id == "a.b"
    assert r2.canonical == "F-m*a"
