"""D1.1/D1.4/D1.7 grammar tests: IR, families, topology, prune, provenance."""
import pytest

from apedp_dyn.grammar import (
    TERM_FAMILIES,
    CandidateDynamics,
    Term,
    canonical_form,
    deduplicate,
    enumerate_candidates,
    full_topology_candidate,
    materialize,
    prune_admissible,
    validate,
)
from apedp_dyn.problems import mass_spring_damper


def test_ir_rejects_unknown_terms_and_dup_params():
    with pytest.raises(ValueError):
        validate(CandidateDynamics("xddot", (Term("x", "a0"), Term("nope", "a1"))))
    with pytest.raises(ValueError):
        validate(CandidateDynamics("xddot", (Term("x", "a0"), Term("v", "a0"))))


def test_term_families_cover_vocabulary():
    assert TERM_FAMILIES["restoring"] == ("x", "x2", "x3", "sin_x", "grad_x", "laplacian_x")
    assert TERM_FAMILIES["dissipative"] == ("v", "abs_v_v", "v2")
    assert TERM_FAMILIES["forcing"] == ("u",)
    assert Term("x", "k1").family == "restoring"
    assert Term("v", "c1").family == "dissipative"
    assert Term("u", "b1").family == "forcing"


def test_full_topology_template():
    c = full_topology_candidate()
    assert [t.name for t in c.terms] == ["one", "x", "x3", "v", "abs_v_v", "sin_x", "u"]
    assert c.metadata["family"] == "mechanical_2nd_order"


def test_enumeration_is_controlled():
    cands = enumerate_candidates(required=("u",), max_terms=3)
    # C(6,0)+C(6,1)+C(6,2) = 22, every candidate carries the forcing term.
    assert len(cands) == 22
    assert all("u" in c.term_set() for c in cands)
    assert len({c.candidate_id for c in cands}) == 22


def test_canonical_dedup():
    a = enumerate_candidates(required=("u",), optional=("x",), max_terms=2)
    b = enumerate_candidates(required=("u",), optional=("x",), max_terms=2,
                             id_prefix="other")
    assert len(deduplicate(a + b)) == len(a)
    assert canonical_form(a[0]) == canonical_form(b[0])


def test_materialize_derives_units():
    prob = mass_spring_damper.spec()
    c = enumerate_candidates(required=("u",), optional=("x", "v"), max_terms=3)[-1]
    assert set(c.term_set()) == {"u", "x", "v"}
    m = materialize(prob, c)
    assert [e.state for e in m.equations] == ["x", "v"]
    by_name = {p.name: p for p in m.parameters}
    # x-term coefficient has units 1/s^2 (i.e. k/m); u-term 1/kg (i.e. 1/m).
    assert "s" in by_name["a1"].units and "meter" not in by_name["a1"].units
    assert "kilogram" in by_name["a0"].units


def test_prune_drops_dimensionally_meaningless_before_integration():
    prob = mass_spring_damper.spec()
    cands = enumerate_candidates(required=("u",), max_terms=3)
    kept = prune_admissible(prob, cands)
    # sin(x) with x in meters is physically meaningless: 6 subsets die at gate 1.
    assert len(cands) == 22 and len(kept) == 16
    for s in kept:
        assert s.provenance["gate1_status"]["passed"]
        assert s.provenance["gate2_status"]["passed"]
        assert s.provenance["canonical_form"].startswith("mechanical_2nd_order")
    truth = [s for s in kept if set(s.candidate.term_set()) == {"u", "x", "v"}]
    assert len(truth) == 1
