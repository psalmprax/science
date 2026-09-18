"""Tests for the prospective/composer roadmap (A1-A4, B)."""
from __future__ import annotations

import csv

import sympy as sp

from apedp.composer import THEORETICAL_MATURITY, HypothesisComposer
from apedp.core import ClosedLoopAssuranceSystem, DimensionalAssuranceEngine
from apedp.discovery import MultiVariableSRE
from apedp.epistemic import DataPartitionVault, PriorArtAuditor
from apedp.safety import safe_parse_expr


def test_a1_rename_isomorphism_matches_ohm():
    r = PriorArtAuditor.audit_novelty("I*R")
    assert r.status.value == "KNOWN_MODEL"
    assert len(PriorArtAuditor.STANDARD_PHYSICS_CORPUS) >= 20
    assert PriorArtAuditor.corpus_hash() == PriorArtAuditor.corpus_hash()


def test_a1_trivial_parabola_stays_novel():
    r = PriorArtAuditor.audit_novelty("x**2")
    assert r.status.value == "APPARENTLY_NOVEL_WITHIN_CORPUS"
    assert "does NOT constitute" in r.epistemic_disclaimer


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y"])
        for x, y in rows:
            w.writerow([x, y])


def test_a2_prospective_persists_on_matching_holdout(tmp_path):
    from apedp.pipeline import DiscoveryPipeline

    tr = tmp_path / "train.csv"
    ho = tmp_path / "hold.csv"
    _write_csv(str(tr), [(float(x), float(x**2)) for x in range(1, 11)])
    _write_csv(str(ho), [(float(x), float(x**2)) for x in range(11, 16)])
    pipe = DiscoveryPipeline(output_dir=str(tmp_path / "out"))
    r = pipe.run_prospective(str(tr), str(ho), "y", variable_units={"x": "dimensionless"})
    assert r["status"] == "PERSISTED"
    assert r["holdout_r_squared"] > 0.99


def test_a2_prospective_refuses_shifted_holdout(tmp_path):
    from apedp.pipeline import DiscoveryPipeline

    tr = tmp_path / "train.csv"
    ho = tmp_path / "hold.csv"
    _write_csv(str(tr), [(float(x), float(x**2)) for x in range(1, 11)])
    _write_csv(str(ho), [(float(x), float(x**3)) for x in range(11, 16)])
    pipe = DiscoveryPipeline(output_dir=str(tmp_path / "out"))
    r = pipe.run_prospective(str(tr), str(ho), "y", variable_units={"x": "dimensionless"})
    assert r["status"] == "REFUSED"
    assert pipe.store.list_models() == []


def test_a2_vault_blocks_development_touching_holdout():
    v = DataPartitionVault("train", "train", "hold", "hold")
    assert v.verify_isolation("train", "development") is True
    try:
        v.verify_isolation("hold", "development")
    except PermissionError:
        return
    raise AssertionError("vault should have raised")


def test_a3_erf_parses_and_gates_dimensionless():
    expr = safe_parse_expr("erf(x)")
    eng = DimensionalAssuranceEngine()
    eng.register_variable("x", "dimensionless")
    assert eng.check_expression_consistency(expr).dimensionality == eng.registry("dimensionless").dimensionality


def test_a3_thirds_monomial_reachable():
    import random
    from fractions import Fraction

    random.seed(12345)
    eng = DimensionalAssuranceEngine()
    plat = ClosedLoopAssuranceSystem(eng)
    sre = MultiVariableSRE(assurance_system=plat, independent_vars=["a"], target_unit_str="dimensionless")
    eng.register_variable("a", "meter")

    block_dims = {sp.Symbol("a"): {"meter": 1.0}}
    got = sre._monomial_for_dim(
        {"meter": float(Fraction(1, 3))}, block_dims, tries=2000
    )
    assert got is not None
    assert got.has(sp.Symbol("a"))


def test_a4_metrology_passthrough_and_lambda_cache(tmp_path):
    from apedp.pipeline import DiscoveryPipeline

    p = tmp_path / "data.csv"
    _write_csv(str(p), [(float(x), float(2 * x + 1)) for x in range(1, 11)])
    pipe = DiscoveryPipeline(output_dir=str(tmp_path / "out"))
    r = pipe.run(str(p), "y", variable_units={"x": "dimensionless"}, param_uncertainties={"x": 0.01})
    assert r["status"] == "PERSISTED"
    assert r["metrology"] is not None
    assert r["metrology"]["method"] in ("gum", "monte_carlo")

    eng = DimensionalAssuranceEngine()
    plat = ClosedLoopAssuranceSystem(eng)
    sre = MultiVariableSRE(assurance_system=plat, independent_vars=["x"])
    syms = [sp.Symbol("x")]
    e = sp.Symbol("x") ** 2
    assert sre._lambdify_cached(syms, e) is sre._lambdify_cached(syms, e)


def test_b_composer_returns_ranked_speculation():
    c = HypothesisComposer()
    res = c.compose(["m*v**2/2", "G*m1*m2/r**2"], max_candidates=5)
    assert res
    assert all(r.maturity == THEORETICAL_MATURITY for r in res)
    scores = [r.score for r in res]
    assert scores == sorted(scores, reverse=True)
    # Handoff: templates must parse in the fitter without crashing seeding.
    eng = DimensionalAssuranceEngine()
    plat = ClosedLoopAssuranceSystem(eng)
    priors = [r.as_template_prior() for r in res[:2]]
    sre = MultiVariableSRE(
        assurance_system=plat, independent_vars=["m", "v"],
        target_unit_str="joule", template_priors=priors,
    )
    assert isinstance(sre.template_exprs, list)
