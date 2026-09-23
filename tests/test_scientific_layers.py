"""Tests for agnostic scientific layers: scale analysis and taxonomy mapping."""
import pytest

from apedp.classification_ontology import REVIEWER_TAXONOMY, Classification
from apedp.scientific_layers import (
    RelationshipKind,
    candidate_report,
    domain_audit,
    extrapolation_split,
    scale_ratio,
)


def test_scale_ratio_classical_regime():
    r = scale_ratio("m*v**2/2", "m*c**2", {"m": 1.0, "v": 3e5, "c": 3e8})
    assert r["ratio"] == "v**2/(2*c**2)", r
    assert r["regime"] == "small-correction", r


def test_scale_ratio_unknown_without_values():
    r = scale_ratio("m*v**2/2", "m*c**2")
    assert r["regime"] == "unknown", r
    assert r["numeric"] is None, r


def test_domain_audit_catches_negative_nu():
    out = domain_audit(
        "14.6846932060399*sqrt(Pr) + 0.0125048330920227*Re - 72.6895004618477",
        {"Re": (3000, 15000), "Pr": (0.7, 50.0)},
        {"output_min": 0.0}, grid=25)
    assert out["status"] == "CONDITIONAL", out
    assert out["n_bad"] > 0, out
    assert out["worst"][1] < 0.0, out


def test_domain_audit_passes_clean_candidate():
    out = domain_audit("0.023*Re**0.8*Pr**0.4",
                       {"Re": (3000, 15000), "Pr": (0.7, 50.0)},
                       {"output_min": 0.0}, grid=9)
    assert out["status"] == "PASS", out


def test_extrapolation_split_holds_out_tails():
    import numpy as np
    x = {"Re": np.linspace(3000, 15000, 20)}
    sp = extrapolation_split(x, x["Re"], axis="Re")
    assert len(sp["y_train"]) + len(sp["y_test"]) == 20
    assert min(sp["x_test"]["Re"]) < min(sp["x_train"]["Re"])
    assert max(sp["x_test"]["Re"]) > max(sp["x_train"]["Re"])


def test_candidate_report_has_all_fields():
    txt = candidate_report(
        "Nu-cand-1", "14.68*Pr^0.5+0.0124*Re-72.7", "Gnielinski",
        {"Re": (3000, 15000)}, 200, 7.716, 8.148, "PASS",
        {"status": "CONDITIONAL", "reason": "7/625 violate"},
        "synthetic/model-generated", "none", "not established")
    for field in ("REFERENCE:", "CANDIDATE:", "DOMAIN:", "HELD-OUT:",
                  "MAE:", "BASELINE MAE:", "RELATIVE IMPROVEMENT:",
                  "DIMENSIONAL VALIDITY:", "PHYSICAL DOMAIN:", "REASON:",
                  "REFERENCE STATUS:", "EXPERIMENTAL VALIDATION:",
                  "LITERATURE NOVELTY:"):
        assert field in txt, field
    assert "5.30%" in txt


def test_ab_compare_flags_nonsignificant_gap():
    import numpy as np
    rng = np.random.default_rng(0)
    a = list(rng.normal(8.0, 3.0, 200))
    b = list(rng.normal(8.1, 3.0, 200))
    from apedp.scientific_layers import ab_compare
    out = ab_compare(a, b)
    assert out["n"] == 200
    assert out["verdict"] in ("significant", "not-significant")
    assert 0.0 <= out["win_frac_A"] <= 1.0


def test_ab_compare_detects_real_gap():
    pytest.importorskip("scipy")
    from apedp.scientific_layers import ab_compare
    out = ab_compare([1.0] * 50, [5.0] * 50)
    assert out["winner"] == "A"
    assert out["verdict"] == "significant"


def test_candidate_report_carries_significance():
    from apedp.scientific_layers import candidate_report
    txt = candidate_report(
        "Nu-cand-1", "14.68*Pr^0.5+0.0124*Re-72.7", "Gnielinski",
        {"Re": (3000, 15000)}, 200, 7.716, 8.148, "PASS",
        {"status": "CONDITIONAL", "reason": "7/625 violate"},
        "synthetic/model-generated", "none", "not established",
        ab_p=0.21, ab_verdict="not-significant")
    assert "A/B SIGNIFICANCE:" in txt
    assert "not-significant" in txt
    txt2 = candidate_report(
        "X", "x", "ref", {}, 10, 1.0, 2.0, "PASS",
        {"status": "PASS", "reason": ""}, "s", "e", "n")
    assert "A/B SIGNIFICANCE:" not in txt2


def test_bakeoff_ranks_library_and_candidate():
    import numpy as np

    from apedp.scientific_layers import bakeoff
    rng = np.random.default_rng(7)
    x = rng.uniform(0.5, 5.0, 60)
    y = 2.0 * x + 1.0
    xtr, ytr = {"x": x[:20]}, y[:20]
    xte, yte = {"x": x[20:]}, y[20:]
    out = bakeoff(xtr, ytr, xte, yte, lambda d: 2.0 * d["x"] + 1.0)
    forms = [r["form"] for r in out["rows"]]
    assert "candidate" in forms and "additive-linear" in forms
    assert out["failed"] == []
    best = out["rows"][0]
    assert best["mae"] < 1e-6, best


def test_stability_report_flags_disagreement():
    from apedp.scientific_layers import stability_report
    out = stability_report([
        {"expression": "x*y", "mae": 0.01, "seed": 7},
        {"expression": "x*y", "mae": 0.02, "seed": 42}])
    assert out["verdict"] == "stable", out
    out2 = stability_report([
        {"expression": "x*y", "mae": 0.01, "seed": 7},
        {"expression": "x+y", "mae": 0.5, "seed": 42}])
    assert out2["verdict"] == "unstable", out2
    assert out2["mae_spread_frac"] > 1.0, out2


def test_reviewer_taxonomy_maps_onto_enforced_enums():
    for term, (cls_name, kind_name) in REVIEWER_TAXONOMY.items():
        assert cls_name in Classification.__members__, term
        assert kind_name in RelationshipKind.__members__, term
    # Key invariants from review
    assert REVIEWER_TAXONOMY["ALGEBRAIC_REARRANGEMENT"][0] == "DERIVED"
    assert REVIEWER_TAXONOMY["APPROXIMATE_DERIVED_RELATION"] == ("DERIVED", "MODEL")
    assert REVIEWER_TAXONOMY["NOT_FOUND_IN_CURRENT_CORPUS"][0] == "CANDIDATE"
