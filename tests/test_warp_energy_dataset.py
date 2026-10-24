"""Warp energy-estimate dataset: provenance integrity + scaling-law check."""
import json

import sympy as sp

PATH = "data/warp_energy_estimates.jsonl"
M_SUN_KG = 1.98847e30
L_PLANCK_M = 1.616255e-35


def _rows():
    with open(PATH) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_rows_have_provenance():
    rows = _rows()
    assert len(rows) >= 8, rows
    for r in rows:
        assert r["id"] and r["source"], r
        assert r["credibility"], r
        assert r["license"], r
        assert r["year"] is None or isinstance(r["year"], int), r


def test_wall_scaling_across_33_orders():
    """E ~ R^2/w: Ford-Pfenning (Planck-thin wall) vs Lentz (w=1m), same R=100m.

    Measured ratio must agree with the 1/w prediction within 3 orders of
    magnitude. This tests the SCALING structure shared with candidate #5's
    core form R^2*s*v_s^2 (s ~ 1/w), not the candidate itself.
    """
    rows = {r["id"]: r for r in _rows()}
    fp = rows["ford-pfenning-1997-R100"]
    lz = rows["lentz-2021-soliton-R100"]
    e_fp = abs(fp["E_total_kg_per_vs"])  # at vs = 1
    e_lz = abs(lz["E_total_solar_masses_per_vs2"]) * M_SUN_KG
    measured_ratio = e_fp / e_lz
    predicted_ratio = 1.0 / L_PLANCK_M  # w_Lentz / w_FP ~ 1m / Lp
    assert abs(__import__("math").log10(measured_ratio / predicted_ratio)) < 3.0, (
        measured_ratio, predicted_ratio)


def test_candidate5_scales_correctly():
    """Candidate #5 R^2*s*v_s^2/gamma: finite, grows in v_s and R."""
    from apedp.gated_assurance import safe_parse_expr
    expr = safe_parse_expr(
        "R**2*s*v_s**2/sqrt(1-v**2/c**2)",
        {"I": sp.Symbol("I")},
    )
    def subs(R, s, vs, v, c):
        return float(expr.subs(
            {"R": R, "s": s, "v_s": vs, "v": v, "c": c}))
    base = subs(10.0, 1.0, 1.0, 0.1, 1.0)
    assert abs(subs(10.0, 1.0, 2.0, 0.1, 1.0) / base - 4.0) < 0.05
    assert subs(20.0, 1.0, 1.0, 0.1, 1.0) / base > 3.9
    assert subs(10.0, 1.0, 1.0, 0.5, 1.0) > base  # gamma growth
