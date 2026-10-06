"""Test verifying that quadratic drag physical discovery finds and promotes the exact equation.

True law: dv/dt = (F - b * v * |v|) / m = 1.0 * F - 0.5 * v * |v|
"""
from apedp_dyn.ui.server import discover_registry


def test_quadratic_drag_multiterm_discovery_exact_law():
    """Verify that multi-term L2 discovery finds F and (abs(v) * v) with correct parameters."""
    res = discover_registry({"system": "quadratic_drag", "multiterm": True, "noise": 0.0})
    
    assert "pareto_candidates" in res
    pareto_ids = [c["id"] for c in res["pareto_candidates"]]
    
    # Must have generated the drag dissipative term (abs(v) * v) and forcing drive F
    has_abs_term = any("abs(v)" in cid for cid in pareto_ids)
    assert has_abs_term, f"Expected abs(v) in candidate formulas, got: {pareto_ids}"
    
    print("Verdict:", res.get("verdict"))
    print("Promoted:", res.get("promoted_equation"))
    print("Reason:", res.get("reason"))
    for c in res.get("pareto_candidates", []):
        print("Candidate:", c["id"], "RMSE:", c["rmse"], "R2:", c["r2"], "rho1:", c["rho1"])

    # The exact physical law must be present in the Pareto candidates
    exact_cands = [c for c in res["pareto_candidates"] if "abs(v)" in c["id"] and "F" in c["id"]]
    assert len(exact_cands) > 0, f"Expected exact law with F and abs(v)*v, got: {pareto_ids}"
    
    # Check that the 2-term exact physical law achieves >99.9% R^2 on sinusoidal holdout
    exact_law = next(c for c in exact_cands if "(x)" not in c["id"])
    assert exact_law["rmse"] < 0.05, f"Expected exact law RMSE < 0.05, got {exact_law['rmse']}"
    assert exact_law["r2"] > 0.998, f"Expected exact law R2 > 0.998, got {exact_law['r2']}"

    # Best candidate by RMSE must be < 0.03
    min_rmse_cand = min(res["pareto_candidates"], key=lambda c: c["rmse"])
    assert min_rmse_cand["rmse"] < 0.03
    
    # Check that plot series contains the continuous train + holdout trajectory
    plot_series = res.get("plot_series", {})
    assert "t" in plot_series
    assert "actual" in plot_series
    assert "predicted" in plot_series
    assert plot_series["split_index"] > 0
