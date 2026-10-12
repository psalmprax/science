"""EP-OPT-1 Unit Test Suite: Sphere Packing & Computational Optimality.

Verifies:
  1. Exact theoretical densities for 2D Hexagonal, 3D FCC, and 3D HCP.
  2. Overlap feasibility guard (hard sphere constraint).
  3. Polytype equivalence class between FCC and HCP (observational non-identifiability).
  4. Numerical overlap super-density rejection.
  5. Amorphous jamming trapping detection.
  6. Clean global optimum promotion.
  7. Full benchmark scorecard matrix execution.
"""
from __future__ import annotations

import numpy as np

from apedp_dyn.sandbox import sphere_packing_sandbox as SP


def test_theoretical_bounds_and_generators():
    hex2d = SP.generate_hexagonal_2d(n_x=4, n_y=4)
    assert np.isclose(hex2d.density, SP.THEORETICAL_MAX_DENSITY[2], atol=1e-5)

    fcc3d = SP.generate_fcc_3d(n_cells=2)
    assert np.isclose(fcc3d.density, SP.THEORETICAL_MAX_DENSITY[3], atol=1e-5)

    hcp3d = SP.generate_hcp_3d(n_cells=2)
    assert np.isclose(hcp3d.density, SP.THEORETICAL_MAX_DENSITY[3], atol=1e-5)

    bcc3d = SP.generate_bcc_3d(n_cells=2)
    assert np.isclose(bcc3d.density, np.pi * np.sqrt(3.0) / 8.0, atol=1e-5)


def test_feasibility_guard_detects_overlap():
    fcc = SP.generate_fcc_3d(n_cells=2)
    is_feas, ov, _msg = SP.audit_feasibility(fcc)
    assert is_feas is True and ov == 0.0

    # Introduce intentional overlap by expanding radius
    fcc_bad = SP.generate_fcc_3d(n_cells=2)
    fcc_bad.radius *= 1.05
    is_feas_bad, ov_bad, msg_bad = SP.audit_feasibility(fcc_bad)
    assert is_feas_bad is False
    assert ov_bad > 0.0
    assert "infeasible" in msg_bad.lower()


def test_polytype_fcc_hcp_equivalence():
    """FCC and HCP achieve identical density and form an equivalence class."""
    res = SP.run_sphere_packing_scenario(trap_id=3, dim=3)
    assert res["apedp_status"] == "POLYTYPE_EQUIVALENCE_CLASS"
    assert res["apedp_verdict"] == "MULTI_OPTIMAL"
    assert "HCP_candidate" in res["equivalence_class"]
    assert "FCC_candidate" in res["equivalence_class"]


def test_trap6_numerical_overlap_rejection():
    """An optimizer claiming super-optimum via overlap is rejected."""
    res = SP.run_sphere_packing_scenario(trap_id=6, dim=3)
    # The promoted winner must be FCC, NOT the overlapping fake
    assert res["apedp_promoted"] == "FCC_candidate"
    assert "OVERLAPPING" not in str(res["apedp_promoted"])


def test_trap4_jammed_rcp_suboptimal():
    """Random Close Packing is flagged as suboptimal, preventing false claim."""
    res = SP.run_sphere_packing_scenario(trap_id=4, dim=3)
    assert res["apedp_status"] in ("LOCALLY_OPTIMAL_TRAPPED", "FEASIBLE_SUBOPTIMAL")
    assert res["apedp_verdict"] in ("SUBOPTIMAL_LOCAL_MINIMUM", "UNRESOLVED")
    assert res["dual_gap"] > 0.05


def test_clean_global_optimality_promotion():
    """In clean 2D and 3D benchmarks, true optimum is promoted as globally optimal."""
    res2d = SP.run_sphere_packing_scenario(trap_id=0, dim=2)
    assert res2d["apedp_status"] == "GLOBALLY_PROVEN_OPTIMAL"
    assert res2d["apedp_promoted"] == "HEXAGONAL_candidate"

    res3d = SP.run_sphere_packing_scenario(trap_id=0, dim=3)
    assert res3d["apedp_status"] == "GLOBALLY_PROVEN_OPTIMAL"
    assert res3d["apedp_promoted"] == "FCC_candidate"


def test_trap7_micro_overlap_adversary():
    """Adversarial candidate with epsilon=5e-4 overlap is caught by feasibility audit."""
    res = SP.run_sphere_packing_scenario(trap_id=7, dim=3)
    assert res["apedp_promoted"] != "MICRO_OVERLAP_ADVERSARY_candidate"
    rec_map = {r["candidate_id"]: r for r in res["forensic_records"]}
    assert rec_map["MICRO_OVERLAP_ADVERSARY_candidate"]["is_feasible"] is False


def test_trap8_local_basin_zero_gradient_trap():
    """Local basin with zero gradient at phi=0.640 is flagged as suboptimal local minimum."""
    res = SP.run_sphere_packing_scenario(trap_id=8, dim=3)
    assert res["apedp_status"] in ("LOCALLY_OPTIMAL_TRAPPED", "FEASIBLE_SUBOPTIMAL")
    assert res["dual_gap"] > 0.05


def test_benchmark_matrix_scorecard():
    """Full benchmark matrix scorecard executes cleanly and proves 0% false claims."""
    bench = SP.run_sphere_packing_benchmark_matrix(verbose=False)
    scorecard = bench["scorecard"]
    assert scorecard["total_scenarios"] == 10
    assert scorecard["apedp_false_optimality_rate"] == 0.0
    assert scorecard["apedp_abstention_safety_rate"] > 0.40

