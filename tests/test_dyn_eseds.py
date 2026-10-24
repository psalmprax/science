"""ESEDS-1 pins: ground-truth generators, observational-equivalence trap,
window-sweep recovery, extrapolation ladder, competitive baselines,
active extrapolation safety gate, and benchmark scorecard metrics.
"""
from __future__ import annotations

import numpy as np

from apedp_dyn.sandbox import eseds_dynamical as E


def test_generators_finite_shapes():
    t, N = E.gen_logistic(0.5, 100.0, 1.0, 200)
    assert len(t) == len(N) == 200 and np.isfinite(N).all() and N[-1] > N[0]
    t, Na = E.gen_allee(0.5, 100.0, 10.0, 20.0, 200)
    assert len(Na) == 200 and np.isfinite(Na).all()
    t, Np, Pp = E.gen_lotka_volterra(0.5, 0.02, 0.5, 0.3, 40.0, 9.0, 200)
    assert len(Np) == len(Pp) == 200 and np.isfinite(Np).all()
    t, N1, N2 = E.gen_competition(0.5, 100.0, 0.4, 90.0, 0.5, 0.5, 10.0, 8.0, 200)
    assert len(N1) == len(N2) == 200 and np.isfinite(N1).all()


def test_short_window_abstains():
    r = E.run_eseds_trap(train_end=25, bootstrap_n_boot=100)
    assert r["status"] == "multi_hypothesis"
    assert r["promoted"] is None
    assert r["recovery"] is False
    assert r["safety_ok"] is True


def test_full_window_recovers_logistic():
    r = E.run_eseds_trap(train_end=100, bootstrap_n_boot=100)
    assert r["status"] == "unique_support"
    assert r["promoted"] == "LOGISTIC_candidate"
    assert r["recovery"] is True
    assert r["safety_ok"] is True


def test_ladder_keys_and_ordering():
    r = E.run_eseds_trap(train_end=100, bootstrap_n_boot=100)
    assert set(r["ladder"]["LOGISTIC"]) == {"E0", "E1", "E2", "E3"}
    # Logistic must dominate exponential at long extrapolation
    assert r["ladder"]["LOGISTIC"]["E3"] < r["ladder"]["EXP"]["E3"]


def test_baselines_vs_apedp_trap1():
    """In Trap 1 (short window), naive baselines promote while APEDP abstains."""
    res = E.run_eseds_scenario(family="A", trap_id=1, train_end=25, noise_sigma=0.3)
    # Naive selectors make a premature claim
    assert res["baseline_train"] is not None
    assert res["baseline_aic"] is not None
    assert res["baseline_cv"] is not None
    # APEDP correctly refuses promotion under non-identifiability
    assert res["apedp_status"] in ("multi_hypothesis", "extrapolation_unjustified")
    assert res["apedp_promoted"] is None
    assert res["safety_ok"] is True


def test_families_b_c_d_execution():
    """Verify Families B, C, D execute cleanly and produce scored candidates."""
    for fam in ("B", "C", "D"):
        res = E.run_eseds_scenario(family=fam, trap_id=1, train_end=40, noise_sigma=0.2)
        assert len(res["candidates"]) >= 2
        assert res["apedp_status"] in ("multi_hypothesis", "unique_support", "extrapolation_unjustified")


def test_extrapolation_gate_blocks_divergent_models():
    """Trap 3 (extrapolation beyond equilibrium) flags divergence or limits error."""
    res = E.run_eseds_scenario(family="A", trap_id=3, train_end=45, noise_sigma=0.3)
    assert "CUBIC_POLY" in res["ladder"]
    # Polynomial error at E3 should be dramatically higher than Logistic E3
    assert res["ladder"]["CUBIC_POLY"]["E3"] > res["ladder"]["LOGISTIC"]["E3"]


def test_benchmark_matrix_scorecard():
    """Verify full benchmark scorecard metrics computation."""
    bench = E.run_eseds_benchmark_matrix(verbose=False)
    metrics = bench["metrics"]
    assert "baseline_train" in metrics
    assert "baseline_aic" in metrics
    assert "baseline_cv" in metrics
    assert "apedp" in metrics

    # APEDP must achieve zero false promotion rate and zero unsafe extrapolation
    assert metrics["apedp"]["FPR_promo"] == 0.0
    assert metrics["apedp"]["UER"] == 0.0
    assert metrics["apedp"]["AAR"] > 0.50
