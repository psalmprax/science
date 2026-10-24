"""ESO-1 pins: ground-truth verdicts. Deterministic (fixed seeds)."""

from apedp_dyn.sandbox import eso_packing as P


def test_interval_exact_global():
    r = P.run_eso_case("interval", 5, budget=200)
    assert r["verdict"] == "GLOBALLY_OPTIMAL"
    assert abs(r["best_r"] - 0.1) <= 0.01


def test_square4_global():
    r = P.run_eso_case("square", 4, budget=200)
    assert r["verdict"] == "GLOBALLY_OPTIMAL"
    assert abs(r["best_r"] - 0.25) <= 0.01


def test_square9_best_observed_not_global():
    # Random starts cannot reach the 3x3 grid: must refuse GLOBAL.
    r = P.run_eso_case("square", 9, budget=2000)
    assert r["verdict"] == "BEST_OBSERVED"
    assert r["gap"] > r["tau"]
    assert r["best_r"] < 1.0 / 6.0


def test_unknown_instance_rejected():
    try:
        P.run_eso_case("square", 7, budget=100)
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown ground truth")
