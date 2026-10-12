"""Cross-agent validation: independent traps vs independent referees.

Pins agreement between the two ESEDS entry points (run_eseds_trap and
run_eseds_scenario) on the same traps, the baselines-vs-referee safety gap,
and consistent verdict behavior across the two sphere-packing implementations.
Deterministic (fixed seeds).
"""

from apedp_dyn.sandbox import eseds_dynamical as E
from apedp_dyn.sandbox import eso_packing as P
from apedp_dyn.sandbox.sphere_packing_sandbox import (
    run_sphere_packing_scenario as sphere_scenario,
)


def test_eseds_dual_api_agree_short_window():
    # Trap: N<<K throughout train -> both paths must abstain.
    a = E.run_eseds_trap(train_end=25, bootstrap_n_boot=100)
    b = E.run_eseds_scenario(family="A", trap_id=2)
    assert a["status"] == "multi_hypothesis" and a["promoted"] is None
    assert b["apedp_status"] == "multi_hypothesis" and b["apedp_promoted"] in (None, "None")
    assert a["recovery"] is False and str(b["recovery"]) == "False"


def test_eseds_dual_api_agree_adequate_window():
    # Adequate window -> both paths must recover LOGISTIC.
    a = E.run_eseds_trap(train_end=40, bootstrap_n_boot=100)
    b = E.run_eseds_scenario(family="A", trap_id=1)
    assert a["status"] == "unique_support" and a["promoted"] == "LOGISTIC_candidate"
    assert b["apedp_status"] == "unique_support" and b["apedp_promoted"] == "LOGISTIC_candidate"
    assert a["recovery"] is True and str(b["recovery"]) == "True"


def test_baselines_promote_where_referee_abstains():
    # Safety gap: point-metric baselines promote on the trap window
    # while the referee abstains. This is the behavior being pinned.
    b = E.run_eseds_scenario(family="A", trap_id=2)
    assert b["apedp_status"] == "multi_hypothesis"
    assert b["baseline_train"] == "LOGISTIC_candidate"
    assert b["baseline_aic"] == "LOGISTIC_candidate"


def test_sphere_both_sides_refuse_hard_cases():
    # Hard cases: both implementations must withhold GLOBAL claims.
    assert P.run_eso_case("square", 9, budget=200)["verdict"] == "BEST_OBSERVED"
    assert sphere_scenario(trap_id=3)["apedp_verdict"] == "MULTI_OPTIMAL"


def test_sphere_both_sides_promote_resolved_cases():
    assert P.run_eso_case("square", 4, budget=200)["verdict"] == "GLOBALLY_OPTIMAL"
    assert sphere_scenario(trap_id=2)["apedp_promoted"] == "FCC_candidate"
