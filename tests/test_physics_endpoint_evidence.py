"""The physics page's evidence is what its run measured (outside audit of 27 November 2026).

The page used to grant "interventional invariance" from the request alone (a pendulum amplitude of 1.5 or more, or the
kinetic system; the amplitude's default gave it to the other systems as well) although the route runs no intervention,
hung the two fit flags on one fixed number whatever the scale of the quantity, and said the eight repair invariants
were verified whether or not the stage that audits them had run. These tests fix what each piece now rests on.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from apedp_dyn.ui import physics_endpoint

CAUSAL = "interventional_invariance"


@pytest.fixture(autouse=True)
def _without_the_slow_routes(monkeypatch: pytest.MonkeyPatch) -> None:
    """The route in force and the engine competition are shown beside the result and take no part in the evidence."""
    monkeypatch.setattr(physics_endpoint, "route_reading", lambda *a, **k: {"action": "skipped in this test"})
    monkeypatch.setattr(physics_endpoint, "competition_records", lambda *a, **k: [])


def run(**request: Any) -> dict[str, Any]:
    fields = {"system": "pendulum", "theta0": 1.8, "noise": 0.01, "multiterm": False, "n_samples": 80, "search_seconds": None}
    out: dict[str, Any] = physics_endpoint.run_physics_discovery(SimpleNamespace(**{**fields, **request}))
    return out


@pytest.mark.parametrize("request_", [
    {"system": "pendulum", "theta0": 2.0}, {"system": "pendulum", "theta0": 2.5, "multiterm": True},
    {"system": "duffing"}, {"system": "harmonic"}, {"system": "kinetic"}, {"system": "kinetic", "theta0": 2.5},
], ids=lambda r: "-".join(str(v) for v in r.values()))
def test_no_request_is_granted_a_causal_tier_because_no_intervention_is_run(request_: dict[str, Any]) -> None:
    framework = run(**request_)["epistemic_framework"]
    assert framework["intervention_run"] is False and CAUSAL not in framework["evidence"]
    assert framework["claimed_tier"] == "C4_CAUSAL_MECHANISTIC"
    assert framework["justified_tier"] in ("C3_STRUCTURAL", "C2_PREDICTIVE")
    assert framework["referee_verdict"] == "CAUSAL_UNRESOLVED" and framework["eedr_status"] == "PROMOTED_REPAIRED"
    assert framework["eedr_repair"]["repair_operator_applied"] == "claim_level_calibration"


def test_a_law_that_beat_a_compared_rival_is_structural_and_one_that_stood_alone_is_predictive() -> None:
    pendulum = run(system="pendulum", theta0=2.0)
    assert pendulum["verdict"] == "unique_support" and pendulum["rival_b"] is not None
    assert pendulum["delta_rmse"] > pendulum["tau_epi"]
    assert pendulum["epistemic_framework"]["justified_tier"] == "C3_STRUCTURAL"
    assert pendulum["epistemic_framework"]["evidence"] == ["equivalence_rivals_rejected", "in_sample_convergence",
                                                           "out_of_sample_holdout"]
    kinetic = run(system="kinetic")                       # its units leave one shape: nothing was compared with it
    assert kinetic["verdict"] == "unique_support" and kinetic["rival_b"] is None
    assert kinetic["epistemic_framework"]["justified_tier"] == "C2_PREDICTIVE"
    assert kinetic["epistemic_framework"]["evidence"] == ["in_sample_convergence", "out_of_sample_holdout"]
    assert kinetic["epistemic_framework"]["evidence_failures"] == ["sole_survivor_no_rival_compared"]


def test_the_fit_flags_do_not_hang_on_the_scale_of_the_quantity() -> None:
    """Noise of 0.5 on an acceleration near 9, or of 1 on an energy near 250: the right law, and a holdout error
    above the old fixed 0.25. Both were refused their fit evidence and still handed the causal flag."""
    pendulum = run(system="pendulum", theta0=2.0, noise=0.5)
    assert pendulum["promoted_equation"].endswith("sin(theta)") and pendulum["promoted_candidate"]["holdout_rmse"] > 0.25
    assert pendulum["epistemic_framework"]["justified_tier"] == "C3_STRUCTURAL"
    kinetic = run(system="kinetic", noise=1.0)
    assert kinetic["promoted_candidate"]["holdout_rmse"] > 0.25 and kinetic["promoted_candidate"]["r2"] > 0.99
    assert {"in_sample_convergence", "out_of_sample_holdout"} <= set(kinetic["epistemic_framework"]["evidence"])


def test_two_laws_the_rows_cannot_tell_apart_are_written_down_as_that() -> None:
    small = run(system="pendulum", theta0=0.4)            # sin(theta) and theta agree at a small angle
    assert small["verdict"] == "multi_hypothesis" and not small["promoted_equation"]
    framework = small["epistemic_framework"]
    assert framework["evidence_failures"] == ["multi_hypothesis_identifiability_barrier"]
    assert "equivalence_rivals_rejected" not in framework["evidence"] and framework["justified_tier"] == "C2_PREDICTIVE"


def test_a_candidate_that_fails_the_engine_s_holdout_gate_has_no_holdout_evidence() -> None:
    drowned = run(system="kinetic", noise=200.0)
    assert drowned["verdict"] == "insufficient_evidence"
    framework = drowned["epistemic_framework"]
    assert "out_of_sample_holdout" not in framework["evidence"]
    assert framework["justified_tier"] in ("C1_DESCRIPTIVE", "C0_NO_SUPPORT") and framework["eedr_status"] == "TERMINAL_ABSTAIN"


def test_the_eight_invariants_are_reported_verified_only_when_the_stage_that_audits_them_ran(
        monkeypatch: pytest.MonkeyPatch) -> None:
    assert run(system="harmonic")["epistemic_framework"]["eight_invariants_verified"] is True
    from apedp_dyn.epistemic.claim_ladder import (
        ClaimAdjudicationResult,
        EpistemicAdjudicationVerdict,
    )

    def promoted(candidate_id: str, proposed_claim: Any, evidence: Any, **_: Any) -> Any:
        return ClaimAdjudicationResult(candidate_id=candidate_id, proposed_claim=proposed_claim, justified_claim=proposed_claim,
                                       verdict=EpistemicAdjudicationVerdict.PROMOTED, claim_inflated=False,
                                       justification="stand-in: promoted without a repair", missing_evidence=[],
                                       metrics=evidence.metrics)

    monkeypatch.setattr(physics_endpoint, "adjudicate_scientific_claim", promoted)
    framework = run(system="harmonic")["epistemic_framework"]
    assert framework["eedr_status"] == "PROMOTED_ORIGINAL" and framework["eedr_repair"] is None
    assert framework["eight_invariants_verified"] is False


def test_the_number_of_rows_is_bounded_whatever_is_asked() -> None:
    assert len(run(system="harmonic", n_samples=10 ** 9)["plot_series"]["t"]) == 200
    assert len(run(system="harmonic", n_samples=-5)["plot_series"]["t"]) == 30
