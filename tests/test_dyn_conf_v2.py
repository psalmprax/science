"""D5-CONF v2 regression tests (frozen): collinearity ladder + misspecification.

Locks the v2 freeze: kappa-band thresholds, ladder transition shape
(UNIQUE at kappa≈10 → abstention at higher kappa, zero false promotions),
and the misspecification three-way split (generation failure recorded,
selection ambiguity, systematic residual structure for ALL power laws).
"""
from __future__ import annotations

from apedp_dyn.conf import benchmark_v2 as v2
from apedp_dyn.epistemic.adjudicate import EpistemicStatus


class TestCollinearityLadder:
    def test_transition_unique_to_abstention_no_false_promotion(self):
        lad = v2.run_ladder()
        rung1 = lad["by_rung"]["10.0"]
        assert rung1["recovery_rate"] == 1.0, rung1
        assert rung1["false_promotion_rate"] == 0.0
        for key, rung in lad["by_rung"].items():
            if float(key) >= 100.0:
                assert rung["abstention_rate"] == 1.0, (key, rung)
                assert rung["false_promotion_rate"] == 0.0, (key, rung)

    def test_measured_kappa_hits_rungs_and_bands_frozen(self):
        lad = v2.run_ladder(seeds=(0,), kappas=(10.0, 100000.0))
        cells = {c["kappa_target"]: c for c in lad["cells"]}
        assert abs(cells[10.0]["kappa_measured"] - 10.0) / 10.0 < 0.05
        assert abs(cells[100000.0]["kappa_measured"] - 100000.0) / 100000.0 < 0.05
        assert cells[10.0]["kappa_band"] == "well-conditioned"
        assert cells[100000.0]["kappa_band"] == "severe"
        assert v2.KAPPA_BANDS == {"well": 1e2, "severe": 1e3}


class TestMisspecification:
    def test_three_failure_modes_split(self):
        ms = v2.run_misspec(seeds=(0,))
        c = ms["cells"][0]
        assert c["generation_failure"] is True  # structural fact, by construction
        assert c["misspecification"] is True  # all power laws carry residual structure
        for name, m in c["models"].items():
            assert abs(m["resid_loss_corr"]) > v2.MISSPEC_RESID_CORR_FLAG, (name, m)
        # Rival power laws must be mutually ambiguous, not decisively ranked.
        assert c["selection_ambiguity"] or c["adjudication"] == EpistemicStatus.INSUFFICIENT_EVIDENCE.value


class TestCollinearFollowupLoop:
    def test_targeted_excitation_recovers(self):
        """Second closed loop (static domain): MULTI @ kappa=1e5 ->
        re-excite at kappa=10 -> UNIQUE M_true, all seeds."""
        from apedp_dyn.conf import benchmark_v2 as v2

        for seed in (0, 1, 2):
            out = v2.followup_collinear(seed)
            assert out["before"] == "multi_hypothesis", out
            assert out["after"] == "unique_support", out
            assert out["promoted_after"] == "M_true", out
            assert out["recovered"] is True
