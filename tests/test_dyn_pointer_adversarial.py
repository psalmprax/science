"""Second-order adversarial pointer tournament (frozen v1).

Attacks the recommendation layer itself: contradictory evidence, multiple
simultaneous deficits, borderline thresholds, deliberately unsupported
diagnostics. Pass criterion: the pointer stays conservative and
deterministic — exactly one deficit per the frozen precedence, unclassified
when nothing is supported, never a crash, never a status rewrite, never a
revision of L2 evidence.

Frozen precedence under attack:
  kappa >= 1e3 -> |resid corr| >= 0.5 -> drive ratio < 2 -> unclassified.
"""
from __future__ import annotations

import dataclasses

from apedp_dyn.epistemic.adjudicate import EpistemicStatus, adjudicate
from apedp_dyn.epistemic.pointer import (
    POINTER_VERSION,
    Recommendation,
    recommend,
)


class TestContradictoryEvidence:
    def test_all_three_deficits_yields_exactly_one_collinearity(self):
        """Every signal fires at once: precedence must pick collinearity
        deterministically, not hedge across explanations."""
        rec = recommend(status="multi_hypothesis", delta_rmse=0.01, tau_epi=0.05,
                        kappa_measured=1e5, resid_corr=-0.96,
                        resid_corr_label="loss_frac", drive_ratio=1.2)
        assert rec.evidence_deficit == "collinear_predictors"
        assert rec.status == "multi_hypothesis"

    def test_resid_beats_drive_when_kappa_absent(self):
        rec = recommend(status="multi_hypothesis", resid_corr=0.9,
                        resid_corr_label="x", drive_ratio=1.1)
        assert rec.evidence_deficit == "missing_structure_or_regime"

    def test_unique_status_with_high_kappa_still_routes_by_rules(self):
        """Pointer does not gate on status except where frozen: UNIQUE + high
        kappa and no residual signal -> unclassified (no deficit invented to
        match an abstention narrative)."""
        rec = recommend(status="unique_support", kappa_measured=5e3)
        assert rec.status == "unique_support"
        assert rec.evidence_deficit == "unclassified_deficit"


class TestBorderlines:
    def test_kappa_boundary(self):
        assert recommend(status="multi_hypothesis", kappa_measured=999.9).evidence_deficit \
            == "unclassified_deficit"
        assert recommend(status="multi_hypothesis", kappa_measured=1000.0).evidence_deficit \
            == "collinear_predictors"

    def test_resid_boundary(self):
        assert recommend(status="multi_hypothesis",
                         resid_corr=0.499).evidence_deficit == "unclassified_deficit"
        assert recommend(status="multi_hypothesis",
                         resid_corr=0.5).evidence_deficit == "missing_structure_or_regime"

    def test_drive_boundary(self):
        assert recommend(status="insufficient_evidence",
                         drive_ratio=2.0).evidence_deficit == "unclassified_deficit"
        assert recommend(status="insufficient_evidence",
                         drive_ratio=1.999).evidence_deficit == "weak_intervention"


class TestUnsupportedDiagnostics:
    def test_all_none_multi_is_unclassified(self):
        rec = recommend(status="multi_hypothesis", delta_rmse=0.01, tau_epi=0.05)
        assert rec.evidence_deficit == "unclassified_deficit"
        assert rec.confidence == 0.2

    def test_nan_inputs_neither_crash_nor_invent(self):
        rec = recommend(status="multi_hypothesis", kappa_measured=float("nan"),
                        resid_corr=float("nan"), drive_ratio=float("nan"))
        assert rec.evidence_deficit == "unclassified_deficit"

    def test_negative_kappa_is_unclassified(self):
        rec = recommend(status="multi_hypothesis", kappa_measured=-5.0)
        assert rec.evidence_deficit == "unclassified_deficit"


class TestConfidenceHonesty:
    def test_confidence_bounded_and_unclassified_low(self):
        cases = [
            recommend(status="multi_hypothesis", kappa_measured=1e5),
            recommend(status="multi_hypothesis", resid_corr=0.9),
            recommend(status="insufficient_evidence", drive_ratio=1.0),
            recommend(status="no_valid_candidate"),
        ]
        for rec in cases:
            assert 0.0 <= rec.confidence <= 1.0, rec
        assert cases[-1].confidence == 0.2

    def test_no_truth_claim_fields(self):
        """Recommendation-confidence, not truth-confidence: the schema must
        not grow probability-of-truth fields without a frozen revision."""
        names = {f.name for f in dataclasses.fields(Recommendation)}
        assert names == {"status", "evidence_deficit", "suggested_intervention",
                         "diagnostic_basis", "confidence", "pointer_version"}
        assert POINTER_VERSION == "1.1.0-evidence-only"


class TestLayerInvariant:
    def test_readjudication_identity_after_pointer(self):
        """L3 does not feed back into L2: adjudicating the same fitted pair
        after running the pointer yields the identical status. Uses the live
        small-holdout DUFF MULTI pair (real pipeline evidence)."""
        import dataclasses as _dc

        import apedp_dyn
        from apedp_dyn import make_signal
        from apedp_dyn.problems import duffing
        from apedp_dyn.problems.base import finalize
        from tests.test_dyn_pointer import _duff_pair

        base = duffing.spec()
        holdout = dict(base.holdout)
        holdout["u_of_t"] = make_signal({"kind": "sine", "amplitude": 0.4, "frequency": 1.0})
        prob = finalize(_dc.replace(base, holdout=holdout))
        truth_s, linear_s = _duff_pair(prob)
        fitted = [apedp_dyn.fit_candidate(prob, truth_s.candidate),
                  apedp_dyn.fit_candidate(prob, linear_s.candidate)]
        before = adjudicate(fitted)
        rec = recommend(status=before.status.value, delta_rmse=before.delta_rmse,
                        tau_epi=before.tau_epi, drive_ratio=0.4 / 0.3)
        after = adjudicate(fitted)
        assert before.status == after.status == EpistemicStatus.MULTI_HYPOTHESIS
        assert rec.evidence_deficit == "weak_intervention"
        assert (before.best_rmse, before.second_rmse) == (after.best_rmse, after.second_rmse)
