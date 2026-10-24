"""Informative-abstention pointer tests (independent boundary v1).

The pointer is observationally downstream: it inspects frozen evidence and
returns recommendations. It cannot modify gates, admissibility, bootstrap
adjudication, statuses, or any artifact. Each test anchors one information
deficit to real measured evidence (never synthetic deficits):

  collinearity       -> v2 ladder high-kappa MULTI cell -> excite variables
  residual structure -> v2b misspec cell (|corr|~0.96)   -> probe regime slice
  weak intervention  -> small-holdout DUFF MULTI         -> stronger drive
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

from apedp_dyn import fit_candidate, make_signal
from apedp_dyn.conf import benchmark_v2 as v2
from apedp_dyn.epistemic.adjudicate import EpistemicStatus, adjudicate
from apedp_dyn.epistemic.pointer import (
    DEFICIT_VOCABULARY,
    POINTER_VERSION,
    recommend,
)
from apedp_dyn.problems import duffing
from apedp_dyn.problems.base import finalize

TOURNAMENT_JSON = Path(__file__).resolve().parent.parent / "outputs" / "dyn_tournament_duff3.json"


def _duff_pair(prob):
    from apedp_dyn import enumerate_candidates, prune_admissible

    cands = enumerate_candidates(required=("u",), optional=("x", "x3", "v"), max_terms=4)
    kept = {tuple(sorted(s.candidate.term_set())): s for s in prune_admissible(prob, cands)}
    return kept[("u", "v", "x", "x3")], kept[("u", "v", "x")]


class TestCollinearityPointer:
    def test_high_kappa_multi_recommends_variable_excitation(self):
        cell = v2.run_ladder_cell(seed=0, kappa_target=1e5)
        assert cell["status"] == EpistemicStatus.MULTI_HYPOTHESIS.value, cell
        assert cell["kappa_measured"] >= 1e3
        before = dict(cell)
        rec = recommend(status=cell["status"], delta_rmse=cell["delta_rmse"],
                        tau_epi=cell["tau_epi"], kappa_measured=cell["kappa_measured"])
        assert cell == before  # read-only: input evidence untouched
        assert rec.status == cell["status"]  # echo, never a new verdict
        assert rec.evidence_deficit == "collinear_predictors"
        assert "excite variable space" in rec.suggested_intervention
        assert rec.pointer_version == POINTER_VERSION


class TestMissingStructurePointer:
    def test_misspec_cell_recommends_regime_probe(self):
        cell = v2.run_misspec_cell(seed=0)
        corrs = {k: m["resid_loss_corr"] for k, m in cell["models"].items()}
        worst = max(corrs.values(), key=abs)
        assert abs(worst) >= 0.5, corrs
        rec = recommend(status=cell["adjudication"], resid_corr=worst,
                        resid_corr_label="loss_frac_Prad_over_Pin")
        assert rec.status == cell["adjudication"]
        assert rec.evidence_deficit == "missing_structure_or_regime"
        assert "regime slice" in rec.suggested_intervention
        assert rec.diagnostic_basis["resid_corr"] == worst


class TestWeakInterventionPointer:
    def test_small_holdout_multi_recommends_stronger_drive(self):
        base = duffing.spec()
        holdout = dict(base.holdout)
        holdout["u_of_t"] = make_signal({"kind": "sine", "amplitude": 0.4, "frequency": 1.0})
        prob = finalize(dataclasses.replace(base, holdout=holdout))
        truth_s, linear_s = _duff_pair(prob)
        rt = fit_candidate(prob, truth_s.candidate)
        rl = fit_candidate(prob, linear_s.candidate)
        adj = adjudicate([rt, rl])
        assert adj.status is EpistemicStatus.MULTI_HYPOTHESIS, adj
        rec = recommend(status=adj.status.value, delta_rmse=adj.delta_rmse,
                        tau_epi=adj.tau_epi, drive_ratio=0.4 / 0.3)
        assert rec.status == adj.status.value
        assert rec.evidence_deficit == "weak_intervention"
        assert "stronger intervention" in rec.suggested_intervention


class TestNeverMutates:
    def test_tournament_artifact_bit_identical(self):
        raw_before = TOURNAMENT_JSON.read_bytes()
        digest_before = hashlib.sha256(raw_before).hexdigest()
        data = json.loads(raw_before)
        for tr in data["traces"]:
            # DUFF frozen spec amplitudes: train 0.3, holdout 2.5.
            rec = recommend(status=tr["adjudication"], drive_ratio=2.5 / 0.3)
            assert rec.status == tr["adjudication"]
            if tr["adjudication"] in ("no_valid_candidate", "multi_hypothesis"):
                # No kappa/residual diagnostics travel with these traces, and
                # the drive is already strong: honest output is unclassified.
                assert rec.evidence_deficit == "unclassified_deficit", (tr["rep"], rec)
        assert hashlib.sha256(TOURNAMENT_JSON.read_bytes()).hexdigest() == digest_before
        assert TOURNAMENT_JSON.read_bytes() == raw_before


class TestPointerContract:
    def test_version_and_vocab_pinned(self):
        # Superseded by TestPointerV11Deficits.test_version_and_vocab_v11;
        # kept as a tripwire documenting the v1.0 -> v1.1 vocabulary growth.
        assert POINTER_VERSION == "1.1.0-evidence-only"
        assert len(DEFICIT_VOCABULARY) == 6

    def test_pointer_imports_no_pipeline(self):
        import apedp_dyn.epistemic.pointer as pm

        assert not hasattr(pm, "fit_candidate")
        assert not hasattr(pm, "simulate")
        assert not hasattr(pm, "adjudicate")


class TestFollowupLoop:
    def test_weak_intervention_followup_recovers(self):
        """Closed loop on the small-holdout DUFF MULTI case: the pointer's
        weak-intervention recommendation is EXECUTED (holdout drive x6.25 to
        the decisive amplitude), both structures refit, re-adjudicated.
        Expectation from the frozen decisive-holdout behavior: UNIQUE truth."""
        import dataclasses as _dc

        from apedp_dyn import fit_candidate, make_signal
        from apedp_dyn.epistemic.adjudicate import adjudicate as adjudicate_fn
        from apedp_dyn.epistemic.followup import FOLLOWUP_VERSION, followup_weak_intervention
        from apedp_dyn.epistemic.pointer import POINTER_VERSION
        from apedp_dyn.problems import duffing
        from apedp_dyn.problems.base import finalize

        assert FOLLOWUP_VERSION == "1.0.0-escalate"
        base = duffing.spec()
        holdout = dict(base.holdout)
        holdout["u_of_t"] = make_signal({"kind": "sine", "amplitude": 0.4, "frequency": 1.0})
        prob = finalize(_dc.replace(base, holdout=holdout))
        truth_s, linear_s = _duff_pair(prob)
        before = adjudicate_fn(
            [fit_candidate(prob, truth_s.candidate),
             fit_candidate(prob, linear_s.candidate)])
        assert before.status is EpistemicStatus.MULTI_HYPOTHESIS, before
        rec = recommend(status=before.status.value, delta_rmse=before.delta_rmse,
                        tau_epi=before.tau_epi, drive_ratio=0.4 / 0.3)
        assert rec.evidence_deficit == "weak_intervention"
        out = followup_weak_intervention(prob, truth_s.candidate, linear_s.candidate,
                                         rec, factor=2.5 / 0.4)
        assert out["acted"] is True
        assert out["before"] == "multi_hypothesis"
        assert out["after"] == "unique_support", out
        assert out["recovered"] is True
        # Pointer artifact untouched by execution.
        assert rec.pointer_version == POINTER_VERSION

    def test_followup_refuses_other_deficits(self):
        from apedp_dyn.epistemic.followup import followup_weak_intervention
        from apedp_dyn.epistemic.pointer import recommend

        rec = recommend(status="multi_hypothesis", kappa_measured=1e5)
        out = followup_weak_intervention(None, None, None, rec, factor=2.0)
        assert out["acted"] is False


class TestPointerV11Deficits:
    def test_qualified_support_on_promoted_structured_winner(self):
        from apedp_dyn.epistemic.pointer import recommend

        rec = recommend(status="unique_support", winner_rho1=0.7)
        assert rec.status == "unique_support"
        assert rec.evidence_deficit == "qualified_support"
        assert "promotion stands" in rec.suggested_intervention

    def test_hypothesis_space_on_unexplained_remainder(self):
        from apedp_dyn.epistemic.pointer import recommend

        rec = recommend(status="insufficient_evidence", resid_rho1=0.8)
        assert rec.evidence_deficit == "hypothesis_space"
        assert "vocabulary" in rec.suggested_intervention

    def test_corr_signal_beats_rho_remainder(self):
        """A measured channel explanation outranks vocabulary doubt."""
        from apedp_dyn.epistemic.pointer import recommend

        rec = recommend(status="insufficient_evidence", resid_corr=0.9,
                        resid_corr_label="loss_frac", resid_rho1=0.8)
        assert rec.evidence_deficit == "missing_structure_or_regime"

    def test_version_and_vocab_v11(self):
        from apedp_dyn.epistemic.pointer import DEFICIT_VOCABULARY, POINTER_VERSION

        assert POINTER_VERSION == "1.1.0-evidence-only"
        assert DEFICIT_VOCABULARY == ("collinear_predictors", "missing_structure_or_regime",
                                      "qualified_support", "hypothesis_space",
                                      "weak_intervention", "unclassified_deficit")
