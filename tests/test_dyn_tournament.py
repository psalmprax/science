"""DUFF 3% adversarial tournament contract (frozen boundary v1).

Additive only: this file plus apedp_dyn/dyn_tournament.py. adjudicate.py,
stage_c, Adjudication semantics, and the informative-abstention pointer are
untouched (see test_rmse_rel_is_audit_only).

Pins what the trace has in every environment measured, and says what it does
not. Until 2026-11-16 this file pinned the whole trace of one machine: C1, C4
and C8 at gate 4 on reps 0-1 and at adjudication on rep 2. Those three sit
near the gate-4 tolerance, where the layer a candidate ends at follows the
optimizer's path through an unconverged fit, and the path follows the library
build. Same code, same seeds, C1/C4/C8 only (g = gate 4, a = adjudication):

    recorded pin              rep0 g g g   rep1 g g g   rep2 a a a
    numpy 2.1.3 scipy 1.14.1  rep0 g g g   rep1 g g g   rep2 g a g
    numpy 2.2.6 scipy 1.15.3  rep0 g g g   rep1 g g g   rep2 g a g
    numpy 2.4.6 scipy 1.17.1  rep0 g g g   rep1 g g g   rep2 g g a
    numpy 2.5.x scipy 1.18.1  rep0 a a a   rep1 g g g   rep2 a a a

The recorded pin reproduces under none of the four, so the two tests that
carried it failed on every interpreter. The fitted truth's holdout error on
rep 0 ranges from 0.085 to 0.344 of the observed RMS across them: finding D1
of the earlier docstring, that gate 4 rejects the fitted truth at 3 % noise,
is a statement about where one optimizer stopped.

What all five agree on, and what is pinned here:
  S1. C5 dies at gate 1 and C6 at gate 3 (structural), and C2, C3 and C7 miss
      the gate-4 tolerance (by 6x, 6x and 1.27x) on every rep.
  S2. No adversary is promoted on any rep. Every environment above promoted
      nothing at all; the truth being promoted would not be a regression.
  S3. The status is the one the survivor count implies.

A change in S1 fails even when the final status is unchanged.
"""
from __future__ import annotations

import inspect

import pytest

from apedp_dyn import dyn_tournament as tour
from apedp_dyn.epistemic.adjudicate import adjudicate as adjudicate_fn

FROZEN_SLATE_HASH = "98ac77e181afc6cc06df382459abd6bcf9068735da67db8e39da1b42510e3abe"

pytestmark = pytest.mark.slow  # 8-candidate slate fits; ~85s

#: S1. Attribution that does not depend on which optimum a fit stopped at.
STRUCTURAL_ATTRIBUTION = {"C2": "gate4", "C3": "gate4", "C5": "gate1", "C6": "gate3", "C7": "gate4"}
#: The truth (C1) and the two adversaries close enough to it to share the
#: gate-4 boundary. Where each ends is environment-dependent; see the module
#: docstring for the five traces.
BOUNDARY_CANDIDATES = ("C1", "C4", "C8")
TRUTH = "C1"


@pytest.fixture(scope="module")
def traces():
    return tour.run_tournament()


def _by_rep(traces, rep):
    return next(t for t in traces["traces"] if t["rep"] == rep)


class TestSlateImmutability:
    def test_slate_hash(self, traces):
        assert traces["slate_hash"] == FROZEN_SLATE_HASH == tour.slate_hash()

    def test_eps_phys_frozen(self, traces):
        assert traces["eps_phys"] == 0.05 == tour.EPS_PHYS


class TestLayerAttribution:
    def test_structural_attribution_map(self, traces):
        for rep in (0, 1, 2):
            rows = _by_rep(traces, rep)["rows"]
            assert set(rows) == set(STRUCTURAL_ATTRIBUTION) | set(BOUNDARY_CANDIDATES)
            observed = {cid: rows[cid]["eliminated_at"] for cid in STRUCTURAL_ATTRIBUTION}
            assert observed == STRUCTURAL_ATTRIBUTION, f"rep {rep}: {observed}"

    def test_boundary_candidates_reach_the_holdout_and_no_adversary_passes_it(self, traces):
        for rep in (0, 1, 2):
            rows = _by_rep(traces, rep)["rows"]
            for cid in BOUNDARY_CANDIDATES:
                allowed = {"gate4", "adjudication"} | ({"promotion"} if cid == TRUTH else set())
                assert rows[cid]["eliminated_at"] in allowed, (rep, cid, rows[cid])

    def test_structural_kills_rep_invariant(self, traces):
        for rep in (0, 1, 2):
            rows = _by_rep(traces, rep)["rows"]
            assert rows["C5"]["eliminated_at"] == "gate1"
            assert rows["C5"]["rmse_rel"] is None
            assert rows["C6"]["eliminated_at"] == "gate3"
            assert rows["C6"]["rmse_rel"] is None


class TestPromotionSet:
    def test_no_adversary_is_promoted(self, traces):
        for rep in (0, 1, 2):
            tr = _by_rep(traces, rep)
            assert tr["promoted"] in (None, TRUTH), (rep, tr["promoted"])

    def test_the_status_is_the_one_the_survivor_count_implies(self, traces):
        for rep in (0, 1, 2):
            tr = _by_rep(traces, rep)
            reached = [c for c, r in tr["rows"].items()
                       if r["eliminated_at"] in ("adjudication", "promotion")]
            assert tr["n_survivors"] == len(reached), (rep, reached)
            assert (tr["adjudication"] == "no_valid_candidate") is (not reached), (rep, tr["adjudication"])
            assert (tr["adjudication"] == "unique_support") is (tr["promoted"] is not None)
            if tr["promoted"] is not None:
                assert tr["rows"][tr["promoted"]]["eliminated_at"] == "promotion"


class TestRmseRel:
    def test_present_iff_holdout_reached(self, traces):
        for rep in (0, 1, 2):
            for cid, r in _by_rep(traces, rep)["rows"].items():
                if r["eliminated_at"] in ("gate4", "adjudication", "promotion"):
                    if r["rmse_rel"] is None:
                        # The holdout was reached and could not be simulated:
                        # there is no error to report, and the candidate must
                        # stop at gate 4 and say why. CI's 3.11 leg on
                        # 2026-11-16 fitted C1 on rep 2 to parameters whose
                        # holdout run overflowed (an environment-dependent
                        # stopping point, docs/SYSTEM_FLAWS_AUDIT.md J-16);
                        # until then this branch assumed it could not happen.
                        assert r["eliminated_at"] == "gate4", (rep, cid, r)
                        assert r["holdout_rmse"] is None, (rep, cid, r)
                        assert "holdout simulation invalid" in r["detail"], (rep, cid, r)
                        continue
                    assert r["rmse_rel"] >= 0, (rep, cid, r)
                    assert abs(r["rmse_rel"] - r["holdout_rmse"] / _obs_rms(rep)) < 1e-9
                else:
                    assert r["rmse_rel"] is None, (rep, cid, r)


    def test_rmse_rel_is_audit_only(self):
        assert "rmse_rel" not in inspect.getsource(adjudicate_fn)
        assert "EPS_PHYS" not in inspect.getsource(adjudicate_fn)


class TestDeterminism:
    def test_rep2_rerun_identical(self, traces):
        again = tour.run_rep(2)
        first = _by_rep(traces, 2)
        assert again["rows"] == first["rows"]
        assert again["adjudication"] == first["adjudication"]
        assert again["promoted"] == first["promoted"]


def _obs_rms(rep: int) -> float:
    import numpy as np

    from apedp_dyn import dyn_sweep as ds

    slate = ds.build_slates()
    prob = ds._with_noise(slate["DUFF"]["prob_clean"], tour.PCT,
                          seed=ds.cell_seed("DUFF", rep, tour.PCT))
    obs = np.asarray(prob.holdout["truth_y"], dtype=float)
    return float(np.sqrt(np.mean(obs ** 2)))
