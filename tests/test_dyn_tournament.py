"""DUFF 3% adversarial tournament contract (frozen boundary v1).

Additive only: this file plus apedp_dyn/dyn_tournament.py. adjudicate.py,
stage_c, Adjudication semantics, and the informative-abstention pointer are
untouched (see test_rmse_rel_is_audit_only).

Pins OBSERVED live-trace behavior, not wished behavior. Deviations from the
naive expectation table are recorded here as findings:
  D1. C1 (ground truth) is NOT promoted at 3% on any rep: gate 4 rejects the
      fitted truth on reps 0-1 (rmse_rel 0.24/0.13 vs auto tolerance), and on
      rep 2 the over-parameterized C4 wins holdout outright (0.0416 < 0.0831)
      with the trio inside tau -> MULTI_HYPOTHESIS. The firewall refuses to
      promote; that refusal is the pinned behavior.
  D2. C4/C8 reach adjudication only on rep 2; on reps 0-1 they die at gate 4.
      Attribution is rep-dependent by design (noise-dependent layers), while
      C5->gate1 and C6->gate3 are structural and rep-invariant.
  D3. reps 0-1 end NO_VALID_CANDIDATE (zero gate-4 survivors).

A change in layer attribution fails even when the final status is unchanged.
"""
from __future__ import annotations

import inspect

import pytest

from apedp_dyn import dyn_tournament as tour
from apedp_dyn.epistemic.adjudicate import adjudicate as adjudicate_fn

FROZEN_SLATE_HASH = "98ac77e181afc6cc06df382459abd6bcf9068735da67db8e39da1b42510e3abe"

pytestmark = pytest.mark.slow  # 8-candidate slate fits; ~85s

EXPECTED_ATTRIBUTION = {
    0: {"C1": "gate4", "C2": "gate4", "C3": "gate4", "C4": "gate4",
        "C5": "gate1", "C6": "gate3", "C7": "gate4", "C8": "gate4"},
    1: {"C1": "gate4", "C2": "gate4", "C3": "gate4", "C4": "gate4",
        "C5": "gate1", "C6": "gate3", "C7": "gate4", "C8": "gate4"},
    2: {"C1": "adjudication", "C2": "gate4", "C3": "gate4",
        "C4": "adjudication", "C5": "gate1", "C6": "gate3",
        "C7": "gate4", "C8": "adjudication"},
}
EXPECTED_STATUS = {0: "no_valid_candidate", 1: "no_valid_candidate",
                   2: "multi_hypothesis"}
EXPECTED_PROMOTED = {0: None, 1: None, 2: None}


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
    def test_attribution_map(self, traces):
        for rep, expected in EXPECTED_ATTRIBUTION.items():
            rows = _by_rep(traces, rep)["rows"]
            observed = {cid: r["eliminated_at"] for cid, r in rows.items()}
            assert observed == expected, f"rep {rep}: {observed}"

    def test_structural_kills_rep_invariant(self, traces):
        for rep in (0, 1, 2):
            rows = _by_rep(traces, rep)["rows"]
            assert rows["C5"]["eliminated_at"] == "gate1"
            assert rows["C5"]["rmse_rel"] is None
            assert rows["C6"]["eliminated_at"] == "gate3"
            assert rows["C6"]["rmse_rel"] is None


class TestPromotionSet:
    def test_status_and_promotion(self, traces):
        for rep in (0, 1, 2):
            tr = _by_rep(traces, rep)
            assert tr["adjudication"] == EXPECTED_STATUS[rep], (rep, tr)
            assert tr["promoted"] == EXPECTED_PROMOTED[rep], (rep, tr)


class TestRmseRel:
    def test_present_iff_holdout_reached(self, traces):
        for rep in (0, 1, 2):
            for cid, r in _by_rep(traces, rep)["rows"].items():
                if r["eliminated_at"] in ("gate4", "adjudication", "promotion"):
                    assert r["rmse_rel"] is not None and r["rmse_rel"] >= 0, (rep, cid, r)
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
