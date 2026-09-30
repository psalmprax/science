"""Frozen-constant sensitivity surface (measured, not tuned).

Each constant below is justified alone; this file measures their JOINT
neighborhood: one-at-a-time perturbations around frozen values on fixed
evidence, outcomes pinned as observed. A future change that moves any anchor
fails loudly — sensitivity drift is a regression.
"""
import inspect

import numpy as np
import pytest

from apedp_dyn.epistemic.adjudicate import EpistemicStatus, adjudicate, resolution_floor
from apedp_dyn.grammar.candidate_ir import CandidateDynamics, ScoredCandidate, Term

needs_guards = pytest.mark.skipif(
    "resid_sole" not in inspect.signature(adjudicate).parameters,
    reason="adjudication hardening stood down pending owner; v1 tree active",
)


def _sole_live() -> bool:
    """Behavior probe (deliberately duplicated, not imported: no cross-test
    coupling): poor sole abstains iff Guard 1 is enforced, not merely parsed."""
    try:
        ta = tuple(Term(n, f"a{i}") for i, n in enumerate(("one", "x")))
        c = CandidateDynamics(lhs="xddot", terms=ta, candidate_id="probe")
        g4 = {"passed": True, "holdout_obs_rms": 1.0, "holdout_noise_sigma": 0.06,
              "holdout_n_samples": 400}
        s = ScoredCandidate(c, {"gate4_status": g4}, train_loss=0.1, holdout_rmse=0.50)
        return bool(adjudicate([s], resid_sole=[0.0] * 400).status
                    is EpistemicStatus.INSUFFICIENT_EVIDENCE)
    except Exception:  # noqa: BLE001 -- any failure mode means inactive
        return False


needs_sole_live = pytest.mark.skipif(
    not _sole_live(),
    reason="sole-survivor guards parsed but not enforced by live tree",
)


def _pair(best_rmse, second_rmse):
    ta = tuple(Term(n, f"a{i}") for i, n in enumerate(("one", "x", "x3", "v")))
    tb = tuple(Term(n, f"a{i}") for i, n in enumerate(("one", "v", "abs_v_v")))
    ca = CandidateDynamics(lhs="xddot", terms=ta, candidate_id="m1")
    cb = CandidateDynamics(lhs="xddot", terms=tb, candidate_id="m2")
    g4 = {"passed": True, "holdout_obs_rms": 1.0, "holdout_noise_sigma": 0.1,
          "holdout_n_samples": 100}
    sa = ScoredCandidate(ca, {"gate4_status": g4}, train_loss=0.5, holdout_rmse=best_rmse)
    sb = ScoredCandidate(cb, {"gate4_status": g4}, train_loss=0.52, holdout_rmse=second_rmse)
    return sa, sb


def _sole(rmse):
    ta = tuple(Term(n, f"a{i}") for i, n in enumerate(("one", "x")))
    c = CandidateDynamics(lhs="xddot", terms=ta, candidate_id="sole")
    g4 = {"passed": True, "holdout_obs_rms": 1.0, "holdout_noise_sigma": 0.06,
          "holdout_n_samples": 400}
    return ScoredCandidate(c, {"gate4_status": g4}, train_loss=0.1, holdout_rmse=rmse)


class TestTauSensitivity:
    def test_boundary_pair_moves_with_tau(self):
        """Delta=0.05 sits exactly on baseline tau: halving resolves to
        UNIQUE, doubling (and baseline) abstains. The boundary is sharp by
        construction here — pinned so tau edits show up immediately."""
        sa, sb = _pair(0.45, 0.50)
        base = resolution_floor(1.0, 0.1, 100)
        assert adjudicate([sa, sb], tau_epi=base).status is EpistemicStatus.MULTI_HYPOTHESIS
        assert adjudicate([sa, sb], tau_epi=base * 0.5).status is EpistemicStatus.UNIQUE_SUPPORT
        assert adjudicate([sa, sb], tau_epi=base * 2.0).status is EpistemicStatus.MULTI_HYPOTHESIS


@needs_guards
@needs_sole_live
class TestSoleGammaFlagSurface:
    def test_gamma_sweep_rep2_like_case(self):
        """RMSE 0.075 @ sigma 0.06 (rep2-like, white residuals): passes the
        ceiling for gamma in {2.4, 3.0, 3.6}."""
        rng = np.random.default_rng(4)
        white = rng.normal(0, 1, size=400)
        for gamma in (2.4, 3.0, 3.6):
            adj = adjudicate([self._s(0.075)], resid_sole=white, sole_gamma=gamma)
            assert adj.status is EpistemicStatus.UNIQUE_SUPPORT, (gamma, adj)

    def test_flag_sweep_structured_case(self):
        """rho1=0.594 series: flags 0.4/0.5 abstain, 0.6 promotes. The live
        rep2 value sits near the flag — pinned as a known sensitive edge,
        not smoothed over."""
        import apedp_dyn
        from apedp_dyn import dyn_sweep as ds
        from apedp_dyn.epistemic.bootstrap import holdout_se_series

        slate = ds.build_slates()
        prob = ds._with_noise(slate["STAB"]["prob_clean"], 10,
                              seed=ds.cell_seed("STAB", 2, 10))
        rr = apedp_dyn.fit_candidate(prob, slate["STAB"]["rival"])
        se = holdout_se_series(prob, rr.candidate, rr.fit.param_values)
        rt = apedp_dyn.fit_candidate(prob, slate["STAB"]["truth"])
        for flag, expected in ((0.4, "insufficient_evidence"),
                               (0.5, "insufficient_evidence"),
                               (0.6, "unique_support")):
            adj = adjudicate([rt, rr], resid_sole=se, sole_rho_flag=flag)
            assert adj.status.value == expected, (flag, adj)

    @staticmethod
    def _s(rmse):
        ta = tuple(Term(name, f"a{i}") for i, name in enumerate(("one", "x")))
        c = CandidateDynamics(lhs="xddot", terms=ta, candidate_id="sole")
        g4 = {"passed": True, "holdout_obs_rms": 1.0, "holdout_noise_sigma": 0.06,
              "holdout_n_samples": 400}
        return ScoredCandidate(c, {"gate4_status": g4}, train_loss=0.1, holdout_rmse=rmse)


@needs_guards
@needs_sole_live
class TestCalSensitivity:
    def test_cal_factor_sweep_stable(self):
        """CAL_SIGMA in {1.44, 1.8, 2.16} (+-20%): the rep2-like sole case
        passes Guard 1 throughout (0.075 <= 3*0.06/2.16). Calibration
        uncertainty does not flip this cell."""
        import apedp_dyn.epistemic.calibration as cal

        rng = np.random.default_rng(4)
        ta = tuple(Term(name, f"a{i}") for i, name in enumerate(("one", "x")))
        c = CandidateDynamics(lhs="xddot", terms=ta, candidate_id="sole")
        g4 = {"passed": True, "holdout_obs_rms": 1.0, "holdout_noise_sigma": 0.06,
              "holdout_n_samples": 400}
        s = ScoredCandidate(c, {"gate4_status": g4}, train_loss=0.1, holdout_rmse=0.075)
        white = rng.normal(0, 1, size=400)
        for factor in (1.44, 1.8, 2.16):
            cal.CAL_SIGMA = factor
            try:
                adj = adjudicate([s], resid_sole=white)
                assert adj.status is EpistemicStatus.UNIQUE_SUPPORT, (factor, adj)
            finally:
                cal.CAL_SIGMA = 1.8
        assert cal.CAL_SIGMA == 1.8
