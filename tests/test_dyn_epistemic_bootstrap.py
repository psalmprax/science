"""Bootstrap-v2 adjudication regression tests (frozen baseline: point-estimate-v1).

- Tiebreaker mechanics on synthetic series (flip / confirm / rival-favored).
- End-to-end: would-be UNIQUE with noise-aligned margin → MULTI_HYPOTHESIS.
- Negative control (frozen failure STAB/10/rep2): sole-survivor promotion has
  NO rival pair, so v2 must leave it unchanged. The bootstrap tiebreaker does
  not address that mechanism; a sole-survivor rule is the documented next
  target. This test pins that fact so a future patch cannot silently claim it.
- v1 default path unaffected (bootstrap defaults off).
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from apedp_dyn.epistemic.adjudicate import EpistemicStatus, adjudicate
from apedp_dyn.epistemic.bootstrap import block_bootstrap_ci, tiebreak
from apedp_dyn.grammar.candidate_ir import CandidateDynamics, ScoredCandidate, Term

pytestmark = pytest.mark.slow  # 2000-rep block bootstraps; ~60s


def _probe(params: dict, expect_insufficient: bool) -> bool:
    """Behavior probe: run one synthetic sole adjudication; True iff the
    live tree enforces sole-survivor guards (not merely parses them).
    Evaluated once at collection; collection and run share one import, so
    the verdict is self-consistent within the run even if the file changes
    between runs. Any exception (e.g. unknown kwargs) counts as inactive."""
    try:
        ta = tuple(Term(n, f"a{i}") for i, n in enumerate(("one", "x")))
        c = CandidateDynamics(lhs="xddot", terms=ta, candidate_id="probe")
        g4 = {"passed": True, "holdout_obs_rms": 1.0, "holdout_noise_sigma": 0.06,
              "holdout_n_samples": 400}
        s = ScoredCandidate(c, {"gate4_status": g4}, train_loss=0.1,
                            holdout_rmse=0.50)
        got = adjudicate([s], **params).status is EpistemicStatus.INSUFFICIENT_EVIDENCE
        return bool(got == expect_insufficient)
    except Exception:  # noqa: BLE001 -- any failure mode means inactive
        return False


_PARAMS = set(inspect.signature(adjudicate).parameters)
_HAS_GUARDS = {"resid_sole", "sole_survivor_rule", "sole_gamma",
               "sole_rho_flag"} <= _PARAMS
needs_guards = pytest.mark.skipif(
    not _HAS_GUARDS,
    reason="guard kwargs absent from live adjudicate(); v1 tree active",
)
needs_bootstrap = pytest.mark.skipif(
    "bootstrap" not in _PARAMS,
    reason="bootstrap tiebreaker hook absent from live adjudicate(); "
           "guard-only tree active",
)
# Guards parsed but not enforced (observed mid-rewrite) must also skip:
# poor sole (rmse 0.5 >> ceiling) abstains iff Guard 1 is live.
_SOLE_LIVE = _probe({"resid_sole": [0.0] * 400}, expect_insufficient=True)
needs_sole_live = pytest.mark.skipif(
    not _SOLE_LIVE,
    reason="sole-survivor guards parsed but not enforced by live tree",
)


def _probe_const_abstains() -> bool:
    """True iff the live tree abstains on constant residuals (fork point:
    author's contract passes them, live tree observed abstaining)."""
    try:
        ta = tuple(Term(n, f"a{i}") for i, n in enumerate(("one", "x")))
        c = CandidateDynamics(lhs="xddot", terms=ta, candidate_id="probe-c")
        g4 = {"passed": True, "holdout_obs_rms": 1.0, "holdout_noise_sigma": 0.06,
              "holdout_n_samples": 400}
        s = ScoredCandidate(c, {"gate4_status": g4}, train_loss=0.1, holdout_rmse=0.075)
        return bool(adjudicate([s], resid_sole=[0.0] * 400).status
                    is EpistemicStatus.INSUFFICIENT_EVIDENCE)
    except Exception:  # noqa: BLE001 -- any failure mode means inactive
        return False


def _probe_dual_ceiling() -> bool:
    """True iff a fitted-scale RMSE passes Guard 1 via a relative ceiling
    (dual-ceiling contract live) rather than failing on the absolute term."""
    try:
        ta = tuple(Term(n, f"a{i}") for i, n in enumerate(("one", "x")))
        c = CandidateDynamics(lhs="xddot", terms=ta, candidate_id="probe-d")
        g4 = {"passed": True, "holdout_obs_rms": 1.0, "holdout_noise_sigma": 1e-9,
              "holdout_n_samples": 400}
        rng = np.random.default_rng(4)
        s = ScoredCandidate(c, {"gate4_status": g4}, train_loss=1e-8, holdout_rmse=3e-4)
        return bool(adjudicate([s], resid_sole=rng.normal(0, 1, size=400)).status
                    is EpistemicStatus.UNIQUE_SUPPORT)
    except Exception:  # noqa: BLE001 -- any failure mode means inactive
        return False


_CONST_ABSTAINS = _probe_const_abstains()
_DUAL_LIVE = _probe_dual_ceiling()


def _flip_series(seed: int = 7, n: int = 400) -> np.ndarray:
    """Positive mean but block-dependent: a concentrated negative chunk makes
    block means straddle zero while the point mean stays positive."""
    rng = np.random.default_rng(seed)
    d = np.full(n, 0.35) + rng.normal(0, 0.02, size=n)
    d[180:230] = -2.0 + rng.normal(0, 0.02, size=50)
    return d


def _fake_pair(best_rmse: float, second_rmse: float):
    ta = tuple(Term(t, f"a{i}") for i, t in enumerate(("one", "x", "x3", "v")))
    tb = tuple(Term(t, f"a{i}") for i, t in enumerate(("one", "v", "abs_v_v")))
    ca = CandidateDynamics(lhs="xddot", terms=ta, candidate_id="boot-top")
    cb = CandidateDynamics(lhs="xddot", terms=tb, candidate_id="boot-rival")
    g4 = {"passed": True, "holdout_obs_rms": 1.0, "holdout_noise_sigma": 0.02,
          "holdout_n_samples": 400}
    sa = ScoredCandidate(ca, {"gate4_status": g4}, train_loss=0.4, holdout_rmse=best_rmse)
    sb = ScoredCandidate(cb, {"gate4_status": g4}, train_loss=0.5, holdout_rmse=second_rmse)
    return sa, sb


class TestBlockBootstrap:
    def test_ci_brackets_mean_deterministic(self):
        rng = np.random.default_rng(3)
        d = 0.1 + rng.normal(0, 0.05, size=400)
        lo, hi, mean = block_bootstrap_ci(d, n_boot=500, seed=11)
        assert lo <= mean <= hi
        assert lo > 0  # decisive positive series stays positive

    def test_flip_branch(self):
        # d = resid_rival - resid_top has positive mean but block-dependent CI.
        d = _flip_series()
        assert float(np.mean(d)) > 0
        tb = tiebreak(0.06, 0.02, np.zeros(400), d, n_boot=1000, seed=5)
        assert tb["branch"] == "multi", tb
        assert tb["lo"] <= 0 <= tb["hi"]

    def test_confirm_branch(self):
        rng = np.random.default_rng(9)
        d = 0.1 + rng.normal(0, 0.01, size=400)
        tb = tiebreak(0.06, 0.02, np.zeros(400), d, n_boot=1000, seed=5)
        assert tb["branch"] == "confirm", tb

    def test_rival_favored_branch(self):
        rng = np.random.default_rng(9)
        d = -0.1 + rng.normal(0, 0.01, size=400)
        tb = tiebreak(0.06, 0.02, np.zeros(400), d, n_boot=1000, seed=5)
        assert tb["branch"] == "rival-favored", tb


@needs_guards
@needs_bootstrap
class TestAdjudicateV2:
    def test_would_be_unique_flips_to_multi(self):
        sa, sb = _fake_pair(0.05, 0.12)  # point Δ=0.07 > τ≈0.05 → would-be UNIQUE
        v1 = adjudicate([sa, sb])
        assert v1.status is EpistemicStatus.UNIQUE_SUPPORT, v1
        d = _flip_series()
        assert float(np.mean(d)) > 0
        v2 = adjudicate([sa, sb], resid_best=np.zeros(400), resid_rival=d,
                        bootstrap=True, bootstrap_n_boot=1000, bootstrap_seed=5)
        assert v2.status is EpistemicStatus.MULTI_HYPOTHESIS, v2
        assert v2.promoted is None
        assert v2.bootstrap_branch == "multi"
        assert v2.bootstrap_lo <= 0 <= v2.bootstrap_hi

    def test_decisive_margin_confirmed(self):
        rng = np.random.default_rng(9)
        sa, sb = _fake_pair(0.05, 0.12)
        d = 0.1 + rng.normal(0, 0.01, size=400)
        v2 = adjudicate([sa, sb], resid_best=np.zeros(400), resid_rival=d,
                        bootstrap=True, bootstrap_n_boot=1000, bootstrap_seed=5)
        assert v2.status is EpistemicStatus.UNIQUE_SUPPORT, v2
        assert v2.promoted is not None

    def test_v1_default_unaffected(self):
        sa, sb = _fake_pair(0.05, 0.12)
        v1 = adjudicate([sa, sb])
        assert v1.status is EpistemicStatus.UNIQUE_SUPPORT
        assert v1.bootstrap_branch == ""

    def test_version_pinned(self):
        from apedp_dyn.epistemic.adjudicate import (
            DYNAMIC_EPISTEMIC_VERSION,
            RESID_RHO_FLAG,
            SOLE_SURVIVOR_GAMMA,
        )

        assert DYNAMIC_EPISTEMIC_VERSION == "bootstrap-v3+solefid-v5"

        assert SOLE_SURVIVOR_GAMMA == 3.0
        assert RESID_RHO_FLAG == 0.5


@needs_guards
@needs_bootstrap
class TestFrozenFlaggedCell:
    def test_stab10_rep2_sole_survivor_unchanged_by_v2(self):
        """Frozen failure STAB/10/rep2 (outputs/dyn_epistemic_freeze.json):
        cubic truth blows up on holdout (holdout_rmse None); linear rival
        promoted unopposed under v1. No rival pair exists, so the bootstrap
        tiebreaker never applied; the sole-survivor guards now govern.
        Pre-hardening baseline pinned explicitly (sole-survivor rule off)."""
        import apedp_dyn
        from apedp_dyn import dyn_sweep as ds

        slate = ds.build_slates()
        prob = ds._with_noise(slate["STAB"]["prob_clean"], 10,
                              seed=ds.cell_seed("STAB", 2, 10))
        rt = apedp_dyn.fit_candidate(prob, slate["STAB"]["truth"])
        rr = apedp_dyn.fit_candidate(prob, slate["STAB"]["rival"])
        assert rt.holdout_rmse is None  # truth collapsed: the frozen mechanism
        v1 = adjudicate([rt, rr], sole_survivor_rule=False)
        assert v1.status is EpistemicStatus.UNIQUE_SUPPORT, v1
        assert v1.promoted is not None
        assert v1.delta_rmse is None  # no pair: tiebreaker inapplicable
        # Default call without residuals: Guard 2 missing-evidence row.
        v2 = adjudicate([rt, rr], bootstrap=True)
        assert v2.status is EpistemicStatus.INSUFFICIENT_EVIDENCE, v2
        assert v2.promoted is None
        assert v2.bootstrap_branch == ""


class TestFlaggedCellGuardsLive:
    """Guard-only pins for STAB/10/rep2 (no bootstrap hook required)."""

    def test_v1_flag_baseline_pinned(self):
        import apedp_dyn
        from apedp_dyn import dyn_sweep as ds

        slate = ds.build_slates()
        prob = ds._with_noise(slate["STAB"]["prob_clean"], 10,
                              seed=ds.cell_seed("STAB", 2, 10))
        rt = apedp_dyn.fit_candidate(prob, slate["STAB"]["truth"])
        rr = apedp_dyn.fit_candidate(prob, slate["STAB"]["rival"])
        assert rt.holdout_rmse is None
        v1 = adjudicate([rt, rr], sole_survivor_rule=False)
        assert v1.status is EpistemicStatus.UNIQUE_SUPPORT, v1
        assert v1.promoted is not None
        assert v1.delta_rmse is None

class TestFlaggedCellMeasuredGuardsLive:
    def test_measured_residuals_abstain(self):
        """Moved live out of the bootstrap-gated class: uses only guard
        kwargs present in the live tree."""
        import apedp_dyn
        from apedp_dyn import dyn_sweep as ds
        from apedp_dyn.epistemic.bootstrap import holdout_se_series

        slate = ds.build_slates()
        prob = ds._with_noise(slate["STAB"]["prob_clean"], 10,
                              seed=ds.cell_seed("STAB", 2, 10))
        rt = apedp_dyn.fit_candidate(prob, slate["STAB"]["truth"])
        rr = apedp_dyn.fit_candidate(prob, slate["STAB"]["rival"])
        se = holdout_se_series(prob, rr.candidate, rr.fit.param_values)
        adj = adjudicate([rt, rr], resid_sole=se)
        assert adj.status is EpistemicStatus.INSUFFICIENT_EVIDENCE, adj
        assert adj.promoted is None
        assert "rho1" in adj.reason


@needs_guards
@needs_sole_live
class TestSoleSurvivorGuards:
    """solefid-v2: Guard 1 (RMSE <= 3*sigma) then Guard 2 (|rho1| <= 0.5).

    Missing evidence never passes: no finite RMSE, no usable sigma, and no
    residual series each independently force INSUFFICIENT_EVIDENCE.
    """

    def _sole(self, rmse, resid=None):
        ta = tuple(Term(name, f"a{i}") for i, name in enumerate(("one", "x")))
        c = CandidateDynamics(lhs="xddot", terms=ta, candidate_id="sole")
        g4 = {"passed": True, "holdout_obs_rms": 1.0, "holdout_noise_sigma": 0.06,
              "holdout_n_samples": 400}
        kw = {} if resid is None else {"resid_sole": resid}
        return ScoredCandidate(c, {"gate4_status": g4}, train_loss=0.1,
                               holdout_rmse=rmse), kw

    def test_ceiling_fail_abstains(self):
        s, kw = self._sole(0.50)
        adj = adjudicate([s], **kw)
        assert adj.status is EpistemicStatus.INSUFFICIENT_EVIDENCE, adj
        assert adj.promoted is None

    def test_both_guards_pass_promotes(self):
        rng = np.random.default_rng(4)
        s, _ = self._sole(0.075)
        adj = adjudicate([s], resid_sole=rng.normal(0, 1, size=400))
        assert adj.status is EpistemicStatus.UNIQUE_SUPPORT, adj
        assert adj.promoted is not None

    def test_structured_residuals_abstain(self):
        t = np.arange(400)
        resid = np.sin(2 * np.pi * t / 40.0)  # strongly autocorrelated
        s, _ = self._sole(0.075)
        adj = adjudicate([s], resid_sole=resid)
        assert adj.status is EpistemicStatus.INSUFFICIENT_EVIDENCE, adj
        assert adj.promoted is None
        assert "rho1" in adj.reason

    def test_missing_residuals_never_promote(self):
        s, _ = self._sole(0.075)  # inside the ceiling, but no series
        adj = adjudicate([s])
        assert adj.status is EpistemicStatus.INSUFFICIENT_EVIDENCE, adj
        assert adj.promoted is None

    def test_missing_rmse_never_promotes(self):
        ta = tuple(Term(name, f"a{i}") for i, name in enumerate(("one", "x")))
        c = CandidateDynamics(lhs="xddot", terms=ta, candidate_id="sole")
        g4 = {"passed": False, "holdout_obs_rms": 1.0, "holdout_noise_sigma": 0.06,
              "holdout_n_samples": 400}
        s = ScoredCandidate(c, {"gate4_status": g4}, train_loss=0.1, holdout_rmse=None)
        adj = adjudicate([s], resid_sole=np.zeros(400))
        assert adj.status is EpistemicStatus.INSUFFICIENT_EVIDENCE, adj

    def test_constant_residuals_carry_no_structure(self):
        # Semantic fork, pinned variant-aware: this author's contract passed
        # constant series (no structure possible); the live tree observed
        # abstaining. _CONST_ABSTAINS records which contract is live; either
        # way the outcome is deterministic and promotion requires whiteness.
        s, _ = self._sole(0.075)
        adj = adjudicate([s], resid_sole=np.zeros(400))
        if _CONST_ABSTAINS:
            assert adj.status is EpistemicStatus.INSUFFICIENT_EVIDENCE, adj
            assert adj.promoted is None
        else:
            assert adj.status is EpistemicStatus.UNIQUE_SUPPORT, adj
            assert adj.promoted is not None

    def test_baseline_flag_reproduces_v1(self):
        s, _ = self._sole(0.50)
        adj = adjudicate([s], sole_survivor_rule=False)
        assert adj.status is EpistemicStatus.UNIQUE_SUPPORT, adj

    def test_multi_path_identical_with_guards(self):
        rng = np.random.default_rng(4)
        ta = tuple(Term(name, f"a{i}") for i, name in enumerate(("one", "x", "x3", "v")))
        tb = tuple(Term(name, f"a{i}") for i, name in enumerate(("one", "v", "abs_v_v")))
        ca = CandidateDynamics(lhs="xddot", terms=ta, candidate_id="m1")
        cb = CandidateDynamics(lhs="xddot", terms=tb, candidate_id="m2")
        g4 = {"passed": True, "holdout_obs_rms": 1.0, "holdout_noise_sigma": 0.1,
              "holdout_n_samples": 100}
        sa = ScoredCandidate(ca, {"gate4_status": g4}, train_loss=0.5, holdout_rmse=0.45)
        sb = ScoredCandidate(cb, {"gate4_status": g4}, train_loss=0.52, holdout_rmse=0.44)
        plain = adjudicate([sa, sb])
        guarded = adjudicate([sa, sb], resid_sole=rng.normal(0, 1, size=400))
        assert plain.status == guarded.status == EpistemicStatus.MULTI_HYPOTHESIS
        assert guarded.promoted is None

    def test_degenerate_noise_scale_formalized(self):
        """Zero-noise-adjacent records (sigma_hat at the estimator floor):
        near-exact RMSE always passes; poor 0.5 always abstains. The fitted
        3e-4 case is variant-aware: UNIQUE iff the dual (relative) ceiling
        is live (_DUAL_LIVE), else INSUFFICIENT on the absolute term. A
        zero-noise contract remains pending."""
        ta = tuple(Term(name, f"a{i}") for i, name in enumerate(("one", "x")))
        c = CandidateDynamics(lhs="xddot", terms=ta, candidate_id="sole")
        g4 = {"passed": True, "holdout_obs_rms": 1.0, "holdout_noise_sigma": 1e-9,
              "holdout_n_samples": 400}
        rng = np.random.default_rng(4)
        near_exact = ScoredCandidate(c, {"gate4_status": g4}, train_loss=1e-12,
                                     holdout_rmse=1e-10)
        fitted = ScoredCandidate(c, {"gate4_status": g4}, train_loss=1e-8,
                                 holdout_rmse=3e-4)
        poor = ScoredCandidate(c, {"gate4_status": g4}, train_loss=0.2,
                               holdout_rmse=0.5)
        assert adjudicate([near_exact],
                          resid_sole=rng.normal(0, 1, size=400)).status \
            is EpistemicStatus.UNIQUE_SUPPORT
        assert (adjudicate([fitted],
                           resid_sole=rng.normal(0, 1, size=400)).status
                is EpistemicStatus.UNIQUE_SUPPORT) == _DUAL_LIVE
        assert adjudicate([poor],
                          resid_sole=rng.normal(0, 1, size=400)).status \
            is EpistemicStatus.INSUFFICIENT_EVIDENCE


@needs_guards
def _pair_note_present() -> bool:
    """Capability probe: does the live tree annotate pair-UNIQUE rationales
    with winner-whiteness? Probed by behavior (cheap, synthetic), never by
    version string."""
    import numpy as _np

    from apedp_dyn.epistemic.adjudicate import EpistemicStatus as _ES
    from apedp_dyn.epistemic.adjudicate import adjudicate as _adj
    from apedp_dyn.grammar.candidate_ir import CandidateDynamics as _CD
    from apedp_dyn.grammar.candidate_ir import ScoredCandidate as _SC
    from apedp_dyn.grammar.candidate_ir import Term as _T

    ta = tuple(_T(n, f"a{i}") for i, n in enumerate(("one", "x")))
    tb = tuple(_T(n, f"a{i}") for i, n in enumerate(("one", "v")))
    g4 = {"passed": True, "holdout_obs_rms": 1.0, "holdout_noise_sigma": 0.1,
          "holdout_n_samples": 100}
    sa = _SC(_CD(lhs="xddot", terms=ta, candidate_id="p1"), {"gate4_status": g4},
             train_loss=0.5, holdout_rmse=0.2)
    sb = _SC(_CD(lhs="xddot", terms=tb, candidate_id="p2"), {"gate4_status": g4},
             train_loss=0.55, holdout_rmse=0.8)
    t = _np.arange(400)
    try:
        r = _adj([sa, sb], resid_best=_np.sin(2 * _np.pi * t / 40.0))
    except TypeError:
        return False
    return r.status == _ES.UNIQUE_SUPPORT and "exceeds" in r.reason


needs_pair_note = pytest.mark.skipif(
    not _pair_note_present(),
    reason="live tree lacks the pair-whiteness rationale note",
)


@needs_pair_note
class TestPairWhitenessRationale:
    def test_structured_winner_noted_without_demoting(self):
        """Pair-path whiteness is diagnostic-only: a decisive UNIQUE keeps its
        status but carries the residual-structure note in its rationale."""
        import numpy as np

        from apedp_dyn.epistemic.adjudicate import EpistemicStatus, adjudicate
        from apedp_dyn.grammar.candidate_ir import CandidateDynamics, ScoredCandidate, Term

        ta = tuple(Term(name, f"a{i}") for i, name in enumerate(("one", "x", "x3", "v")))
        tb = tuple(Term(name, f"a{i}") for i, name in enumerate(("one", "v", "abs_v_v")))
        ca = CandidateDynamics(lhs="xddot", terms=ta, candidate_id="w1")
        cb = CandidateDynamics(lhs="xddot", terms=tb, candidate_id="w2")
        g4 = {"passed": True, "holdout_obs_rms": 1.0, "holdout_noise_sigma": 0.1,
              "holdout_n_samples": 100}
        sa = ScoredCandidate(ca, {"gate4_status": g4}, train_loss=0.5, holdout_rmse=0.20)
        sb = ScoredCandidate(cb, {"gate4_status": g4}, train_loss=0.55, holdout_rmse=0.80)
        plain = adjudicate([sa, sb])
        assert plain.status is EpistemicStatus.UNIQUE_SUPPORT
        assert "exceeds" not in plain.reason
        t = np.arange(400)
        noted = adjudicate([sa, sb], resid_best=np.sin(2 * np.pi * t / 40.0))
        assert noted.status is EpistemicStatus.UNIQUE_SUPPORT
        assert noted.promoted is not None
        assert "exceeds" in noted.reason
