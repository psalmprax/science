"""Regression tests for the two adjudication contracts that had no direct test.

These pin decisions made while restoring adjudicate.py, and are deliberately
NOT capability-probed. The neighbouring tests/test_dyn_epistemic_bootstrap.py
guards its classes with skipif markers so a tree lacking the feature skips
rather than fails; that is right for a spec that spans tree versions, but wrong
for a regression test. A skipped regression test protects nothing, so these run
unconditionally and fail loudly if either contract is undone.

Two contracts are pinned:

  1. Guard 1 is ABSOLUTE-only. The ceiling is gamma * sigma / CAL_SIGMA. A
     relative alternative (a ceiling on some fraction of the observation RMS)
     was in force briefly and withdrawn. Its failure mode is specific and worth
     naming: a relative ceiling is LOOSER exactly where the noise estimate is
     degenerate, so it waved through mid-quality fits on near-noiseless
     records that the absolute contract exists to refuse. sigma is a
     third-difference estimate; on a smooth record it measures curvature and
     sampling, not noise, and can sit orders of magnitude below any real fit's
     error. Guard 1's job is to refuse survivors that the evidence cannot
     support, and obs_rms must therefore have no say in it.

  2. Pair-path whiteness is DIAGNOSTIC, never enforcing. The residual-structure
     barrier answers "is an unopposed survivor explaining the record?". On the
     pair path a structurally distinct rival existed and was beaten on the same
     holdout, so the residual shape has already been arbitrated by the
     evidence. Demoting a decisive winner over structure the data itself
     resolved would make the guard a veto the holdout cannot contradict.
"""
from __future__ import annotations

import numpy as np
import pytest

from apedp_dyn.epistemic.adjudicate import (
    RESID_RHO_FLAG,
    SOLE_SURVIVOR_GAMMA,
    EpistemicStatus,
    adjudicate,
)
from apedp_dyn.epistemic.calibration import CAL_SIGMA, calibrated_sigma
from apedp_dyn.grammar.candidate_ir import CandidateDynamics, ScoredCandidate, Term


def _sole(rmse, sigma, obs_rms=1.0, resid=None):
    """A sole structural survivor: no rival survives scoring."""
    terms = tuple(Term(n, f"a{i}") for i, n in enumerate(("one", "x")))
    cand = CandidateDynamics(lhs="xddot", terms=terms, candidate_id="sole")
    g4 = {"passed": True, "holdout_obs_rms": obs_rms,
          "holdout_noise_sigma": sigma, "holdout_n_samples": 400}
    kw = {} if resid is None else {"resid_sole": resid}
    scored = ScoredCandidate(cand, {"gate4_status": g4}, train_loss=1e-8,
                             holdout_rmse=rmse)
    return adjudicate([scored], **kw)


def _pair(resid_best=None):
    """Two structurally distinct survivors, decisively separated (Δ=0.6, τ=0.05).

    Identical structures would be de-duplicated before reaching the pair path;
    these differ in term set, so the pair branch is genuinely exercised.
    """
    ta = tuple(Term(n, f"a{i}") for i, n in enumerate(("one", "x", "x3", "v")))
    tb = tuple(Term(n, f"a{i}") for i, n in enumerate(("one", "v", "abs_v_v")))
    g4 = {"passed": True, "holdout_obs_rms": 1.0, "holdout_noise_sigma": 0.1,
          "holdout_n_samples": 100}
    best = ScoredCandidate(
        CandidateDynamics(lhs="xddot", terms=ta, candidate_id="w1"),
        {"gate4_status": g4}, train_loss=0.5, holdout_rmse=0.20)
    rival = ScoredCandidate(
        CandidateDynamics(lhs="xddot", terms=tb, candidate_id="w2"),
        {"gate4_status": g4}, train_loss=0.55, holdout_rmse=0.80)
    kw = {} if resid_best is None else {"resid_best": resid_best}
    return adjudicate([best, rival], **kw)


def _white(seed=4, n=400):
    return np.random.default_rng(seed).normal(0, 1, size=n)


class TestGuard1AbsoluteOnly:
    """Guard 1's ceiling is gamma * (sigma / CAL_SIGMA) and nothing else."""

    def test_degenerate_noise_scale_uses_absolute_term(self):
        """At sigma_hat = 1e-9 the absolute ceiling is ~1.7e-9.

        A near-exact fit passes it; a 3e-4 fit and a 0.5 fit do not. This is the
        row that a relative ceiling (1% of obs_rms = 1e-2 here) would flip:
        under that rule the 3e-4 fit passes, which is the regression.
        """
        ceiling = SOLE_SURVIVOR_GAMMA * calibrated_sigma(1e-9)
        assert ceiling < 1e-8, ceiling
        assert _sole(1e-10, 1e-9, resid=_white()).status \
            is EpistemicStatus.UNIQUE_SUPPORT
        assert _sole(3e-4, 1e-9, resid=_white()).status \
            is EpistemicStatus.INSUFFICIENT_EVIDENCE
        assert _sole(0.5, 1e-9, resid=_white()).status \
            is EpistemicStatus.INSUFFICIENT_EVIDENCE

    @pytest.mark.parametrize("obs_rms", [1.0, 1e3, 1e6])
    def test_observation_scale_cannot_loosen_the_ceiling(self, obs_rms):
        """The sharpest guard on the withdrawn relative ceiling.

        A relative term scales with obs_rms, so inflating the observation scale
        would raise the bar's ceiling and admit the same bad fit. The absolute
        contract must be blind to obs_rms: at any observation scale, a fit 3e8
        times the noise floor does not earn promotion.
        """
        adj = _sole(0.5, 1e-9, obs_rms=obs_rms, resid=_white())
        assert adj.status is EpistemicStatus.INSUFFICIENT_EVIDENCE
        assert adj.promoted is None

    def test_no_relative_ceiling_constant_is_exported(self):
        """SOLE_SURVIVOR_REL was withdrawn with the dual-ceiling contract.

        Asserting its absence keeps the decision from being quietly reversed by
        re-adding the constant and leaving it unused-but-present.
        """
        import apedp_dyn.epistemic.adjudicate as adj_mod

        assert not hasattr(adj_mod, "SOLE_SURVIVOR_REL")

    def test_ceiling_tracks_sigma(self):
        """The ceiling is a function of the noise scale, so it must move with it.

        Guards against the ceiling accidentally becoming a fixed constant or
        an obs_rms-derived quantity: at fixed rmse, more noise buys more room.
        """
        assert _sole(0.5, 0.06, resid=_white()).status \
            is EpistemicStatus.INSUFFICIENT_EVIDENCE
        assert _sole(0.5, 0.6, resid=_white()).status \
            is EpistemicStatus.UNIQUE_SUPPORT

    def test_ceiling_tracks_gamma(self):
        """sole_gamma is the sole knob on the ceiling; it must actually bind.

        Same candidate, same evidence, same residuals — only gamma moves. If
        gamma were ignored the two calls would agree and this test would pass
        for the wrong reason, so the strict/refused pair is asserted too.
        """
        strict = adjudicate([_sole_candidate(0.5, 0.06)], resid_sole=_white(),
                            sole_gamma=1.0)
        assert strict.status is EpistemicStatus.INSUFFICIENT_EVIDENCE
        loose = adjudicate([_sole_candidate(0.5, 0.06)], resid_sole=_white(),
                           sole_gamma=100.0)
        assert loose.status is EpistemicStatus.UNIQUE_SUPPORT
        # Default must equal the frozen constant, not some private default.
        default = _sole(0.5, 0.06, resid=_white())
        assert default.status is adjudicate(
            [_sole_candidate(0.5, 0.06)], resid_sole=_white(),
            sole_gamma=SOLE_SURVIVOR_GAMMA).status
        assert CAL_SIGMA > 0.0


def _sole_candidate(rmse, sigma):
    terms = tuple(Term(n, f"a{i}") for i, n in enumerate(("one", "x")))
    cand = CandidateDynamics(lhs="xddot", terms=terms, candidate_id="sole")
    g4 = {"passed": True, "holdout_obs_rms": 1.0,
          "holdout_noise_sigma": sigma, "holdout_n_samples": 400}
    return ScoredCandidate(cand, {"gate4_status": g4}, train_loss=1e-8,
                           holdout_rmse=rmse)


class TestPairPathWhitenessIsDiagnostic:
    """Structure is reported on a promoted winner; it never demotes one."""

    def test_structured_winner_noted_without_demoting(self):
        adj = _pair(resid_best=np.sin(2 * np.pi * np.arange(400) / 40.0))
        assert adj.status is EpistemicStatus.UNIQUE_SUPPORT
        assert adj.promoted is not None
        assert "exceeds" in adj.reason
        # The note must not be mistaken for the abstention rationale.
        assert "not explanatory support" not in adj.reason

    def test_white_winner_carries_no_note(self):
        """A note on every verdict would train readers to ignore it."""
        adj = _pair(resid_best=_white())
        assert adj.status is EpistemicStatus.UNIQUE_SUPPORT
        assert "exceeds" not in adj.reason

    def test_absent_series_carries_no_note(self):
        """No evidence is not a finding; it must not manufacture one."""
        adj = _pair()
        assert adj.status is EpistemicStatus.UNIQUE_SUPPORT
        assert "exceeds" not in adj.reason

    @pytest.mark.parametrize("series", [
        np.zeros(400),                                          # constant
        np.array([1.0, -1.0, 1.0]),                             # too short
        np.where(np.arange(400) % 50 == 0, np.nan, 1.0),        # non-finite
    ], ids=["constant", "too-short", "non-finite"])
    def test_unassessable_series_is_silent(self, series):
        """Unassessable is not structured.

        These are the same inputs the sole-path Guard 2 refuses to judge. On
        the pair path they must stay silent rather than claim a finding, and
        must certainly not demote.
        """
        adj = _pair(resid_best=series)
        assert adj.status is EpistemicStatus.UNIQUE_SUPPORT
        assert adj.promoted is not None
        assert "exceeds" not in adj.reason

    def test_sole_path_still_enforces_where_pair_path_only_notes(self):
        """The load-bearing contrast: same structured series, opposite outcomes.

        The pair note is justified by the pair path being a different question.
        If the sole path ever stopped refusing structured residuals too, the
        note would no longer be a scoped exception but a second, quieter guard
        — and accuracy-without-whiteness would promote again.
        """
        structured = np.sin(2 * np.pi * np.arange(400) / 40.0)

        sole_adj = _sole(0.075, 0.06, resid=structured)
        assert sole_adj.status is EpistemicStatus.INSUFFICIENT_EVIDENCE
        assert sole_adj.promoted is None
        assert "rho1" in sole_adj.reason

        pair_adj = _pair(resid_best=structured)
        assert pair_adj.status is EpistemicStatus.UNIQUE_SUPPORT
        assert "exceeds" in pair_adj.reason

    def test_flag_is_the_documented_constant(self):
        """The note quotes RESID_RHO_FLAG; the flag must be the pinned 0.5."""
        assert RESID_RHO_FLAG == 0.5

    def test_flag_survives_squaring_a_white_residual_series(self):
        """The fallback's series is SQUARED error, so the flag must still
        separate structure from a correct model after squaring.

        This was worth measuring rather than assuming. Squaring a white
        residual does not leave it white: e^2 is non-negative and skewed, so
        its lag-1 correlation is a different quantity from e's. If its null
        value sat at or above 0.5, every correct model would abstain roughly
        half the time and the fallback would be a coin flip. Measured over 40
        seeds, rho1(e^2) is ~0.01, far below the flag, and never once reached
        it -- so the flag transfers and the abstention stays meaningful.
        """
        vals = []
        for seed in range(40):
            e = np.random.default_rng(seed).normal(0, 1, size=400)
            sq = e ** 2
            c = sq - sq.mean()
            denom = float(np.dot(c, c))
            vals.append(float(np.dot(c[:-1], c[1:]) / denom))
        vals = np.asarray(vals)
        assert vals.mean() < 0.1, vals.mean()
        assert (np.abs(vals) >= RESID_RHO_FLAG).sum() == 0
        adj = _pair(resid_best=np.sin(2 * np.pi * np.arange(400) / 40.0))
        assert f"{RESID_RHO_FLAG:g} whiteness flag" in adj.reason


class TestProvenanceFallbackFiresEndToEnd:
    """Guard 2's provenance fallback, on a real fitted candidate.

    The fallback is what makes the sole-survivor contract reachable outside
    dyn_sweep: callers that forward their fits and pass no explicit series get
    the one gate 4 already recorded. Every other test in this file feeds
    hand-built candidates, which proves the key is read but not that a real
    fitter-produced record drives it. This closes that gap.
    """

    @staticmethod
    def _noisy_duffing(pct=5.0, seed=11):
        """Duffing at 5% noise.

        Noise level is load-bearing, not decoration. On the clean record the
        third-difference sigma sits ~2e-5 while the true model still fits to
        ~6e-4, so Guard 1 refuses and Guard 2 is never reached -- the two
        guards are ordered, so Guard 1's verdict hides Guard 2's. 5% noise
        lifts sigma (0.042) above a third of the fit error, so Guard 1 passes
        and the fallback becomes observable.
        """
        import dataclasses

        from apedp_dyn.problems import duffing

        rng = np.random.default_rng(seed)
        prob = duffing.load()
        train, holdout = dict(prob.train), dict(prob.holdout)
        for reg in (train, holdout):
            y = np.asarray(reg["truth_y"], dtype=float)
            reg["truth_y"] = y + rng.normal(0.0, pct / 100.0 * np.std(y, axis=0),
                                           y.shape)
        return dataclasses.replace(prob, train=train, holdout=holdout)

    @classmethod
    def _best_fit(cls):
        from apedp_dyn.fitting.parameter_fit import fit_candidate
        from apedp_dyn.grammar import enumerate_candidates, prune_admissible

        prob = cls._noisy_duffing()
        kept = prune_admissible(prob, enumerate_candidates(
            required=("u",), optional=("x", "x3", "v"), max_terms=4))
        scored = [fit_candidate(prob, k.candidate) for k in kept]
        scored = [s for s in scored
                  if s.train_loss is not None and np.isfinite(s.train_loss)]
        return min(scored, key=lambda s: s.train_loss)

    def test_gate4_records_the_series_in_provenance(self):
        """Precondition: the series exists, as a JSON-safe list."""
        import json

        best = self._best_fit()
        se = best.provenance["gate4_status"]["holdout_sq_error"]
        assert len(se) == 400
        # Must survive the artifact write path; a raw ndarray here silently
        # breaks every provenance dump.
        json.dumps(best.provenance)

    def test_guard1_passes_so_guard2_is_reachable(self):
        """Ordering precondition: the fidelity ceiling must admit this fit.

        Asserted separately because it is the easy thing to break silently --
        tighten the ceiling and this whole class degenerates into testing
        Guard 1 while appearing to test the fallback.
        """
        from apedp_dyn.epistemic.adjudicate import _sole_survivor_fidelity
        from apedp_dyn.epistemic.calibration import calibrated_sigma

        best = self._best_fit()
        g4 = best.provenance["gate4_status"]
        scale = {"obs_rms": g4["holdout_obs_rms"],
                 "noise_sigma": g4["holdout_noise_sigma"]}
        ok, why = _sole_survivor_fidelity(best, scale, SOLE_SURVIVOR_GAMMA)
        assert ok, why
        # And the ceiling is genuinely tight, not vacuously satisfied.
        assert best.holdout_rmse < SOLE_SURVIVOR_GAMMA * calibrated_sigma(
            g4["holdout_noise_sigma"])

    def test_fallback_abstains_without_any_explicit_series(self):
        """The end-to-end claim: no resid_sole, real fit, guard-2 abstention.

        adjudicate() is called with a single candidate and no residual series
        at all. Any rho1 in the rationale can therefore only have come from
        gate 4's recorded series, read through provenance.
        """
        best = self._best_fit()
        adj = adjudicate([best])
        assert adj.status is EpistemicStatus.INSUFFICIENT_EVIDENCE
        assert adj.promoted is None
        # Guard 1 passed, then Guard 2 refused -- both stated in the reason.
        assert "absolute ceiling" in adj.reason
        assert "rho1" in adj.reason

    def test_fallback_reaches_the_same_verdict_as_an_explicit_series(self):
        """Provenance and an explicit series must not disagree.

        Two routes to the same evidence, one verdict. A divergence would mean
        the fallback is judging something other than what callers pass
        explicitly -- the exact inconsistency that made the guard unreachable
        outside one script in the first place.
        """
        best = self._best_fit()
        series = best.provenance["gate4_status"]["holdout_sq_error"]

        via_provenance = adjudicate([best])
        via_argument = adjudicate([best], resid_sole=series)
        assert via_provenance.status is via_argument.status
        assert via_provenance.promoted is None
        assert via_argument.promoted is None
