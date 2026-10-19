"""Regression pins for the blind-audit remediation round 2 (2026-10-19).

Pins, at the public interface level, the two semantic fixes round 2 introduced
(see CHANGELOG "Blind-audit remediation round 2"):

- Finding 14 (`apedp_dyn/epistemic/adjudicate.py`): the pair evaluation between
  the promotion head and its nearest rival is anchored on the gate-4-certified
  holdout-lowest-RMSE candidate, never on the train-loss ordering. The tests
  here FAIL against the pre-fix code:
    * with a train-loss winner whose holdout RMSE is far worse than a
      generalizing rival, the pre-fix anchor produced a NEGATIVE
      ``delta = rival_ho - best_ho``, so ``delta <= tau`` tripped for any
      tau > 0 and wrongly returned MULTI_HYPOTHESIS;
    * after the fix the verdict is UNIQUE_SUPPORT for the holdout head, with
      ``delta_rmse`` equal to the true holdout separation and ``tau_epi``
      honored.
  Both directions are pinned: the genuine-separation promotion AND the honest
  refusal of an actual holdout tie (delta <= tau -> MULTI_HYPOTHESIS, verified
  against the holdout delta, never the train delta).

- Finding 15 (`apedp_dyn/proposers/ncm_structural_proposer.py`): every
  archetype branch emits a real ``compute_score`` closure that reproduces the
  fit-time in-sample predictor on the fit frame and stays nonzero on unseen
  frames; only a missing-column fallback legitimately returns all zeros.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from apedp_dyn.epistemic.adjudicate import EpistemicStatus, adjudicate
from apedp_dyn.grammar.candidate_ir import CandidateDynamics, ScoredCandidate, Term
from apedp_dyn.proposers.ncm_structural_proposer import (
    MacroConceptArchetype,
    NCMStructuralProposer,
)

# ── F14: adjudicate() pair anchoring is holdout-based ────────────────────────

def _scored(cid: str, train_loss: float, holdout_rmse: float, terms: tuple[str, ...],
            gate4: bool = True) -> ScoredCandidate:
    fit = type(
        "F",
        (),
        {
            "param_values": {f"a{i}": 1.0 for i in range(len(terms))},
            "train_loss": train_loss,
            "nfev": 1,
            "success": True,
            "message": "ols",
        },
    )()
    c = CandidateDynamics(
        lhs="y",
        terms=tuple(Term(t, f"a{i}") for i, t in enumerate(terms)),
        candidate_id=cid,
    )
    return ScoredCandidate(
        candidate=c,
        provenance={"gate4_status": {
            "passed": gate4,
            "holdout_obs_rms": 0.5,
            "holdout_noise_sigma": 0.05,
            "holdout_n_samples": 400,
        }},
        fit=fit,
        train_loss=train_loss,
        holdout_rmse=holdout_rmse,
    )


class TestF14HoldoutAnchoredPair:
    def _assert_unique_holdout_head(self, adj) -> None:
        assert adj.status is EpistemicStatus.UNIQUE_SUPPORT, adj.reason
        # The PROMOTED candidate is the holdout winner, NOT the train winner.
        assert adj.promoted.candidate.candidate_id == "c_generalizing"
        # delta is the true holdout separation, nonnegative and above tau.
        assert adj.delta_rmse is not None
        assert adj.delta_rmse == pytest.approx(2.0 - 0.07)
        assert adj.delta_rmse > adj.tau_epi
        # The re-anchored head leads the ranking; the train winner remains.
        ranking_ids = [s.candidate.candidate_id for s in adj.ranking]
        assert ranking_ids[0] == "c_generalizing"
        assert "c_overfitted" in ranking_ids

    def test_overfitted_train_winner_cannot_refuse_generalizing_rival(self):
        """The exact F14 case: the train-loss winner is the overfit candidate.

        Pre-fix: delta = rival_ho - best_ho < 0 tripped ``delta <= tau`` and
        returned MULTI_HYPOTHESIS for any tau > 0 even with a huge holdout
        separation.
        """
        overfitted = _scored("c_overfitted", 0.01, 2.00, ("x",))
        generalizing = _scored("c_generalizing", 0.90, 0.07, ("x", "v"))
        adj = adjudicate([overfitted, generalizing],
                         sole_survivor_rule=True, data_origin="simulation")
        self._assert_unique_holdout_head(adj)

    def test_order_invariance_over_candidate_submission(self):
        """The fix must not depend on submission order (clients permute pools)."""
        for fitted in ([_scored("c_overfitted", 0.01, 2.00, ("x",)),
                        _scored("c_generalizing", 0.90, 0.07, ("x", "v"))],
                       [_scored("c_generalizing", 0.90, 0.07, ("x", "v")),
                        _scored("c_overfitted", 0.01, 2.00, ("x",))]):
            adj = adjudicate(fitted, sole_survivor_rule=True, data_origin="simulation")
            self._assert_unique_holdout_head(adj)

    def test_true_holdout_tie_still_refuses_unique(self):
        """Both directions: delta <= tau (now measured on holdout) -> MULTI."""
        a = _scored("c_a", 0.50, 0.07, ("x",))
        b = _scored("c_b", 0.61, 0.072, ("x", "v"))
        adj = adjudicate([a, b], sole_survivor_rule=True, data_origin="simulation")
        assert adj.status is EpistemicStatus.MULTI_HYPOTHESIS
        assert adj.delta_rmse == pytest.approx(0.002)
        assert adj.delta_rmse <= adj.tau_epi
        assert adj.promoted is None
        assert "observationally indistinguishable" in adj.reason
        assert "abstains from unique promotion" in adj.reason

    def test_reanchor_requires_gate4_certified_holdout_head(self):
        """A holdout winner without its own gate-4 certification cannot depose
        the train-loss anchor, and the pair cannot claim it either: the
        certified head stays in the evaluation and the verdict is a refusal
        (never UNIQUE_SUPPORT for the uncertified head)."""
        certified_overfitted = _scored("c_certified", 0.01, 2.00, ("x",))
        uncertified_generalizing = _scored("c_uncertified", 0.90, 0.07, ("x", "v"),
                                           gate4=False)
        adj = adjudicate([certified_overfitted, uncertified_generalizing],
                         sole_survivor_rule=True, data_origin="simulation")
        assert adj.status is EpistemicStatus.MULTI_HYPOTHESIS
        assert adj.promoted is None
        # The refusal must be pair-consistent: the reported delta is the
        # anchored-pair separation, never the contradictory global recomputed
        # delta_ho the pre-fix code printed ("delta=1.93 <= tau=0.025").
        assert adj.delta_rmse == pytest.approx(0.07 - 2.00)
        # The contradiction the pre-fix code printed (a huge recomputed
        # delta_ho quoted next to a refusal) must not reappear: the anchored
        # pair separation (−1.93 here) is what is reported.
        assert f"Δ={0.07 - 2.00:.3g}" in adj.reason
        assert "abstains from unique promotion" in adj.reason

    def test_reanchored_pass_skips_bootstrap_confirmation(self):
        """A re-anchored pass falls back to the point-estimate contract instead
        of asserting a bootstrap confirmation it cannot attribute to the
        re-anchored head."""
        overfitted = _scored("c_overfitted", 0.01, 2.00, ("x",))
        generalizing = _scored("c_generalizing", 0.90, 0.07, ("x", "v"))
        adj = adjudicate(
            [overfitted, generalizing],
            sole_survivor_rule=True,
            data_origin="simulation",
            bootstrap=True,
            bootstrap_n_boot=200,
            bootstrap_seed=7,
        )
        assert adj.status is EpistemicStatus.UNIQUE_SUPPORT, adj.reason
        assert getattr(adj, "bootstrap_branch", "") == ""


# ── F15: NCMStructuralProposer compute_score contract ───────────────────────

def _fit_frame(n: int = 140, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    x1 = np.sin(t / 6) + 0.05 * rng.normal(size=n)
    x2 = np.cos(t / 9) + 0.06 * rng.normal(size=n)
    y = 0.5 * x1 + 0.3 * x2 + 0.02 * rng.normal(size=n)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


def _archetype_pool(df: pd.DataFrame) -> dict[str, object]:
    proposer = NCMStructuralProposer()
    proposals = proposer.fit_and_propose(df, feature_cols=["x1", "x2"],
                                         target_col="y")
    return {p.metadata["archetype"]: p for p in proposals}


class TestF15NcmComputeScore:
    @pytest.mark.parametrize("arch", [a.value for a in MacroConceptArchetype])
    def test_every_archetype_emits_nonzero_forward_score(self, arch):
        """No NCM proposal may fall back to the all-zero default closure."""
        df = _fit_frame()
        pool = _archetype_pool(df)
        assert arch in pool, f"{arch} proposal absent"
        p = pool[arch]
        scores = p.compute_score(df)
        assert isinstance(scores, np.ndarray) and scores.shape == (len(df),)
        assert np.all(np.isfinite(scores))
        assert np.any(np.abs(scores) > 1e-12), arch

    def test_scores_are_honest_forward_functions_on_unseen_frames(self):
        """The score must be derived from the frame it receives, not replay a
        frozen fit-time array: a disjoint unseen slice also gets predictions."""
        df = _fit_frame()
        unseen = df.iloc[60:100].reset_index(drop=True)
        pool = _archetype_pool(df)
        for arch, p in pool.items():
            s_unseen = p.compute_score(unseen)
            assert s_unseen.shape == (len(unseen),)
            assert np.any(np.abs(s_unseen) > 1e-12), arch
            # Not a replay of the fit-frame vector (different length alone
            # already proves recomputation; also check values change).
            s_fit = p.compute_score(df)
            assert not np.array_equal(np.asarray(s_unseen), np.asarray(s_fit))

    def test_in_sample_score_is_deterministic_across_identical_frames(self):
        """The closure recomputes from the frame: two numerically identical
        frames yield identical scores (fit-time in-sample prediction is the
        same number, not a cached vector keyed by identity)."""
        df = _fit_frame()
        pool = _archetype_pool(df)
        df_copy = df.copy()
        for _arch, p in pool.items():
            np.testing.assert_allclose(p.compute_score(df), p.compute_score(df_copy))

    def test_missing_columns_fail_closed_to_zero_fallback(self):
        """The only legitimate zeros: the frame lacks the declared inputs."""
        df = _fit_frame()
        pool = _archetype_pool(df)
        empty = pd.DataFrame({"unrelated": np.arange(90.0)})
        for arch, p in pool.items():
            s = np.asarray(p.compute_score(empty))
            assert np.all(s == 0.0), arch

    def test_full_archetype_pool_emitted(self):
        """Sanity: 4/4 archetypes produce proposals for this fixture."""
        df = _fit_frame()
        pool = _archetype_pool(df)
        assert len(pool) == 4
        assert set(pool) == {a.value for a in MacroConceptArchetype}
