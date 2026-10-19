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

import math
import pathlib
import typing

import numpy as np
import pandas as pd
import pytest
import torch

from apedp_dyn.cuda.batch_physics_steppers import batch_multi_agent_cbf_step
from apedp_dyn.epistemic.adjudicate import EpistemicStatus, adjudicate
from apedp_dyn.epistemic.trial_ledger import (
    CandidateTrialLedger,
    compute_deflated_sharpe_ratio,
)
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


# ── External-review round: F14 τ-rule, selection-reuse disclosure, F4/F5/F16 ──

class TestF14TauRuleIsASeparationTest:
    """External-review case: holdout RMSE 0.1000 vs 0.1001 (Δ=10^-4).

    A nonnegative separation is guaranteed by construction (the rival is the
    holdout minimum), so uniqueness must come from the PRE-SPECIFIED τ rule,
    not from the sign. Pinned both ways: Δ ≤ τ -> refusal; a genuine margin
    (τ raised above Δ by the caller would be odd, so instead compare the
    τ-separation direction in the other review case which clears τ) -- and
    the same fixture with τ shrunk below Δ still resolves via the τ rule.
    """

    def test_delta_at_tau_boundary_refuses_unique(self):
        a = _scored("c_tie_a", 0.50, 0.1000, ("x",))
        b = _scored("c_tie_b", 0.61, 0.1001, ("x", "v"))
        adj = adjudicate([a, b], sole_survivor_rule=True,
                         data_origin="simulation", tau_epi=1e-4)
        assert adj.status is EpistemicStatus.MULTI_HYPOTHESIS
        assert adj.delta_rmse == pytest.approx(1e-4)
        assert adj.delta_rmse <= adj.tau_epi
        assert adj.promoted is None
        assert "abstains from unique promotion" in adj.reason

    def test_tau_shrunk_below_delta_resolves_unique(self):
        """Same 10^-4 separation, but τ stepped an order-of-magnitude lower:
        the referee's verdict follows the PRE-SPECIFIED τ, not the sign."""
        a = _scored("c_tie_a", 0.50, 0.1000, ("x",))
        b = _scored("c_tie_b", 0.61, 0.1001, ("x", "v"))
        adj = adjudicate([a, b], sole_survivor_rule=True,
                         data_origin="simulation", tau_epi=1e-6)
        assert adj.status is EpistemicStatus.UNIQUE_SUPPORT, adj.reason
        assert adj.delta_rmse == pytest.approx(1e-4)
        assert adj.delta_rmse > adj.tau_epi


class TestF14SelectionReuseDisclosure:
    """Every pair-path UNIQUE_SUPPORT discloses that the winner was SELECTED on
    the same holdout slice that certifies it, with the pool size, so callers
    must charge it to the DSR trial ledger (or hold a sealed confirmation set).
    """

    def test_unique_verdict_carries_selection_reuse_flag(self):
        overfitted = _scored("c_overfitted", 0.01, 2.00, ("x",))
        generalizing = _scored("c_generalizing", 0.90, 0.07, ("x", "v"))
        extra = _scored("c_third", 0.5, 0.5, ("w", "v", "z"))
        adj = adjudicate([overfitted, generalizing, extra],
                         sole_survivor_rule=True, data_origin="simulation")
        assert adj.status is EpistemicStatus.UNIQUE_SUPPORT
        flags = tuple(adj.diagnostic_flags)
        assert any(
            f.startswith("selection_reused_confirmation_holdout(n_candidates=")
            for f in flags
        ), flags
        pool = next(
            f for f in flags if f.startswith("selection_reused_confirmation_holdout"))
        # 3 holdout-scorable candidates were compared on this slice.
        assert pool == "selection_reused_confirmation_holdout(n_candidates=3)"


class TestF4TrialLedgerChargesSeedsAndHyperparameters:
    """The multiplicity correction must count every recorded trial, including
    repeated evaluations of the same structure under different seeds or
    hyperparameters: charging per *structural* candidate would hide them."""

    def test_seed_and_hyperparameter_repeats_are_charged(self):
        ledger = CandidateTrialLedger()
        for seed in (1, 2, 3):
            for lr in (1e-2, 1e-3):
                ledger.record_trial(
                    candidate_id=f"h_{seed}_{lr}",
                    proposer_id="p",
                    formula="y = f(x)",
                    strategy_archetype="tactical",
                    in_sample_sharpe=0.4 + 0.01 * seed,
                    n_bars=200,
                    metadata={"search_id": "sweep_run_1", "seed": seed, "lr": lr},
                )
        assert ledger.trial_count == 6
        assert ledger.effective_trial_count == 6  # shared search_id, all distinct trials charged

    def test_hidden_declared_pool_charged_beyond_records(self):
        """A proposer that searched 40 seeds, recorded 2 survivors: charged 40."""
        ledger = CandidateTrialLedger()
        for i in (0, 1):
            ledger.record_trial(
                candidate_id=f"c_{i}",
                proposer_id="p",
                formula="y = f(x)",
                strategy_archetype="tactical",
                in_sample_sharpe=0.3,
                n_bars=200,
                metadata={"search_id": "sweep_run_2", "selection_pool_size": 40},
            )
        assert ledger.trial_count == 2
        assert ledger.effective_trial_count == 40
        # Same variance for both comparisons: inflation from the hidden pool
        # is the only difference between sel and naive here.
        ledger_var = ledger.get_sharpe_distribution()[1]
        sel = ledger.dsr_with_selection_adjustment(
            observed_sharpe=2.5, n_bars=252, annualization_factor=252.0)
        naive = compute_deflated_sharpe_ratio(
            observed_sharpe=2.5, n_trials=2, var_trials=ledger_var, n_bars=252,
            annualization_factor=252.0)
        assert sel["n_trials"] == 40
        assert sel["hidden_trials"] == 38
        assert sel["dsr"] < naive["dsr"]  # hidden multiplicity lowers the verdict


class TestF5DSRAnnualizationNumericConsistency:
    """Numerical pin: feeding an ANNUALIZED Sharpe with annualization_factor=1.0
    deflates the standard error by ~10x and manufactures significance
    (p = 0); the F5-corrected calls pass annualization_factor=252 and the
    verdict flips to non-significant for this fixture."""

    _COMMON: typing.ClassVar[dict] = {"observed_sharpe": 2.5, "n_trials": 500,
                                      "var_trials": 0.15, "n_bars": 252,
                                      "skewness": 0.1, "kurtosis": 3.5}

    def test_unannualized_path_fabricates_significance(self):
        naive = compute_deflated_sharpe_ratio(**self._COMMON, annualization_factor=1.0)
        assert naive["is_significant"] is True
        assert naive["p_value"] < 1e-6

    def test_annualized_path_is_calibrated(self):
        fixed = compute_deflated_sharpe_ratio(**self._COMMON, annualization_factor=252.0)
        assert fixed["is_significant"] is False
        assert 0.05 < fixed["p_value"] < 0.15
        # The SE is on the annualized scale (O(1) for this fixture), not the
        # ~0.14 of the wrongly de-annualized path.
        assert fixed["se_sharpe"] > 0.5


class TestF16CBFSafetyEnvelope:
    """Safety-envelope battery for the batched repulsion stepper (review round).

    The artificial-potential deflection is NOT a certified CBF-QP (the stepper
    docstring says so); these pins establish the behavioral envelope an CBF
    audit would start from: finiteness, a measured inter-agent separation
    floor, exact index symmetry, dt-robustness, progress through a collinear
    head-on encounter, and degenerate coincident starts. A formal
    ``dot(h) + alpha(h) >= 0`` certificate remains future work and the
    docstring disclosure must stay honest.
    """

    _ATOL = 1e-5

    def _run(self, pose_grid, dt=0.1, steps=60, r_safe=0.5, r_warning=1.0, targets=None, roles=None):
        poses = torch.tensor([pose_grid], dtype=torch.float32)
        n = poses.shape[1]
        if targets is None:
            targets = torch.zeros(1, n, 2)
        if roles is None:
            roles = torch.ones(1, n)
        min_sep = float("inf")
        for _ in range(steps):
            poses = batch_multi_agent_cbf_step(
                poses, targets, roles, dt, r_safe=r_safe, r_warning=r_warning)
            assert torch.isfinite(poses).all(), "non-finite dynamics"
            p = poses[:, :, :2]
            d = (p.unsqueeze(2) - p.unsqueeze(1)).norm(dim=-1)
            eye = torch.eye(n, dtype=torch.bool).unsqueeze(0)
            if n > 1:
                min_sep = min(min_sep, float(d[0][~eye[0]].min()))
        return poses[0], min_sep

    def test_documented_uncertified_disclosure(self):
        from apedp_dyn.cuda.batch_physics_steppers import batch_multi_agent_cbf_step as fn
        assert "not a certified quadratic-programming" in fn.__doc__

    def test_head_on_collinear_encounter_does_not_collapse_separation(self):
        poses, min_sep = self._run(
            [[1.0, 0.0, math.pi], [-1.0, 0.0, 0.0]])
        # Near-collision is deflected and never violates the safety floor.
        assert min_sep > 0.6  # floor 0.696 measured; r_safe=0.5
        final = float((poses[0, :2] - poses[1, :2]).norm())
        assert final > 0.6

    def test_high_relative_velocity_head_on(self):
        _, min_sep = self._run(
            [[2.5, 0.0, math.pi], [-2.5, 0.0, 0.0]])
        assert min_sep > 0.6

    @pytest.mark.parametrize("dt", [0.05, 0.1, 0.2])
    def test_separation_floor_robust_to_time_step(self, dt):
        _, min_sep = self._run(
            [[2.5, 0.0, math.pi], [-2.5, 0.0, 0.0]], dt=dt, steps=int(8 / dt))
        assert min_sep > 0.6

    def test_agent_index_symmetry(self):
        poses, _ = self._run([[1.0, 0.0, math.pi], [-1.0, 0.0, 0.0]], steps=1)
        poses_sw, _ = self._run([[-1.0, 0.0, 0.0], [1.0, 0.0, math.pi]], steps=1)
        np.testing.assert_allclose(
            poses.cpu().numpy(), poses_sw[[1, 0]].cpu().numpy(), atol=self._ATOL)

    def test_degenerate_coincident_starts_stay_finite(self):
        poses, min_sep = self._run(
            [[0.0, 0.0, 0.0], [1e-6, 0.0, 0.0]], steps=40)
        assert torch.isfinite(poses).all()
        assert min_sep >= 0.0

    def test_simultaneous_multi_pair_engagement(self):
        _, min_sep = self._run(
            [[0.6, 0.6, 3.9], [-0.6, 0.6, 2.4],
             [-0.6, -0.6, 0.8], [0.6, -0.6, -0.8]], steps=120)
        assert min_sep > 0.6  # four pairwise engagements, floor 0.702 measured

    def test_head_on_breaks_static_deadlock(self):
        """The F16 regression class itself: with heading-opposing repulsion the
        agents make forward progress into the encounter zone (pre-fix
        radial-only deflection left the symmetric head-on pair motionless)."""
        poses0 = torch.tensor([[[1.0, 0.0, math.pi], [-1.0, 0.0, 0.0]]], dtype=torch.float32)
        targets = torch.zeros(1, 2, 2)
        roles = torch.ones(1, 2)
        poses = poses0.clone()
        for _ in range(20):
            poses = batch_multi_agent_cbf_step(poses, targets, roles, 0.1)
        advance = float(poses0[0, 0, 0] - poses[0, 0, 0])
        assert advance > 0.4  # dead-stamp would keep x ≈ 1.0; measured 0.55+


class TestF17ScaleAwareExactGate:
    """F17 reviewer ask: the exact-recovery RMSE gate must be scale-aware.

    The Rust RIME implicit kernel measures exactness RELATIVE to the RMS
    magnitude of the relation's own terms, so O(1e5)-scale exact relations are
    certifiable while small-scale near-misses are refused. Verified through
    the rebuilt pyo3 binding (the same interface downstream consumers call).
    """

    def _discover_json(self, x: np.ndarray, y: np.ndarray):
        import json as _json

        import apedp_kernel as _k
        cfg = _json.dumps({
            "max_degree": 2, "include_cross_terms": False,
            "include_rational_products": False,
            "sparsity_threshold": 1e-3, "rcond_threshold": 1e-8,
        })
        raw = _k.rime_implicit_discover_json(
            _json.dumps(list(map(float, x))), _json.dumps(list(map(float, y))), cfg)
        return _json.loads(raw) if isinstance(raw, str) else raw

    def test_o1_scale_exact_relation_certified(self):
        t = np.linspace(0.0, 2.0 * math.pi, 200)
        out = self._discover_json(np.cos(t), np.sin(t))
        assert out["is_exact_recovery"] is True
        assert out["num_active_terms"] >= 2

    def test_large_scale_exact_relation_certifiable(self):
        """A truly exact relation on O(1e5)-scale data: y = x + 10 with
        x ~ 1e5. The former absolute `rmse < 1e-5` gate could never certify
        this (rmse in those units is impossible to satisfy); the
        scale-aware gate measures rmse relative to the term magnitude."""
        x = np.random.default_rng(0).normal(size=200) * 1e5
        y = x + 10.0
        out = self._discover_json(x, y)
        assert out["formula"].count("=") == 1
        assert out["is_exact_recovery"] is True
        assert out["algebraic_rmse"] < 1e-9

    def test_small_scale_near_miss_refused(self):
        t = np.linspace(0.0, 2.0 * math.pi, 200)
        out = self._discover_json(np.cos(t), np.sin(t) + 1e-4)
        # rmse ~1.4e-4 is far above the scale-relative 1e-5 ceiling -> refused.
        assert out["is_exact_recovery"] is False

    def test_tautology_still_refused_at_any_scale(self):
        """An all-zero pruned vector ('0 = 0') stays rejected regardless of
        the data scale (the active_count guard is scale-independent)."""
        rng = np.random.default_rng(5)
        out = self._discover_json(rng.normal(size=120), rng.normal(size=120))
        assert out["num_active_terms"] >= 0
        if out["num_active_terms"] < 2:
            assert out["is_exact_recovery"] is False


class TestF8FrozenLibraryTermAudit:
    """F8 reviewer asks: (a) the frozen (cold-end) scoring library must cover
    every dev-discovered active term — a dropped column silently corrupts the
    frozen R^2, so unresolved terms now fail closed; (b) term names are
    classified correctly (u^3 is a cubic STATE term, not a third-order
    derivative)."""

    def test_unresolvable_active_term_fails_closed(self):
        from apedp_dyn.epistemic.pde_discovery import _frozen_residual_r2
        ut = np.linspace(0.0, 1.0, 50)
        cols = {"u": ut, "u_x": ut * 2.0}
        # A dev-invented term absent from the cold library must raise, not
        # silently drop the coefficient.
        with pytest.raises(ValueError, match="u_xxx"):
            _frozen_residual_r2(ut, cols, ["u", "u_xxx"], [1.0, 0.5])
        # Complete coverage still scores normally.
        r2 = _frozen_residual_r2(ut, cols, ["u", "u_x"], [0.0, 0.5])
        assert np.isfinite(r2)

    def test_attemptterm_classification_u3_is_state_not_derivative(self):
        """Term-name classification: only *_xxx / laplacian-family names are
        third-order; 'u^3' is a cubic state nonlinearity."""
        from apedp_dyn.epistemic.pde_discovery import _eval_frozen_1d_library
        rng = np.random.default_rng(9)
        u = np.cumsum(rng.normal(size=(12, 40)), axis=1)
        _, cols = _eval_frozen_1d_library(
            u, dt=0.1, dx=0.05, max_spatial_order=3,
            max_polynomial_degree=3, include_advection_terms=True)
        assert "u_xxx" in cols          # third-order SPATIAL derivative
        assert "u^3" in cols            # cubic state nonlinearity, unchanged
        # And the two are genuinely different series:
        assert not np.allclose(cols["u^3"], cols["u_xxx"])

    def test_mirror_covers_rust_1d_catalog(self):
        """Every term name the Rust 1D discovery library can emit resolves in
        the cold-end mirror built with the same options."""
        from apedp_dyn.epistemic.pde_discovery import _eval_frozen_1d_library
        rng = np.random.default_rng(9)
        u = np.cumsum(rng.normal(size=(12, 40)), axis=1)
        _, cols = _eval_frozen_1d_library(
            u, dt=0.1, dx=0.05, max_spatial_order=3,
            max_polynomial_degree=3, include_advection_terms=True)
        expected = {"u", "u^2", "u^3", "u_x", "u*u_x", "u^2*u_x",
                    "u_xx", "u*u_xx", "u_xxx"}
        missing = expected - set(cols)
        assert not missing, f"cold-end mirror drifted: {missing}"


_LEAN_FORMAL_DIR = pathlib.Path(__file__).resolve().parents[1] / "apedp_dyn" / "lean_formal"
if not _LEAN_FORMAL_DIR.is_dir():  # pragma: no cover - layout guard
    raise AssertionError("lean_formal directory missing from repo workspace")


class TestLeanProofChainIntegrity:
    """Reviewer ask (F12/Lean): compile the complete dependency chain and check
    for unintended axioms or unproved assumptions. Pin: every P2-verified
    record's emitted Lean file is free of `sorry`, and every file containing
    `sorry` belongs to a P0-refuted record — the sorry-channel cannot certify."
    """

    def _aggregate(self):
        import json as _json

        from apedp_dyn.sandbox.build_all_system_equations_lean_proofs import ARTIFACT_DIR
        path = ARTIFACT_DIR / "all_system_equations_lean_provenance.json"
        with open(path) as f:
            return _json.load(f)

    def test_p2_records_have_sorry_free_lean_sources(self):
        data = self._aggregate()
        checked = 0
        for rec in data["results"]:
            if rec["proof_claim_status"] != "P2":
                continue
            lean_basename = rec.get("lean_basename")
            assert lean_basename, rec["candidate_id"]
            lean_path = _LEAN_FORMAL_DIR / lean_basename
            text = lean_path.read_text()
            assert "sorry" not in text, (
                f"P2 record {rec['candidate_id']} emits {lean_basename} containing sorry")
            checked += 1
        assert checked > 300  # the 315-record P2 family is fully pinned

    def test_sorry_files_correspond_only_to_refuted_records(self):
        from apedp_dyn.sandbox.build_all_system_equations_lean_proofs import (
            ARTIFACT_DIR,
            LEAN_FORMAL_DIR,
        )
        data = self._aggregate()
        p2_ids = {r["candidate_id"] for r in data["results"]
                  if r["proof_claim_status"] == "P2"}
        for lean_path in LEAN_FORMAL_DIR.glob("*.lean"):
            if "sorry" not in lean_path.read_text():
                continue
            cid = lean_path.stem.removesuffix("_verification").upper()
            assert cid not in p2_ids, f"sorry-carrying file claimed as P2: {cid}"
        assert ARTIFACT_DIR is not None
