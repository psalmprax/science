"""Tests for APEDP Level 2 Financial Discovery & Epistemic Attribution Sandbox.

Verifies:
  1. L2 Factor Discovery on volatility-scaled momentum:
     L2 open discovery constructs (r_prev / vol) and promotes with UNIQUE_SUPPORT.
     L0 does NOT starve on this regime: see the note on
     test_volatility_scaled_momentum_recovers_interaction_term for what changed
     in bdc1595 and why a linear AR(1) is a legitimate survivor here.
  2. Immunity to Factor Zoo Overfitting:
     In pure random-walk regimes, epistemic guards prevent spurious factor promotion,
     so the verdict admits no position and exposure is 0%.
  3. Factor Collinearity as Persistent Ambiguity:
     Correlated technical indicators trigger MULTI_HYPOTHESIS under
     INTRINSIC_OBSERVATIONAL_NONIDENTIFIABILITY; the sizing rule then holds a
     corner rather than inventing a split between near-identical factors.
"""
from __future__ import annotations

import numpy as np
import pytest

from apedp_dyn.epistemic.adjudicate import (
    EpistemicAttribution,
    EpistemicStatus,
)
from apedp_dyn.sandbox.finance_l2_demo import (
    run_factor_collinearity_experiment,
    run_finance_l2_comparison,
    size_positions,
)


class TestFinanceL2Sandbox:
    """Rigorous verification of L2 open discovery and epistemic attribution in finance."""

    def test_volatility_scaled_momentum_recovers_interaction_term(self):
        """L2 discovers volatility-scaled momentum r_prev/vol with full sizing.

        L0 no longer starves on this regime, and the old expectation is not
        reachable through any of the guards. The demo hand-builds its gate-4
        record, and `_residual_series` used to look up `holdout_sq_error`,
        which that record does not populate -- so Guard 2 reported the winner's
        whiteness *unassessable* and refused the sole survivor for the wrong
        reason. bdc1595 made the lookup signed-only (`holdout_residuals`), the
        series is now found, and its lag-1 autocorrelation is 0.157 against the
        0.5 flag: white, so Guard 2 passes on the evidence. Guard 1 cannot
        refuse it either -- the record's noise scale is the fit's own residual
        scale, and `r_t = 0.015 * r_{t-1}/sigma_{t-1} + noise` is near-linear
        here because the rolling sigma is ~= noise_sigma. An AR(1) on this data
        is a legitimate survivor, so the starvation narrative is recorded as
        not holding for these parameters rather than forced back into place.
        """
        res = run_finance_l2_comparison("volatility_scaled_momentum", bootstrap=False, seed=42)

        # L0: the AR(1) is the sole scored survivor and passes both guards.
        assert res.l0_status == EpistemicStatus.UNIQUE_SUPPORT
        assert res.l0_attribution == EpistemicAttribution.DECISIVE_EVIDENTIAL_SUPPORT

        # L2 discovers the non-linear interaction term
        assert res.l2_status == EpistemicStatus.UNIQUE_SUPPORT
        assert res.l2_attribution == EpistemicAttribution.DECISIVE_EVIDENTIAL_SUPPORT
        assert "r_prev" in res.l2_promoted_expr and "vol" in res.l2_promoted_expr
        # The verdict admits sizing, but the holdout mean over this window is
        # negative, so the cost-aware rule declines the long. Pinned as measured.
        assert res.sizing.admitted is True
        assert res.sizing.gross_exposure == 0.0
        assert "no trade" in res.sizing.reason

    def test_random_walk_regime_immune_to_factor_zoo_overfitting(self):
        """In random-walk noise, L2 rejects overfitted factor candidates and stays in cash."""
        res = run_finance_l2_comparison("random_walk", bootstrap=False, seed=42)

        # Refuses promotion of spurious non-linear combinations
        assert res.l2_status != EpistemicStatus.UNIQUE_SUPPORT
        assert res.l2_promoted_expr == ""
        assert res.sizing.admitted is False
        assert res.sizing.gross_exposure == 0.0

    def test_factor_collinearity_triggers_persistent_ambiguity_and_sizes_down(self):
        """Collinear factors trigger MULTI_HYPOTHESIS and downsize risk allocation to 50%."""
        res = run_factor_collinearity_experiment(seed=42)

        assert res.l2_status == EpistemicStatus.MULTI_HYPOTHESIS
        assert res.l2_attribution == EpistemicAttribution.INTRINSIC_OBSERVATIONAL_NONIDENTIFIABILITY
        assert res.sizing.admitted is True
        assert max(res.sizing.weights) > 0.99  # a corner, not a 50/50 split
        assert res.sizing.gross_exposure <= 1.0 + 1e-9
        assert "collinearity" in res.epistemic_insight.lower()

    def test_size_positions_responds_to_covariance_and_fees(self):
        """The size is computed from forecast, covariance and fee, not a verdict."""
        forecast = [2e-3, 2e-3]
        covariance = [[1e-4, 0.0], [0.0, 4e-4]]

        free = size_positions(forecast, covariance, fee_bps=0.0)
        assert free.admitted is True
        assert free.cost == 0.0
        assert free.gross_exposure == pytest.approx(1.0)
        assert free.weights[0] > free.weights[1]  # covariance moves the split
        assert free.converged is True
        assert free.iterations > 0
        assert free.kkt_residual is not None and free.kkt_residual < 1e-8

        thin_edge = size_positions(forecast, covariance, fee_bps=50.0)
        assert thin_edge.turnover == 0.0  # 2e-3 forecast < 50 bps = 5e-3 fee
        assert thin_edge.cost == 0.0
        assert thin_edge.gross_exposure == 0.0

        costly = size_positions(forecast, covariance, fee_bps=5.0)
        assert costly.turnover == pytest.approx(1.0)
        assert costly.cost == pytest.approx(5e-4)
        assert costly.cap_binding is True

        with pytest.raises(ValueError, match="covariance must be symmetric"):
            size_positions(forecast, [[1e-4, 1e-5], [0.0, 4e-4]])

    def test_size_positions_converges_on_ill_conditioned_eight_asset_book(self):
        """The active-set polish resolves the n=8 cases the fixed step cannot."""
        # Seeds from scripts/audit_sizing_solver_convergence.py: these two exact
        # draws exhausted the 4,000-iteration budget before the polish existed.
        seeds = {1e4: 20261133, 1e12: 20261141}
        for kappa, seed in seeds.items():
            rng = np.random.default_rng(seed)
            basis, _ = np.linalg.qr(rng.normal(size=(8, 8)))
            covariance = basis @ np.diag(np.logspace(0.0, -np.log10(kappa), 8)) @ basis.T
            covariance = 0.5 * (covariance + covariance.T)
            forecast = rng.uniform(0.005, 0.02, size=8)

            decision = size_positions(forecast, covariance, fee_bps=5.0)

            assert decision.converged is True, kappa
            assert decision.kkt_residual is not None
            assert decision.kkt_residual <= 1e-10
            assert min(decision.weights) >= 0.0
            assert decision.gross_exposure <= 1.0 + 1e-9

    def test_previous_weight_dust_is_not_treated_as_a_held_position(self):
        """A 1e-18 previous weight must not fake a fee-driven KKT violation.

        Found on the real equity panel: a coordinate sold to zero in an earlier
        solve retained ~1e-18, and the sold-out KKT rule charged the full fee
        for re-entering it, so optimal books were reported unconverged.
        """
        decision = size_positions(
            [0.5, 0.5, 0.0],
            np.eye(3),
            fee_bps=5.0,
            previous_weights=[0.5, 0.5, 1e-18],
        )

        assert decision.converged is True
        assert decision.kkt_residual is not None
        assert decision.kkt_residual <= 1e-10
        assert decision.turnover == 0.0
        assert decision.cost == 0.0

    def test_unconstrained_size_reports_native_scalars(self):
        """A non-cap solve must not leak numpy scalars into the decision.

        The KKT check's early return skipped the float conversion, so a
        converged decision carried ``numpy.bool_`` and the audit record could
        not be serialized to JSON.
        """
        decision = size_positions([1e-3], [[1e-2]], fee_bps=0.0)

        assert decision.gross_exposure < 1.0
        assert isinstance(decision.kkt_residual, float)
        assert isinstance(decision.converged, bool)

    def test_bootstrap_instability_is_refused_even_under_multi_hypothesis(self):
        """A sign that flips under resampling is not a direction to size."""
        from apedp_dyn.sandbox.finance_l2_demo import sizing_from_adjudication

        decision = sizing_from_adjudication(
            EpistemicStatus.MULTI_HYPOTHESIS,
            EpistemicAttribution.BOOTSTRAP_INSTABILITY,
            [1e-2, 1e-2],
            [[1e-4, 0.0], [0.0, 1e-4]],
        )
        assert decision.admitted is False
        assert decision.gross_exposure == 0.0
        assert "admits no position" in decision.reason


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
