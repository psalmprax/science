"""Tests for APEDP Level 2 Financial Discovery & Epistemic Attribution Sandbox.

Verifies:
  1. L0 Model Starvation vs L2 Factor Discovery:
     L0 linear AR(1) fails on volatility-scaled momentum with HYPOTHESIS_SPACE_LIMITED.
     L2 open discovery constructs (r_prev / vol) and promotes with UNIQUE_SUPPORT.
  2. Immunity to Factor Zoo Overfitting:
     In pure random-walk regimes, epistemic guards prevent spurious factor promotion,
     allocating 0.0% capital.
  3. Factor Collinearity as Persistent Ambiguity:
     Correlated technical indicators trigger MULTI_HYPOTHESIS under
     INTRINSIC_OBSERVATIONAL_NONIDENTIFIABILITY and downsize position to 50%.
"""
from __future__ import annotations

import pytest

from apedp_dyn.epistemic.adjudicate import (
    EpistemicAttribution,
    EpistemicStatus,
)
from apedp_dyn.sandbox.finance_l2_demo import (
    epistemic_position_sizing,
    run_factor_collinearity_experiment,
    run_finance_l2_comparison,
)


class TestFinanceL2Sandbox:
    """Rigorous verification of L2 open discovery and epistemic attribution in finance."""

    def test_volatility_scaled_momentum_recovers_interaction_term(self):
        """L0 starves; L2 discovers volatility-scaled momentum r_prev/vol with full sizing."""
        res = run_finance_l2_comparison("volatility_scaled_momentum", bootstrap=False, seed=42)

        # L0 suffers hypothesis-space starvation
        assert res.l0_status == EpistemicStatus.INSUFFICIENT_EVIDENCE
        assert res.l0_attribution == EpistemicAttribution.HYPOTHESIS_SPACE_LIMITED

        # L2 discovers the non-linear interaction term
        assert res.l2_status == EpistemicStatus.UNIQUE_SUPPORT
        assert res.l2_attribution == EpistemicAttribution.DECISIVE_EVIDENTIAL_SUPPORT
        assert "r_prev" in res.l2_promoted_expr and "vol" in res.l2_promoted_expr
        assert res.recommended_position_size == 1.0

    def test_random_walk_regime_immune_to_factor_zoo_overfitting(self):
        """In random-walk noise, L2 rejects overfitted factor candidates and stays in cash."""
        res = run_finance_l2_comparison("random_walk", bootstrap=False, seed=42)

        # Refuses promotion of spurious non-linear combinations
        assert res.l2_status != EpistemicStatus.UNIQUE_SUPPORT
        assert res.l2_promoted_expr == ""
        assert res.recommended_position_size == 0.0

    def test_factor_collinearity_triggers_persistent_ambiguity_and_sizes_down(self):
        """Collinear factors trigger MULTI_HYPOTHESIS and downsize risk allocation to 50%."""
        res = run_factor_collinearity_experiment(seed=42)

        assert res.l2_status == EpistemicStatus.MULTI_HYPOTHESIS
        assert res.l2_attribution == EpistemicAttribution.INTRINSIC_OBSERVATIONAL_NONIDENTIFIABILITY
        assert res.recommended_position_size == 0.5
        assert "collinearity" in res.epistemic_insight.lower()

    def test_epistemic_position_sizing_logic(self):
        """Unit test of risk management mapping from epistemic states to capital allocation."""
        assert epistemic_position_sizing(EpistemicStatus.UNIQUE_SUPPORT, "decisive_evidential_support") == 1.0
        assert epistemic_position_sizing(EpistemicStatus.MULTI_HYPOTHESIS, "intrinsic_observational_nonidentifiability") == 0.5
        assert epistemic_position_sizing(EpistemicStatus.MULTI_HYPOTHESIS, "bootstrap_instability") == 0.0
        assert epistemic_position_sizing(EpistemicStatus.INSUFFICIENT_EVIDENCE, "measurement_noise_swamped") == 0.0
        assert epistemic_position_sizing(EpistemicStatus.INSUFFICIENT_EVIDENCE, "hypothesis_space_limited") == 0.0
        assert epistemic_position_sizing(EpistemicStatus.NO_VALID_CANDIDATE, "out_of_grammar_or_misspecified") == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
