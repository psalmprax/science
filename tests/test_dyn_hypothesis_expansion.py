"""Tests for Hypothesis-Space Expansion: Model Starvation vs Genuine Non-Identifiability.

Validates the formal distinction:
    Abstention = Epistemic Uncertainty OR Hypothesis-Space Insufficiency

Verifies:
  1. Model-Space Starvation:
     Restricted H_0 yields INSUFFICIENT_EVIDENCE with attribution HYPOTHESIS_SPACE_LIMITED.
     Expanded H_L2 yields UNIQUE_SUPPORT with attribution DECISIVE_EVIDENTIAL_SUPPORT.
  2. Persistent Ambiguity:
     Restricted H_0 yields MULTI_HYPOTHESIS.
     Expanded H_L2 also yields MULTI_HYPOTHESIS with attribution INTRINSIC_OBSERVATIONAL_NONIDENTIFIABILITY.
     Feature covariance condition number kappa >> 10^3 demonstrates genuine observational equivalence.
  3. L3 Active Interventional Resolution:
     Synthesizing an optimal interventional probe breaks the observational manifold
     and authorizes promotion to UNIQUE_SUPPORT under active discovery.
"""
from __future__ import annotations

import pytest

from apedp_dyn.epistemic.adjudicate import (
    EpistemicAttribution,
    EpistemicStatus,
)
from apedp_dyn.sandbox.hypothesis_space_experiment import (
    run_model_space_starvation_experiment,
    run_persistent_ambiguity_experiment,
)


class TestHypothesisSpaceExpansion:
    """Rigorous tests for model-space starvation vs persistent non-identifiability."""

    def test_model_space_starvation_stage1_fails_and_stage2_discovers(self):
        """Restricted H_0 fails due to model starvation; expanded H_L2 recovers true law."""
        res = run_model_space_starvation_experiment(seed=42)

        # Stage 1: Linear candidates fail fidelity ceiling on quadratic physics
        assert res.stage1_status == EpistemicStatus.INSUFFICIENT_EVIDENCE
        assert res.stage1_attribution == EpistemicAttribution.HYPOTHESIS_SPACE_LIMITED

        # Stage 2: Open L2 typed grammar discovers m * v^2
        assert res.stage2_status == EpistemicStatus.UNIQUE_SUPPORT
        assert res.stage2_attribution == EpistemicAttribution.DECISIVE_EVIDENTIAL_SUPPORT
        assert "m" in res.discovered_expression and "v" in res.discovered_expression
        assert res.demonstrates_starvation is True

    def test_persistent_ambiguity_survives_hypothesis_expansion(self):
        """Collinear mechanisms remain MULTI_HYPOTHESIS across both H_0 and expanded H_L2."""
        res = run_persistent_ambiguity_experiment(seed=42)

        # Stage 1: Tied under H_0
        assert res.stage1_status == EpistemicStatus.MULTI_HYPOTHESIS
        assert res.stage1_attribution == EpistemicAttribution.INTRINSIC_OBSERVATIONAL_NONIDENTIFIABILITY

        # Stage 2: Ambiguity persists under expanded H_L2
        assert res.stage2_status == EpistemicStatus.MULTI_HYPOTHESIS
        assert res.stage2_attribution == EpistemicAttribution.INTRINSIC_OBSERVATIONAL_NONIDENTIFIABILITY

        # Independent physical diagnostic: severe ill-conditioning
        assert res.condition_number > 1000.0
        assert res.feature_correlation > 0.999
        assert res.demonstrates_persistent_ambiguity is True

    def test_l3_active_intervention_resolves_persistent_ambiguity(self):
        """Active experimental intervention breaks collinearity manifold and promotes winner."""
        res = run_persistent_ambiguity_experiment(seed=42)

        # Stage 3: Active interventional data separates hypotheses
        assert res.l3_status == EpistemicStatus.UNIQUE_SUPPORT
        assert res.l3_attribution == EpistemicAttribution.DECISIVE_EVIDENTIAL_SUPPORT
        assert res.l3_resolved_by_intervention is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
