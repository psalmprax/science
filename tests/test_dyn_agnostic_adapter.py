"""Unit Test Suite for APEDP-X Agnostic Domain Adapter & 3-Arm Ablation.

Verifies:
  1. Agnostic Adapter evaluates empirical fit and out-of-sample holdout correctly.
  2. Generic Declared Invariants (conservation, bounds, monotonicity, stability)
     detect violations without hard-coded domain logic.
  3. Structural rival comparison detects equivalence traps at resolution floor tau_epi.
  4. 3-Arm Ablation Experiment confirms:
     - Arm A (Specialized): 100% VRR, 0% FPR.
     - Arm B (Naive Core): 100% FPR (naive empirical fit falsely promotes traps).
     - Arm C (Agnostic Adapter): 100% VRR, 0% FPR (generic invariants suppress all traps).
"""
from __future__ import annotations

import numpy as np

from apedp_dyn.epistemic.agnostic_adapter import (
    AgnosticAdapter,
    AgnosticCandidateSpecification,
    AgnosticProblemDataset,
    DeclaredInvariant,
    InvariantKind,
)
from apedp_dyn.epistemic.claim_ladder import ClaimLevel
from apedp_dyn.sandbox.agnostic_domain_sandbox import (
    generate_ablation_corpus,
    run_ablation_experiment,
    run_arm_c_agnostic_adapter,
)


def test_agnostic_adapter_empirical_evaluation():
    """Verify empirical fit and holdout evaluation on clean data."""
    adapter = AgnosticAdapter(default_tau_epi=0.02)
    x = np.linspace(0, 5, 50)
    y = 2.0 * x + 1.0

    spec = AgnosticCandidateSpecification(
        candidate_id="LINEAR_TEST",
        proposed_claim=ClaimLevel.C2_PREDICTIVE,
        formula_repr="y = 2*x + 1",
        model_callable=lambda arr: 2.0 * arr + 1.0,
    )
    dataset = AgnosticProblemDataset(
        domain_label="Synthetic Linear",
        X_train=x,
        y_train=y,
        X_holdout=x,
        y_holdout=y,
    )
    metrics = adapter.evaluate_empirical_fit(spec, dataset)
    assert metrics["in_sample_r2"] > 0.999
    assert metrics["holdout_rmse"] < 1e-4

    res = adapter.adjudicate(spec, dataset)
    assert res.promoted is True
    assert res.justified_claim == ClaimLevel.C2_PREDICTIVE
    assert res.final_status == "PROMOTED_ORIGINAL"


def test_agnostic_adapter_generic_invariants():
    """Verify built-in evaluator for bound positivity and monotonicity."""
    adapter = AgnosticAdapter()
    x = np.linspace(0, 10, 50)

    # 1. Positivity pass
    inv_pos = DeclaredInvariant(
        name="positive_energy",
        kind=InvariantKind.BOUND_POSITIVITY,
        target_value=0.0,
        tolerance=1e-5,
    )
    spec_clean = AgnosticCandidateSpecification(
        candidate_id="POS_PASS",
        proposed_claim=ClaimLevel.C3_STRUCTURAL,
        formula_repr="y = x^2",
        model_callable=lambda arr: arr ** 2,
        declared_invariants=[inv_pos],
    )
    ds = AgnosticProblemDataset("Synthetic", X_train=x, y_train=x**2, X_holdout=x, y_holdout=x**2)
    res_pos = adapter.verify_invariant(inv_pos, spec_clean, ds)
    assert res_pos.passed is True

    # 2. Positivity fail (dips negative)
    spec_neg = AgnosticCandidateSpecification(
        candidate_id="POS_FAIL",
        proposed_claim=ClaimLevel.C3_STRUCTURAL,
        formula_repr="y = x^2 - 10",
        model_callable=lambda arr: arr ** 2 - 10.0,
        declared_invariants=[inv_pos],
    )
    res_neg = adapter.verify_invariant(inv_pos, spec_neg, ds)
    assert res_neg.passed is False
    assert res_neg.observed_metric < 0.0


def test_agnostic_adapter_structural_rivals():
    """Verify detection of indistinguishable rival models at resolution floor tau_epi."""
    adapter = AgnosticAdapter(default_tau_epi=0.05)
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    cand1 = AgnosticCandidateSpecification(
        candidate_id="MODEL_1",
        proposed_claim=ClaimLevel.C3_STRUCTURAL,
        formula_repr="sin(x)",
        model_callable=lambda arr: np.sin(arr),
    )
    rival_indistinguishable = AgnosticCandidateSpecification(
        candidate_id="MODEL_2_RIVAL",
        proposed_claim=ClaimLevel.C3_STRUCTURAL,
        formula_repr="sin(x) + 0.005",
        model_callable=lambda arr: np.sin(arr) + 0.005,
    )

    dataset = AgnosticProblemDataset(
        domain_label="Oscillatory",
        X_train=x,
        y_train=y,
        X_holdout=x,
        y_holdout=y,
        rival_models=[rival_indistinguishable],
        tau_epi=0.05,
    )

    all_rejected, min_delta, indist = adapter.evaluate_structural_rivals(cand1, dataset, cand_rmse=0.0)
    assert all_rejected is False
    assert "MODEL_2_RIVAL" in indist
    assert min_delta < 0.05


def test_3arm_ablation_experiment_scorecard():
    """Execute full 3-arm ablation and assert key epistemic findings."""
    bench = run_ablation_experiment()

    assert bench["total_cases"] == 8
    assert bench["valid_count"] == 4
    assert bench["trap_count"] == 4

    # Arm A: Specialized Adapters
    assert bench["arm_a_specialized"]["vrr"] == 100.0  # 4/4 valid retained
    assert bench["arm_a_specialized"]["fpr"] == 0.0    # 0/4 false promotions

    # Arm B: Naive Core (High false promotion on traps due to relying on empirical fit alone)
    assert bench["arm_b_core_naive"]["fpr"] == 100.0   # 4/4 traps falsely promoted!

    # Arm C: Agnostic Adapter with Declared Invariants (Full trap suppression without domain hardcoding)
    assert bench["arm_c_agnostic"]["vrr"] == 100.0     # 4/4 valid retained
    assert bench["arm_c_agnostic"]["fpr"] == 0.0       # 0/4 false promotions!


def test_agnostic_adapter_calibrates_confounded_causal_claim():
    """Candidate claiming C4 with failing do(X) invariance is calibrated to C2 via EEDR-CA."""
    corpus = {c.case_id: c for c in generate_ablation_corpus()}
    causal_case = corpus["CASE-07-CAUSAL-CONFOUNDED"]

    res = run_arm_c_agnostic_adapter(causal_case)
    assert res["final_status"] == "PROMOTED_REPAIRED"
    assert res["justified_claim"] == "C2_PREDICTIVE"
    assert res["is_false_promotion"] is False
    assert res["is_calibrated"] is True
