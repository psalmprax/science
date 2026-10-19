"""Tests for Quantum Magic State Cultivation & Distillation.

Verifies:
1. Bravyi-Kitaev 15-to-1 cubic error suppression analytics and asymptotic scaling.
2. Continuous magic state cultivation filter contraction and suppression properties.
3. Multi-round distillation cascade convergence to fault-tolerant thresholds.
4. MagicStateCultivationProposer conforms to ProposerBase and emits valid CandidateProposal objects.
"""
from __future__ import annotations

import pytest

from apedp_dyn.grammar.proposer_interface import CandidateProposal, ProposerBase
from apedp_dyn.quantum.magic_state import (
    DistillationProtocol,
    DistillationRoundResult,
    DistillationScheduleResult,
    MagicStateCultivationProposer,
    compute_15_to_1_round,
    compute_cultivation_filter_round,
    simulate_distillation_cascade,
)


def test_15_to_1_cubic_suppression_and_scaling():
    """Verify Bravyi-Kitaev 15-to-1 exhibits exact O(eps^3) cubic scaling and pseudo-threshold ceiling."""
    eps1 = 0.01
    eps2 = 0.005  # Halved input infidelity
    res1 = compute_15_to_1_round(eps1, code_distance=5)
    res2 = compute_15_to_1_round(eps2, code_distance=5)

    assert isinstance(res1, DistillationRoundResult)
    assert res1.input_infidelity == 0.01

    # Exact cubic scaling assertion: (eps1 / eps2)^3 = 2^3 = 8.0
    scaling_ratio = res1.output_infidelity / res2.output_infidelity
    assert pytest.approx(scaling_ratio, rel=1e-5) == 8.0

    # Pseudo-threshold boundary assertion:
    res_above_thresh = compute_15_to_1_round(0.20, code_distance=5)
    assert res_above_thresh.output_infidelity >= res_above_thresh.input_infidelity
    assert res_above_thresh.suppression_factor == 1.0


def test_continuous_cultivation_filter_contraction():
    """Verify continuous cultivation filter acts as a contraction mapping on non-Clifford infidelity."""
    # Test across realistic operational range
    for eps in [0.001, 0.005, 0.01, 0.02, 0.05]:
        res = compute_cultivation_filter_round(eps, cultivation_filter_strength=40.0, code_distance=5)
        assert isinstance(res, DistillationRoundResult)
        # Contraction mapping: output strictly smaller than input
        assert res.output_infidelity < res.input_infidelity
        assert res.suppression_factor > 1.0
        # Yield is valid probability
        assert 0.0 < res.yield_probability <= 1.0

    # Monotonic yield: higher input error implies lower acceptance probability
    r_clean = compute_cultivation_filter_round(0.005)
    r_noisy = compute_cultivation_filter_round(0.05)
    assert r_clean.yield_probability > r_noisy.yield_probability


def test_distillation_cascade_convergence():
    initial_infidelity = 0.05
    target_infidelity = 1e-5

    sched = simulate_distillation_cascade(
        initial_infidelity=initial_infidelity,
        target_infidelity=target_infidelity,
        protocol=DistillationProtocol.WILLOW_CULTIVATION,
        max_rounds=4,
    )

    assert isinstance(sched, DistillationScheduleResult)
    assert sched.final_infidelity <= target_infidelity
    assert sched.meets_threshold is True
    assert len(sched.rounds) >= 1
    assert sched.cumulative_yield > 0.0
    assert sched.total_physical_qubits > 0
    assert "MagicStateSchedule" in sched.summary_formula


def test_magic_state_cultivation_proposer_interface():
    proposer = MagicStateCultivationProposer(target_infidelity=1e-4)
    assert isinstance(proposer, ProposerBase)
    assert proposer.proposer_id == "magic_state_cultivation_proposer_v1"

    proposals = proposer.fit_and_propose(input_infidelities=(0.01, 0.005))
    assert len(proposals) == 6  # 2 noise regimes x 3 protocols

    for p in proposals:
        assert isinstance(p, CandidateProposal)
        assert p.candidate_id.startswith("H-QMAGIC-")
        assert p.proposer_id == "magic_state_cultivation_proposer_v1"
        assert 0.0 <= p.parameters["heuristic_score"] <= 1.0
        assert "protocol" in p.metadata
        assert "final_infidelity" in p.metadata
        assert "cumulative_yield" in p.metadata
