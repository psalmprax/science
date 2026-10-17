"""Tests for Expanded Proposers, Power Curve Calibration, Regime Diagnostics, and Execution Models.

Validates the reviewer's 5 engineering priorities:
  P0: Discovery calibration: empirical vs theoretical power curves with Wilson CIs.
  P1: Proposer expansion: DistributedLagInteractionProposer, RegimeSwitchingProposer, HawkesOrderFlowProposer.
  P2: Search accountability via CandidateProposal contracts.
  P3: Regime robustness diagnostic testing predictive incrementality across environments.
  P4: Economic validation via decoupled execution models (Spread/Slippage, Almgren-Chriss, Limit Order Queue).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from apedp_dyn.epistemic.novelty_adjudication import (
    evaluate_regime_robustness_diagnostic,
)
from apedp_dyn.grammar.proposer_interface import StrategyArchetype
from apedp_dyn.proposers.distributed_lag_interaction_proposer import (
    DistributedLagInteractionProposer,
)
from apedp_dyn.proposers.hawkes_orderflow_proposer import HawkesOrderFlowProposer
from apedp_dyn.proposers.regime_switching_proposer import RegimeSwitchingProposer
from apedp_dyn.sandbox.calibrated_benchmark_suite import (
    compute_theoretical_gaussian_power,
    generate_power_curve_matrix,
)
from apedp_dyn.sandbox.execution_cost_models import (
    AlmgrenChrissImpactCostModel,
    FixedPlusSlippageCostModel,
    LimitOrderQueueCostModel,
)

# ── P0: Power Curve Calibration Tests ────────────────────────────────────────

def test_theoretical_power_computation_monotonicity():
    """Theoretical power must strictly increase with sample size and effect size."""
    p_small = compute_theoretical_gaussian_power(n=250, ic=0.02, alpha=0.05)
    p_large = compute_theoretical_gaussian_power(n=2500, ic=0.02, alpha=0.05)
    assert p_large > p_small
    assert 0.0 <= p_small <= 1.0
    assert 0.0 <= p_large <= 1.0

    # At IC = 0.15, power at N=1000 should exceed 95%
    p_strong = compute_theoretical_gaussian_power(n=1000, ic=0.15, alpha=0.05)
    assert p_strong > 0.95


def test_power_curve_matrix_generation():
    """Generate empirical vs theoretical power grid with confidence intervals."""
    def mock_proposer(f, y):
        corr = float(np.corrcoef(f, y)[0, 1])
        t_stat = corr * np.sqrt((len(f) - 2) / max(1.0 - corr ** 2, 1e-8))
        return bool(t_stat > 1.65), corr

    cells = generate_power_curve_matrix(
        mock_proposer,
        sample_sizes=[250, 500],
        target_ics=[0.05, 0.15],
        trials_per_cell=5,
        seed=42,
    )
    assert len(cells) == 4
    for cell in cells:
        assert 0.0 <= cell.empirical_power <= 1.0
        assert 0.0 <= cell.theoretical_max_power <= 1.0
        assert cell.power_ci[0] <= cell.power_ci[1]


# ── P1: Expanded Proposer Tests ──────────────────────────────────────────────

def test_distributed_lag_interaction_proposer_generates_valid_proposals():
    """DistributedLagInteractionProposer generates polynomial lag and interaction features."""
    rng = np.random.RandomState(42)
    n = 200
    x1 = rng.randn(n)
    x2 = rng.randn(n)
    # Target depends on lag1 of x1 and interaction x1*x2
    y = 0.3 * np.roll(x1, 1) + 0.2 * (x1 * x2) + 0.5 * rng.randn(n)

    df = pd.DataFrame({"x1": x1, "x2": x2, "target": y})
    proposer = DistributedLagInteractionProposer(lags=(1, 2), include_interactions=True)
    proposals = proposer.fit_and_propose(df, feature_cols=["x1", "x2"], target_col="target")

    assert len(proposals) >= 1
    p = proposals[0]
    assert p.strategy_archetype == StrategyArchetype.CONTINUOUS_FACTOR
    assert p.declared_scale_invariant is True
    # Evaluate compute_score
    scores = p.compute_score(df)
    assert len(scores) == n
    assert np.std(scores) > 0.0


def test_regime_switching_proposer_generates_valid_proposals():
    """RegimeSwitchingProposer generates piecewise threshold candidate models."""
    rng = np.random.RandomState(42)
    n = 300
    vol = np.abs(rng.randn(n))
    x = rng.randn(n)
    # High vol has positive beta, low vol has negative beta
    y = np.where(vol > np.median(vol), 0.4 * x, -0.4 * x) + 0.3 * rng.randn(n)

    df = pd.DataFrame({"feat": x, "vol": vol, "target": y})
    proposer = RegimeSwitchingProposer(vol_window=10, regime_threshold_quantile=0.50)
    proposals = proposer.fit_and_propose(df, feature_cols=["feat"], target_col="target", vol_col="vol")

    assert len(proposals) >= 1
    p = proposals[0]
    assert p.strategy_archetype == StrategyArchetype.TACTICAL
    assert "IF vol >" in p.formula
    scores = p.compute_score(df)
    assert len(scores) == n
    assert np.std(scores) > 0.0


def test_hawkes_orderflow_proposer_fits_intensity():
    """HawkesOrderFlowProposer fits self-exciting arrival intensities."""
    # Synthesize clustered timestamps
    rng = np.random.RandomState(42)
    dt = rng.exponential(scale=0.5, size=100)
    timestamps = np.cumsum(dt)

    proposer = HawkesOrderFlowProposer(decay_beta=1.5, intensity_threshold_quantile=0.80)
    proposals = proposer.fit_and_propose(timestamps)

    assert len(proposals) >= 1
    p = proposals[0]
    assert p.strategy_archetype == StrategyArchetype.TAIL_RISK
    assert 0.0 < p.parameters["branching_ratio"] < 1.0
    assert "hawkes_intensity" in p.formula


# ── P3: Regime Robustness Diagnostic Tests ───────────────────────────────────

def test_regime_robustness_diagnostic_evaluates_cross_environment_survival():
    """Regime robustness diagnostic assesses stability across volatility/temporal slices."""
    rng = np.random.RandomState(42)
    n = 300
    regimes = np.array(["regime_A"] * 100 + ["regime_B"] * 100 + ["regime_C"] * 100)

    y_true = rng.randn(n)
    y_base = np.zeros(n)
    # Candidate provides positive loss reduction in regimes A and B, slight negative in C
    y_cand = np.zeros(n)
    y_cand[:200] = 0.20 * y_true[:200] + 0.1 * rng.randn(200)
    y_cand[200:] = 0.05 * rng.randn(100)

    diag = evaluate_regime_robustness_diagnostic(
        y_true=y_true,
        y_candidate_pred=y_cand,
        y_incumbent_pred=y_base,
        regime_labels=regimes,
        min_survival_rate=0.60,
    )
    assert diag.status in ("REGIME_STABLE", "REGIME_VULNERABLE")
    assert diag.survival_fraction >= 0.60
    assert "regime_A" in diag.per_regime_results
    assert "regime_B" in diag.per_regime_results


# ── P4: Decoupled Execution Cost Models Tests ─────────────────────────────────

def test_fixed_plus_slippage_model():
    """FixedPlusSlippageCostModel scales with turnover, spread, and local volatility."""
    model = FixedPlusSlippageCostModel(fee_bps=2.0, slippage_coeff=0.10)
    pos = np.array([0.0, 1.0, 1.0, 0.0, -1.0])
    prices = np.array([100.0, 101.0, 102.0, 101.0, 100.0])

    drag = model.compute_cost_drag(positions=pos, prices=prices)
    assert len(drag) == len(pos)
    assert drag[0] == 0.0  # Initial flat position
    assert drag[1] > 0.0  # Turn from 0 to 1 incur costs
    assert drag[2] == 0.0  # Held position incurs 0 turnover fee


def test_almgren_chriss_impact_model():
    """AlmgrenChrissImpactCostModel separates temporary and permanent price impact."""
    model = AlmgrenChrissImpactCostModel(eta_temp=0.05, gamma_perm=0.02)
    pos = np.array([0.0, 5000.0, 10000.0, 10000.0])
    prices = np.array([50.0, 50.1, 50.2, 50.2])
    volumes = np.array([1e5, 1e5, 1e5, 1e5])

    drag = model.compute_cost_drag(positions=pos, prices=prices, volumes=volumes)
    assert len(drag) == len(pos)
    assert drag[1] > 0.0
    assert drag[2] > 0.0
    assert drag[3] == 0.0  # Zero turnover incurs zero execution drag


def test_limit_order_queue_model():
    """LimitOrderQueueCostModel applies maker rebate and penalizes adverse selection."""
    model = LimitOrderQueueCostModel(adverse_selection_bps=5.0, maker_rebate_bps=1.0)
    pos = np.array([0.0, 1.0, 1.0, -1.0])
    prices = np.array([10.0, 10.0, 10.0, 10.0])
    # OFI opposing position changes produces adverse selection
    ofi = np.array([0.0, -100.0, 0.0, 100.0])

    drag = model.compute_cost_drag(positions=pos, prices=prices, ofi=ofi)
    assert len(drag) == len(pos)
    assert drag[1] > 0.0  # Opposing OFI triggers adverse selection
    assert drag[2] == 0.0  # Zero turnover
