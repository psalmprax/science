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
    # The fitted model carries an intercept and raw-scale weights, so it is NOT
    # invariant under input rescaling; claiming invariance here used to make
    # referee invariance tests falsify a true statement.
    assert p.declared_scale_invariant is False
    # Evaluate compute_score. The first max(lag) rows are unknowable and are
    # scored NaN rather than fabricated, so the signal check uses the finite
    # overlap (the same rows the fit used).
    scores = p.compute_score(df)
    assert len(scores) == n
    finite_scores = scores[np.isfinite(scores)]
    assert finite_scores.size >= n - 2
    assert np.std(finite_scores) > 0.0


def test_distributed_lag_proposer_scores_fitted_model_faithfully():
    """Regression: coefficients must be mapped back to the raw feature scale.

    Features with large offsets/scales expose any mismatch between the
    standardized ElasticNet fit and the scorer: applying standardized-fit
    coefficients directly to raw feature values yields predictions wildly
    inconsistent with the fitted signal.
    """
    rng = np.random.RandomState(7)
    n = 400
    x1 = 100.0 + 5.0 * rng.randn(n)
    x2 = 0.5 + 0.1 * rng.randn(n)
    lag_x1 = np.roll(x1, 1)
    lag_x1[0] = 0.0
    y = 0.3 * lag_x1 + 50.0 * (x1 * x2) + 0.5 * rng.randn(n)

    df = pd.DataFrame({"x1": x1, "x2": x2, "target": y})
    proposer = DistributedLagInteractionProposer(lags=(1,), include_interactions=True)
    proposals = proposer.fit_and_propose(df, feature_cols=["x1", "x2"], target_col="target")

    assert len(proposals) >= 1
    p = proposals[0]
    assert p.metadata["coefficients_mapped_to_raw_scale"] is True
    pred = np.asarray(p.compute_score(df), dtype=float)
    # The first lag rows are unknowable (no earlier observation) and are scored
    # NaN rather than fabricated, so faithfulness is measured on the finite
    # overlap -- the same rows the fit itself used.
    finite = np.isfinite(pred)
    assert finite.sum() >= n - 2, finite.sum()
    assert not finite[0], "the first lag row must not be fabricated"
    ss_res = float(np.sum((y[finite] - pred[finite]) ** 2))
    ss_tot = float(np.sum((y[finite] - y[finite].mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    assert r2 > 0.90, f"scorer does not reproduce the fitted model: R2={r2}"


def test_distributed_lag_scorer_is_streaming_safe():
    """Chunked scoring must equal single-pass scoring (state buffer, not zeros).

    The lag scorer used to zero-fill the first `lag` rows of whatever frame it
    was handed, so a streaming chunk shorter than the lag lost 100% of the lag
    signal. With the carried state buffer, scoring a series in short chunks
    through the same proposal reproduces the single-pass scores exactly.
    """
    rng = np.random.RandomState(5)
    n = 300
    x1 = rng.randn(n)
    y = 0.4 * np.roll(x1, 2) + 0.1 * rng.randn(n)
    df = pd.DataFrame({"x1": x1, "target": y})
    proposer = DistributedLagInteractionProposer(lags=(2,), include_interactions=False)
    proposals = proposer.fit_and_propose(df, feature_cols=["x1"], target_col="target")
    assert len(proposals) >= 1
    p = proposals[0]

    full = np.asarray(p.compute_score(df), dtype=float)

    chunks = []
    for start in range(0, n, 7):
        piece = df.iloc[start:start + 7]
        chunks.append(np.asarray(p.compute_score(piece), dtype=float))
    streamed = np.concatenate(chunks)

    both = np.isfinite(full) & np.isfinite(streamed)
    assert both.sum() > n - 5, both.sum()
    assert np.allclose(full[both], streamed[both], atol=1e-12), float(np.max(np.abs(full[both] - streamed[both])))


def test_distributed_lag_multi_lag_same_column_streaming_safe():
    """FLAW-A09 pin: sibling lag terms on the SAME base column must not leak.

    The old buffer handling wrote ``lag_state["buffers"][base_col]`` on the
    first lag term of a call, so a second lag term on the same base column
    (e.g. lag_1_x AND lag_2_x both selected) read this call's own tail as
    prior history and prepended future batch data to the head of the array —
    corrupting the sibling term both on the full frame and on every chunk.
    The fix snapshots the pre-call buffer once per (call, base col) and
    updates it once per (call, base col), so chunked scoring equals single-pass
    scoring exactly for multi-lag selections too.

    Mutation demonstration: under the old code the sibling term's head rows
    shift with the chunking, so full-vs-chunked equality fails.
    """
    from apedp_dyn.proposers.distributed_lag_interaction_proposer import (
        DistributedLagInteractionProposer,
    )

    rng = np.random.RandomState(11)
    n = 240
    x1 = rng.randn(n)
    # Target depends on BOTH lag 1 and lag 2 of the same column so the
    # ElasticNet selection retains two lag terms sharing the base column.
    y = np.roll(x1, 1) + 0.5 * np.roll(x1, 2) + 0.05 * rng.randn(n)
    y[0] = y[1] = np.nan  # raw lag rows hold nothing; scorer fabricates no values
    df = pd.DataFrame({"x1": x1, "target": y})
    proposer = DistributedLagInteractionProposer(
        lags=(1, 2), include_interactions=False
    )
    proposals = proposer.fit_and_propose(df, feature_cols=["x1"], target_col="target")
    assert len(proposals) >= 1
    p = proposals[0]
    selected = p.metadata["selected_terms"]
    assert sum(1 for t in selected if "lag" in t) >= 2, (
        "fixture must retain two lag terms on the same column" + str(selected)
    )
    full = np.asarray(p.compute_score(df), dtype=float)
    for head in (2, 7):
        streamed = np.concatenate([
            np.asarray(p.compute_score(df.iloc[i:i + head]), dtype=float)
            for i in range(0, n, head)
        ])
        both = np.isfinite(full) & np.isfinite(streamed)
        assert both.sum() > n - 8, both.sum()
        assert np.allclose(full[both], streamed[both], atol=1e-10), (
            f"head={head}: sibling lag-term buffer leaked "
            f"(max diff {float(np.max(np.abs(full[both] - streamed[both]))):.3g})"
        )


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


def test_regime_switching_proposer_excludes_unobservable_warmup_rows():
    """Regression: warmup rows with no observable volatility must not be fitted.

    Before the fix the rolling-std warmup was backfilled, assigning fabricated
    volatility (and therefore a regime) to early rows and leaking their target
    values into the low-regime fit.
    """
    rng = np.random.RandomState(11)
    n = 300
    vol_window = 10
    warmup = 4  # rolling(min_periods=5) has no value for the first 4 rows
    x = np.empty(n)
    x[:200] = 0.5 * rng.randn(200)  # low-amplitude (low-vol) segment
    x[200:] = 5.0 * rng.randn(100)  # high-amplitude (high-vol) segment
    y = np.where(np.arange(n) >= 200, 0.5 * x, -0.5 * x)
    # Misleading early rows: if (wrongly) included in the low regime they
    # drag the fitted low-regime slope far away from -0.5.
    y[:warmup] = 10.0 * x[:warmup]

    df = pd.DataFrame({"feat": x, "target": y})
    proposer = RegimeSwitchingProposer(vol_window=vol_window, regime_threshold_quantile=0.50)
    proposals = proposer.fit_and_propose(df, feature_cols=["feat"], target_col="target")

    assert len(proposals) >= 1
    p = proposals[0]
    # The warmup rows lack a volatility reading: excluded from both fits.
    assert p.metadata["samples_metric_missing"] == warmup
    assert p.metadata["samples_low"] + p.metadata["samples_high"] + warmup == n

    import re
    m = re.search(r"THEN ([+-][\d.]+)\*feat ELSE ([+-][\d.]+)\*feat", p.formula)
    assert m is not None, p.formula
    beta_high, beta_low = float(m.group(1)), float(m.group(2))
    assert abs(beta_high - 0.5) < 0.1, beta_high
    assert abs(beta_low + 0.5) < 0.1, beta_low

    # Scorer must refuse to classify rows without observable volatility.
    scores = p.compute_score(df)
    assert np.all(np.isnan(scores[:warmup]))
    assert np.all(np.isfinite(scores[warmup:]))


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
    assert p.metadata["binary_surge_viable_on_fit"] is True
    assert 0.0 < p.parameters["branching_ratio"] < 1.0
    assert "hawkes_intensity" in p.formula


def test_hawkes_degenerate_threshold_declares_continuous_factor():
    """Regression: a non-binding threshold emits a continuous signal and must
    be declared CONTINUOUS_FACTOR, not TAIL_RISK (ledger mis-adjudication)."""
    rng = np.random.RandomState(3)
    ts = np.cumsum(rng.exponential(scale=0.5, size=200))

    proposer = HawkesOrderFlowProposer(decay_beta=1.5, intensity_threshold_quantile=1.0)
    proposals = proposer.fit_and_propose(ts)

    assert len(proposals) == 1
    cand = proposals[0]
    assert cand.strategy_archetype == StrategyArchetype.CONTINUOUS_FACTOR
    assert cand.metadata["binary_surge_viable_on_fit"] is False
    assert cand.metadata["score_mode"] == "continuous_standardized_intensity"

    scores = cand.compute_score(pd.DataFrame({"timestamp": ts}))
    assert np.std(scores) > 0.0
    # Continuous, not a binary 0/1 vector.
    assert len(np.unique(np.round(scores, 10))) > 2


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
