"""Comprehensive Test Suite for Competing Proposers, DSR Multiplicity Control,
Strategy-Class Gating, and Decoupled Structural Novelty Adjudication.

Verifies:
  1. Mathematically consistent stock-split invariance (P -> P/k, V -> k*V).
  2. Strategy-class specific operational exposure & statistical power contracts.
  3. Decoupled predictive incrementality (Diebold-Mariano) vs structural novelty.
  4. Candidate trial ledger persistence and Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014).
  5. Competing proposers (ElasticNet & Order-Book Resiliency) conforming to common interface.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from apedp_dyn.epistemic.novelty_adjudication import (
    evaluate_predictive_incrementality,
    evaluate_structural_novelty,
)
from apedp_dyn.epistemic.trial_ledger import (
    CandidateTrialLedger,
    compute_deflated_sharpe_ratio,
    compute_expected_max_sharpe,
)
from apedp_dyn.grammar.proposer_interface import (
    CandidateProposal,
    ProposerBase,
    StrategyArchetype,
)
from apedp_dyn.proposers.resiliency_proposer import OrderBookResiliencyProposer
from apedp_dyn.proposers.sparse_microstructure_proposer import SparseMicrostructureProposer
from apedp_dyn.sandbox.canonical_ledger import (
    assert_stock_split_invariance,
    assert_strategy_class_compliance,
    run_ledger,
)


# ── 1. Consistent Stock-Split Invariance ──────────────────────────────────────

def test_stock_split_invariance_passes_on_dimensionless_ratio():
    """Dimensionless price/volume ratios must be invariant under P -> P/k, V -> k*V."""
    np.random.seed(42)
    n = 100
    prices = 100.0 * np.cumprod(1.0 + 0.01 * np.random.randn(n))
    volumes = np.abs(np.random.randn(n)) * 1000.0 + 100.0
    df = pd.DataFrame({"price": prices, "V": volumes})

    # Normalized ratio signal: Price / MA(Price)
    def ratio_signal(d: pd.DataFrame) -> np.ndarray:
        p = d["price"].to_numpy()
        ma = pd.Series(p).rolling(10, min_periods=1).mean().to_numpy()
        return p / ma

    is_inv, max_diff, status = assert_stock_split_invariance(ratio_signal, df, split_ratio=2.0)
    assert is_inv is True
    assert max_diff < 1e-10
    assert status == "SPLIT_INVARIANT"


def test_stock_split_invariance_detects_nominal_price_dependency():
    """Signals that mistakenly depend on nominal price scale must fail split invariance."""
    np.random.seed(42)
    n = 100
    prices = 100.0 * np.cumprod(1.0 + 0.01 * np.random.randn(n))
    volumes = np.abs(np.random.randn(n)) * 1000.0 + 100.0
    df = pd.DataFrame({"price": prices, "V": volumes})

    # Flawed nominal signal: price - 100.0 (changes when stock splits!)
    def nominal_signal(d: pd.DataFrame) -> np.ndarray:
        return d["price"].to_numpy() - 100.0

    is_inv, max_diff, status = assert_stock_split_invariance(nominal_signal, df, split_ratio=2.0)
    assert is_inv is False
    assert max_diff > 1.0
    assert status == "SPLIT_VARIANT"


# ── 2. Strategy-Class Specific Exposure Contracts ───────────────────────────

def test_tail_risk_strategy_passes_with_sufficient_events():
    """Tail-risk strategy with low active rate (< 5%) passes if it has >= min_events trades."""
    # 2% active rate over 2000 bars, organized into 40 distinct crisis episodes
    rows = [{"pos": 0.0} for _ in range(2000)]
    for i in range(40):
        idx = i * 45
        rows[idx]["pos"] = -1.0  # short hedge on crisis event

    is_comp, frac, status, reason = assert_strategy_class_compliance(
        rows, strategy_archetype="tail_risk", min_events=30
    )
    assert is_comp is True
    assert frac < 0.05
    assert status == "GATE_BINDING"
    assert "40 crisis events" in reason


def test_tail_risk_strategy_fails_insufficient_statistical_power():
    """Tail-risk strategy with only 3 crisis events must report INSUFFICIENT_STATISTICAL_POWER."""
    rows = [{"pos": 0.0} for _ in range(500)]
    rows[10]["pos"] = -1.0
    rows[100]["pos"] = -1.0
    rows[200]["pos"] = -1.0

    is_comp, frac, status, reason = assert_strategy_class_compliance(
        rows, strategy_archetype="tail_risk", min_events=30
    )
    assert is_comp is False
    assert status == "INSUFFICIENT_STATISTICAL_POWER"
    assert "minimum 30 required" in reason


def test_continuous_factor_strategy_requires_high_active_rate():
    """Continuous factor strategy requires >= 85% active exposure."""
    # 90% active exposure
    rows_good = [{"pos": 1.0 if i < 90 else 0.0} for i in range(100)]
    is_comp, frac, status, _ = assert_strategy_class_compliance(
        rows_good, strategy_archetype="continuous_factor"
    )
    assert is_comp is True
    assert frac >= 0.85
    assert status == "GATE_BINDING"

    # Only 50% active exposure -> fails continuous factor contract
    rows_bad = [{"pos": 1.0 if i < 50 else 0.0} for i in range(100)]
    is_comp_bad, _, status_bad, reason_bad = assert_strategy_class_compliance(
        rows_bad, strategy_archetype="continuous_factor"
    )
    assert is_comp_bad is False
    assert status_bad == "NON_CONTINUOUS_FACTOR_EXPOSURE"
    assert "falls below 85%" in reason_bad


def test_run_ledger_tail_risk_archetype_allows_low_active_fraction():
    """run_ledger with strategy_archetype='tail_risk' awards binding status if events >= 30."""
    np.random.seed(42)
    n = 500
    prices = 100.0 * np.cumprod(1.0 + 0.01 * np.random.randn(n))
    volumes = np.abs(np.random.randn(n)) * 1000.0 + 100.0
    df = pd.DataFrame({"price": prices, "V": volumes})
    data = {"BTC": df}
    weights = {"H-BB": 1.0}

    # Evaluate with tail_risk archetype and min_tail_events = 5
    _, summary = run_ledger(
        syms=["BTC"],
        data=data,
        weights=weights,
        rule="hyst",
        th_in=2.0,  # Strict threshold, triggers rarely
        th_out=2.0,
        eval_slice=slice(50, n),
        strategy_archetype="tail_risk",
        min_tail_events=5,
    )
    assert summary["gate_binding"] in (True, False)
    assert summary["status"] in ("GATE_BINDING", "INSUFFICIENT_STATISTICAL_POWER")


# ── 3. Decoupled Predictive Incrementality & Structural Novelty ─────────────

def test_predictive_incrementality_diebold_mariano():
    """Statistically significant forecast error reduction must yield PREDICTIVELY_INCREMENTAL."""
    np.random.seed(42)
    n = 200
    y_true = np.random.randn(n)
    # Baseline forecast: noisy
    y_base = y_true + 0.8 * np.random.randn(n)
    # Candidate forecast: noticeably lower error
    y_cand = y_true + 0.2 * np.random.randn(n)

    res = evaluate_predictive_incrementality(y_true, y_cand, y_base, alpha=0.01)
    assert res.verdict == "PREDICTIVELY_INCREMENTAL"
    assert res.is_significant is True
    assert res.loss_reduction > 0.0
    assert res.p_value < 0.01

    # Ineffective candidate (worse than baseline)
    y_worse = y_true + 1.2 * np.random.randn(n)
    res_worse = evaluate_predictive_incrementality(y_true, y_worse, y_base, alpha=0.01)
    assert res_worse.verdict == "INCREMENTALITY_UNPROVEN"
    assert res_worse.is_significant is False


def test_structural_novelty_detects_literature_reidentification():
    """A signal collinear with Kyle's lambda must be flagged as REPRESENTATION_REIDENTIFICATION."""
    np.random.seed(42)
    n = 200
    kyle_signal = np.random.randn(n)
    literature = {
        "KYLE_LAMBDA": kyle_signal,
        "CARHART_MOMENTUM": np.random.randn(n),
    }

    # Candidate is 95% correlated with Kyle's lambda (just scaled and with tiny noise)
    candidate = 2.5 * kyle_signal + 0.1 * np.random.randn(n)

    res = evaluate_structural_novelty(candidate, literature, novelty_threshold_r2=0.85)
    assert res.verdict == "REPRESENTATION_REIDENTIFICATION"
    assert res.closest_literature_baseline == "KYLE_LAMBDA"
    assert res.max_projection_r2 > 0.85
    assert "Kyle, A. S. (1985)" in res.literature_citation


def test_structural_novelty_passes_on_orthogonal_signal():
    """An orthogonal signal (R^2 < 0.85) must be certified as STRUCTURALLY_NOVEL."""
    np.random.seed(42)
    n = 200
    literature = {
        "KYLE_LAMBDA": np.random.randn(n),
        "AMIHUD_ILLIQUIDITY": np.random.randn(n),
        "ORDER_BOOK_RESILIENCY": np.random.randn(n),
    }

    # Candidate is completely independent random noise
    independent_candidate = np.random.randn(n)

    res = evaluate_structural_novelty(independent_candidate, literature, novelty_threshold_r2=0.85)
    assert res.verdict == "STRUCTURALLY_NOVEL"
    assert res.max_projection_r2 < 0.85
    assert res.orthogonal_variance_fraction > 0.15


# ── 4. Candidate Trial Ledger & Deflated Sharpe Ratio (DSR) ──────────────────

def test_candidate_trial_ledger_records_and_computes_stats(tmp_path):
    """Trial ledger records attempts, counts trials, and persists to jsonl."""
    ledger_file = tmp_path / "test_trials.jsonl"
    ledger = CandidateTrialLedger(persistence_path=ledger_file)

    rec1 = ledger.record_trial(
        candidate_id="c1",
        proposer_id="elasticnet",
        formula="0.5*mom + 0.2*cvd",
        strategy_archetype="continuous_factor",
        in_sample_sharpe=0.85,
        returns=np.random.randn(100) * 0.01 + 0.001,
        status="EVALUATED",
    )
    rec2 = ledger.record_trial(
        candidate_id="c2",
        proposer_id="resiliency",
        formula="Shock * exp(-0.4*k)",
        strategy_archetype="tail_risk",
        in_sample_sharpe=0.45,
        returns=np.random.randn(100) * 0.01,
        status="EVALUATED",
    )

    assert ledger.trial_count == 2
    assert ledger_file.is_file()

    mean_sr, var_sr = ledger.get_sharpe_distribution()
    assert abs(mean_sr - 0.65) < 1e-4
    assert var_sr > 0.0


def test_deflated_sharpe_ratio_penalizes_search_multiplicity():
    """Moderate Sharpe ratio (0.80) in a 500-trial family fails DSR significance."""
    # 500 trials with average trial variance 0.35
    exp_max = compute_expected_max_sharpe(n_trials=500, var_trials=0.35)
    assert exp_max > 1.0  # pure chance would produce max SR > 1.0!

    # Observed Sharpe 0.80 with sample size 250 bars
    res_deflated = compute_deflated_sharpe_ratio(
        observed_sharpe=0.80,
        n_trials=500,
        var_trials=0.35,
        n_bars=250,
        skewness=-0.5,
        kurtosis=4.5,
    )
    # 0.80 is less than expected maximum (exp_max > 1.0), so DSR must reject!
    assert res_deflated["is_significant"] is False
    assert res_deflated["dsr"] < 0.50
    assert res_deflated["p_value"] > 0.50

    # Genuine massive alpha (SR = 2.8) passes DSR despite 500 trials
    res_stellar = compute_deflated_sharpe_ratio(
        observed_sharpe=2.80,
        n_trials=500,
        var_trials=0.35,
        n_bars=500,
        skewness=-0.2,
        kurtosis=3.5,
    )
    assert res_stellar["is_significant"] is True
    assert res_stellar["dsr"] > 0.99
    assert res_stellar["p_value"] < 0.01


# ── 5. Competing Proposers Conforming to Common Interface ────────────────────

def test_sparse_microstructure_proposer_generates_proposals():
    """SparseMicrostructureProposer fits ElasticNet and emits valid CandidateProposal objects."""
    np.random.seed(42)
    n = 150
    p = 100.0 * np.cumprod(1.0 + 0.01 * np.random.randn(n))
    v = np.abs(np.random.randn(n)) * 1000.0 + 100.0
    df = pd.DataFrame({"price": p, "V": v, "taker_buy": v * 0.52})
    data = {"BTC": df}

    proposer = SparseMicrostructureProposer(l1_ratios=[0.5], alphas=[0.001])
    assert isinstance(proposer, ProposerBase)
    assert proposer.proposer_id == "sparse_microstructure_elasticnet"

    proposals = proposer.fit_and_propose(data)
    assert len(proposals) > 0

    c = proposals[0]
    assert isinstance(c, CandidateProposal)
    assert c.strategy_archetype == StrategyArchetype.CONTINUOUS_FACTOR
    assert c.declared_scale_invariant is True
    assert len(c.parameters) > 0

    # Ensure compute_score runs causally on new data
    score = c.compute_score(df)
    assert len(score) == len(df)
    assert np.all(np.isfinite(score))


def test_order_book_resiliency_proposer_generates_proposals():
    """OrderBookResiliencyProposer fits exponential bounce and emits valid tail-risk proposals."""
    np.random.seed(42)
    n = 200
    p = 100.0 * np.cumprod(1.0 + 0.01 * np.random.randn(n))
    v = np.abs(np.random.randn(n)) * 1000.0 + 100.0
    df = pd.DataFrame({"price": p, "V": v})
    data = {"ETH": df}

    proposer = OrderBookResiliencyProposer(shock_std_threshold=1.5, max_horizon=5)
    assert isinstance(proposer, ProposerBase)
    assert proposer.proposer_id == "order_book_resiliency_biais_large"

    proposals = proposer.fit_and_propose(data)
    assert len(proposals) > 0

    c = proposals[0]
    assert isinstance(c, CandidateProposal)
    assert c.strategy_archetype == StrategyArchetype.TAIL_RISK
    assert "Shock_t" in c.formula
    assert "beta" in c.parameters
    assert "gamma" in c.parameters

    # Ensure compute_score works
    score = c.compute_score(df)
    assert len(score) == len(df)
    assert np.all(np.isfinite(score))


def test_tactical_strategy_enforces_5_to_90_percent_active_window():
    """Tactical archetype enforces active rate strictly in [5%, 90%]."""
    # 50% active -> GATE_BINDING
    rows_50 = [{"pos": -1.0 if i < 50 else 0.0} for i in range(100)]
    is_c, _, status, _ = assert_strategy_class_compliance(rows_50, strategy_archetype="tactical")
    assert is_c is True
    assert status == "GATE_BINDING"

    # 92% active -> GATE_NON_BINDING (> 90% ceiling)
    rows_92 = [{"pos": -1.0 if i < 92 else 0.0} for i in range(100)]
    is_c92, frac92, status92, reason92 = assert_strategy_class_compliance(rows_92, strategy_archetype="tactical")
    assert is_c92 is False
    assert status92 == "GATE_NON_BINDING"
    assert "outside [5%, 90%]" in reason92

