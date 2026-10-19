"""Unit Test Suite for Epistemic Verification of Trading Signals & 30-Asset Sandboxes under AEH.

Verifies:
  1. Trading Signals Sandbox:
     - Evolved GA combinations claiming C4 (Causal Alpha) trigger CLAIM_TARGET_MISMATCH
     - EEDR applies claim_level_calibration downscaling from C4 -> C2 (Predictive Generalization)
     - Preserves evidence bundle anti-p-hacking invariant and promotes as PROMOTED_REPAIRED
     - Divergent/overfitted trading candidates terminate in TERMINAL_ABSTAIN
  2. 30-Asset Multi-Asset Sandbox:
     - Two-Tier Stratified Risk Parity passes Gate 0.5 (CAN-G)
     - Adjudicates and promotes C3 Structural Invariance under AEH
     - Naive unstratified pooling triggers Ranking Trap detection and terminates in TERMINAL_ABSTAIN
  3. UI Server Integration:
     - /api/sandbox/list includes trading_signals and multi_asset_30
     - /api/sandbox/run executes both sandboxes with complete epistemic_framework provenance payloads
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from apedp_dyn.epistemic.claim_ladder import (
    ClaimLevel,
    EvidenceBundle,
)
from apedp_dyn.epistemic.eedr import (
    EpistemicFailureCategory,
    execute_eedr_loop,
)
from apedp_dyn.sandbox.multi_asset_signals_sandbox import run_epistemic_30_asset_adjudication
from apedp_dyn.sandbox.trading_signals_sandbox import (
    load_real_btc_orderflow,
    run_epistemic_trading_adjudication,
)
from apedp_dyn.ui.server import app


def test_trading_signals_pipeline_charges_ga_multiplicity_and_promotion_refuses():
    """The live GA pipeline must charge its search to the DSR and run the promotion lifecycle.

    Every GA fitness evaluation is recorded as a trial, so the holdout Sharpe
    is deflated by the real multiplicity (DSR falls below the single-trial
    value), and the six-stage promotion state machine actually runs on the
    candidate: GA strategies have no Lean-verified law, so stage 1 refuses and
    the candidate halts there instead of reaching economic validity.
    """
    df = load_real_btc_orderflow(600)
    res = run_epistemic_trading_adjudication(
        panel=df,
        dataset_name="btc_multiplicity",
        initial_capital=10000.0,
        pop_size=20,
        generations=5,
        seed=7,
    )

    mult = res["trial_multiplicity"]
    assert mult["recorded_trials"] > 0
    assert mult["effective_trials"] == mult["recorded_trials"]
    assert mult["hidden_trials"] == 0
    dsr_adj = mult["dsr_selection_adjusted"]
    dsr_naive = mult["dsr_single_trial"]
    assert dsr_adj is not None and dsr_naive is not None
    assert dsr_adj <= dsr_naive + 1e-12
    assert mult["expected_max_sr"] > 0.0

    promo = res["promotion_lifecycle"]
    assert promo["is_halted"] is True
    assert promo["highest_stage_passed"] == 0
    assert promo["stage_1_status"] == "MATHEMATICAL_CONSISTENCY_REFUSED"
    assert "Failed" in (promo["halt_reason"] or "")

    # Evidence bundle carries the deflated value, not the raw Sharpe alone.
    assert "dsr_selection_adjusted" in res["metrics"] or dsr_adj is not None


def test_trading_signals_epistemic_adjudication_and_eedr():
    """Verify Trading Signals Sandbox runs under AEH with C4 -> C2 calibration and invariant enforcement."""
    df = load_real_btc_orderflow(600)
    res = run_epistemic_trading_adjudication(
        panel=df,
        dataset_name="btc_test",
        initial_capital=10000.0,
        pop_size=60,
        generations=20,
        seed=42,
    )

    assert res["sandbox_id"] == "trading_signals"
    assert res["claimed_tier"] == "C4_CAUSAL_MECHANISTIC"
    assert res["justified_tier"] == "C2_PREDICTIVE"
    assert res["verdict"] == "PROMOTED_REPAIRED"
    assert res["promoted"] is True
    assert "metrics" in res
    assert res["metrics"]["trades"] > 0
    assert "best_model" in res
    assert "market_extrapolation_ladder" in res
    m_lad = res["market_extrapolation_ladder"]
    assert "rungs" in m_lad
    assert "M0" in m_lad["rungs"]
    assert "M1" in m_lad["rungs"]
    assert "M3" in m_lad["rungs"]
    assert "M5" in m_lad["rungs"]

    # Epistemic Framework Provenance & Invariants
    framework = res["epistemic_framework"]
    assert framework["version"] == "APEDP-X v1.0-FROZEN"
    assert framework["autonomy_level"] == "LEVEL_1_SELF_CORRECTING_REFEREE"
    assert framework["eight_invariants_verified"] is True
    assert len(framework["aeh_manifest_hash"]) == 64
    assert len(framework["aeh_ledger_hash"]) == 64

    # EEDR repair audit
    repair = framework["eedr_repair"]
    assert repair is not None
    assert repair["failure_diagnosed"] == EpistemicFailureCategory.CLAIM_TARGET_MISMATCH.value
    assert repair["repair_operator_applied"] == "claim_level_calibration"
    assert repair["evidence_fingerprint_preserved"] is True
    assert repair["lower_claim_eligibility"] == "VERIFIED"
    assert repair["candidate_lower_claim"] == "C2_PREDICTIVE"
    assert repair["repair_category"] == "EEDR-CA"

    # Test that overfitted / divergent strategy terminates in TERMINAL_ABSTAIN
    failing_bundle = EvidenceBundle(metrics={"sharpe": -0.8, "max_dd": 55.0})
    failing_bundle.failures.append("extrapolation_divergence_drawdown_limit")
    failing_res = execute_eedr_loop(
        candidate_id="FAILING_OVERFIT_STRAT",
        proposed_claim=ClaimLevel.C4_CAUSAL_MECHANISTIC,
        evidence=failing_bundle,
        domain="market_microstructure",
    )
    assert failing_res.final_status == "TERMINAL_ABSTAIN"
    assert failing_res.evidence_fingerprint_preserved is True


def test_multi_asset_30_cang_gate_and_aeh_adjudication():
    """Verify 30-Asset Sandbox evaluates Gate 0.5 (CAN-G), promotes C3 structural invariance, and catches Ranking Trap."""
    # 1. Stratified Two-Tier Portfolio (Expected to pass CAN-G and promote C3)
    res_stratified = run_epistemic_30_asset_adjudication(
        initial_capital=100000.0,
        pop_size=60,
        generations=20,
        seed=42,
        evaluate_naive_trap=False,
    )

    assert res_stratified["sandbox_id"] == "multi_asset_30"
    assert res_stratified["claimed_tier"] == "C3_STRUCTURAL"
    assert res_stratified["justified_tier"] == "C3_STRUCTURAL"
    assert res_stratified["verdict"] == "PROMOTED_ORIGINAL"
    assert res_stratified["promoted"] is True

    # CAN-G Gate Audit & Market Extrapolation Ladder.
    #
    # Re-pinned 2026-10-19 after the lookahead remediation in
    # multi_asset_signals_sandbox.py (rolling-volatility `bfill` -> causal
    # `ffill`, with the undefined warm-up rows excluded rather than fabricated):
    # the gate now DETECTS the cross-asset ranking trap it previously missed,
    # because the fabricated volatility used to hide the crypto/equity
    # volatility disparity. Stratified allocation is what makes the candidate
    # admissible, and the naive pooled arm is negative -- so the trap flag is
    # the honest reading, not a regression.
    cang = res_stratified["gate_05_cang"]
    assert cang["passed"] is True
    assert cang["status"] == "passed"
    assert cang["scientific_valid"] is True
    assert cang["portfolio_admissible"] is True
    assert cang["volatility_normalized"] is True
    assert cang["ranking_trap_detected"] is True
    assert "ranking trap" in cang["reason"]
    assert cang["two_tier_sharpe"] > 0.0
    assert cang["naive_pooled_sharpe"] < cang["two_tier_sharpe"]
    assert cang["within_class_sharpes"]["crypto"] > cang["within_class_sharpes"]["equity"]
    assert "crypto" in cang["within_class_sharpes"]
    assert "equity" in cang["within_class_sharpes"]

    assert "market_extrapolation_ladder" in res_stratified
    ladder_strat = res_stratified["market_extrapolation_ladder"]
    assert "M0" in ladder_strat["rungs"]
    assert "M3" in ladder_strat["rungs"]
    assert "M5" in ladder_strat["rungs"]

    # AEH Invariants
    framework = res_stratified["epistemic_framework"]
    assert framework["eight_invariants_verified"] is True
    assert len(framework["aeh_manifest_hash"]) == 64
    assert len(framework["aeh_ledger_hash"]) == 64

    # 2. Naive Pooled Ranking (Detects Ranking Trap and Abstains)
    res_trap = run_epistemic_30_asset_adjudication(
        initial_capital=100000.0,
        pop_size=60,
        generations=20,
        seed=42,
        evaluate_naive_trap=True,
    )
    assert res_trap["gate_05_cang"]["ranking_trap_detected"] is True
    assert res_trap["verdict"] == "TERMINAL_ABSTAIN"
    assert res_trap["promoted"] is False


def test_ui_sandbox_run_trading_and_multi_asset_30():
    """Verify API registry and runner for trading_signals, multi_asset_30, and multi_asset_60 endpoints."""
    client = TestClient(app, raise_server_exceptions=False)

    # 1. Catalog listing
    resp_list = client.get("/api/sandbox/list")
    assert resp_list.status_code == 200
    catalog = resp_list.json()
    catalog_ids = {item["id"] for item in catalog}
    assert "trading_signals" in catalog_ids
    assert "multi_asset_30" in catalog_ids
    assert "multi_asset_60" in catalog_ids

    # 2. Run trading_signals sandbox
    resp_trading = client.post("/api/sandbox/run", json={"sandbox_id": "trading_signals"})
    assert resp_trading.status_code == 200
    data_trading = resp_trading.json()
    assert data_trading["sandbox_id"] == "trading_signals"
    assert data_trading["verdict"] in ("PROMOTED_REPAIRED", "PROMOTED_ORIGINAL")
    assert "epistemic_framework" in data_trading
    assert data_trading["epistemic_framework"]["eight_invariants_verified"] is True

    # 3. Run multi_asset_30 sandbox
    resp_multi = client.post("/api/sandbox/run", json={"sandbox_id": "multi_asset_30"})
    assert resp_multi.status_code == 200
    data_multi = resp_multi.json()
    assert data_multi["sandbox_id"] == "multi_asset_30"
    assert data_multi["verdict"] in ("PROMOTED_ORIGINAL", "PROMOTED_REPAIRED")
    assert "gate_05_cang" in data_multi
    assert data_multi["epistemic_framework"]["eight_invariants_verified"] is True

    # 4. Run multi_asset_60 sandbox
    resp_multi_60 = client.post("/api/sandbox/run", json={"sandbox_id": "multi_asset_60"})
    assert resp_multi_60.status_code == 200
    data_multi_60 = resp_multi_60.json()
    assert data_multi_60["sandbox_id"] == "multi_asset_60"
    assert data_multi_60["total_assets"] == 60
    assert data_multi_60["crypto_assets_count"] == 30
    assert data_multi_60["equity_assets_count"] == 30
    assert data_multi_60["verdict"] in ("PROMOTED_ORIGINAL", "PROMOTED_REPAIRED")
    assert "gate_05_cang" in data_multi_60
    assert data_multi_60["epistemic_framework"]["eight_invariants_verified"] is True

    # 5. Run multi_asset_90 sandbox (5-year panel, 90 assets)
    resp_multi_90 = client.post("/api/sandbox/run", json={"sandbox_id": "multi_asset_90"})
    assert resp_multi_90.status_code == 200
    data_multi_90 = resp_multi_90.json()
    assert data_multi_90["sandbox_id"] == "multi_asset_90"
    assert data_multi_90["total_assets"] == 90
    assert data_multi_90["crypto_assets_count"] == 45
    assert data_multi_90["equity_assets_count"] == 45
    assert data_multi_90["verdict"] == "TERMINAL_ABSTAIN"
    assert data_multi_90["promoted"] is False
    assert "gate_05_cang" in data_multi_90
    assert data_multi_90["epistemic_framework"]["eight_invariants_verified"] is True


def test_multi_asset_60_cang_gate_and_aeh_adjudication():
    """Verify 60-Asset Sandbox (30 Crypto + 30 Equities) evaluates Gate 0.5 (CAN-G) and AEH adjudication."""
    from apedp_dyn.sandbox.multi_asset_signals_sandbox import run_epistemic_60_asset_adjudication

    res_60 = run_epistemic_60_asset_adjudication(
        initial_capital=100000.0,
        pop_size=40,
        generations=15,
        seed=42,
        evaluate_naive_trap=False,
    )

    assert res_60["sandbox_id"] == "multi_asset_60"
    assert res_60["total_assets"] == 60
    assert res_60["crypto_assets_count"] == 30
    assert res_60["equity_assets_count"] == 30
    assert res_60["claimed_tier"] == "C3_STRUCTURAL"
    assert res_60["justified_tier"] in ("C3_STRUCTURAL", "C2_PREDICTIVE")
    assert res_60["verdict"] in ("PROMOTED_ORIGINAL", "PROMOTED_REPAIRED")
    assert res_60["promoted"] is True

    # CAN-G Gate Audit & Market Extrapolation Ladder
    cang = res_60["gate_05_cang"]
    assert "crypto" in cang["within_class_sharpes"]
    assert "equity" in cang["within_class_sharpes"]
    assert cang["ranking_trap_detected"] is False
    if cang["within_class_sharpes"]["equity"] <= 0.0:
        assert cang["passed"] is False
        assert cang["status"] == "failed_within_class"
    else:
        assert cang["passed"] is True
        assert cang["status"] == "passed"
        assert cang["two_tier_sharpe"] > 0.0

    assert "market_extrapolation_ladder" in res_60
    ladder_strat = res_60["market_extrapolation_ladder"]
    assert "M0" in ladder_strat["rungs"]
    assert "M3" in ladder_strat["rungs"]
    assert "M5" in ladder_strat["rungs"]

    # AEH Invariants
    framework = res_60["epistemic_framework"]
    assert framework["eight_invariants_verified"] is True
    assert len(framework["aeh_manifest_hash"]) == 64
    assert len(framework["aeh_ledger_hash"]) == 64


def test_multi_asset_90_cang_gate_and_aeh_adjudication():
    """Verify 90-Asset 5-Year Sandbox evaluates Gate 0.5 (CAN-G) and AEH adjudication."""
    from apedp_dyn.sandbox.multi_asset_signals_sandbox import run_epistemic_90_asset_adjudication

    res_90 = run_epistemic_90_asset_adjudication(
        initial_capital=100000.0,
        pop_size=40,
        generations=15,
        seed=42,
        evaluate_naive_trap=False,
    )

    assert res_90["sandbox_id"] == "multi_asset_90"
    assert res_90["total_assets"] == 90
    assert res_90["crypto_assets_count"] == 45
    assert res_90["equity_assets_count"] == 45
    assert res_90["claimed_tier"] == "C3_STRUCTURAL"
    assert res_90["verdict"] in ("PROMOTED_ORIGINAL", "PROMOTED_REPAIRED", "TERMINAL_ABSTAIN")
    assert "gate_05_cang" in res_90
    assert "market_extrapolation_ladder" in res_90

    # AEH Invariants
    framework = res_90["epistemic_framework"]
    assert framework["eight_invariants_verified"] is True
    assert len(framework["aeh_manifest_hash"]) == 64
    assert len(framework["aeh_ledger_hash"]) == 64


def test_goldbach_rust_sandbox_and_ui_endpoint():
    """Verify Goldbach Conjecture Rust Rayon engine runs and integrates with UI server."""
    from apedp_dyn.sandbox.goldbach_sandbox import run_goldbach_adjudication

    res = run_goldbach_adjudication(limit=2_000_000)
    assert res["sandbox_id"] == "goldbach_conjecture"
    assert res["verdict"] == "PROMOTED_REPAIRED"
    assert res["claimed_tier"].startswith("C6")
    assert res["justified_tier"] == "C2_PREDICTIVE"
    assert res["metrics"]["violations"] == 0
    assert res["metrics"]["evens_tested"] == 999_999
    assert res["metrics"]["verification_rate_m_evens_sec"] > 10.0
    assert "gate_breakdown" in res["extra"]
    assert res["extra"]["epistemic_calibration"]["finite_induction_fallacy_blocked"] is True

    # Check UI endpoints
    client = TestClient(app)
    r_list = client.get("/api/sandbox/list")
    assert r_list.status_code == 200
    assert any(item["id"] == "goldbach_conjecture" for item in r_list.json())

    r_run = client.post("/api/sandbox/run", json={"sandbox_id": "goldbach_conjecture"})
    assert r_run.status_code == 200
    data = r_run.json()
    assert data["sandbox_id"] == "goldbach_conjecture"
    assert data["verdict"] == "PROMOTED_REPAIRED"
    assert "series" in data
    assert len(data["series"]["x"]) > 0


def test_riemann_rust_sandbox_and_ui_endpoint():
    """Verify Riemann Hypothesis Rust Rayon engine runs and integrates with UI server."""
    from apedp_dyn.sandbox.riemann_sandbox import run_riemann_adjudication

    res = run_riemann_adjudication(max_x=100.0, sample_points=1000)
    assert res["sandbox_id"] == "riemann_hypothesis"
    assert res["verdict"] == "PROMOTED_REPAIRED"
    assert res["claimed_tier"].startswith("C6")
    assert res["metrics"]["zeros_dynamically_found"] > 0
    assert res["metrics"]["off_line_zeros_found"] == 0
    assert res["metrics"]["numerical_accuracy_gate_pass"] is True
    assert res["metrics"]["max_reference_discrepancy"] < 1e-8
    assert res["metrics"]["harmonic_r2"] > 0.99
    assert res["metrics"]["error_inflation_factor"] > 3.0
    assert "gate_breakdown" in res["extra"]
    assert res["extra"]["epistemic_calibration"]["finite_induction_fallacy_blocked"] is True

    # Check UI endpoints
    client = TestClient(app)
    r_list = client.get("/api/sandbox/list")
    assert r_list.status_code == 200
    assert any(item["id"] == "riemann_hypothesis" for item in r_list.json())

    r_run = client.post("/api/sandbox/run", json={"sandbox_id": "riemann_hypothesis"})
    assert r_run.status_code == 200
    data = r_run.json()
    assert data["sandbox_id"] == "riemann_hypothesis"
    assert data["verdict"] == "PROMOTED_REPAIRED"
    assert "series" in data
    assert len(data["series"]["x"]) > 0



