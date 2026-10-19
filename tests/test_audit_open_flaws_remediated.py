"""Regression tests for the four open fault areas identified in the external re-audit:

Fault 1: Rank ablation – reimplement feature-rank ablation with live executable code path.
Fault 5: Stale freezes – make drift fail loudly (verify_all_frozen_records_or_raise & provenance check).
Fault 6: Missing-data masking – mandate mask-first path (invalid data strictly forces flat with reason).
Fault 7: Non-binding thresholds – binding check asserts active fraction in [5%, 90%], else GATE_NON_BINDING.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from apedp_dyn.epistemic.frozen_spec import (
    PINNED_FROZEN_REGISTRY,
    FreezeDriftError,
    MissingProvenanceError,
    require_table_provenance_record,
    verify_all_frozen_records_or_raise,
)
from apedp_dyn.sandbox.canonical_ledger import (
    assert_mask_first_compliance,
    assert_threshold_binding,
    run_feature_rank_ablation,
    run_ledger,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


# ── Fault 1: Feature-Rank Ablation Live Execution ───────────────────────────

def test_feature_rank_ablation_live_code_path():
    """Verify feature-rank ablation executes live, comparing ranked vs unranked feature."""
    np.random.seed(42)
    n = 200
    prices = 100.0 * np.cumprod(1.0 + 0.01 * np.random.randn(n))
    volumes = np.abs(np.random.randn(n)) * 1000.0 + 50.0

    df = pd.DataFrame({
        "price": prices,
        "V": volumes,
        "taker_buy": volumes * 0.52,
    })
    data = {"BTC": df}
    weights = {"H-BB": 0.6, "H-RV": 0.4}

    res = run_feature_rank_ablation(
        syms=["BTC"],
        data=data,
        weights=weights,
        rule="base",
        th_in=0.0,
        th_out=0.0,
        eval_slice=slice(50, n),
    )

    # Must contain both arms and explicit deltas
    assert "ranked_summary" in res
    assert "unranked_summary" in res
    assert "delta_net_return" in res
    assert "delta_turnover" in res
    assert "delta_sharpe" in res
    assert isinstance(res["is_noop"], bool)
    assert res["ablation_verdict"] in ("FEATURE_RANK_INFORMATIVE", "FEATURE_RANK_EQUIVALENT")

    # Verify both arms executed non-empty ledgers
    assert res["ranked_summary"]["n_obs"] > 0
    assert res["unranked_summary"]["n_obs"] > 0
    assert res["ranked_summary"]["turnover"] >= 0.0
    assert res["unranked_summary"]["turnover"] >= 0.0


# ── Fault 5: Stale Freezes Fail Loudly ──────────────────────────────────────

def test_all_frozen_records_verified_against_pinned_hashes():
    """All registered frozen files in repository must exactly match their pinned SHA256 digests."""
    verified = verify_all_frozen_records_or_raise()
    assert len(verified) == len(PINNED_FROZEN_REGISTRY)
    for path, h in verified.items():
        assert len(h) == 64
        assert h == PINNED_FROZEN_REGISTRY[path]


def test_freeze_drift_fails_loudly_on_mutation(monkeypatch):
    """Mutation test: Any drift in a frozen file must raise FreezeDriftError immediately."""
    tampered_registry = dict(PINNED_FROZEN_REGISTRY)
    tampered_registry["outputs/trading_np93_freeze.json"] = "0000000000000000000000000000000000000000000000000000000000000000"

    monkeypatch.setattr("apedp_dyn.epistemic.frozen_spec.PINNED_FROZEN_REGISTRY", tampered_registry)

    with pytest.raises(FreezeDriftError, match="CRITICAL FREEZE DRIFT DETECTED"):
        verify_all_frozen_records_or_raise()


def test_ledger_table_requires_provenance_make_record():
    """Any table entering the ledger must have valid provenance/make_record.py artifacts."""
    prov_dir = REPO_ROOT / "provenance" / "selection_stability_and_ablation_20261009"
    assert prov_dir.is_dir()

    manifest = require_table_provenance_record(
        table_id="SELECTION_STABILITY_BLOCK_B",
        record_dir=prov_dir,
    )
    assert manifest["returncode"] == 0
    assert "script_hashes" in manifest

    # Missing directory must fail loudly
    with pytest.raises(MissingProvenanceError, match="cannot enter ledger: missing"):
        require_table_provenance_record("UNATTESTED_TABLE", REPO_ROOT / "non_existent_prov_dir")


# ── Fault 6: Missing-Data Masking Mandate ───────────────────────────────────

def test_missing_data_strictly_forces_flat_with_reason():
    """Invalid input data (NaN/inf) must strictly force flat positions with reason='invalid_score'."""
    np.random.seed(42)
    n = 100
    prices = 100.0 * np.cumprod(1.0 + 0.01 * np.random.randn(n))
    volumes = np.abs(np.random.randn(n)) * 1000.0 + 50.0

    # Corrupt bars at t=60, 61, 75
    prices[60] = np.nan
    prices[61] = np.inf
    volumes[75] = np.nan

    df = pd.DataFrame({"price": prices, "V": volumes})
    data = {"TEST": df}
    weights = {"H-BB": 1.0}

    rows, summary = run_ledger(
        syms=["TEST"],
        data=data,
        weights=weights,
        rule="base",
        th_in=0.0,
        th_out=0.0,
        eval_slice=slice(50, n),
    )

    # Check that all invalid rows are strictly flat
    invalid_rows = [r for r in rows if not r["valid"]]
    assert len(invalid_rows) > 0
    for r in invalid_rows:
        assert r["pos"] == 0.0
        assert r["reason"] == "invalid_score"

    assert summary["n_invalid"] == len(invalid_rows)


def test_mask_first_compliance_assertion_detects_violations():
    """Mutation test: If an invalid bar produces a non-zero position, compliance check must raise."""
    corrupted_rows = [
        {"asset": "A", "t": 10, "valid": False, "pos": 1.0, "reason": "base_rule"},
    ]
    with pytest.raises(ValueError, match="MASK-FIRST COMPLIANCE VIOLATION"):
        assert_mask_first_compliance(corrupted_rows, raise_on_violation=True)


# ── Fault 7: Non-Binding Threshold Assertion ───────────────────────────────

def test_binding_threshold_passes_assertion():
    """Active fraction within [5%, 90%] is deemed binding and reports Sharpe."""
    # 50% active positions
    rows = [{"pos": -1.0 if i % 2 == 0 else 0.0} for i in range(100)]
    is_binding, active_frac, status = assert_threshold_binding(rows)
    assert is_binding is True
    assert active_frac == 0.50
    assert status == "GATE_BINDING"


def test_decorative_high_threshold_reports_gate_non_binding():
    """Threshold too strict (< 5% trades active) reports GATE_NON_BINDING and suppresses Sharpe."""
    # Only 1% active positions (threshold theta >> |S|)
    rows = [{"pos": -1.0 if i == 0 else 0.0} for i in range(100)]
    is_binding, active_frac, status = assert_threshold_binding(rows)
    assert is_binding is False
    assert active_frac == 0.01
    assert status == "GATE_NON_BINDING"


def test_decorative_low_threshold_reports_gate_non_binding():
    """Threshold too loose (> 90% trades active) reports GATE_NON_BINDING and suppresses Sharpe."""
    # 98% active positions (threshold theta << |S|)
    rows = [{"pos": -1.0 if i < 98 else 0.0} for i in range(100)]
    is_binding, active_frac, status = assert_threshold_binding(rows)
    assert is_binding is False
    assert active_frac == 0.98
    assert status == "GATE_NON_BINDING"


def test_run_ledger_decorative_threshold_suppresses_sharpe():
    """In run_ledger, an extreme non-binding threshold must report GATE_NON_BINDING and sharpe=None."""
    np.random.seed(42)
    n = 100
    prices = 100.0 * np.cumprod(1.0 + 0.01 * np.random.randn(n))
    volumes = np.abs(np.random.randn(n)) * 1000.0 + 50.0
    df = pd.DataFrame({"price": prices, "V": volumes})
    data = {"TEST": df}
    weights = {"H-BB": 1.0}

    # Extremely high threshold: th_in = 999.0 (never binds)
    _rows, summary = run_ledger(
        syms=["TEST"],
        data=data,
        weights=weights,
        rule="hyst",
        th_in=999.0,
        th_out=999.0,
        eval_slice=slice(50, n),
        check_binding=True,
    )

    assert summary["status"] == "GATE_NON_BINDING"
    assert summary["gate_binding"] is False
    assert summary["sharpe"] is None
    assert "Gate is non-binding/decorative" in summary["notes"]



def test_rime_ad_dag_candidate_single_application_of_slope_intercept():
    """Verify RimeAdProposer._make_dag_candidate does not apply slope/intercept twice."""
    from apedp_dyn.proposers.rime_ad_proposer import RimeAdProposer

    proposer = RimeAdProposer()
    item = {
        "formula": "2.5000 * cos(feat) + 1.2000",
        "slope": 2.5,
        "intercept": 1.2,
        "r2": 1.0,
    }
    cand = proposer._make_dag_candidate(item, "feat")
    assert cand is not None
    assert cand.formula == "2.5000 * cos(feat) + 1.2000"

    # Evaluate on test values
    xv = np.linspace(0.1, 2.0, 50)
    expected_y = 2.5 * np.cos(xv) + 1.2
    df = pd.DataFrame({"feat": xv})
    computed_y = cand.compute_score(df)
    max_err = float(np.max(np.abs(computed_y - expected_y)))
    assert max_err < 1e-4, f"max error {max_err} indicates slope/intercept was applied twice!"


def test_rime_implicit_circle_max_degree_3_minimal_relation():
    """Verify RIME implicit on a circle with max_degree=3 recovers minimal x^2+y^2-1=0, not xy(x^2+y^2-1)=0."""
    import json

    import apedp_kernel as ak

    theta = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    x = np.cos(theta)
    y = np.sin(theta)
    cfg = {
        "max_degree": 3,
        "include_cross_terms": True,
        "include_rational_products": False,
        "sparsity_threshold": 0.05,
        "rcond_threshold": 1e-12,
    }
    res = json.loads(ak.rime_implicit_discover_json(json.dumps(x.tolist()), json.dumps(y.tolist()), json.dumps(cfg)))
    assert res.get("algebraic_rmse", 1.0) < 1e-8
    assert res.get("is_exact_recovery", False) is True
    assert res.get("num_active_terms") == 3
    formula = res.get("formula", "")
    assert "x*y" not in formula, f"Formula must be minimal, got non-minimal: {formula}"


def test_hawkes_orderflow_dynamic_on_regular_bars():
    """Verify HawkesOrderFlowProposer emits dynamic non-constant signals on regular timestamped bars."""
    from apedp_dyn.proposers.hawkes_orderflow_proposer import HawkesOrderFlowProposer

    ts = np.arange(100, dtype=float)
    vol = 1000.0 + 500.0 * np.sin(np.linspace(0, 10, 100))
    df = pd.DataFrame({"timestamp": ts, "volume": vol})

    proposer = HawkesOrderFlowProposer()
    props = proposer.fit_and_propose(event_data=df)
    assert len(props) > 0
    score = props[0].compute_score(df)
    assert score.shape == (100,)
    assert float(np.std(score)) > 0.0, "Hawkes must not emit a constant [0.] signal on regular bars"


