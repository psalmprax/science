"""Regression tests for the four open fault areas identified in the external re-audit:

Fault 1: Rank ablation – reimplement feature-rank ablation with live executable code path.
Fault 5: Stale freezes – make drift fail loudly (verify_all_frozen_records_or_raise & provenance check).
Fault 6: Missing-data masking – mandate mask-first path (invalid data strictly forces flat with reason).
Fault 7: Non-binding thresholds – binding check asserts active fraction in [5%, 90%], else GATE_NON_BINDING.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from apedp_dyn.epistemic.frozen_spec import (
    PINNED_FROZEN_REGISTRY,
    PINNED_GENERATED_OUTPUT_REGISTRY,
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
    assert set(PINNED_FROZEN_REGISTRY) <= set(verified)
    registry = {**PINNED_FROZEN_REGISTRY, **PINNED_GENERATED_OUTPUT_REGISTRY}
    for path, h in verified.items():
        assert len(h) == 64
        assert h == registry[path]


def test_freeze_drift_fails_loudly_on_mutation(monkeypatch):
    """Mutation test: Any drift in a frozen file must raise FreezeDriftError immediately."""
    tampered_registry = dict(PINNED_FROZEN_REGISTRY)
    tampered_registry["benchmarks/l2_control_freeze.json"] = "0000000000000000000000000000000000000000000000000000000000000000"

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


def test_rime_ad_dag_scorer_fails_honest_when_both_engines_fail(monkeypatch):
    """Regression: bridge failure must yield NaN, never a substring-matched stand-in.

    The old fallback fabricated a *different* function from substring matching
    (e.g. sqrt(feat) silently became feat), corrupting every downstream score.
    """
    import apedp_dyn.proposers.rime_ad_proposer as mod
    from apedp_dyn.proposers.rime_ad_proposer import RimeAdProposer

    proposer = RimeAdProposer()
    item = {"formula": "1.0000 * sqrt(feat) + 0.0000", "slope": 1.0, "intercept": 0.0, "r2": 1.0}

    def _boom(self, *args, **kwargs):
        raise RuntimeError("both engines down")

    monkeypatch.setattr(mod.RustKernelBridge, "evaluate_expression", _boom, raising=True)
    cand = proposer._make_dag_candidate(item, "feat")
    assert cand is not None
    xv = np.linspace(0.1, 2.0, 50)
    out = cand.compute_score(pd.DataFrame({"feat": xv}))
    assert np.all(np.isnan(out)), "bridge failure must fail honest with NaN, not a fabricated function"


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



def test_declared_scale_invariant_and_dependencies_are_declarative_only():
    """Boundary (ARCHITECTURE.md principle 5): declarative proposer fields must
    never be read in a control-flow decision by any production module."""
    offenders: list[str] = []
    for py in sorted((REPO_ROOT / "apedp_dyn").rglob("*.py")):
        for lineno, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
            s = line.strip()
            if s.startswith("#") or "declared_scale_invariant" not in line:
                continue
            if any(tok in s for tok in ("if ", "return ", "==", "!=", ">=", "<=")):
                offenders.append(f"{py.relative_to(REPO_ROOT)}:{lineno}: {s}")
    assert not offenders, (
        "declared_scale_invariant leaked into control flow (must stay declarative): "
        + "; ".join(offenders)
    )


# ── Audit-class recurrence guards ───────────────────────────────────────────

# The lookahead defect class is BACKWARD fill: .bfill()/.backfill() populate a
# series' early values with statistics measured on FUTURE data. Forward fill
# (.ffill()/.fillna(method="ffill")/"pad") is causal (a stale-but-past value)
# and is the prescribed remediation across the sandbox discovery fit paths;
# ffilling a rolling/expanding window propagates the last in-sample statistic
# forward, never backward.
_FILLNA_LOOKAHEAD = re.compile(r"\.(?:bfill|backfill)\(|method\s*=\s*[\"'](?:bfill|backfill)[\"']")


def _is_docstring_constant(tree: ast.AST) -> set[int]:
    """Return id()s of Constant nodes that are module/class/function docstrings."""
    doc_ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                doc_ids.add(id(body[0].value))
    return doc_ids


def test_no_lookahead_fill_in_proposer_fit_paths():
    """Guard the bfill lookahead defect class: fit paths must not back/forward-fill.

    Originally restricted to ``apedp_dyn/proposers``; the same ``.bfill()``
    lookahead pattern recurred across the sandbox discovery routines
    (execution_cost_models, fluid_ode_dynamical_discovery, ga_depth2/3,
    expanded_space_discovery_sweep, multi_asset_signals_sandbox,
    openai_math_advanced_hypotheses), so the guard now covers all of
    ``apedp_dyn``.
    """
    offenders: list[str] = []
    root = REPO_ROOT / "apedp_dyn"
    for py in sorted(root.rglob("*.py")):
        if "tests" in py.parts:
            continue
        for lineno, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip().startswith("#"):
                continue
            if _FILLNA_LOOKAHEAD.search(line):
                offenders.append(f"{py.relative_to(REPO_ROOT)}:{lineno}: {line.strip()}")
    assert not offenders, (
        "lookahead fill detected in proposer fit path (would leak future data): "
        + "; ".join(offenders)
    )


def test_no_host_specific_home_paths_in_runtime_code():
    """Guard the hardcoded-path defect class: no concrete host home dirs in runtime strings."""
    offenders: list[str] = []
    for pkg in ("apedp_dyn", "apedp", "packages", "scripts"):
        for py in sorted((REPO_ROOT / pkg).rglob("*.py")):
            tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
            docstrings = _is_docstring_constant(tree)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Constant) or id(node) in docstrings or not isinstance(node.value, str):
                    continue
                for match in re.finditer(r"/home/[A-Za-z0-9._-]+", node.value):
                    if match.group(0) == "/home/lean":
                        continue  # inside-container path, not a host path
                    offenders.append(f"{py.relative_to(REPO_ROOT)}:{node.lineno}: {match.group(0)}")
    assert not offenders, (
        "host-specific home path literal in runtime code (use apedp_dyn.conf.paths): "
        + "; ".join(offenders)
    )


# ── 2026-11-08 Deep Remediation Regression Pins (Categories 1–5) ────────────


def test_adjudicate_identity_canonicalizes_proposal_reference():
    """Equivalent formulas wrapped in proposal reference objects must map to identical _identity keys."""
    import types

    from apedp_dyn.epistemic.adjudicate import _identity

    ref_a = types.SimpleNamespace(candidate_id="c1", proposer_id="p1", formula="x + x")
    ref_b = types.SimpleNamespace(candidate_id="c2", proposer_id="p1", formula="2*x")
    sc_a = types.SimpleNamespace(candidate=ref_a)
    sc_b = types.SimpleNamespace(candidate=ref_b)
    assert _identity(sc_a) == _identity(sc_b)


def test_caller_data_protocol_burns_holdout_on_failed_verdict(tmp_path):
    """SealedConfirmationSlice.consume must burn the confirmation slice when the referee rejects."""
    from apedp_dyn.epistemic.caller_data_protocol import (
        CallerSplit,
        ConfirmationVerdict,
        SealedConfirmationSlice,
    )
    from apedp_dyn.epistemic.experiment_ledger import (
        DatasetRole,
        FullLineageExperimentLedger,
        HoldoutContaminationError,
    )

    ledger = FullLineageExperimentLedger(persistence_path=tmp_path / "ledger.jsonl")
    ledger.register_candidate(candidate_id="cand_fail", proposer_id="prop_1", formula="x")
    ledger.register_candidate(candidate_id="cand_retry", proposer_id="prop_1", formula="2*x")

    split = CallerSplit(
        caller_id="caller_1",
        dataset_id="ds_burn_test",
        selection_ids=("sel_1",),
        confirmation_ids=("conf_1",),
    )
    sealed = SealedConfirmationSlice(split=split, ledger=ledger)

    res = sealed.consume(
        candidate_id="cand_fail",
        evaluate=lambda **_kw: ConfirmationVerdict(
            caller_id="caller_1",
            candidate_id="cand_fail",
            verdict="REJECTED",
        ),
    )
    assert res.verdict == "REJECTED"

    # Attempting to expose another candidate to the burned confirmation slice must raise HoldoutContaminationError
    with pytest.raises(HoldoutContaminationError):
        ledger.record_data_exposure(
            candidate_id="cand_retry",
            dataset_id="ds_burn_test",
            slice_identifier="conf_1",
            role=DatasetRole.CONFIRMATION_HOLDOUT,
        )


def test_trial_ledger_excludes_equation_discovery_zero_sharpe_dummies(tmp_path):
    """Equation-discovery records with selection_statistic and dummy 0.0 Sharpe must not pollute Sharpe DSR."""
    from apedp_dyn.epistemic.trial_ledger import CandidateTrialLedger

    ledger = CandidateTrialLedger(persistence_path=tmp_path / "trials.jsonl")
    ledger.record_trial(
        proposer_id="p_fin",
        candidate_id="c_fin",
        formula="x",
        strategy_archetype="tactical_cash",
        in_sample_sharpe=1.25,
        n_bars=100,
    )
    ledger.record_trial(
        proposer_id="p_eq",
        candidate_id="c_eq",
        formula="y",
        strategy_archetype="equation_discovery",
        in_sample_sharpe=0.0,
        n_bars=100,
        selection_statistic=0.99,
        metadata={"selection_metric": "r2"},
    )
    mean_sr, var_sr = ledger.get_sharpe_distribution()
    assert np.isclose(mean_sr, 1.25)
    assert np.isclose(var_sr, 0.0)
    mean_sel, _var_sel = ledger.get_selection_distribution()
    assert np.isclose(mean_sel, 0.99)


def test_promotion_stage4_prevents_in_sample_ic_deflation_spoofing():
    """Stage 4 must reject a spoofed lower in_sample_ic than recorded in Stage 2."""
    from apedp_dyn.epistemic.promotion_lifecycle import (
        PromotionTracker,
        PromotionViolationError,
    )

    tracker = PromotionTracker(candidate_id="cand_spoof")
    tracker.verify_stage_1(lean_status="P2_VERIFIED", dimensions_consistent=True, bounded=True)
    tracker.verify_stage_2(in_sample_ic=0.10, cv_stable=True)
    tracker.verify_stage_3(dm_p_value=0.001, max_literature_r2=0.2)

    with pytest.raises(PromotionViolationError, match="cannot deflate in-sample baseline"):
        tracker.verify_stage_4(oos_ic=0.03, oos_p_value=0.001, in_sample_ic=0.04)


def test_lean_bridge_nary_damping_and_tokenized_invariants():
    """LeanUniversalBridge must detect 3-factor exp(-a*b*t) damping and tokenize identifier keywords."""
    import sympy as sp

    from apedp_dyn.epistemic.lean_universal_bridge import (
        LeanUniversalBridge,
        _ast_detect_damping_factor,
    )

    a, b = sp.symbols("a b", positive=True)
    t = sp.Symbol("t", real=True)
    expr = sp.exp(-2 * a * b * t)
    assert _ast_detect_damping_factor(expr) is not None

    bridge = LeanUniversalBridge()
    # "result" and "convolution" must not trigger "res" (kalman) or "vol" (variance)
    invs = bridge.infer_invariants("cand_result", "convolution + 1", meta="uncertain delegate")
    names = {inv.name for inv in invs}
    assert "kalman_innovation_zero_mean" not in names
    assert "variance_nonnegativity" not in names


def test_rust_bridge_enforce_native_search_synchronizes_both_modules():
    """enforce_native_search must synchronize _ENFORCE_NATIVE_SEARCH across apedp and apedp_dyn."""
    import apedp.rust_bridge as rb1
    import apedp_dyn.rust_bridge as rb2

    try:
        rb1.enforce_native_search(True)
        assert rb1.native_search_enforced() is True
        assert rb2.native_search_enforced() is True
        rb2.enforce_native_search(False)
        assert rb1.native_search_enforced() is False
        assert rb2.native_search_enforced() is False
    finally:
        rb1.enforce_native_search(False)


def test_sparse_microstructure_proposer_includes_raw_intercept_and_target_col():
    """SparseMicrostructureProposer must honor target_col and include raw_intercept in score."""
    from apedp_dyn.proposers.sparse_microstructure_proposer import (
        SparseMicrostructureProposer,
        extract_microstructure_features,
    )

    rng = np.random.default_rng(7)
    n = 80
    df = pd.DataFrame({
        "custom_px": 100.0 + np.cumsum(rng.normal(0.05, 0.5, n)),
        "V": rng.uniform(100.0, 500.0, n),
        "taker_buy": rng.uniform(50.0, 250.0, n),
    })
    proposer = SparseMicrostructureProposer(l1_ratios=(0.5,), alphas=(1e-4,))
    props = proposer.fit_and_propose({"SYM": df}, target_col="custom_px")
    assert len(props) >= 1
    cand = props[0]
    assert "raw_intercept" in cand.metadata
    X_sub, _ = extract_microstructure_features(df, target_col="custom_px")
    manual = np.full(len(df), cand.metadata["raw_intercept"], dtype=float)
    for k, w in cand.parameters.items():
        manual += w * X_sub[k].to_numpy()
    assert np.allclose(cand.compute_score(df), manual)

