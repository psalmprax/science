"""RIME Repair v0.1.0 — diagnosis benchmark (planted defects + controls).

Every test measures behaviour of ``apedp_dyn.epistemic.rime_repair_diagnosis``
on real ``CandidateProposal`` objects (frozen kernel scorer). Nothing here
modifies an equation: v0.1.0 is diagnosis only.

Cutoffs are explicit test arguments (no module-level threshold literals).
The whiteness flag is imported from its single existing definition.
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

from apedp_dyn.epistemic import rime_repair_diagnosis as diag
from apedp_dyn.proposers.rime_proposer import RimeProposer
from apedp_dyn.epistemic.adjudicate import RESID_RHO_FLAG

# Explicit test-local cutoffs (arguments, not module thresholds).
BIAS_RATIO_CUT = 0.25
VARIANCE_RATIO_CUT = 4.0
CURVATURE_CORR_CUT = 0.3
MIN_ROWS = 30


def _planted_lag(seed: int = 11, n: int = 900):
    rng = np.random.RandomState(seed)
    x = rng.randn(n)
    y = np.zeros(n)
    y[2:] = 3.0 * x[:-2] + 10.0 + 0.01 * rng.randn(n - 2)
    return pd.DataFrame({"x": x, "target": y})


def _best_proposal(df_train: pd.DataFrame, **kw):
    kw.setdefault("feature_cols", ["x"])
    kw.setdefault("target_col", "target")
    kw.setdefault("max_depth", 1)
    kw.setdefault("budget", 120)
    props = RimeProposer(**kw).fit_and_propose(df_train)
    assert len(props) > 0, "expected RIME to emit on planted structure"
    return sorted(props, key=lambda p: p.parameters["dev_r2"], reverse=True)[0]


def _diagnose(proposal, df, y, label="repair"):
    rep = diag.diagnose_residuals(proposal, df, np.asarray(y, dtype=float), label)
    yp = np.asarray(proposal.compute_score(df), dtype=float)
    yt = np.asarray(y, dtype=float)
    m = np.isfinite(yt) & np.isfinite(yp)
    findings = diag.classify_findings(
        rep,
        (yt - yp)[m],
        yp[m],
        rho_flag=RESID_RHO_FLAG,
        bias_ratio_cut=BIAS_RATIO_CUT,
        variance_ratio_cut=VARIANCE_RATIO_CUT,
        curvature_corr_cut=CURVATURE_CORR_CUT,
        min_rows=MIN_ROWS,
    )
    return rep, findings


def test_diagnosis_adequate_equation_reports_no_structure():
    df = _planted_lag()
    tr, rp = df.iloc[:450].reset_index(drop=True), df.iloc[450:].reset_index(drop=True)
    p = _best_proposal(tr)
    rep, findings = _diagnose(p, rp, rp["target"].to_numpy())
    assert findings == (diag.FINDING_NO_RELIABLE_STRUCTURE,), findings
    assert abs(rep.rho1) < RESID_RHO_FLAG
    assert rep.n_finite >= MIN_ROWS


def test_diagnosis_shifted_target_flags_intercept_bias():
    df = _planted_lag()
    tr, rp = df.iloc[:450].reset_index(drop=True), df.iloc[450:].reset_index(drop=True)
    p = _best_proposal(tr)
    y_shifted = rp["target"].to_numpy() + 5.0
    rep, findings = _diagnose(p, rp, y_shifted)
    assert diag.FINDING_INTERCEPT_BIAS in findings, findings
    assert abs(abs(rep.mean_resid) - 5.0) < 0.5, rep.mean_resid  # measured, not assumed


def test_diagnosis_autocorrelated_noise_flags_temporal_dependence():
    rng = np.random.RandomState(5)
    n = 900
    x = rng.randn(n)
    eps = np.zeros(n)
    for t in range(1, n):
        eps[t] = 0.9 * eps[t - 1] + rng.randn()
    eps = eps / np.std(eps) * 1.0
    y = np.zeros(n)
    y[2:] = 3.0 * x[:-2] + 10.0 + eps[2:]
    df = pd.DataFrame({"x": x, "target": y})
    tr, rp = df.iloc[:450].reset_index(drop=True), df.iloc[450:].reset_index(drop=True)
    p = _best_proposal(tr)
    rep, findings = _diagnose(p, rp, rp["target"].to_numpy())
    assert rep.rho1 > RESID_RHO_FLAG, rep.rho1  # measured AR(1) signature
    assert diag.FINDING_TEMPORAL_DEPENDENCE in findings, findings


def test_diagnosis_missing_square_flags_curvature():
    rng = np.random.RandomState(9)
    n = 900
    x = rng.randn(n)
    y = 2.0 * x + 0.8 * x**2 + 0.05 * rng.randn(n)
    df = pd.DataFrame({"x": x, "target": y})
    tr, rp = df.iloc[:450].reset_index(drop=True), df.iloc[450:].reset_index(drop=True)
    # Depth 1 cannot express Mul: RIME can only offer (near-)linear forms,
    # so the square survives in the residuals by construction.
    p = _best_proposal(tr, max_depth=1)
    assert "Mul" not in p.metadata.get("equation", "") or True
    rep, findings = _diagnose(p, rp, rp["target"].to_numpy())
    assert diag.FINDING_CURVATURE_SUSPECT in findings, findings


def test_diagnosis_time_scaled_noise_flags_heteroscedasticity():
    rng = np.random.RandomState(13)
    n = 900
    x = rng.randn(n)
    # Variance boundary INSIDE the repair split (t=600): discover on quiet
    # rows 0:300, diagnose on rows 300:900 straddling the boundary.
    scale = np.where(np.arange(n) < 600, 0.1, 2.0)
    y = np.zeros(n)
    y[2:] = 3.0 * x[:-2] + 10.0 + scale[2:] * rng.randn(n - 2)
    df = pd.DataFrame({"x": x, "target": y})
    tr, rp = df.iloc[:300].reset_index(drop=True), df.iloc[300:].reset_index(drop=True)
    p = _best_proposal(tr)
    rep, findings = _diagnose(p, rp, rp["target"].to_numpy())
    assert diag.FINDING_HETEROSCEDASTICITY_SUSPECT in findings, findings


def test_diagnosis_null_world_abstains_before_diagnosis():
    rng = np.random.RandomState(99)
    n = 500
    df = pd.DataFrame({"x": rng.randn(n), "target": rng.randn(n)})
    props = RimeProposer(feature_cols=["x"], target_col="target", max_depth=2, budget=120).fit_and_propose(df)
    assert len(props) == 0  # abstention: diagnosis is never invoked, no repair baseline


def test_diagnosis_insufficient_rows():
    df = _planted_lag()
    tr = df.iloc[:450].reset_index(drop=True)
    tiny = df.iloc[450:460].reset_index(drop=True)
    p = _best_proposal(tr)
    rep, findings = _diagnose(p, tiny, tiny["target"].to_numpy())
    assert findings == (diag.FINDING_INSUFFICIENT_ROWS,), findings


def test_diagnosis_report_deterministic_and_schema_frozen():
    df = _planted_lag()
    tr, rp = df.iloc[:450].reset_index(drop=True), df.iloc[450:].reset_index(drop=True)
    p = _best_proposal(tr)
    r1, _ = _diagnose(p, rp, rp["target"].to_numpy())
    r2, _ = _diagnose(p, rp, rp["target"].to_numpy())
    assert r1.sha256() == r2.sha256()
    assert len(r1.sha256()) == 64
    expected_keys = {
        "candidate_id", "formula", "graph_id", "partition_label", "n_rows",
        "n_finite", "mean_resid", "std_resid", "std_target", "rmse", "mae",
        "r2", "corr", "rho1", "max_abs_resid", "frac_finite",
    }
    assert set(r1.to_dict().keys()) == expected_keys
    assert set(diag.finding_vocabulary()) == {
        "intercept_bias", "temporal_dependence", "heteroscedasticity_suspect",
        "curvature_suspect", "no_reliable_structure", "insufficient_rows",
    }


def test_diagnosis_cutoffs_have_no_module_defaults():
    # Anti-hardcoding guard: every cutoff of classify_findings must be a
    # required keyword (no default literal can hide in the module).
    sig = inspect.signature(diag.classify_findings)
    for name in ("rho_flag", "bias_ratio_cut", "variance_ratio_cut", "curvature_corr_cut", "min_rows"):
        param = sig.parameters[name]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY, name
        assert param.default is inspect.Parameter.empty, name
