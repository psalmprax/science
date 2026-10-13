"""RIME Repair v0.1.3 milestone: real-data controls + adversarial gates.

Test D (reviewer): frozen walk-forward windows on the real BTC panel —
original vs repaired vs persistence, predictive kept separate from any
structural claim. Measured expectation: all-abstain (consistent with
TRADING-NP93), persistence metrics reported per window.

Adversarial: a chance predictive pass on a tiny validation split is
blocked by the stability gate (halves below min_rows cannot pass).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from apedp_dyn.epistemic.adjudicate import RESID_RHO_FLAG
from apedp_dyn.epistemic import rime_repair_diagnosis as diagmod
from apedp_dyn.epistemic.rime_repair_engine import (
    RepairConfig,
    RepairOperator,
    RepairPartitions,
    RepairStatus,
    repair,
)
from apedp_dyn.grammar.proposer_interface import CandidateProposal
from apedp_dyn.proposers.rime_proposer import RimeProposer
from apedp_dyn.sandbox.trading_signals_sandbox import (
    extract_all_22_features,
    load_real_btc_orderflow,
    SIGNAL_REGISTRY,
)

BIAS_RATIO_CUT = 0.25
VARIANCE_RATIO_CUT = 4.0
CURVATURE_CORR_CUT = 0.3
MIN_ROWS = 30

CONFIG = RepairConfig(
    max_candidates_per_round=6,
    max_rounds=2,
    max_added_terms=1,
    min_validation_r2_gain=0.02,
    equivalence_tol=1e-9,
    operators=(RepairOperator.R1_PARAM_REFIT, RepairOperator.R2_TERM_INSERT, RepairOperator.R3_LAG_INSERT),
)

CUTS = dict(rho_flag=RESID_RHO_FLAG, bias_ratio_cut=BIAS_RATIO_CUT,
            variance_ratio_cut=VARIANCE_RATIO_CUT, curvature_corr_cut=CURVATURE_CORR_CUT,
            min_rows=MIN_ROWS)


def _btc_frame():
    panel = load_real_btc_orderflow(600)
    X = extract_all_22_features(panel)
    names = [s[0] for s in SIGNAL_REGISTRY]
    px = panel["price"].to_numpy(float)
    r = np.append(0.0, np.diff(np.log(px)))
    fwd = np.roll(r, -1)
    v = np.arange(5, len(panel) - 1)
    return pd.DataFrame({"obv": X[v, names.index("H-OBV_MOM")],
                         "bb": X[v, names.index("H-BB")],
                         "target": fwd[v]})


def test_btc_panel_abstention_regression():
    # Encodes the measured live run: rejected cubic H-BB form, refit on the
    # discovery split only; repair split R2 < 0, no finding, zero candidates,
    # sealed never opened.
    df = _btc_frame()
    parts = RepairPartitions.from_frame(df, df["target"].to_numpy())
    tr, ytr = parts.discovery_df, parts.discovery_y
    bb = tr["bb"].to_numpy()
    lag5 = np.full(len(bb), np.nan)
    lag5[5:] = bb[:-5]
    t3 = bb * (bb * lag5)
    m = np.isfinite(t3) & np.isfinite(ytr)
    M = np.column_stack([t3[m], np.ones(int(m.sum()))])
    coef, *_ = np.linalg.lstsq(M, ytr[m], rcond=None)
    a, b = float(coef[0]), float(coef[1])

    def scorer(dfx, _a=a, _b=b):
        bb = np.asarray(dfx["bb"], dtype=float)
        l5 = np.full(len(bb), np.nan)
        l5[5:] = bb[:-5]
        return _a * (bb * (bb * l5)) + _b

    orig = CandidateProposal(proposer_id="demo_rejected_cubic", candidate_id="rej_bb_cubic",
                             formula=f"{a:+.6g}*((bb*(bb*bb[t-5]))){b:+.6g}", strategy_archetype=None,
                             declared_scale_invariant=True, declared_nominal_dependencies=("obv", "bb"),
                             compute_score=scorer, parameters={"alpha": a, "beta": b}, metadata={})
    rep = diagmod.diagnose_residuals(orig, parts.repair_df, parts.repair_y, "repair")
    assert rep.r2 < 0, rep.r2  # worse than the repair-split mean
    out = repair(orig, parts, CONFIG, **CUTS)
    assert out.status == RepairStatus.NO_IMPROVEMENT, out.status
    assert out.candidates_evaluated == 0
    assert out.winner is None and out.sealed is None
    print(f"\n[btc] repair-R2={rep.r2:.4f} rho1={rep.rho1:.3f} outcome={out.status.value} sealed=shut")


def test_walkforward_all_abstain_with_persistence_baseline():
    # Five frozen walk-forward windows on the real panel. Expectation from
    # TRADING-NP93: RIME emits nothing on any window (all-abstain).
    # Persistence (P_t = P_{t-1}, i.e. zero forward return) reported per
    # window as the predictive baseline, separate from any structure claim.
    df = _btc_frame()
    y = df["target"].to_numpy()
    n, w, step = len(df), 300, 60
    abstains, persist_rmses = 0, []
    for k in range(5):
        a = k * step
        seg = df.iloc[a:a + w].reset_index(drop=True)
        segy = y[a:a + w]
        props = RimeProposer(feature_cols=["obv", "bb"], target_col="target",
                             max_depth=2, budget=120).fit_and_propose(
                                 seg.assign(target=segy))
        abstains += int(len(props) == 0)
        persist_rmses.append(float(np.sqrt(np.mean(segy**2))))  # zero-prediction RMSE
    print(f"\n[walkforward] abstain {abstains}/5 windows; persistence RMSE: " +
          ", ".join(f"{v:.6f}" for v in persist_rmses))
    assert abstains == 5, abstains


def test_chance_predictive_pass_blocked_by_stability_gate():
    # Adversarial gate interaction: n=200 forces validation halves (20 rows)
    # below min_rows, so even a chance predictive pass cannot clear stability.
    rng = np.random.RandomState(77)
    n = 200
    x = rng.randn(n)
    shift = np.where(np.arange(n) < 80, 0.0, 5.0)
    y = np.zeros(n)
    y[2:] = 3.0 * x[:-2] + 10.0 + shift[2:] + 0.01 * rng.randn(n - 2)
    df = pd.DataFrame({"x": x, "target": y})
    parts = RepairPartitions.from_frame(df, df["target"].to_numpy())
    tr = parts.discovery_df.assign(target=parts.discovery_y)
    props = RimeProposer(feature_cols=["x"], target_col="target", max_depth=1, budget=120).fit_and_propose(tr)
    assert len(props) > 0
    orig = sorted(props, key=lambda p: p.parameters["dev_r2"], reverse=True)[0]
    out = repair(orig, parts, CONFIG, **CUTS)
    assert out.candidates_evaluated >= 1
    assert out.status == RepairStatus.NO_IMPROVEMENT, out.status
    assert all(g.stability_pass is False for g in out.validation.values()), out.validation
    assert out.sealed is None
    print(f"\n[adversarial] predictive-gated candidates blocked at stability; evals={out.candidates_evaluated}")
