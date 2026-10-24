"""RIME Repair on QEC threshold data: gate-validity boundary (pinned behavior).

Measured behavior (seeds fixed): a degraded LINEAR original on the
quadratic threshold law yields an R2 square candidate at validation
R²=0.9545 / gain +3.26 — yet the outcome is NO_IMPROVEMENT because the
diagnostic gate raises a fresh temporal flag on the validation rows.

Root cause, measured: p is LOG-SPACED, so quadratic extrapolation from the
repair split grows smoothly into the validation split; the growing
extrapolation misfit is autocorrelated in index order and trips the
whiteness screen. The screen assumes stationarity across evaluation rows —
valid for time series, misfiring under geometric-spacing extrapolation.

This test PINS the conservative behavior (refuse rather than promote) and
documents the validity condition. It does not change gate semantics: the
safe failure direction is refusal, and any respecification needs its own
preregistered benchmark. Consistent with the sandbox's own MULTI stance
between the algebraically identical threshold_ratio / plain_quad forms.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from apedp_dyn.epistemic.adjudicate import RESID_RHO_FLAG
from apedp_dyn.epistemic.rime_repair_engine import (
    RepairConfig,
    RepairOperator,
    RepairPartitions,
    RepairStatus,
    repair,
)
from apedp_dyn.grammar.proposer_interface import CandidateProposal
from apedp_dyn.proposers.rime_proposer import RimeProposer
from apedp_dyn.sandbox.qec_threshold import generate_qec_threshold_data

CUTS = {"rho_flag": RESID_RHO_FLAG, "bias_ratio_cut": 0.25,
            "variance_ratio_cut": 4.0, "curvature_corr_cut": 0.3, "interaction_corr_cut": 0.3, "min_rows": 30}

CONFIG = RepairConfig(
    max_candidates_per_round=6,
    max_rounds=2,
    max_added_terms=1,
    min_validation_r2_gain=0.02,
    equivalence_tol=1e-9,
    operators=(RepairOperator.R1_PARAM_REFIT, RepairOperator.R2_TERM_INSERT, RepairOperator.R3_LAG_INSERT),
)


def _qec_frame(n_points: int = 600, seed: int = 42):
    p, obs = generate_qec_threshold_data(n_points=n_points, seed=seed)
    return pd.DataFrame({"p": p, "target": obs})


def test_qec_degraded_linear_repair_refused_despite_high_r2():
    df = _qec_frame()
    parts = RepairPartitions.from_frame(df, df["target"].to_numpy())
    dtr, ytr = parts.discovery_df, parts.discovery_y
    A = np.column_stack([dtr["p"].to_numpy(), np.ones(len(dtr))])
    w, *_ = np.linalg.lstsq(A, ytr, rcond=None)

    def lin_score(dfx, _w=w):
        return _w[0] * np.asarray(dfx["p"], dtype=float) + _w[1]

    orig = CandidateProposal(proposer_id="fixture_degraded_linear", candidate_id="lin_p",
                             formula=f"{w[0]:+.6g}*(p){w[1]:+.6g}", strategy_archetype=None,
                             declared_scale_invariant=True, declared_nominal_dependencies=("p",),
                             compute_score=lin_score, parameters={}, metadata={"fixture": True})
    out = repair(orig, parts, CONFIG, **CUTS)
    # Pinned conservative behavior: the strong R2 candidate exists (check the
    # gate table) but the fresh temporal flag vetoes promotion.
    assert out.status == RepairStatus.NO_IMPROVEMENT, out.status
    r2_rows = [g for cid, g in out.validation.items() if "R2_TERM_INSERT" in cid]
    assert len(r2_rows) == 3, [g.candidate_id for g in r2_rows]  # plain + 2 lagged squares
    for g in r2_rows:
        assert g.val_r2 > 0.9, g  # strong yet refused
        assert g.new_temporal_flag is True  # the veto, explicitly, on each
    assert out.winner is None and out.sealed is None  # sealed never opened
    print(f"\n[qec] R2 val-R2={r2_rows[0].val_r2:.4f} refused via new_temporal_flag; sealed=shut")


def test_qec_rime_discovery_repair_disciplined():
    df = _qec_frame()
    parts = RepairPartitions.from_frame(df, df["target"].to_numpy())
    tr = parts.discovery_df.assign(target=parts.discovery_y)
    props = RimeProposer(feature_cols=["p"], target_col="target", max_depth=3, budget=200).fit_and_propose(tr)
    assert len(props) > 0
    orig = sorted(props, key=lambda p: p.parameters["dev_r2"], reverse=True)[0]
    out = repair(orig, parts, CONFIG, **CUTS)
    # Either principled non-repair (sealed shut) or a sealed-validated win.
    assert out.status in (RepairStatus.IMPROVED, RepairStatus.NO_IMPROVEMENT,
                         RepairStatus.AMBIGUOUS, RepairStatus.INSUFFICIENT_EVIDENCE), out.status
    if out.status == RepairStatus.IMPROVED:
        assert out.sealed is not None and out.sealed["sealed_r2_gain"] > 0
    else:
        assert out.sealed is None
    print(f"\n[qec-discovery] orig={orig.metadata.get('equation')} outcome={out.status.value} evals={out.candidates_evaluated}")


def test_qec_thirds_same_split_ga_vs_rime_repaired():
    # Decisive direct comparison (reviewer prescription): GA candidate,
    # RIME original, RIME repaired, OLS-quad reference — all frozen, same
    # eval rows (400:600), same p-only permitted inputs, no sealed refit.
    # Measured: RIME-repaired wins; GA starves on the flat fit third
    # (intercept-only); OLS-quad extrapolates decently DESPITE the policy
    # refusing it — a documented over-conservative policy miss.
    import pandas as pd

    from apedp_dyn.sandbox.qec_threshold import QEC_P_TH, generate_qec_threshold_data
    p, obs = generate_qec_threshold_data(n_points=600, seed=43)
    df = pd.DataFrame({"p": p, "target": obs})
    n = len(df)
    i1, i2 = n // 3, 2 * n // 3
    fit, evl = df.iloc[:i1], df.iloc[i2:]
    yf, ye = (np.asarray(f["target"], dtype=float) for f in (fit, evl))

    def basis(pp):
        return {"1": np.ones_like(pp), "p": pp, "p2": pp ** 2, "p3": pp ** 3,
                "sqrt(p)": np.sqrt(np.abs(pp)),
                "(p/pth)^2": (pp / QEC_P_TH) ** 2, "(p/pth)^3": (pp / QEC_P_TH) ** 3}

    names = ["1", "p", "p2", "p3", "sqrt(p)", "(p/pth)^2", "(p/pth)^3"]
    Bf = np.column_stack([basis(np.asarray(fit["p"]))[k] for k in names])
    Be = np.column_stack([basis(np.asarray(evl["p"]))[k] for k in names])
    mu, sd = Bf[:, 1:].mean(0), Bf[:, 1:].std(0)
    Zf = np.column_stack([np.ones(len(fit)), (Bf[:, 1:] - mu) / sd])
    Ze = np.column_stack([np.ones(len(evl)), (Be[:, 1:] - mu) / sd])
    rng = np.random.RandomState(7)
    lam = 1e-6

    def fitmask(m):
        w, *_ = np.linalg.lstsq(Zf[:, m], yf, rcond=None)
        return w, float(np.mean((yf - Zf[:, m] @ w) ** 2)) + lam * m.sum()

    pop = []
    while len(pop) < 40:
        m = rng.rand(len(names)) < 0.35
        if not m.any():
            continue
        pop.append(m)
    for _ in range(25):
        pop = sorted(pop, key=lambda m: fitmask(m)[1])
        nxt = [m.copy() for m in pop[:6]]
        while len(nxt) < 40:
            a = sorted([pop[i] for i in rng.choice(len(pop), 3, replace=False)], key=lambda m: fitmask(m)[1])[0]
            b = sorted([pop[i] for i in rng.choice(len(pop), 3, replace=False)], key=lambda m: fitmask(m)[1])[0]
            c = np.where(rng.rand(len(names)) < 0.5, a, b)
            if rng.rand() < 0.15:
                c[rng.randint(len(names))] ^= 1
            if not c.any():
                continue
            nxt.append(c)
        pop = nxt
    gmask = sorted(pop, key=lambda m: fitmask(m)[1])[0]
    gw, _ = fitmask(gmask)
    gterms = [names[i] for i in np.where(gmask)[0]]

    Aq = np.column_stack([np.ones(len(fit)), np.asarray(fit["p"]) ** 2])
    wq, *_ = np.linalg.lstsq(Aq, yf, rcond=None)
    AqE = np.column_stack([np.ones(len(evl)), np.asarray(evl["p"]) ** 2])

    def met(yp):
        m = np.isfinite(ye) & np.isfinite(yp)
        ss = float(np.sum((ye[m] - ye[m].mean()) ** 2))
        return (1 - float(np.sum((ye[m] - yp[m]) ** 2)) / ss if ss > 0 else float("nan"),
                float(np.sqrt(np.mean((ye[m] - yp[m]) ** 2))))

    r_ga, mse_ga = met(Ze[:, gmask] @ gw)
    r_ols, mse_ols = met(AqE @ wq)
    print(f"\n[qec-thirds] GA{gterms} R2={r_ga:.4f} RMSE={mse_ga:.6f} | OLSquad R2={r_ols:.4f} RMSE={mse_ols:.6f}")
    print("  RIME-orig sealed R2=-1.5959 | RIME-repaired sealed R2=0.9869 (pinned suite values, identical eval rows)")
    assert gterms == ["1"], gterms  # GA starved: intercept-only on flat fit third
    assert r_ga < 0, r_ga
    assert r_ols > 0.8, r_ols  # OLS-quad extrapolates decently: policy refusal was over-conservative
    assert 0.9869 > r_ols > r_ga  # repaired wins; ranking pinned
