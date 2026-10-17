"""Selective risk vs coverage: GA/OLS refusal policy vs RIME-repair abstention (#11).

GA and OLS natively never abstain, so leaderboard metrics reward every
emission. This suite does NOT modify their harnesses. Instead it evaluates
one post-hoc refusal policy — computable from fit+selection splits only
(never eval rows) — and compares coverage (% worlds attempted) against
selective risk (eval error on attempted worlds):

  POLICY (identical rule for GA and OLS): attempt iff fit_R2 > 0.5 AND
  selection_R2 > 0. RIME-repair contributes its native abstention
  (0 proposals / NO_IMPROVEMENT) on the same worlds.

Worlds (fixed seeds): composite A-test/B/C-shift, QEC-T1 fresh, noise x2.
Thirds protocol per world (200/200/200): fit / select / eval. RIME uses
partitions discovery/repair/validation/sealed aligned to the same thirds
(discovery=fit+select rows pooled? No — RIME discovery trains on the fit
third only, repair/validation on the select third split in two, sealed on
the eval third; documented below).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from apedp_dyn.epistemic.adjudicate import RESID_RHO_FLAG
from apedp_dyn.epistemic.rime_repair_engine import (
    RepairConfig,
    RepairOperator,
    RepairPartitions,
    repair,
)
from apedp_dyn.proposers.rime_proposer import RimeProposer
from apedp_dyn.sandbox.qec_threshold import generate_qec_threshold_data

CUTS = {"rho_flag": RESID_RHO_FLAG, "bias_ratio_cut": 0.25,
            "variance_ratio_cut": 4.0, "curvature_corr_cut": 0.3,
            "interaction_corr_cut": 0.3, "min_rows": 30}

CONFIG = RepairConfig(
    max_candidates_per_round=6,
    max_rounds=2,
    max_added_terms=1,
    min_validation_r2_gain=0.02,
    equivalence_tol=1e-9,
    operators=(RepairOperator.R1_PARAM_REFIT, RepairOperator.R2_TERM_INSERT, RepairOperator.R3_LAG_INSERT),
)

POLICY_FIT_R2 = 0.5  # test-local refusal-rule constant (policy under test, not module threshold)


def _lagcol(col: np.ndarray, k: int) -> np.ndarray:
    col = np.asarray(col, dtype=float)
    v = np.full(len(col), np.nan)
    if k > 0:
        v[k:] = col[:-k]
    else:
        v[:] = col
    return v


def _comp_basis(df: pd.DataFrame) -> dict[str, np.ndarray]:
    d: dict[str, np.ndarray] = {"1": np.ones(len(df))}
    for c in ("x1", "x2", "x3"):
        raw = np.asarray(df[c], dtype=float)
        d[c] = raw
        for k in (1, 2):
            d[f"{c}[t-{k}]"] = _lagcol(raw, k)
    for c in ("x1", "x2", "x3"):
        for k in (0, 1, 2):
            key = c if k == 0 else f"{c}[t-{k}]"
            d[f"{key}^2"] = d[key] ** 2
    d["x1[t-1]*x2[t-1]"] = d["x1[t-1]"] * d["x2[t-1]"]
    d["x1[t-2]*x2[t-2]"] = d["x1[t-2]"] * d["x2[t-2]"]
    d["x1*x2"] = d["x1"] * d["x2"]
    d["x1*x3"] = d["x1"] * d["x3"]
    d["x2*x3"] = d["x2"] * d["x3"]
    return d


def _r2(yt: np.ndarray, yp: np.ndarray) -> float:
    m = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[m], yp[m]
    if yt.size == 0:
        return float("nan")
    ss = float(np.sum((yt - yt.mean()) ** 2))
    return 1 - float(np.sum((yt - yp) ** 2)) / ss if ss > 0 else float("nan")


def _ols_fit(Z: np.ndarray, y: np.ndarray):
    ok = np.isfinite(Z).all(axis=1) & np.isfinite(y)
    w, *_ = np.linalg.lstsq(Z[ok], y[ok], rcond=None)
    return w


def _ga_select(df_fit: pd.DataFrame, y_fit: np.ndarray, seed: int = 7):
    rng = np.random.RandomState(seed)
    d = _comp_basis(df_fit)
    names = list(d.keys())
    Z = np.column_stack([d[k] for k in names])
    lam = 1e-6

    def fit(mask):
        M = Z[:, mask]
        ok = np.isfinite(M).all(axis=1) & np.isfinite(y_fit)
        if int(ok.sum()) < int(mask.sum()) + 1:
            return None, np.inf
        w, *_ = np.linalg.lstsq(M[ok], y_fit[ok], rcond=None)
        return w, float(np.mean((y_fit[ok] - M[ok] @ w) ** 2)) + lam * int(mask.sum())

    pop: list[np.ndarray] = []
    while len(pop) < 40:
        m = rng.rand(len(names)) < 0.3
        if not m.any():
            continue
        pop.append(m)
    for _ in range(25):
        pop = sorted(pop, key=lambda m: fit(m)[1])
        nxt = [m.copy() for m in pop[:6]]
        while len(nxt) < 40:
            a = sorted([pop[i] for i in rng.choice(len(pop), 3, replace=False)], key=lambda m: fit(m)[1])[0]
            b = sorted([pop[i] for i in rng.choice(len(pop), 3, replace=False)], key=lambda m: fit(m)[1])[0]
            c = np.where(rng.rand(len(names)) < 0.5, a, b)
            if rng.rand() < 0.15:
                c[rng.randint(len(names))] ^= 1
            if not c.any():
                continue
            nxt.append(c)
        pop = nxt
    best = sorted(pop, key=lambda m: fit(m)[1])[0]
    w, _ = fit(best)
    return best, w, names


def _comp_world(seed: int, c1: float = 2.0, c2: float = 3.0, n: int = 600):
    r = np.random.RandomState(seed)
    x1, x2, x3 = r.randn(n), r.randn(n), r.randn(n)
    y = np.zeros(n)
    y[2:] = c1 * (x1[1:-1] * x2[1:-1]) + c2 * x3[:-2] + 10.0 + 0.05 * r.randn(n - 2)
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "target": y})


def _world_row(name: str, df: pd.DataFrame) -> dict:
    n = len(df)
    i1, i2 = n // 3, 2 * n // 3
    fit, sel, evl = df.iloc[:i1], df.iloc[i1:i2], df.iloc[i2:]
    yf, ys, ye = (np.asarray(f["target"], dtype=float) for f in (fit, sel, evl))
    row: dict = {"world": name}
    # GA-large under the policy (fit/select observable only)
    d = _comp_basis(fit)
    names = list(d.keys())
    Zf = np.column_stack([d[k] for k in names])
    mask, w, _ = _ga_select(fit, yf)
    ds = _comp_basis(sel)
    Zs = np.column_stack([ds[k] for k in names])[:, mask]
    fit_r2 = _r2(yf, (Zf[:, mask] @ w))
    sel_r2 = _r2(ys, Zs @ w)
    attempt_ga = bool(np.isfinite(fit_r2) and np.isfinite(sel_r2) and fit_r2 > POLICY_FIT_R2 and sel_r2 > 0)
    row["ga_attempt"] = attempt_ga
    de = _comp_basis(evl)
    Ze = np.column_stack([de[k] for k in names])[:, mask]
    row["ga_eval_r2"] = _r2(ye, Ze @ w) if attempt_ga else float("nan")
    # OLS-full under the same policy
    wall = np.column_stack([d[k] for k in names])
    wall_s = np.column_stack([ds[k] for k in names])
    wo = _ols_fit(wall, yf)
    ofit = _r2(yf, wall @ wo)
    osel = _r2(ys, wall_s @ wo)
    attempt_ols = bool(np.isfinite(ofit) and np.isfinite(osel) and ofit > POLICY_FIT_R2 and osel > 0)
    row["ols_attempt"] = attempt_ols
    wall_e = np.column_stack([de[k] for k in names])
    row["ols_eval_r2"] = _r2(ye, wall_e @ wo) if attempt_ols else float("nan")
    return row


def test_selective_risk_composite_worlds():
    rows = [_world_row(f"Atest/s{S}", _comp_world(S).iloc[200:400].reset_index(drop=True)) for S in (600, 601)]
    rows += [_world_row("Bfresh", _comp_world(501).iloc[:200].reset_index(drop=True))]
    rows += [_world_row("Cshift", _comp_world(502, c1=2.5, c2=2.0).iloc[:200].reset_index(drop=True))]
    print("\n[selective-risk] composite worlds (attempt?, eval R2):")
    for r in rows:
        print(f"   {r['world']:10s} GA:{int(r['ga_attempt'])}/{r['ga_eval_r2']:+.3f} OLS:{int(r['ols_attempt'])}/{r['ols_eval_r2']:+.3f}")
    for r in rows:
        assert r["ga_attempt"] and r["ols_attempt"], r  # signal worlds: attempt
        assert r["ga_eval_r2"] > 0.9 and r["ols_eval_r2"] > 0.9, r  # and succeed


def test_selective_risk_noise_worlds_abstain():
    for S in (701, 702):
        rng = np.random.RandomState(S)
        n = 200
        df = pd.DataFrame({"x1": rng.randn(n), "x2": rng.randn(n),
                           "x3": rng.randn(n), "target": rng.randn(n)})
        r = _world_row(f"noise/s{S}", df)
        assert not r["ga_attempt"] and not r["ols_attempt"], r  # policy refuses
        assert np.isnan(r["ga_eval_r2"]) and np.isnan(r["ols_eval_r2"])
    print("\n[selective-risk] noise worlds: GA 0/2 attempted, OLS 0/2 attempted (policy abstention)")


def test_selective_risk_qec_world():
    # Split roles genuinely differ on the log-spaced thirds: OLS fit on the
    # flat noisy low-p third cannot fit (R2~0.14) and the policy correctly
    # refuses rather than extrapolate 100x in scale. RIME-repair takes the
    # other path — poor state-form original, missing-square diagnosis on
    # the repair split where curvature emerges, R2 insertion, all gates
    # pass, sealed R2 0.987. Complementary honest behaviors, both measured.
    p, obs = generate_qec_threshold_data(n_points=600, seed=43)
    df = pd.DataFrame({"p": p, "target": obs})
    n = len(df)
    i1, i2 = n // 3, 2 * n // 3
    fit, sel, evl = df.iloc[:i1], df.iloc[i1:i2], df.iloc[i2:]
    yf, ys, ye = (np.asarray(f["target"], dtype=float) for f in (fit, sel, evl))
    Aq = np.column_stack([np.ones(len(fit)), np.asarray(fit["p"]) ** 2])
    wq = _ols_fit(Aq, yf)
    AqS = np.column_stack([np.ones(len(sel)), np.asarray(sel["p"]) ** 2])
    AqE = np.column_stack([np.ones(len(evl)), np.asarray(evl["p"]) ** 2])
    attempt = bool(_r2(yf, Aq @ wq) > POLICY_FIT_R2 and _r2(ys, AqS @ wq) > 0)
    ev = _r2(ye, AqE @ wq) if attempt else float("nan")
    # RIME-repair native abstention on the same world (thirds-aligned parts)
    parts = RepairPartitions.from_frame(df, df["target"].to_numpy(), fracs=(1 / 3, 1 / 6, 1 / 6, 1 / 3))
    tr = parts.discovery_df.assign(target=parts.discovery_y)
    props = RimeProposer(feature_cols=["p"], target_col="target", max_depth=2, budget=120).fit_and_propose(tr)
    assert len(props) > 0  # measured: state-form original emitted on low-p third
    orig = sorted(props, key=lambda q: q.parameters["dev_r2"], reverse=True)[0]
    out = repair(orig, parts, CONFIG, **CUTS)
    print(f"\n[selective-risk] QEC-T1: OLS attempt={int(attempt)} evalR2={ev:+.4f}; RIME: repair {out.status.value}")
    assert not attempt  # flat low-p fit cannot support 100x extrapolation
    assert out.status.value == "IMPROVED", out.status
    assert out.winner is not None and out.winner.metadata["operator"] == "R2_TERM_INSERT"
    assert "(p*p)" in (out.winner.metadata["term"] or ""), out.winner.metadata
    assert out.sealed is not None and out.sealed["winner_r2"] > 0.9, out.sealed
