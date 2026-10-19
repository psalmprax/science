"""Three-Hypothesis Incremental Information Test (forensics doc S80): does H-A
still contain new predictive information after H-P and H-RV had every chance?
Frozen stages A-F. Transfer-matrix and regime-conditioned ablation explicitly
DEFERRED (not run here).
  Stage A combos: P | RV | A | P+RV | P+A | P+RV+A (OLS, train blocks only).
  Stage B rolling OOS: 6 assets x 11 origins, forward-month blocks.
  Stage C contrasts (QLIKE primary, R2 secondary): A|P, A|P,RV, RV|P,A.
  Stage D coefficient stability: beta trajectories, sign flips, near-zero.
  Stage E redundancy: pairwise forecast correlations + incremental-signal
    correlation corr(yhat_{P+A}-yhat_P, yhat_{P+RV}-yhat_P).
  Stage F headline permutation: M3 - M(P+RV) with full refit inside each
    of 200 month-block permutations. Plus Diebold-Mariano (HAC-4) on the
    headline QLIKE differential, and orthogonalized HA-perp c estimates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.linalg as la
import pandas as pd

from apedp_dyn.epistemic.active_disambiguation.ensemble_gated import qlike
from apedp_dyn.epistemic.active_disambiguation.trading_e4t1_v2 import ORIGINS, _blocks, _d5

COMBOS = ("P", "RV", "A", "P+RV", "P+A", "P+RV+A")
MEMBERS = {"P": ("sig",), "RV": ("rankV",), "A": ("HA",),
           "P+RV": ("sig", "rankV"), "P+A": ("sig", "HA"), "P+RV+A": ("sig", "rankV", "HA")}
N_PERM = 200
PERM_SEED = 8080
QLIKE_FLOOR_Q = 0.05


def _fit(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w, *_ = la.lstsq(X, y, rcond=None)
    dof = max(len(y) - X.shape[1], 1)
    s2 = float(np.sum((y - X @ w) ** 2)) / dof
    try:
        cov = s2 * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.maximum(np.diag(cov), 0))
    except np.linalg.LinAlgError:
        se = np.full(X.shape[1], np.nan)
    return w, se


def _qvec(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    y = np.maximum(np.asarray(y, dtype=float), 1e-8)
    p = np.maximum(np.asarray(pred, dtype=float), 1e-8)
    r = y / p
    return np.asarray(r - np.log(r) - 1.0)


def _dm_hac4(d: np.ndarray) -> tuple[float, float]:
    """Diebold-Mariano t-stat and two-sided normal p-value, HAC(4) rectangular."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    if n < 8:
        return float("nan"), float("nan")
    m = float(np.mean(d))
    g = d - m
    L = 4
    w = 1 - np.arange(L + 1) / (L + 1)
    # Canonical Newey-West estimator: every lagged autocovariance divides by the
    # TOTAL sample size n, not by (n - k). Dividing by n - k breaks positive
    # semi-definiteness of the Bartlett-weighted variance (the same defect fixed
    # earlier in novelty_adjudication.py), producing v <= 0 and spurious NaN
    # DM statistics whenever the longest lag consumes a visible fraction of
    # short forward blocks.
    acov = [float(np.sum(g[: n - k] * g[k:]) / n) for k in range(L + 1)]
    v = float(sum(w[k] * (acov[k] if k == 0 else 2 * acov[k]) for k in range(L + 1)))
    if v <= 0:
        return float("nan"), float("nan")
    from math import erf, sqrt

    t = m / sqrt(v / n)
    p = 1 - erf(abs(t) / sqrt(2))
    return t, max(min(p, 1.0), 0.0)


@dataclass
class AssetBattery:
    asset: str
    qlike: dict = field(default_factory=dict)  # pooled forward QLIKE per combo
    r2: dict = field(default_factory=dict)  # pooled forward R2 per combo
    contrast_q: dict = field(default_factory=dict)  # A|P, A|P,RV, RV|P,A (QLIKE diffs)
    contrast_r2: dict = field(default_factory=dict)
    wins: dict = field(default_factory=dict)  # per-origin pooled-loss wins per combo
    beta_traj: dict = field(default_factory=dict)  # combo -> list of dicts per origin
    sign_flip: dict = field(default_factory=dict)  # fraction of origins with beta<0 per term (full model)
    near_zero: dict = field(default_factory=dict)  # fraction |t|<1 per term (full model)
    corr_pred: dict = field(default_factory=dict)  # pairwise forecast correlations
    corr_incr: float = float("nan")  # incremental-signal correlation
    dm_t: float = float("nan")
    dm_p: float = float("nan")
    perm_p: float = float("nan")  # headline: M3 - M(P+RV)
    ha_perp_t: float = float("nan")  # mean t of c in y = a+b|r|+c HA-perp


def _origin_data(a: pd.DataFrame, tgt: np.ndarray, dates: pd.Series, origin: str):
    nxt = str(pd.Timestamp(origin) + pd.offsets.MonthBegin(1))
    tr = np.where((dates < origin).to_numpy())[0]
    fw = np.where((dates >= origin).to_numpy() & (dates < nxt).to_numpy())[0]
    tb = np.intersect1d(_blocks(len(dates)), tr)
    fb = np.intersect1d(_blocks(len(dates)), fw)
    return tb, fb


def _cols(a: pd.DataFrame, idx: np.ndarray) -> dict[str, np.ndarray]:
    return {c: a[c].to_numpy()[idx] for c in ("sig", "rankV", "HA")}


def run_asset(panels: dict[str, pd.DataFrame], asset: str) -> AssetBattery:
    a = panels[asset]
    dates = a["date"]
    tgt = _d5(a["sig"].to_numpy())
    out = AssetBattery(asset)
    qnum = dict.fromkeys(COMBOS, 0.0)
    sse = dict.fromkeys(COMBOS, 0.0)
    sst = 0.0
    wins = dict.fromkeys(COMBOS, 0)
    betas: dict[str, list] = {c: [] for c in COMBOS}
    preds_all: dict[str, list] = {c: [] for c in ("P", "RV", "A", "P+RV", "P+A", "P+RV+A")}
    d_head: list[float] = []  # headline QLIKE differential M3 - M(P+RV), forward blocks
    ha_t: list[float] = []
    for origin in ORIGINS:
        tb, fb = _origin_data(a, tgt, dates, origin)
        Ct, yt = _cols(a, tb), tgt[tb]
        okt = np.isfinite(yt)
        for c in ("sig", "rankV", "HA"):
            okt &= np.isfinite(Ct[c])
        okt = np.isfinite(yt)
        for c in ("sig", "rankV", "HA"):
            okt &= np.isfinite(Ct[c])
        mu = float(np.mean(yt[okt]))
        qfloor = float(np.quantile(yt[okt], QLIKE_FLOOR_Q))
        W = {}
        for c in COMBOS:
            X = np.column_stack([np.ones(okt.sum())] + [Ct[m][okt] for m in MEMBERS[c]])
            W[c] = _fit(X, yt[okt])
        Cf, yf = _cols(a, fb), tgt[fb]
        okf = np.isfinite(yf)
        for c in ("sig", "rankV", "HA"):
            okf &= np.isfinite(Cf[c])
        yf = yf[okf]
        if len(yf) < 3:
            continue
        p = {}
        for c in COMBOS:
            Xf = np.column_stack([np.ones(okf.sum())] + [Cf[m][okf] for m in MEMBERS[c]])
            p[c] = Xf @ W[c][0]
            betas[c].append(dict(zip(("a",) + MEMBERS[c], [float(v) for v in W[c][0]], strict=True)))
        for c in COMBOS:
            qnum[c] += qlike(yf, p[c], qfloor) * len(yf)
            sse[c] += float(np.sum((yf - p[c]) ** 2))
        sst += float(np.sum((yf - mu) ** 2))
        ntot = sum(len(yf) for _ in [0])
        _ = ntot
        q_o = {c: qlike(yf, p[c], qfloor) for c in COMBOS}
        best = min(COMBOS, key=lambda c: q_o[c])
        wins[best] += 1
        for c in COMBOS:
            preds_all[c].append(p[c])
        d_head.extend(list(_qvec(yf, p["P+RV+A"]) - _qvec(yf, p["P+RV"])))
        # Orthogonalized HA-perp c estimate (train fit, recorded per origin).
        ha = Ct["HA"][okt]
        s = Ct["sig"][okt]
        X = np.column_stack([np.ones(okt.sum()), s])
        g, *_ = la.lstsq(X, ha, rcond=None)
        hap = ha - X @ g
        X2 = np.column_stack([np.ones(okt.sum()), s, hap])
        w2, se2 = _fit(X2, yt[okt])
        ha_t.append(float(w2[2] / se2[2]) if se2[2] > 0 else float("nan"))
    ntot = sum(len(v) for v in preds_all["P"])
    out.qlike = {c: qnum[c] / ntot for c in COMBOS}
    out.r2 = {c: 1 - sse[c] / sst for c in COMBOS}
    out.contrast_q = {"A|P": out.qlike["P"] - out.qlike["P+A"],
                      "A|P,RV": out.qlike["P+RV"] - out.qlike["P+RV+A"],
                      "RV|P,A": out.qlike["P+A"] - out.qlike["P+RV+A"]}
    out.contrast_r2 = {"A|P": out.r2["P+A"] - out.r2["P"],
                       "A|P,RV": out.r2["P+RV+A"] - out.r2["P+RV"],
                       "RV|P,A": out.r2["P+RV+A"] - out.r2["P+A"]}
    out.wins = wins
    out.beta_traj = betas
    full = betas["P+RV+A"]
    for term in ("sig", "rankV", "HA"):
        vals = np.array([b[term] for b in full])
        out.sign_flip[term] = float(np.mean(vals < 0))
        out.near_zero[term] = float(np.mean(np.abs(vals) < 1e-6))
    P = np.concatenate(preds_all["P"])
    R = np.concatenate(preds_all["RV"])
    A = np.concatenate(preds_all["A"])
    out.corr_pred = {"P,RV": float(np.corrcoef(P, R)[0, 1]),
                     "P,A": float(np.corrcoef(P, A)[0, 1]),
                     "RV,A": float(np.corrcoef(R, A)[0, 1])}
    PA = np.concatenate(preds_all["P+A"])
    PR = np.concatenate(preds_all["P+RV"])
    out.corr_incr = float(np.corrcoef(PA - P, PR - P)[0, 1])
    out.dm_t, out.dm_p = _dm_hac4(np.asarray(d_head))
    out.perm_p = _perm_pvalue(a, tgt, dates)
    ht = np.asarray(ha_t)
    out.ha_perp_t = float(np.nanmean(ht))
    return out


def _forward_contrast(a: pd.DataFrame, tgt: np.ndarray, dates: pd.Series, order: list[int]) -> float:
    num = den = 0.0
    months = [tgt[np.intersect1d(_blocks(len(dates)),
                                 np.where((dates >= o).to_numpy()
                                           & (dates < str(pd.Timestamp(o) + pd.offsets.MonthBegin(1))).to_numpy())[0])]
              for o in ORIGINS]
    for k, origin in enumerate(ORIGINS):
        nxt = str(pd.Timestamp(origin) + pd.offsets.MonthBegin(1))
        tr = np.where((dates < origin).to_numpy())[0]
        fw = np.where((dates >= origin).to_numpy() & (dates < nxt).to_numpy())[0]
        tb = np.intersect1d(_blocks(len(dates)), tr)
        fb = np.intersect1d(_blocks(len(dates)), fw)
        Ct, yt = {c: a[c].to_numpy()[tb] for c in ("sig", "rankV", "HA")}, tgt[tb]
        okt = np.isfinite(yt) & np.isfinite(Ct["sig"]) & np.isfinite(Ct["rankV"]) & np.isfinite(Ct["HA"])
        qfloor = float(np.quantile(yt[okt], QLIKE_FLOOR_Q))
        W = {}
        for c in ("P+RV", "P+RV+A"):
            X = np.column_stack([np.ones(okt.sum())] + [Ct[m][okt] for m in MEMBERS[c]])
            W[c] = _fit(X, yt[okt])[0]
        Cf = {c: a[c].to_numpy()[fb] for c in ("sig", "rankV", "HA")}
        yf = months[order[k]][: len(fb)]
        n = min(len(yf), len(Cf["sig"]))
        yf = yf[:n]
        Cf = {c: Cf[c][:n] for c in Cf}
        okf = np.isfinite(yf) & np.isfinite(Cf["sig"]) & np.isfinite(Cf["rankV"]) & np.isfinite(Cf["HA"])
        yf = yf[okf]
        if len(yf) < 3:
            continue
        p3 = np.column_stack([np.ones(okf.sum())] + [Cf[m][okf] for m in MEMBERS["P+RV+A"]]) @ W["P+RV+A"]
        p2 = np.column_stack([np.ones(okf.sum())] + [Cf[m][okf] for m in MEMBERS["P+RV"]]) @ W["P+RV"]
        num += qlike(yf, p3, qfloor) * len(yf) - qlike(yf, p2, qfloor) * len(yf)
        den += len(yf)
    return num / den if den else 0.0


def _perm_pvalue(a: pd.DataFrame, tgt: np.ndarray, dates: pd.Series) -> float:
    rng = np.random.default_rng(8080)
    obs = _forward_contrast(a, tgt, dates, list(range(len(ORIGINS))))
    n_ge = sum(1 for _ in range(N_PERM) if _forward_contrast(a, tgt, dates, list(rng.permutation(len(ORIGINS)))) <= obs)
    return (n_ge + 1) / (N_PERM + 1)
