"""§79 conditional ensemble (forensics doc): can incremental utility be gated
prospectively without hindsight? Frozen, no new hypotheses (H-RV dropped:
§78 showed it dead weight once H-A present).
  Models: M0 = always H-P | M2 = always H-P+H-A | MAD = adaptive H-P/H-P+H-A.
  Gate (train-only, per origin): split train blocks 70/30 fit/val (time-ordered);
  fit M0/M2 on fit-split; admit H-A iff M2 val-MSE < M0 val-MSE AND M2 wins a
  strict majority of val blocks. No OOS lookahead, no tuned constants.
  Primary metric: pooled forward QLIKE (y/yhat - log(y/yhat) - 1, eps-guarded).
  Secondary: pooled forward R2 vs train mean. Control: month-block permutation
  of the forward target (200 draws, statistic = pooled dR2 M2-M0). Report:
  per-origin tables, never pooled-only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.linalg as la
import pandas as pd

from apedp_dyn.epistemic.active_disambiguation.trading_e4t1_v2 import ORIGINS, _blocks, _d5

EPS = 1e-8
N_PERM = 200
PERM_SEED = 6161


def qlike(y: np.ndarray, pred: np.ndarray, floor: float) -> float:
    y = np.maximum(np.asarray(y, dtype=float), EPS)
    p = np.maximum(np.asarray(pred, dtype=float), floor)
    r = y / p
    return float(np.mean(r - np.log(r) - 1.0))


def _fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    w, *_ = la.lstsq(X, y, rcond=None)
    return np.asarray(w)


@dataclass
class OriginOut:
    origin: str
    gate_open: bool
    r2_m0: float
    r2_m2: float
    r2_ad: float
    q_m0: float
    q_m2: float
    q_ad: float


@dataclass
class AssetOut:
    asset: str
    origins: list[OriginOut] = field(default_factory=list)
    perm_p: float = float("nan")


def _cols(a: pd.DataFrame, idx: np.ndarray) -> dict[str, np.ndarray]:
    return {c: a[c].to_numpy()[idx] for c in ("sig", "HA")}


def run_asset(panels: dict[str, pd.DataFrame], asset: str) -> AssetOut:
    a = panels[asset]
    dates = a["date"]
    tgt = _d5(a["sig"].to_numpy())
    out = AssetOut(asset)
    for origin in ORIGINS:
        nxt = str(pd.Timestamp(origin) + pd.offsets.MonthBegin(1))
        tr = np.where((dates < origin).to_numpy())[0]
        fw = np.where((dates >= origin).to_numpy() & (dates < nxt).to_numpy())[0]
        tb = np.intersect1d(_blocks(len(dates)), tr)
        fb = np.intersect1d(_blocks(len(dates)), fw)
        # Train-only gate: time-ordered 70/30 fit/val split within train blocks.
        cut = tb[: int(len(tb) * 0.7)]
        val = tb[int(len(tb) * 0.7):]
        yc, yv = tgt[cut], tgt[val]
        Cc, Cv = _cols(a, cut), _cols(a, val)
        okc = np.isfinite(yc) & np.isfinite(Cc["sig"]) & np.isfinite(Cc["HA"])
        okv = np.isfinite(yv) & np.isfinite(Cv["sig"]) & np.isfinite(Cv["HA"])
        w0 = _fit(np.column_stack([np.ones(okc.sum()), Cc["sig"][okc]]), yc[okc])
        w2 = _fit(np.column_stack([np.ones(okc.sum()), Cc["sig"][okc], Cc["HA"][okc]]), yc[okc])
        Xv0 = np.column_stack([np.ones(okv.sum()), Cv["sig"][okv]])
        Xv2 = np.column_stack([np.ones(okv.sum()), Cv["sig"][okv], Cv["HA"][okv]])
        e0, e2 = yv[okv] - Xv0 @ w0, yv[okv] - Xv2 @ w2
        gate = bool(np.mean(e2**2) < np.mean(e0**2) and np.mean(e2**2 < e0**2) > 0.5)
        # Refit both on full train; evaluate forward.
        yt = tgt[tb]
        Ct = _cols(a, tb)
        okt = np.isfinite(yt) & np.isfinite(Ct["sig"]) & np.isfinite(Ct["HA"])
        mu = float(np.mean(yt[okt]))
        W0 = _fit(np.column_stack([np.ones(okt.sum()), Ct["sig"][okt]]), yt[okt])
        W2 = _fit(np.column_stack([np.ones(okt.sum()), Ct["sig"][okt], Ct["HA"][okt]]), yt[okt])
        yf = tgt[fb]
        Cf = _cols(a, fb)
        okf = np.isfinite(yf) & np.isfinite(Cf["sig"]) & np.isfinite(Cf["HA"])
        yf = yf[okf]
        if len(yf) < 3:
            continue
        p0 = np.column_stack([np.ones(okf.sum()), Cf["sig"][okf]]) @ W0
        p2 = np.column_stack([np.ones(okf.sum()), Cf["sig"][okf], Cf["HA"][okf]]) @ W2
        pad = p2 if gate else p0
        den = float(np.sum((yf - mu) ** 2))
        qfloor = float(np.quantile(yt[okt], 0.05))

        def r2(p, yf=yf, den=den):
            return 1 - float(np.sum((yf - p) ** 2)) / den if den > 0 else float("nan")
        out.origins.append(OriginOut(origin, gate, r2(p0), r2(p2), r2(pad),
                                     qlike(yf, p0, qfloor), qlike(yf, p2, qfloor), qlike(yf, pad, qfloor)))
    # Month-block permutation control on pooled forward dR2 (M2-M0).
    out.perm_p = _perm_pvalue(a, tgt, dates)
    return out


def _forward_dr2(a: pd.DataFrame, tgt: np.ndarray, dates: pd.Series, order: list[int]) -> float:
    num = den = 0.0
    for k, origin in enumerate(ORIGINS):
        nxt = str(pd.Timestamp(origin) + pd.offsets.MonthBegin(1))
        tr = np.where((dates < origin).to_numpy())[0]
        fw = np.where((dates >= origin).to_numpy() & (dates < nxt).to_numpy())[0]
        tb = np.intersect1d(_blocks(len(dates)), tr)
        fb = np.intersect1d(_blocks(len(dates)), fw)
        yt = tgt[tb]
        Ct = {c: a[c].to_numpy()[tb] for c in ("sig", "HA")}
        okt = np.isfinite(yt) & np.isfinite(Ct["sig"]) & np.isfinite(Ct["HA"])
        mu = float(np.mean(yt[okt]))
        W0 = _fit(np.column_stack([np.ones(okt.sum()), Ct["sig"][okt]]), yt[okt])
        W2 = _fit(np.column_stack([np.ones(okt.sum()), Ct["sig"][okt], Ct["HA"][okt]]), yt[okt])
        Cf = {c: a[c].to_numpy()[fb] for c in ("sig", "HA")}
        # Permuted forward target: whole-month blocks reordered by `order`.
        months = [tgt[np.intersect1d(_blocks(len(dates)),
                                     np.where((dates >= o).to_numpy()
                                               & (dates < str(pd.Timestamp(o) + pd.offsets.MonthBegin(1))).to_numpy())[0])]
                  for o in ORIGINS]
        yf = months[order[k]][: len(fb)]
        n = min(len(yf), len(Cf["sig"]))
        yf = yf[:n]
        Cf = {c: Cf[c][:n] for c in Cf}
        okf = np.isfinite(yf) & np.isfinite(Cf["sig"]) & np.isfinite(Cf["HA"])
        yf = yf[okf]
        if len(yf) < 3:
            continue
        p0 = np.column_stack([np.ones(okf.sum()), Cf["sig"][okf]]) @ W0
        p2 = np.column_stack([np.ones(okf.sum()), Cf["sig"][okf], Cf["HA"][okf]]) @ W2
        num += float(np.sum((yf - p2) ** 2) - np.sum((yf - p0) ** 2))
        den += float(np.sum((yf - mu) ** 2))
    return -num / den if den > 0 else 0.0


def _perm_pvalue(a: pd.DataFrame, tgt: np.ndarray, dates: pd.Series) -> float:
    rng = np.random.default_rng(PERM_SEED)
    obs = _forward_dr2(a, tgt, dates, list(range(len(ORIGINS))))
    n_ge = 0
    for _ in range(N_PERM):
        order = list(rng.permutation(len(ORIGINS)))
        if _forward_dr2(a, tgt, dates, order) >= obs:
            n_ge += 1
    return (n_ge + 1) / (N_PERM + 1)
