"""M0-M3 ensemble forecasting test (forensics doc S78): incremental predictive
utility, NOT structural truth. Frozen: OLS stacking weights fit per origin
on train blocks only; evaluated on forward-month blocks; OOS R2 against the
train mean (can go negative); naive baseline = train mean (M_hist).
  M0 = H-P | M1 = H-P+H-RV | M2 = H-P+H-A | M3 = H-P+H-RV+H-A.
Question: does M3 beat M0 out-of-sample after overlap/complexity? Overlap
(H-P subset H-A; logV/rankV ~99% corr) is handled by joint OLS, not ignored.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.linalg as la
import pandas as pd

from apedp_dyn.epistemic.active_disambiguation.trading_e4t1_v2 import (
    ORIGINS,
    _blocks,
    _d5,
    _train_blocks_no_leak,
)

ENSEMBLES = ("M_hist", "M0", "M1", "M2", "M3")
MEMBERS = {"M0": ("sig",), "M1": ("sig", "rankV"), "M2": ("sig", "HA"), "M3": ("sig", "rankV", "HA")}


@dataclass(frozen=True)
class EnsembleResult:
    asset: str
    r2: dict
    delta_m3_m0: float
    win_rate_m3_m0: float
    n_origins: int


def run_asset(panels: dict[str, pd.DataFrame], asset: str) -> EnsembleResult:
    a = panels[asset]
    dates = a["date"]
    tgt = _d5(a["sig"].to_numpy())
    sse: dict[str, float] = dict.fromkeys(ENSEMBLES, 0.0)
    sst = 0.0
    wins = 0
    n = 0
    per_origin_m3: list[float] = []
    per_origin_m0: list[float] = []
    for origin in ORIGINS:
        nxt = str(pd.Timestamp(origin) + pd.offsets.MonthBegin(1))
        fw = np.where((dates >= origin).to_numpy() & (dates < nxt).to_numpy())[0]
        # FLAW-A01: d5 carries a forward 5-day span; train blocks within 5
        # days of the origin leak forward data through shift(-5).
        tb = _train_blocks_no_leak(dates, origin)
        fb = np.intersect1d(_blocks(len(dates)), fw)
        cols_tr = {c: a[c].to_numpy()[tb] for c in ("sig", "rankV", "HA")}
        y_tr = tgt[tb]
        ok = np.isfinite(y_tr)
        for c in cols_tr:
            ok &= np.isfinite(cols_tr[c])
        mu_tr = float(np.mean(y_tr[ok]))
        w: dict[str, np.ndarray] = {}
        for e, members in MEMBERS.items():
            X = np.column_stack([np.ones(ok.sum())] + [cols_tr[c][ok] for c in members])
            w[e], *_ = la.lstsq(X, y_tr[ok], rcond=None)
        cols_fw = {c: a[c].to_numpy()[fb] for c in ("sig", "rankV", "HA")}
        y_fw = tgt[fb]
        okf = np.isfinite(y_fw)
        for c in cols_fw:
            okf &= np.isfinite(cols_fw[c])
        yf = y_fw[okf]
        if len(yf) < 3:
            continue
        preds = {"M_hist": np.full(len(yf), mu_tr)}
        for e, members in MEMBERS.items():
            Xf = np.column_stack([np.ones(okf.sum())] + [cols_fw[c][okf] for c in members])
            preds[e] = Xf @ w[e]
        for e in ENSEMBLES:
            sse[e] += float(np.sum((yf - preds[e]) ** 2))
        sst += float(np.sum((yf - mu_tr) ** 2))
        per_origin_m3.append(float(np.sum((yf - preds["M3"]) ** 2)))
        per_origin_m0.append(float(np.sum((yf - preds["M0"]) ** 2)))
        wins += per_origin_m3[-1] < per_origin_m0[-1]
        n += 1
    r2 = {e: (1 - sse[e] / sst if sst > 0 else float("nan")) for e in ENSEMBLES}
    return EnsembleResult(asset, r2, r2["M3"] - r2["M0"], wins / n if n else float("nan"), n)
