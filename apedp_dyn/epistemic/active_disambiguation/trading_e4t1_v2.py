"""E4-T1 trading transfer v2 (forensics doc S72): active disambiguation, block basis.

Frozen v2 (v1 abandoned: d1 target carries ~3% signal; hypotheses are
near-constant predictors there, D<=0.06σ everywhere -> systematic abstention;
July-2022 smoke origin discarded, origins start 2022-08-01):
  Hypotheses (5-day mean volatility level; OLS a+b*f per origin on train
  NON-OVERLAPPING 5d blocks; overlapping windows would overstate precision):
    H1 log-volume | H2 rank-volume | H3 H-A composite (same panels as v1)
  Candidate measurements (observable consequences, 8):
    m1 d5 BTC | m2 d5 BTC|hi-trailvol | m3 d5 BTC|hi-volrank | m4 d5 BTC|neg-start
    m5 d1 BTC (negative control) | m6 d5 ETH transfer | m7 d5 SOL transfer
    m8 d5 BTC|low-trailvol
  Per rolling origin (2022-08..2023-06, 11 origins): train blocks -> fit +
  MULTI check (P1) + D_sigma table + frozen selection -> forward-month blocks
  -> re-adjudicate forward-only. Policies: active/random/worst/
  persistence-fixed(m1)/oracle. Endpoint epistemic only, never a strategy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.linalg as la
import pandas as pd

from apedp_dyn.epistemic.adjudicate import adjudicate
from apedp_dyn.grammar.candidate_ir import CandidateDynamics, ScoredCandidate, Term

ORIGINS = pd.date_range("2022-08-01", "2023-06-01", freq="MS").strftime("%Y-%m-%d").tolist()
DETECT_K = 3.0
FEATS = {"H1": "logV", "H2": "rankV", "H3": "HA"}
HIDS = ("H1", "H2", "H3")
MEASUREMENT_IDS = ("m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8")


@dataclass(frozen=True)
class OriginResult:
    origin: str
    pre_status: str
    policy: str
    mid: str | None
    d_sigma: float
    post_status: str
    winner: str | None


def _score(hid: str, resid: np.ndarray) -> ScoredCandidate:
    resid = np.asarray(resid, dtype=float)
    resid = resid[np.isfinite(resid)]
    c = CandidateDynamics(lhs="y", terms=(Term(hid, "a0"),), candidate_id=hid)
    rmse = float(np.sqrt(np.mean(resid**2)))
    return ScoredCandidate(c, {"gate4_status": {"passed": True, "holdout_residuals": list(resid),
                                                "holdout_rmse": rmse, "holdout_obs_rms": 1.0}},
                           train_loss=float(np.mean(resid**2)), holdout_rmse=rmse)


def _adjudicate(scored: list[ScoredCandidate]) -> tuple[str, str | None]:
    scored = sorted(scored,
                    key=lambda s: float(s.holdout_rmse)
                    if s.holdout_rmse is not None else float("inf"))
    adj = adjudicate(list(scored),
                     resid_best=np.asarray(scored[0].provenance["gate4_status"]["holdout_residuals"]),
                     resid_rival=np.asarray(scored[1].provenance["gate4_status"]["holdout_residuals"]),
                     bootstrap=False, data_origin="real")
    status = str(getattr(adj, "status", adj))
    winner = scored[0].candidate.candidate_id if "unique_support" in status.lower() else None
    return status, winner


def _blocks(n: int) -> np.ndarray:
    idx = np.arange(n)
    return np.asarray(idx[idx % 5 == 0])


def _d5(sig: np.ndarray) -> np.ndarray:
    return np.asarray(pd.Series(sig).rolling(5).mean().shift(-5).to_numpy())


def _design(panel: pd.DataFrame, asset: str, coefs: dict[str, np.ndarray],
            idx: np.ndarray, target: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Predictions + residuals per hypothesis on index set (transfer-ready)."""
    pred, resid = {}, {}
    y = target[idx]
    for h in HIDS:
        f = panel[FEATS[h]].to_numpy()[idx]
        p = coefs[h][0] + coefs[h][1] * f
        pred[h], resid[h] = p, y - p
    return pred, resid


def run_origin(panels: dict[str, pd.DataFrame], origin: str, policy: str, seed: int) -> OriginResult:
    dates = panels["BTC"]["date"]
    nxt = str(pd.Timestamp(origin) + pd.offsets.MonthBegin(1))
    tr_idx = np.where((dates < origin).to_numpy())[0]
    fw_idx = np.where((dates >= origin).to_numpy() & (dates < nxt).to_numpy())[0]
    tr_blocks = np.intersect1d(_blocks(len(dates)), tr_idx)
    btc = panels["BTC"]
    d5_btc = _d5(btc["sig"].to_numpy())
    # Fit on train blocks.
    coefs: dict[str, np.ndarray] = {}
    for h in HIDS:
        f = btc[FEATS[h]].to_numpy()[tr_blocks]
        y = d5_btc[tr_blocks]
        ok = np.isfinite(f) & np.isfinite(y)
        X = np.column_stack([np.ones(ok.sum()), f[ok]])
        w, *_ = la.lstsq(X, y[ok], rcond=None)
        coefs[h] = w
    # P1: pre-status on train-block residuals.
    _, r_tr = _design(btc, "BTC", coefs, tr_blocks, d5_btc)
    pre_status, _ = _adjudicate([_score(h, r_tr[h]) for h in HIDS])
    # Regime cuts frozen from train blocks.
    tv = btc["trailvol20"].to_numpy()
    tv_tr = tv[tr_blocks]
    tv_tr = tv_tr[np.isfinite(tv_tr)]
    hi_cut, lo_cut = np.quantile(tv_tr, [2 / 3, 1 / 3])
    vr_tr = btc["rankV"].to_numpy()[tr_blocks]
    vr_tr = vr_tr[np.isfinite(vr_tr)]
    vr_cut = np.quantile(vr_tr, 2 / 3)
    # Candidate table (train blocks only).
    d1_btc = btc["sig"].shift(-1).to_numpy()
    rows: dict[str, tuple[dict[str, float], float, float]] = {}
    for mid in MEASUREMENT_IDS:
        if mid in ("m6", "m7"):
            asset = "ETH" if mid == "m6" else "SOL"
            a = panels[asset]
            tgt = _d5(a["sig"].to_numpy())
            sub = tr_blocks
        elif mid == "m5":
            asset, a, tgt, sub = "BTC", btc, d1_btc, tr_idx
        else:
            asset, a, tgt, sub = "BTC", btc, d5_btc, tr_blocks
        tvi = tv[sub]
        if mid in ("m2",):
            sub = sub[tvi > hi_cut]
        elif mid == "m8":
            sub = sub[tvi < lo_cut]
        elif mid == "m3":
            sub = sub[a["rankV"].to_numpy()[sub] > vr_cut]
        elif mid == "m4":
            sub = sub[a["r"].to_numpy()[sub] < 0]
        pred, _resid = _design(a, asset, coefs, sub, tgt)
        fin = {h: pred[h][np.isfinite(pred[h]) & np.isfinite(tgt[sub])] for h in HIDS}
        n_min = min(len(v) for v in fin.values())
        if n_min < 5:
            rows[mid] = ({h: float("nan") for h in HIDS}, 0.0, float("nan"))
            continue
        fin = {h: v[:n_min] for h, v in fin.items()}
        rr = {h: (tgt[sub][np.isfinite(pred[h]) & np.isfinite(tgt[sub])][:n_min] - fin[h]) for h in HIDS}
        sig_m = float(np.sqrt(np.mean(rr["H1"] ** 2)))
        sep = min(float(np.sqrt(np.mean((fin[a] - fin[b]) ** 2))) for a in HIDS for b in HIDS if a < b)
        rows[mid] = ({h: float(np.mean(fin[h])) for h in HIDS}, sep, sig_m)
    preds = {mid: v[0] for mid, v in rows.items() if np.isfinite(v[2]) and v[2] > 0}
    # §71 policy in trading form: rank by frozen (sep/sigma), abstain below k.
    ranked = sorted(preds, key=lambda m: rows[m][1] / rows[m][2], reverse=True)
    d_of = {m: rows[m][1] / rows[m][2] for m in ranked}
    order = ranked
    if policy == "active":
        best = order[0]
        if d_of[best] < DETECT_K:
            return OriginResult(origin, pre_status, policy, None, d_of[best],
                                "INSUFFICIENT_DISCRIMINATION", None)
        mid, d = best, d_of[best]
    elif policy == "random":
        rng = np.random.default_rng(seed)
        mid = order[rng.integers(len(order))]
        d = d_of[mid]
    elif policy == "worst":
        mid, d = order[-1], d_of[order[-1]]
    elif policy == "persistence":
        mid, d = "m1", d_of["m1"]
    elif policy == "oracle":
        d = float("nan")
    else:
        raise ValueError(policy)
    if policy == "oracle":
        best_c, best_margin = None, -1.0
        for mid_c in MEASUREMENT_IDS:
            rms, _ = _forward_rms(panels, coefs, mid_c, fw_idx, hi_cut, lo_cut, vr_cut)
            vals = sorted(rms.values())
            margin = vals[1] - vals[0]
            if np.isfinite(margin) and margin > best_margin:
                best_c, best_margin = mid_c, margin
        mid = best_c if best_c is not None else mid
    _, fwd = _forward_rms(panels, coefs, mid, fw_idx, hi_cut, lo_cut, vr_cut)
    post_status, winner = _adjudicate([_score(h, fwd[h]) for h in HIDS])
    return OriginResult(origin, pre_status, policy, mid, float(d), post_status, winner)


def _forward_rms(panels, coefs, mid, fw_idx, hi_cut, lo_cut, vr_cut):
    rms, fwd = {}, {}
    if mid in ("m6", "m7"):
        asset = "ETH" if mid == "m6" else "SOL"
        a = panels[asset]
        tgt = _d5(a["sig"].to_numpy())
        sub = np.intersect1d(_blocks(len(tgt)), fw_idx)
    elif mid == "m5":
        asset, a, tgt, sub = "BTC", panels["BTC"], panels["BTC"]["sig"].shift(-1).to_numpy(), fw_idx
    else:
        asset, a = "BTC", panels["BTC"]
        tgt = _d5(a["sig"].to_numpy())
        sub = np.intersect1d(_blocks(len(tgt)), fw_idx)
        tv = a["trailvol20"].to_numpy()[sub]
        if mid == "m2":
            sub = sub[tv > hi_cut]
        elif mid == "m8":
            sub = sub[tv < lo_cut]
        elif mid == "m3":
            sub = sub[a["rankV"].to_numpy()[sub] > vr_cut]
        elif mid == "m4":
            sub = sub[a["r"].to_numpy()[sub] < 0]
    y = tgt[sub]
    for h in HIDS:
        f = a[FEATS[h]].to_numpy()[sub]
        ok = np.isfinite(y) & np.isfinite(f)
        r = y[ok] - (coefs[h][0] + coefs[h][1] * f[ok])
        fwd[h] = r
        rms[h] = float(np.sqrt(np.mean(r**2))) if len(r) else float("inf")
    return rms, fwd
