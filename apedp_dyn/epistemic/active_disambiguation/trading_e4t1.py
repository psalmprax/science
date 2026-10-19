"""E4-T1 trading transfer (forensics doc S72): active disambiguation on volume-family ambiguity.

Frozen protocol (all definitions fixed before any forward window is opened):
  Hypotheses (BTC next-day volatility level, OLS a+b*f fitted per origin on train):
    H1 log-volume:   f = log(V)
    H2 rank-volume:  f = causal trailing-40d rank of V
    H3 H-A composite: f = z(log RollMean40(V)) + z(|r|) + z(|r|/V), expanding z
  Candidate measurements (observable consequences of H1/H2/H3, NOT features):
    m1 d1 BTC | m2 d5-mean BTC | m3 d1 BTC|hi-trailing-vol | m4 d1 BTC|hi-vol-rank
    m5 d1 BTC|after-negative-day | m6 d5-mean BTC|hi-trailing-vol
    m7 d1 ETH (BTC-fitted transfer) | m8 d1 SOL (BTC-fitted transfer)
  Time order per rolling origin: train (< origin month) -> fit + MULTI check +
  D_sigma table + frozen selection -> forward (origin month) observe ->
  re-adjudicate on forward residuals only.
  Policies: active / random / worst / persistence-fixed(m1) / oracle(post-hoc).
  Endpoint is epistemic (UNIQUE/MULTI/INSUFFICIENT), never a trading strategy.

Constraint honored: no "find another feature" step anywhere; the candidate set
is generated from the observable consequences of the competing mechanisms.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.linalg as la
import pandas as pd

from apedp_dyn.epistemic.active_disambiguation.policy import discrimination_table
from apedp_dyn.epistemic.adjudicate import adjudicate
from apedp_dyn.grammar.candidate_ir import CandidateDynamics, ScoredCandidate, Term

TRAIN_START = "2021-01-01"
ORIGINS = pd.date_range("2022-07-01", "2023-06-01", freq="MS").strftime("%Y-%m-%d").tolist()
DETECT_K = 3.0
ASSETS = ("BTC", "ETH", "SOL")

MEASUREMENT_IDS = ("m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8")


def load_frames(path: str = "data/market/crypto_top15_orderflow.csv") -> dict[str, pd.DataFrame]:
    df = pd.read_csv(path, parse_dates=["observation_date"]).sort_values("observation_date").reset_index(drop=True)
    return {"raw": df}


def build_asset_panel(df: pd.DataFrame, asset: str) -> pd.DataFrame:
    """Causal feature panel for one asset (all rolling quantities min_periods-pure)."""
    px = df[f"{asset}_CLOSE"].to_numpy(float)
    V = df[f"{asset}_VOL"].to_numpy(float)
    r = np.concatenate([[np.nan], np.log(px[1:] / px[:-1])])
    sig = np.abs(r)
    rm40 = pd.Series(V).rolling(40).mean().to_numpy()
    rank40 = pd.Series(V).rolling(40).apply(lambda w: (pd.Series(w).rank().iloc[-1] - 1) / 39, raw=False).to_numpy()
    ami = np.abs(r) / np.maximum(V, 1e-12)
    s = pd.Series
    def z(v):
        return ((s(v) - s(v).expanding(60).mean()) / s(v).expanding(60).std()).to_numpy()
    with np.errstate(all="ignore"):
        HA = z(np.log(np.maximum(rm40, 1e-12)) + z(np.abs(r)) + z(ami))
    return pd.DataFrame({
        "date": df["observation_date"], "r": r, "sig": sig, "V": V,
        "logV": np.log(np.maximum(V, 1e-12)), "rankV": rank40,
        "HA": HA, "trailvol20": s(sig).rolling(20).mean().to_numpy(),
    })


def fit_btc(df_btc: pd.DataFrame, mask: np.ndarray) -> dict[str, np.ndarray]:
    """OLS a+b*f per hypothesis on train mask (target: next-1d |r|)."""
    tgt = df_btc["sig"].shift(-1).to_numpy()
    coefs = {}
    for hid, col in (("H1", "logV"), ("H2", "rankV"), ("H3", "HA")):
        ok = mask & np.isfinite(df_btc[col].to_numpy()) & np.isfinite(tgt)
        X = np.column_stack([np.ones(ok.sum()), df_btc[col].to_numpy()[ok]])
        w, *_ = la.lstsq(X, tgt[ok], rcond=None)
        coefs[hid] = w
    return coefs


def predict_level(coefs: dict[str, np.ndarray], feats: dict[str, float]) -> dict[str, float]:
    return {h: float(w[0] + w[1] * feats[h]) for h, w in coefs.items()}


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
    rmse = float(np.sqrt(np.mean(np.asarray(resid) ** 2)))
    return ScoredCandidate(c, {"gate4_status": {"passed": True, "holdout_residuals": list(resid),
                                                "holdout_rmse": rmse, "holdout_obs_rms": 1.0}},
                           train_loss=float(np.mean(np.asarray(resid) ** 2)), holdout_rmse=rmse)


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


def run_origin(panels: dict[str, pd.DataFrame], origin: str, policy: str, seed: int) -> OriginResult:
    """One rolling origin. Forward data (>origin month) never enters selection."""
    dates = panels["BTC"]["date"]
    m_tr = (dates < origin).to_numpy()
    m_fw = (dates >= origin).to_numpy() & (dates < str(pd.Timestamp(origin) + pd.offsets.MonthBegin(1)))
    btc = panels["BTC"]
    coefs = fit_btc(btc, m_tr)
    feats = {"H1": "logV", "H2": "rankV", "H3": "HA"}
    # P1: pre-status on train next-1d residuals.
    tgt_tr = btc["sig"].shift(-1).to_numpy()
    pre = [_score(h, (tgt_tr - (coefs[h][0] + coefs[h][1] * btc[feats[h]].to_numpy()))[m_tr]) for h in feats]
    pre_status, _ = _adjudicate(pre)
    # Candidate table from train fits (observable consequences only).
    def rm(s):
        return float(np.sqrt(np.mean(s**2))) if len(s) else float("nan")
    rows: dict[str, tuple[dict[str, float], float, float]] = {}
    ftr = {h: btc[feats[h]].to_numpy() for h in feats}
    base_pred = {h: coefs[h][0] + coefs[h][1] * ftr[h] for h in feats}
    d1 = btc["sig"].shift(-1).to_numpy()
    d5 = btc["sig"].rolling(5).mean().shift(-5).to_numpy()
    tv = btc["trailvol20"].to_numpy()
    tv_cut = np.nanquantile(tv[m_tr], 2 / 3)
    hi_tv, hi_vr = tv > tv_cut, btc["rankV"].to_numpy() > 2 / 3
    neg = btc["r"].to_numpy() < 0
    subsets = {"m1": m_tr, "m2": m_tr, "m3": m_tr & hi_tv, "m4": m_tr & hi_vr,
               "m5": m_tr & neg, "m6": m_tr & hi_tv, "m7": m_tr, "m8": m_tr}
    targets = {"m1": d1, "m2": d5, "m3": d1, "m4": d1, "m5": d1, "m6": d5, "m7": None, "m8": None}
    for mid in MEASUREMENT_IDS:
        if mid in ("m7", "m8"):
            asset = "ETH" if mid == "m7" else "SOL"
            a = panels[asset]
            fp = {h: a[feats[h]].to_numpy() for h in feats}
            pred = {h: coefs[h][0] + coefs[h][1] * fp[h] for h in feats}
            ty = a["sig"].shift(-1).to_numpy()
        else:
            pred, ty = base_pred, targets[mid]
        sub = subsets[mid] & np.isfinite(ty)
        for h in feats:
            pv = pred[h] if isinstance(pred[h], np.ndarray) else np.full(len(ty), pred[h])
            sub &= np.isfinite(pv)
        ph, yh = {}, ty[sub]
        for h in feats:
            ph[h] = pred[h][sub] if isinstance(pred[h], np.ndarray) else pred[h]
        ph = {h: (v if isinstance(v, np.ndarray) else np.full(len(yh), v)) for h, v in ph.items()}
        resid = {h: yh - ph[h] for h in feats}
        resid = {h: v[np.isfinite(v)] for h, v in resid.items()}
        sig_m = rm(resid["H1"])
        fin = {h: ph[h][np.isfinite(ph[h])] for h in feats}
        n_min = min(len(v) for v in fin.values())
        fin = {h: v[:n_min] for h, v in fin.items()}
        sep = min(rm(fin[a] - fin[b]) for a in feats for b in feats if a < b)
        rows[mid] = ({h: float(np.mean(ph[h])) for h in feats}, sep, sig_m)
    preds = {mid: v[0] for mid, v in rows.items()}
    sigmas = {mid: v[2] for mid, v in rows.items()}
    table = discrimination_table(preds, sigmas)
    order = [r.mid for r in table]
    d_of = {r.mid: r.d_sigma for r in table}
    if policy == "active":
        # §71 policy, trading form: discrimination_table IS the max-D_sigma
        # ranking (mean-prediction consequences + train noise scale); the
        # k-threshold abstention is the P7 rule. No point-query wrapper.
        best = table[0]
        if best.d_sigma < DETECT_K:
            return OriginResult(origin, pre_status, policy, None, best.d_sigma,
                                "INSUFFICIENT_DISCRIMINATION", None)
        mid, d = best.mid, best.d_sigma
    elif policy == "random":
        rng = np.random.default_rng(seed)
        mid = order[rng.integers(len(order))]
        d = d_of[mid]
    elif policy == "worst":
        mid, d = order[-1], d_of[order[-1]]
    elif policy == "persistence":
        mid, d = "m1", d_of["m1"]
    elif policy == "oracle":
        d = -1.0
    else:
        raise ValueError(policy)
    # Observe on forward month (regime labels causal), re-adjudicate forward-only.
    fw_idx = np.where(m_fw)[0]
    if policy == "oracle":
        best_c, best_margin = None, -1.0
        for mid_c in MEASUREMENT_IDS:
            st, _ = _forward_score(panels, coefs, feats, mid_c, fw_idx)
            vals = sorted(st.values())
            margin = vals[1] - vals[0]
            if margin > best_margin:
                best_c, best_margin = mid_c, margin
        mid = best_c if best_c is not None else mid
        d = float("nan")
    _, fwd = _forward_score(panels, coefs, feats, mid, fw_idx)
    scored = [_score(h, fwd[h]) for h in feats]
    post_status, winner = _adjudicate(scored)
    return OriginResult(origin, pre_status, policy, mid, float(d), post_status, winner)


def _forward_score(panels, coefs, feats, mid, fw_idx):
    rms, fwd = {}, {}
    if mid in ("m7", "m8"):
        asset = "ETH" if mid == "m7" else "SOL"
        a = panels[asset]
        ty = a["sig"].shift(-1).to_numpy()[fw_idx]
        ok = np.isfinite(ty)
        for h in feats:
            pr = (coefs[h][0] + coefs[h][1] * a[feats[h]].to_numpy()[fw_idx])[ok]
            fwd[h] = ty[ok] - pr
            rms[h] = float(np.sqrt(np.mean(fwd[h] ** 2)))
        return rms, fwd
    btc = panels["BTC"]
    ty = (btc["sig"].rolling(5).mean().shift(-5).to_numpy() if mid in ("m2", "m6")
          else btc["sig"].shift(-1).to_numpy())[fw_idx]
    tv = btc["trailvol20"].to_numpy()[fw_idx]
    ok = np.isfinite(ty)
    if mid == "m3" or mid == "m6":
        ok &= tv > np.nanquantile(btc["trailvol20"].to_numpy()[: fw_idx[0]], 2 / 3)
    if mid == "m4":
        ok &= btc["rankV"].to_numpy()[fw_idx] > 2 / 3
    if mid == "m5":
        ok &= btc["r"].to_numpy()[fw_idx] < 0
    for h in feats:
        pr = (coefs[h][0] + coefs[h][1] * btc[feats[h]].to_numpy()[fw_idx])[ok]
        fwd[h] = ty[ok] - pr
        rms[h] = float(np.sqrt(np.mean(fwd[h] ** 2))) if ok.sum() else float("inf")
    return rms, fwd
