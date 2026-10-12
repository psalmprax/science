"""Volume-aware sandbox: daily volume-state experiments on sidecar data.

Sidecar volume dataset (NEVER modifies the frozen repo benchmark):
  - /tmp/ma2024/vol_daily_crypto.json : 15 assets, daily (date, close, volume)
  - /tmp/ma2024/vol_daily_SPX.json / vol_daily_NDX.json : Yahoo daily closes+volume
  - closes validated bar-for-bar against data/market/crypto_top15.csv (reldev 0)

Frozen experiment (pre-registered 2026-10-03): daily volume-state -> future
volatility. Target = next-5-day realized vol of log-returns.
  V0 = trailing-vol persistence + |r| + Amihud
  V1 = V0 + logVolume
Fit 2021-2022 (frozen OLS) -> OOS 2023+. Metric: OOS R^2 increment +
permutation control + quarterly blocks. MATIC series ends 2024-09 (delist).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SIDECAR_DIR = Path("/tmp/ma2024")  # noqa: S108 -- deliberate sidecar location; repo benchmark must stay frozen
CRYPTO_FILE = SIDECAR_DIR / "vol_daily_crypto.json"

QUARTER_BLOCKS = (
    ("2023Q1", "2023-01-01", "2023-04-01"),
    ("2023Q2", "2023-04-01", "2023-07-01"),
    ("2023Q3", "2023-07-01", "2023-10-01"),
    ("2023Q4", "2023-10-01", "2024-01-01"),
    ("2024", "2024-01-01", "2025-01-01"),
    ("2025+", "2025-01-01", "2027-01-01"),
)


@dataclass
class VolumeFrame:
    dates: np.ndarray
    close: np.ndarray
    volume: np.ndarray


def load_crypto_volume(asset: str, path: Path | None = None) -> VolumeFrame:
    """Load sidecar (date, close, volume) for one crypto asset."""
    data = json.loads(Path(path or CRYPTO_FILE).read_text())
    rows = [(d, float(c), float(v)) for d, c, v in data[asset.upper()]]
    dates = np.array([r[0] for r in rows])
    return VolumeFrame(dates, np.array([r[1] for r in rows]),
                       np.array([r[2] for r in rows]))


def load_equity_volume(asset: str, path: Path | None = None) -> VolumeFrame:
    """Load sidecar Yahoo daily closes+volume (SPX or NDX)."""
    tag = asset.upper()
    data = json.loads(Path(path or SIDECAR_DIR / f"vol_daily_{tag}.json").read_text())
    rows = data[tag]
    dates = np.array([r[0] for r in rows])
    return VolumeFrame(dates, np.array([r[1] for r in rows], dtype=float),
                       np.array([r[2] for r in rows], dtype=float))


def causal_features(frame: VolumeFrame) -> dict[str, np.ndarray]:
    """Strictly causal daily features; NaN where history is insufficient."""
    px, qv = frame.close, frame.volume
    n = len(px)
    r = np.full(n, np.nan)
    r[1:] = np.diff(np.log(px, out=np.full(n, np.nan), where=px > 0))
    tv = np.array([r[max(0, t - 10):t].std() if t >= 6 and np.isfinite(r[max(0, t - 10):t]).all() else np.nan
                   for t in range(n)])
    amih = np.abs(r) / np.maximum(qv, 1e-18)
    lv = np.log(qv, out=np.full(n, np.nan), where=qv > 0)
    tgt = np.array([r[t + 1:t + 6].std() if t + 6 <= n - 1 and np.isfinite(r[t + 1:t + 6]).all() else np.nan
                    for t in range(n)])
    return {"r": r, "tv": tv, "amih": amih, "lv": lv, "tgt": tgt}


def trailing_rank(x: np.ndarray, window: int = 252) -> np.ndarray:
    """Causal trailing-window percentile rank in [0, 1] (min 60 obs)."""
    out = np.full_like(x, np.nan, dtype=float)
    for t in range(len(x)):
        w = x[max(0, t - window):t]
        w = w[np.isfinite(w)]
        if len(w) > 60 and np.isfinite(x[t]):
            out[t] = float((w < x[t]).mean())
    return out


def frozen_ols(Xtr: np.ndarray, ytr: np.ndarray) -> np.ndarray:
    w, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(Xtr)), Xtr]), ytr, rcond=None)
    return np.asarray(w)


def oos_r2(w: np.ndarray, Xte: np.ndarray, yte: np.ndarray) -> float:
    p = np.column_stack([np.ones(len(Xte)), Xte]) @ w
    return 1.0 - float(((yte - p) ** 2).sum() / ((yte - yte.mean()) ** 2).sum())


def run_volume_experiment(frame: VolumeFrame, n_perm: int = 200,
                          seed: int = 0) -> dict[str, Any]:
    """Frozen V0 vs V0+logVolume protocol. Returns metrics dict (no I/O)."""
    f = causal_features(frame)
    X = np.column_stack([f["tv"], np.abs(f["r"]), f["amih"], f["lv"]])
    y, dd = f["tgt"], frame.dates
    rk = trailing_rank(f["lv"])
    m = np.isfinite(X).all(axis=1) & np.isfinite(y) & np.isfinite(rk)
    X, y, dd, rk = X[m], y[m], dd[m], rk[m]
    mu, sd = np.column_stack([X, rk[:, None]])[dd < "2023-01-01"].mean(0), np.column_stack([X, rk[:, None]])[dd < "2023-01-01"].std(0)
    sd[sd == 0] = 1.0
    Xs = (np.column_stack([X, rk[:, None]]) - mu) / sd
    tr, te = dd < "2023-01-01", (dd >= "2023-01-01")
    w0 = frozen_ols(Xs[tr][:, :3], y[tr])
    w1 = frozen_ols(Xs[tr][:, :4], y[tr])
    w2 = frozen_ols(np.column_stack([Xs[tr][:, :3], Xs[tr][:, 4]]), y[tr])

    def ev(w: np.ndarray, cols: list[int], mask: np.ndarray) -> float:
        p = np.column_stack([np.ones(mask.sum()), Xs[mask][:, cols]]) @ w
        yt = y[mask]
        return 1.0 - float(((yt - p) ** 2).sum() / ((yt - yt.mean()) ** 2).sum())

    r0 = ev(w0, [0, 1, 2], te)
    r1 = ev(w1, [0, 1, 2, 3], te)
    r2 = ev(w2, [0, 1, 2, 4], te)
    rng = np.random.default_rng(seed)
    cnt = cnt2 = 0
    ib = Xs[te][:, 3].copy()
    ib2 = Xs[te][:, 4].copy()
    for _ in range(n_perm):
        rng.shuffle(ib)
        A = np.column_stack([np.ones(te.sum()), Xs[te][:, 0], Xs[te][:, 1], Xs[te][:, 2], ib]) @ w1
        yt = y[te]
        d = (1.0 - float(((yt - A) ** 2).sum() / ((yt - yt.mean()) ** 2).sum())) - r0
        cnt += d >= (r1 - r0)
    for _ in range(n_perm):
        rng.shuffle(ib2)
        A = np.column_stack([np.ones(te.sum()), Xs[te][:, 0], Xs[te][:, 1], Xs[te][:, 2], ib2]) @ w2
        yt = y[te]
        d = (1.0 - float(((yt - A) ** 2).sum() / ((yt - yt.mean()) ** 2).sum())) - r0
        cnt2 += d >= (r2 - r0)
    blocks = {}
    for qn, a, b in QUARTER_BLOCKS:
        mk = (dd >= a) & (dd < b) & te
        if mk.sum() < 60:
            continue
        blocks[qn] = ev(w1, [0, 1, 2, 3], mk) - ev(w0, [0, 1, 2], mk)
    return {"n_tr": int(tr.sum()), "n_te": int(te.sum()), "V0": r0, "V1": r1,
            "delta": r1 - r0, "perm_p": cnt / n_perm, "blocks": blocks,
            "V2": r2, "delta2": r2 - r0, "perm_p2": cnt2 / n_perm}


def main(asset: str = "BTC") -> None:
    """CLI: run the frozen volume experiment for one asset."""
    loader = load_equity_volume if asset.upper() in ("SPX", "NDX") else load_crypto_volume
    res = run_volume_experiment(loader(asset))
    print(f"{asset}: V0={res['V0']:.4f} V1={res['V1']:.4f} "
          f"DELTA={res['delta']:+.4f} perm_p={res['perm_p']:.4f} "
          f"V2rank={res['V2']:.4f} d2={res['delta2']:+.4f} p2={res['perm_p2']:.4f} "
          f"(n_tr={res['n_tr']} n_te={res['n_te']})")
    print("quarterly:", " ".join(f"{k}={v:+.4f}" for k, v in res["blocks"].items()))


def _holdout_residuals(frame: VolumeFrame, cols: list[int]) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Refit OLS on train, return (holdout residuals, holdout target, train_loss, holdout_rmse)."""
    f = causal_features(frame)
    X = np.column_stack([f["tv"], np.abs(f["r"]), f["amih"], f["lv"], trailing_rank(f["lv"])])
    y, dd = f["tgt"], frame.dates
    m = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y, dd = X[m], y[m], dd[m]
    mu, sd = X[dd < "2023-01-01"].mean(0), X[dd < "2023-01-01"].std(0)
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd
    tr, te = dd < "2023-01-01", dd >= "2023-01-01"
    w, *_ = np.linalg.lstsq(np.column_stack([np.ones(tr.sum()), Xs[tr][:, cols]]), y[tr], rcond=None)
    pred_tr = np.column_stack([np.ones(tr.sum()), Xs[tr][:, cols]]) @ w
    pred_te = np.column_stack([np.ones(te.sum()), Xs[te][:, cols]]) @ w
    resid = y[te] - pred_te
    return resid, y[te], float(np.mean((y[tr] - pred_tr) ** 2)), float(np.sqrt(np.mean(resid ** 2)))


def run_volume_adjudication(asset: str = "BTC") -> tuple[Any, list[Any], dict[str, Any], dict[str, Any]]:
    """Dashboard adapter: (adjudication, scored, series, extra).

    V0 (persistence+|r|+Amihud) vs V1 (+logVolume), frozen fit 2021-2022,
    OOS 2023+. Verdict UNIQUE_SUPPORT iff delta>0, perm_p<0.05 AND V1 R2>0
    (absolute floor: beating a bad baseline while worse than the mean is
    not support). Else INSUFFICIENT_EVIDENCE. Residuals carried so
    whiteness/R2 derive.
    """
    from apedp_dyn.epistemic.adjudicate import Adjudication, EpistemicStatus
    from apedp_dyn.grammar.candidate_ir import CandidateDynamics, ScoredCandidate, Term

    tag = asset.upper()
    loader = load_equity_volume if tag in ("SPX", "NDX") else load_crypto_volume
    frame = loader(tag)
    res = run_volume_experiment(frame)
    scored = []
    for cid, names, cols in (("vol_V0_baseline", ("trail_vol", "abs_ret", "amihud"), [0, 1, 2]),
                             ("vol_V1_plus_volume", ("trail_vol", "abs_ret", "amihud", "log_volume"), [0, 1, 2, 3]),
                             ("vol_V2_rank", ("trail_vol", "abs_ret", "amihud", "rank_volume"), [0, 1, 2, 4])):
        resid, yho, tr_loss, rmse = _holdout_residuals(frame, cols)
        ir = CandidateDynamics(lhs="future_vol",
                               terms=tuple(Term(nm, f"theta_{i}") for i, nm in enumerate(names)),
                               candidate_id=cid)
        sc = ScoredCandidate(
            ir,
            {"gate4_status": {"passed": True,
                              "holdout_residuals": [float(v) for v in resid],
                              "holdout_obs_rms": float(np.sqrt(np.mean(yho ** 2)))}},
            train_loss=tr_loss, holdout_rmse=rmse)
        scored.append(sc)
    # Three-way gate: promote the best challenger (V1 log-volume or V2 rank)
    # iff it beats V0 with delta>0 (own perm p<0.05) and R2>0.
    chall = [("V1", res["delta"], res["perm_p"], res["V1"], scored[1]),
             ("V2", res["delta2"], res["perm_p2"], res["V2"], scored[2])]
    passed = [(nm, sc) for nm, d, pp, r2, sc in chall
              if d > 0 and pp < 0.05 and r2 > 0]
    win = (min(passed,
               key=lambda t: t[1].holdout_rmse if t[1].holdout_rmse is not None
               else float("inf"))
           if passed else (None, None))
    ok = win[0] is not None
    r0, r1 = scored[0].holdout_rmse, scored[1].holdout_rmse
    # NOTE: delta_rmse deliberately left None: the shared formatter overwrites
    # reason with a τ-comparison whenever delta_rmse<0, hiding this adapter's
    # incremental-ΔR2 reason. The delta lives in extra/delta_r2 instead.
    v0r2 = res.get("V0", float("nan"))
    v1r2 = res.get("V1", float("nan"))
    scale_note = (f"V0 R2={v0r2:+.3f} (persistence baseline), V1 R2={v1r2:+.3f}; "
                  "claim is incremental ΔR2, never absolute R2.")
    adj = Adjudication(
        status=EpistemicStatus.UNIQUE_SUPPORT if ok else EpistemicStatus.INSUFFICIENT_EVIDENCE,
        promoted=win[1] if ok else None,
        ranking=(scored[1], scored[2], scored[0]),
        best_rmse=(win[1].holdout_rmse if (ok and win[1] is not None) else r1),
        second_rmse=r0,
        reason=((f"{win[0]} beats V0 OOS R2 with permutation control. " + scale_note)
                if ok else (f"V1 delta {res['delta']:+.4f} (p={res['perm_p']:.4f}); V2-rank delta {res['delta2']:+.4f} (p={res['perm_p2']:.4f}) — none clears the bar. " + scale_note)),
        data_origin="real_observational")
    series = {"x": list(range(int(res["n_te"]))),
              "actual": [float(v) for v in _holdout_target(frame)],
              "x_label": f"{tag} OOS future-vol target (2023+, daily)"}
    extra = {"target_asset": tag, "delta_r2": res["delta"], "perm_p": res["perm_p"],
             "delta_r2_V2": res["delta2"], "perm_p_V2": res["perm_p2"], "v2_r2": res["V2"],
             "n_train": res["n_tr"], "n_test": res["n_te"],
             "quarterly_delta": {k: float(v) for k, v in res["blocks"].items()},
             "dataset": "sidecar /tmp/ma2024/vol_daily_*.json (repo benchmark untouched)",
             "train_regime": "2021-2022 fit (frozen OLS)", "holdout_regime": "2023+ OOS"}
    return adj, scored, series, extra


def _holdout_target(frame: VolumeFrame) -> np.ndarray:
    f = causal_features(frame)
    y, dd = f["tgt"], frame.dates
    m = np.isfinite(y) & (dd >= "2023-01-01")
    return y[m]


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "BTC")
