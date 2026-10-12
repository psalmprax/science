"""PRE-REGISTERED regime-reliability forward evaluation (frozen 2026-10-08).

Protocol (no fitting anywhere in this file):
  Base equation : PINNED_30 4-term core, frozen weights/threshold.
  R_t forms     : (a) trailing median cross-sectional volatility (20d),
                  (b) trailing cross-sectional return dispersion (20d).
                  Both causal (trailing windows only).
  Gate          : G_t = 1.0 if R_t <= trailing median(R_t, 60d) else 0.5.
                  Constants pre-registered, never optimized.
  Window        : 90-asset panel rows 753: (2024-2025), unseen by construction.
  Verdict rule  : gate helps iff gated Sharpe > ungated Sharpe on BOTH legs
                  AND combined; else REGIME TRANSFER UNRESOLVED.

Any edit to the forms, constants, or window below invalidates the
pre-registration. Record the file hash before running.
"""

import numpy as np
import pandas as pd

from apedp_dyn.sandbox.multi_asset_signals_sandbox import (
    PINNED_30_CHROMOSOME,
    SIGNAL_NAMES,
    calc_signals_for_asset,
    load_universe,
    simulate_multi_asset_portfolio,
)

TERMS = ["H-BB", "H-OBV_MOM", "NL_TAKER_MOM", "NL_MASTER_SNR"]
START_ROW = 753  # forward window: 2024-2025 only
FEE_BPS = 5.0


def run_forward():
    uni, _ = load_universe("90")
    from apedp_dyn.sandbox.multi_asset_signals_sandbox import (
        CRYPTO_45_ASSETS, EQUITY_45_ASSETS,
    )
    syms = list(CRYPTO_45_ASSETS) + list(EQUITY_45_ASSETS)
    N = len(syms)
    sub = {s: uni[s].iloc[START_ROW:].reset_index(drop=True) for s in syms}
    T = len(sub[syms[0]])
    rets = np.column_stack([np.asarray(sub[s]["r"], float) for s in syms])
    vols = pd.DataFrame(rets).rolling(20, min_periods=5).std().bfill().to_numpy()
    med = np.median(rets, axis=1)
    F = np.stack([calc_signals_for_asset(sub[s], med) for s in syms], axis=1)
    chrom = PINNED_30_CHROMOSOME
    idx = [SIGNAL_NAMES.index(next(s for s in SIGNAL_NAMES if s.split()[0] == t)) for t in TERMS]
    w = np.array([chrom.weights[i] for i in idx])
    S = F @ (np.zeros(len(SIGNAL_NAMES)))
    W = np.zeros(len(SIGNAL_NAMES))
    for i, v in zip(idx, w):
        W[i] = v
    S = F @ W
    thr = float(chrom.threshold)

    # Regime variables (causal trailing only)
    rv = vols  # per-asset trailing vol
    Rvol = np.median(rv, axis=1)  # (a) market-wide volatility level
    Rdisp = np.nan_to_num(pd.DataFrame(rets).rolling(20, min_periods=20).std().mean(axis=1).to_numpy())  # (b) dispersion
    med60 = lambda x: pd.Series(x).rolling(60, min_periods=60).median().bfill().to_numpy()
    Gvol = np.where(Rvol <= med60(Rvol), 1.0, 0.5)
    Gdisp = np.where(Rdisp <= med60(Rdisp), 1.0, 0.5)

    out = {}
    for gname, G in (("ungated", np.ones(T)), ("Gvol", Gvol), ("Gdisp", Gdisp)):
        legs = {}
        for sl, nm, cap in ((slice(0, 45), "crypto", 50000.0), (slice(45, None), "equity", 50000.0)):
            Sg = (S * G[:, None])[:, sl]
            ret, sr, _, _, tr, _, _, _ = simulate_multi_asset_portfolio(
                Sg, rets[:, sl], vols[:, sl], mode="cross_sectional_ls",
                sizing="risk_parity", threshold=thr, initial_capital=cap)
            legs[nm] = {"ret": ret, "sharpe": sr, "trades": tr}
        out[gname] = legs
    return out


if __name__ == "__main__":
    import hashlib
    print("file sha256:", hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:16])
    r = run_forward()
    for g, legs in r.items():
        print(f"{g}: crypto {legs['crypto']['ret']:+.1f}%/{legs['crypto']['sharpe']:+.2f} "
              f"equity {legs['equity']['ret']:+.1f}%/{legs['equity']['sharpe']:+.2f}")
