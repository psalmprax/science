"""CPMD-HXSC Stages 1+2 — shock concentration predicts dispersion.

Prereg: docs/CPMD_PREREGISTRATION_HXSC.md (60/40 split declared pre-numbers).
Risk-state test only; no trading. Block bootstrap (len 20, 500 reps).
"""
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "apedp_dyn/sandbox")

import numpy as np
import pandas as pd

SW, EPS, NB = 20, 1e-8, 500
BL = 20


def load(path, suf, exc):
    df = pd.read_csv(path)
    cols = [c for c in df.columns if c.endswith(suf)
            and c.replace(suf, "") not in exc]
    P = df[cols].to_numpy(dtype=float)
    return np.diff(np.log(P), axis=0)


def features(R):
    n, N = R.shape
    sig = np.array([[float(np.std(R[max(0, t - SW + 1):t + 1, i]) + EPS)
                     for i in range(N)] for t in range(n)])
    Z = R / (sig + EPS)
    q = np.abs(Z) / (EPS + np.sum(np.abs(Z), axis=1, keepdims=True))
    H = (np.sum(q ** 2, axis=1) - 1.0 / N) / (1.0 - 1.0 / N)
    m = np.median(Z, axis=1)
    Dnow = np.array([float(np.mean((R[t, :] - R[t, :].mean()) ** 2)) for t in range(n)])
    D5 = np.array([float(np.mean([np.mean((R[t + k, :] - R[t + k, :].mean()) ** 2)
                                  for k in range(1, 6)])) if t + 5 < n else np.nan
                   for t in range(n)])
    Dcur = np.array([float(np.mean(Dnow[max(0, t - 4):t + 1])) for t in range(n)])
    vol = np.array([float(np.std(m[max(0, t - 19):t + 1]) + EPS) for t in range(n)])
    return H, D5, Dcur, np.abs(m), vol


def fit_t(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def stage(tag, R, lo, hi):
    H, D5, Dc, am, vol = features(R)
    idx = np.arange(max(lo, 220), min(hi, len(R) - 5))
    y = D5[idx]
    X = np.column_stack([H[idx], Dc[idx], am[idx], vol[idx], np.ones(len(idx))])
    b = fit_t(X, y)
    X0 = X[:, 1:]
    r2 = 1 - float(np.sum((y - X @ b) ** 2)) / float(np.sum((y - y.mean()) ** 2))
    r0 = 1 - float(np.sum((y - X0 @ fit_t(X0, y)) ** 2)) / float(np.sum((y - y.mean()) ** 2))
    # block bootstrap on H
    h = H[idx]
    nb = int(np.ceil(len(h) / BL))
    rng = np.random.RandomState(4242)
    t_obs = b[0]
    cnt = 0
    se_s = []
    for _ in range(NB):
        order = rng.randint(0, len(h), size=nb)
        hb = np.concatenate([h[o:o + BL] for o in order])[:len(h)]
        Xb = np.column_stack([hb, X[:, 1:]])
        bb = fit_t(Xb, y)
        se_s.append(bb[0])
        if abs(bb[0]) >= abs(t_obs):
            cnt += 1
    p = float(cnt / NB)
    print(f"{tag}: n={len(idx)} b={b[0]:+.6f} R2={r2:.3f} dR2={r2 - r0:+.4f} "
          f"p={p:.3f} [{'PASS' if p < 0.05 else 'FAIL'}]")
    return p


def run():
    panels = {
        "EQUITY": load("data/market/equity_top15.csv", "_CLOSE", ("SPY",)),
        "CRYPTO": load("data/market/crypto_top15.csv", "",
                       ("COMPOSITE_TOP15", "observation_date", "BTC_OFI",
                        "ETH_OFI", "SOL_OFI", "BNB_OFI", "XRP_OFI", "ADA_OFI",
                        "AVAX_OFI", "DOGE_OFI", "DOT_OFI", "LINK_OFI",
                        "MATIC_OFI", "NEAR_OFI", "ATOM_OFI", "UNI_OFI",
                        "LTC_OFI", "COMPOSITE_TOP15_OFI")),
    }
    dev_p = {}
    cuts = {}
    for tag, R in panels.items():
        el0, el1 = 220, len(R) - 5
        cut = el0 + int(0.6 * (el1 - el0))
        cuts[tag] = cut
        dev_p[tag] = stage(f"{tag} development[{el0}:{cut}]", R, el0, cut)
    dall = all(p < 0.05 for p in dev_p.values())
    print(f"development: {'PASS both' if dall else 'FAIL — confirmation closed'}")
    if not dall:
        print("H-XSC: REJECTED at development")
        return
    con = {tag: stage(f"{tag} confirmation[{cuts[tag]}:{len(R) - 5}]",
                       R, cuts[tag], len(R) - 5)
           for tag, R in panels.items()}
    if all(p < 0.05 for p in con.values()):
        print("H-XSC: PROCEEDS to Stage 3 (new prereg required)")
    else:
        print("H-XSC: REJECTED at confirmation")


if __name__ == "__main__":
    run()
