"""CPMD-HLIK Stage 3 — real-data economic evaluation.

Prereg: docs/CPMD_PREREGISTRATION_HLIK_S3.md (frozen BEFORE any real-data
evaluation). Candidate frozen (W=200, L=6, lam=1.0). Causal throughout.
"""
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "apedp_dyn/sandbox")

import numpy as np
import pandas as pd

W, L, LAM, SW, EPS, NSW = 200, 6, 1.0, 20, 1e-8, 200
FEE = 0.0005


def load_panel(path, close_suffix, exclude=()):
    df = pd.read_csv(path)
    cols = [c for c in df.columns if c.endswith(close_suffix)
            and c.replace(close_suffix, "") not in exclude]
    names = [c.replace(close_suffix, "") for c in cols]
    P = df[cols].to_numpy(dtype=float)
    R = np.diff(np.log(P), axis=0)
    return R, names


def trailing_std(R, w=SW):
    n, N = R.shape
    return np.array([[float(np.std(R[max(0, t - w + 1):t + 1, i]) + EPS)
                      for i in range(N)] for t in range(n)])


def hlik_forecast(R):
    n, N = R.shape
    sig = trailing_std(R)
    Z = R / (sig + EPS)
    fc = np.full((n, N), np.nan)
    for i in range(N):
        others = [j for j in range(N) if j != i]
        u = np.median(Z[:, others], axis=1)
        for t in range(W, n - 1):
            S = np.arange(t - W, t)
            Y = Z[S + 1, i]
            X = np.column_stack([u[S - l] for l in range(L)])
            try:
                bh = np.linalg.solve(X.T @ X + LAM * np.eye(L), X.T @ Y)
            except np.linalg.LinAlgError:
                continue
            fc[t, i] = sig[t, i] * float(bh @ np.array([u[t - l] for l in range(L)]))
    return fc


def comparators(R):
    n, N = R.shape
    sig = trailing_std(R)
    Z = R / (sig + EPS)
    m = np.median(Z, axis=1)
    mom = np.full((n, N), np.nan)
    com = np.full((n, N), np.nan)
    for t in range(100, n - 1):
        mom[t] = (np.where(np.sum(R[t - 5:t], axis=0) > 0, 1.0, -1.0)
                  * sig[t])
        Zw, mw = Z[t - 99:t + 1], m[t - 99:t + 1]
        den = float(np.sum((mw - mw.mean()) ** 2) + EPS)
        b = np.array([float(np.sum((mw - mw.mean()) * (Zw[:, i] - Zw[:, i].mean())) / den)
                      for i in range(N)])
        com[t] = b * m[t]
    return mom, com


def panel_ic_null(F, R, seed):
    f, r = F[:-1].ravel(), R[1:].ravel()
    ok = np.isfinite(f) & np.isfinite(r)
    f, r = f[ok], r[ok]
    obs = float(np.corrcoef(f, r)[0, 1])
    base = float(np.mean(r ** 2))
    mse = float(np.mean((f - r) ** 2))
    rng = np.random.RandomState(int(seed))
    cnt = sum(float(np.corrcoef(np.concatenate([f[s:], f[:s]]), r)[0, 1]) >= obs
              for s in rng.randint(0, len(f), size=NSW))
    return obs, float(cnt / NSW), mse / base


def evaluate(tag, R, names):
    print(f"== {tag} ({R.shape[0]} days x {len(names)} assets) ==")
    F = hlik_forecast(R)
    M, C = comparators(R)
    ic, p, mser = panel_ic_null(F, R, 31337)
    print(f"3A H-LIK: IC={ic:+.4f} p={p:.3f} MSEratio={mser:.3f} "
          f"[{'PASS' if p < 0.05 and mser < 1 else 'FAIL'}]")
    # 3B residualize on [MOM, COMMON]
    f, r = F[:-1].reshape(-1), R[1:].reshape(-1)
    cc = np.stack([M[:-1].reshape(-1), C[:-1].reshape(-1)], axis=1)
    ok = np.isfinite(f) & np.isfinite(r) & np.all(np.isfinite(cc), axis=1)
    f, r, cc = f[ok], r[ok], cc[ok]
    X = np.column_stack([cc, np.ones(len(cc))])
    fr = f - X @ np.linalg.lstsq(X, f, rcond=None)[0]
    obs = float(np.corrcoef(fr, r)[0, 1])
    rng = np.random.RandomState(777)
    cnt = sum(float(np.corrcoef(np.concatenate([fr[s:], fr[:s]]), r)[0, 1]) >= obs
              for s in rng.randint(0, len(fr), size=NSW))
    pr = float(cnt / NSW)
    print(f"3B residual: IC={obs:+.4f} p={pr:.3f} "
          f"[{'PASS' if pr < 0.05 else 'FAIL'}]")
    return (p < 0.05 and mser < 1 and pr < 0.05), F, M


def trade(tag, R, F, M):
    n, N = R.shape
    t0 = W
    strat = {"hlik": np.zeros((n, N)), "mom": np.zeros((n, N)),
             "bh": np.full((n, N), 1.0 / N)}
    for t in range(t0, n - 1):
        fh = F[t]
        if np.sum(np.isfinite(fh)) >= 6:
            lo = np.argsort(fh)[:3]
            hi = np.argsort(fh)[-3:]
            strat["hlik"][t, hi] = 1.0 / 3
            strat["hlik"][t, lo] = -1.0 / 3
        fm = M[t]
        if np.sum(np.isfinite(fm)) >= 6:
            lo = np.argsort(fm)[:3]
            hi = np.argsort(fm)[-3:]
            strat["mom"][t, hi] = 1.0 / 3
            strat["mom"][t, lo] = -1.0 / 3
    print(f"-- 3C {tag} --")
    res = {}
    for k, P in strat.items():
        p = P[t0:-1]
        r = R[t0 + 1:]
        chg = np.sum(np.abs(np.diff(np.concatenate([np.zeros((1, N)), p]), axis=0)), axis=1)
        nr = np.sum(p * r, axis=1) - chg * FEE
        nav = np.cumsum(nr)
        dd = float(np.max(np.maximum.accumulate(nav) - nav))
        sh = float(np.mean(nr) / (np.std(nr) + 1e-12) * np.sqrt(252))
        # halves
        h = len(nr) // 2
        sh1 = float(np.mean(nr[:h]) / (np.std(nr[:h]) + 1e-12) * np.sqrt(252))
        sh2 = float(np.mean(nr[h:]) / (np.std(nr[h:]) + 1e-12) * np.sqrt(252))
        # monthly paired test basis
        res[k] = (sh, float(np.sum(nr)), dd, sh1, sh2, nr)
        print(f"{k}: sharpe={sh:+.3f} net={np.sum(nr):+.4f} dd={dd:.4f} "
              f" halves=({sh1:+.2f},{sh2:+.2f})")
    # paired t on monthly nets hlik vs mom
    def monthly(nr):
        return np.array([np.sum(nr[i:i + 21]) for i in range(0, len(nr), 21)])
    mh, mm = monthly(res["hlik"][5]), monthly(res["mom"][5])
    d = mh - mm
    se = float(np.std(d) / np.sqrt(len(d)))
    t = float(np.mean(d) / (se + 1e-12))
    from math import erf
    pval = float(2 * (1 - 0.5 * (1 + erf(abs(t) / np.sqrt(2)))))
    g = (res["hlik"][0] > 0 and pval < 0.05
         and res["hlik"][2] < 2 * res["bh"][2])
    print(f"hlik vs mom monthly t={t:+.2f} p~{pval:.4f} | "
          f"GATES sharpe>0:{res['hlik'][0] > 0} beats-mom:{pval < 0.05} dd-ok:{res['hlik'][2] < 2 * res['bh'][2]} "
          f"-> {'PASS' if g else 'FAIL'}")
    return g


def run():
    Re, ne = load_panel("data/market/equity_top15.csv", "_CLOSE", exclude=("SPY",))
    Rc, nc = load_panel("data/market/crypto_top15.csv", "",
                        exclude=("COMPOSITE_TOP15", "observation_date",
                                 "BTC_OFI", "ETH_OFI", "SOL_OFI", "BNB_OFI",
                                 "XRP_OFI", "ADA_OFI", "AVAX_OFI", "DOGE_OFI",
                                 "DOT_OFI", "LINK_OFI", "MATIC_OFI", "NEAR_OFI",
                                 "ATOM_OFI", "UNI_OFI", "LTC_OFI",
                                 "COMPOSITE_TOP15_OFI"))
    ok_e, Fe, Me = evaluate("EQUITY real", Re, ne)
    ok_c, Fc, Mc = evaluate("CRYPTO real", Rc, nc)
    if ok_e and ok_c:
        ge = trade("EQUITY", Re, Fe, Me)
        gc = trade("CRYPTO", Rc, Fc, Mc)
        print(f"STAGE3: {'PROCEEDS to independent replication' if ge and gc else 'no replication — economics fail'}")
    else:
        print("STAGE3: 3A/3B failed — no trading evaluation (verdicts separate)")


if __name__ == "__main__":
    run()
