"""CPMD-HOFD — daily OFI reversal hypothesis (Stages: discovery crypto,
confirmation equity-40 + multi30). Prereg committed BEFORE this file.
"""
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "apedp_dyn/sandbox")

import numpy as np
import pandas as pd

SW, EPS, NSW, ZW = 20, 1e-8, 200, 60


def load_cols(close_path, close_cols, ofi_cols):
    df = pd.read_csv(close_path)
    P = df[close_cols].to_numpy(dtype=float)
    R = np.diff(np.log(P), axis=0)
    O = df[ofi_cols].to_numpy(dtype=float)[1:]  # align with returns
    assert R.shape == O.shape, (R.shape, O.shape)
    return R, O


def trailing_std(R, w=SW):
    n, N = R.shape
    return np.array([[float(np.std(R[max(0, t - w + 1):t + 1, i]) + EPS)
                      for i in range(N)] for t in range(n)])


def hofd_forecast(R, O):
    n, N = R.shape
    sig = trailing_std(R)
    fc = np.full((n, N), np.nan)
    for t in range(ZW, n - 1):
        w = O[t - ZW + 1:t + 1]
        mu, sd = w.mean(0), w.std(0) + EPS
        s = (O[t] - mu) / sd
        fc[t] = -sig[t] * np.tanh(s)
    return fc


def ic_null(F, R, seed):
    f, r = F[:-1].ravel(), R[1:].ravel()
    ok = np.isfinite(f) & np.isfinite(r)
    f, r = f[ok], r[ok]
    obs = float(np.corrcoef(f, r)[0, 1])
    mse = float(np.mean((f - r) ** 2) / np.mean(r ** 2))
    rng = np.random.RandomState(int(seed))
    cnt = sum(float(np.corrcoef(np.concatenate([f[s:], f[:s]]), r)[0, 1]) >= obs
              for s in rng.randint(0, len(f), size=NSW))
    return obs, float(cnt / NSW), mse


def evaluate(tag, R, O, seed, lo=0.0, hi=1.0):
    n = len(R)
    a, b = int(lo * n), int(hi * n)
    F = hofd_forecast(R)
    # restrict to window [a:b] on FORECAST index
    Fw = np.full_like(F, np.nan)
    Fw[a:b] = F[a:b]
    ic, p, mser = ic_null(Fw, R, seed)
    # incremental vs MOM + MARKET
    sig = trailing_std(R)
    Z = R / (sig + EPS)
    m = np.median(Z, axis=1)
    mom = np.where(np.array([np.sum(R[max(0, t - 5):t], axis=0) for t in range(n)]) > 0, 1.0, -1.0) * sig
    com = np.zeros_like(R)
    for t in range(100, n - 1):
        Zw, mw = Z[t - 99:t + 1], m[t - 99:t + 1]
        den = float(np.sum((mw - mw.mean()) ** 2) + EPS)
        com[t] = np.array([float(np.sum((mw - mw.mean()) * (Zw[:, i] - Zw[:, i].mean())) / den)
                           for i in range(R.shape[1])]) * m[t]
    f, r = Fw[:-1].reshape(-1), R[1:].reshape(-1)
    cc = np.stack([mom[:-1].reshape(-1), com[:-1].reshape(-1)], axis=1)
    ok = np.isfinite(f) & np.isfinite(r) & np.all(np.isfinite(cc), axis=1)
    f, r, cc = f[ok], r[ok], cc[ok]
    X = np.column_stack([cc, np.ones(len(cc))])
    fr = f - X @ np.linalg.lstsq(X, f, rcond=None)[0]
    obs = float(np.corrcoef(fr, r)[0, 1])
    rng = np.random.RandomState(seed + 7)
    cnt = sum(float(np.corrcoef(np.concatenate([fr[s:], fr[:s]]), r)[0, 1]) >= obs
              for s in rng.randint(0, len(fr), size=NSW))
    pr = float(cnt / NSW)
    okg = p < 0.05 and mser < 1 and pr < 0.05
    print(f"{tag}: IC={ic:+.4f} p={p:.3f} MSEr={mser:.3f} residIC={obs:+.4f} residP={pr:.3f} "
          f"[{'PASS' if okg else 'FAIL'}]")
    return okg


def run():
    C = ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "AVAX", "DOGE", "DOT",
         "LINK", "MATIC", "NEAR", "ATOM", "UNI", "LTC"]
    Rc, Oc = load_cols("data/market/crypto_top15.csv", C, [a + "_OFI" for a in C])
    E = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM", "V",
         "JNJ", "WMT", "PG", "XOM", "UNH"]
    Re, Oe = load_cols("data/market/equity_top15.csv",
                       [e + "_CLOSE" for e in E], [e + "_OFI" for e in E])
    M30 = pd.read_csv("data/market/multi_asset_30_orderflow.csv")
    mclose = [c for c in M30.columns if c.endswith("_CLOSE")]
    mnames = [c.replace("_CLOSE", "") for c in mclose]
    mofi = [m + "_OFI" for m in mnames if m + "_OFI" in M30.columns]
    mclose = [m + "_CLOSE" for m in mnames if m + "_OFI" in M30.columns]
    Rm, Om = load_cols("data/market/multi_asset_30_orderflow.csv", mclose, mofi)
    print("shapes:", Rc.shape, Oc.shape, Re.shape, Oe.shape, Rm.shape, Om.shape,
          "multi30 assets:", len(mclose))
    d = evaluate("CRYPTO discovery", Rc, Oc, 5150)
    if not d:
        print("H-OFD: REJECTED at discovery")
        return
    e = evaluate("EQUITY confirm(last40)", Re, Oe, 5151, lo=0.6, hi=1.0)
    m = evaluate("MULTI30 confirm", Rm, Om, 5152)
    print("H-OFD:", "PROCEEDS (trading needs new prereg)" if (e and m) else "REJECTED at confirmation")


if __name__ == "__main__":
    run()
