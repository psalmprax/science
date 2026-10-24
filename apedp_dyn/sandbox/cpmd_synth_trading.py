"""CPMD-ST1 — synthetic trading test of OLS-T slope signal.

Prereg: docs/CPMD_PREREGISTRATION_ST1.md (committed BEFORE this file).
Strictly causal; costs 2bp/side; 30 paths, seeds 20000+.
"""
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "apedp_dyn/sandbox")

import numpy as np

LAGS = np.arange(4, 13, dtype=float)
LX = np.log(LAGS)
XC = LX - LX.mean()
DENOM = float(np.sum(XC ** 2))
COST = 0.0002


def ols_slope_width(v):
    ly = np.log(np.maximum(v, 1e-12))
    s = -float(np.sum(XC * ly) / DENOM)
    r = ly - ly.mean() + s * XC
    se = float(np.sqrt(float(np.sum(r ** 2)) / 7 / DENOM))
    return s, 2 * 2.3646 * se


def one_path(seed):
    rng = np.random.RandomState(int(seed))
    n = 3000
    state = np.zeros(n, dtype=int)
    state[0] = 0
    for t in range(1, n):
        state[t] = state[t - 1] if rng.rand() < 0.98 else 1 - state[t - 1]
    phi = np.where(state == 0, 0.3, -0.3)
    eps = rng.normal(0, 0.01, n)
    ret = np.zeros(n)
    for t in range(1, n):
        ret[t] = phi[t] * ret[t - 1] + eps[t]
    px = np.cumsum(ret)
    return ret, px, state


def decay_curve(a):
    a = np.abs(a)
    return np.array([float(np.mean(a[:len(a) - lag_i] * a[lag_i:]))
                     for lag_i in range(4, 13)])


def run_path(seed):
    ret, _px, _state = one_path(seed)
    W = 200
    # burn-in median threshold
    thr = []
    for t in range(W, W + 200):
        c = decay_curve(ret[t - W:t])
        s, _ = ols_slope_width(c)
        thr.append(s)
    med = float(np.median(thr))
    pos = {"ols": np.zeros(len(ret)), "bh": np.ones(len(ret)),
           "mom": np.zeros(len(ret)), "rev": np.zeros(len(ret)),
           "rnd": np.zeros(len(ret))}
    rr = np.random.RandomState(seed + 1)
    for t in range(W + 200, len(ret) - 1):
        c = decay_curve(ret[t - W:t])
        s, w = ols_slope_width(c)
        m5 = float(np.sum(ret[t - 5:t]))
        sig = 1.0 if m5 > 0 else -1.0
        pos["mom"][t] = sig
        pos["rev"][t] = -sig
        pos["rnd"][t] = 1.0 if rr.rand() < 0.5 else -1.0
        if w <= 1.0:  # abstention cap
            pos["ols"][t] = sig if s < med else -sig
    out = {}
    for k, p in pos.items():
        p = p[W + 200:-1]
        r = ret[W + 200 + 1:]
        gross = p * r
        trades = np.sum(np.abs(np.diff(np.concatenate([[0.0], p])))) / 2.0
        float(np.sum(gross) - trades * COST * 2 / len(p) * len(p))
        netrets = gross - np.abs(np.diff(np.concatenate([[0.0], p]))) * COST
        nav = np.cumsum(netrets)
        dd = float(np.max(np.maximum.accumulate(nav) - nav))
        sharpe = float(np.mean(netrets) / (np.std(netrets) + 1e-12)
                       * np.sqrt(252))
        out[k] = {"sharpe": sharpe, "net": float(np.sum(netrets)),
                  "dd": dd, "trades": float(trades)}
    return out


def run():
    keys = ("ols", "bh", "mom", "rev", "rnd")
    acc = {k: {"sharpe": [], "net": [], "dd": []} for k in keys}
    for i in range(30):
        o = run_path(20000 + i)
        for k in keys:
            for m in ("sharpe", "net", "dd"):
                acc[k][m].append(o[k][m])
    for k in keys:
        print(f"{k}: sharpe={np.mean(acc[k]['sharpe']):+.3f} "
              f"net={np.mean(acc[k]['net']):+.5f} dd={np.mean(acc[k]['dd']):.5f}")
    from scipy.stats import wilcoxon
    for base in ("bh", "mom", "rev", "rnd"):
        try:
            _, p = wilcoxon(np.array(acc["ols"]["sharpe"])
                            - np.array(acc[base]["sharpe"]))
        except Exception:  # noqa: BLE001 - deliberate catch-all: best-effort probe path reports or skips the failure, never crashes
            p = 1.0
        print(f"ols vs {base}: Wilcoxon p={p:.4f}")
    g1 = np.mean(acc["ols"]["sharpe"]) > 0
    g3 = np.mean(acc["ols"]["dd"]) < 2 * np.mean(acc["bh"]["dd"])
    print(f"GATE1 positive sharpe: {g1} | GATE3 dd<2xBH: {g3}")


if __name__ == "__main__":
    run()
