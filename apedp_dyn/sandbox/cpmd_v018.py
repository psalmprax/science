"""CPMD v0.18 — signal-strength calibration & controlled discovery.

Prereg + candidates frozen in docs/CPMD_PREREGISTRATION_v018.md BEFORE
gen-C below was written (blindness by commit order).
gen-C: r_{t+1} = 1.5*sigma*tanh(3*Q_t)*sign(C_t) + N(0,sigma).
"""
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "apedp_dyn/sandbox")

import numpy as np

from cpmd_discovery_suite import rolling_std, ic_null, K, M, KK, EPS

SIG = 0.01


def gen_ladder(seed, c, n=3000):
    rng = np.random.RandomState(int(seed))
    ret = np.zeros(n)
    for t in range(10, n - 1):
        m = float(np.mean(ret[t - 9:t + 1]))
        ret[t + 1] = c * (1.0 if m > 0 else -1.0) * SIG + rng.normal(0, SIG)
    return ret


def control_pred(ret):
    n = len(ret)
    sig = rolling_std(ret)
    fc = np.full(n, np.nan)
    for t in range(10, n - 1):
        m = float(np.mean(ret[t - 9:t + 1]))
        fc[t] = (1.0 if m > 0 else -1.0) * sig[t]
    return fc


def gen_C(seed, n=3000):
    rng = np.random.RandomState(int(seed))
    ret = rng.normal(0, SIG, n)
    sig = rolling_std(ret)
    z = ret / (sig + EPS)
    out = ret.copy()
    for t in range(K, n - 1):
        zw = z[t - K + 1:t + 1]
        # recompute adaptively as series extends (causal: uses out ≤ t)
        sigt = float(np.std(out[max(0, t - 19):t + 1]) + EPS)
        zt = out[max(0, t - 19):t + 1] / (rolling_std(out)[max(0, t - 19):t + 1] + EPS)
        zw2 = zt[-K:] if len(zt) >= K else zt
        C = float(np.sum(zw2) / (EPS + np.sum(np.abs(zw2))))
        Q = float(np.sum(zw2 * np.abs(zw2)) / (EPS + np.sum(zw2 ** 2)))
        out[t + 1] = 1.5 * SIG * np.tanh(3 * Q) * (1.0 if C >= 0 else -1.0) \
            + rng.normal(0, SIG)
    return out


def feats(ret):
    n = len(ret)
    sig = rolling_std(ret)
    z = ret / (sig + EPS)
    C = np.full(n, np.nan)
    Q = np.full(n, np.nan)
    V = np.full(n, np.nan)
    Mt = np.full(n, np.nan)
    for t in range(K, n):
        zw = z[t - K + 1:t + 1]
        C[t] = float(np.sum(zw) / (EPS + np.sum(np.abs(zw))))
        Q[t] = float(np.sum(zw * np.abs(zw)) / (EPS + np.sum(zw ** 2)))
        rw = ret[t - KK + 1:t + 1]
        V[t] = float((EPS + np.mean(rw[-M:] ** 2)) / (EPS + np.mean(rw ** 2)))
        Mt[t] = 1.0 if float(np.mean(ret[t - 4:t + 1])) > 0 else -1.0
    return sig, C, Q, V, Mt


def cand_fc(name, ret):
    n = len(ret)
    sig, C, Q, V, Mt = feats(ret)
    fc = np.full(n, np.nan)
    for t in range(K, n - 1):
        if name == "ctp":
            fc[t] = sig[t] * (1.0 if C[t] >= 0 else -1.0) * (abs(C[t]) - abs(C[t - 1]))
        elif name == "qsc":
            fc[t] = sig[t] * Q[t] * abs(C[t])
        elif name == "vti":
            fc[t] = sig[t] * (1.0 if V[t] >= V[t - 1] else -1.0) * Mt[t]
    return fc


def run():
    print("== 1. strength ladder (oracle IC + harness p + IC coverage) ==")
    for c in (0.0, 0.1, 0.5, 2.0):
        oics, ps, cov = [], [], 0
        for i in range(20):
            sd = 80000 + int(c * 100) + i
            ret = gen_ladder(sd, c)
            fc = control_pred(ret)
            ic, p, _ = ic_null(fc[:-1], ret[1:len(fc)], sd)
            oics.append(ic)
            ps.append(p)
        oics = np.array(oics)
        se = float(np.std(oics) / np.sqrt(len(oics)))
        # CI coverage of the GROUP mean IC across... report per-path CI hit rate
        print(f"c={c}: oracleIC={np.mean(oics):+.4f}±{se:.4f} "
              f"reject-rate(p<0.025)={np.mean(np.array(ps) < 0.025):.2f} "
              f"median-p={np.median(ps):.4f}")
    print("== 2+3. gen-C discovery (fresh seeds 81000+) ==")
    for name in ("ctp", "qsc", "vti"):
        ics, ps = [], []
        for i in range(20):
            sd = 81000 + i
            ret = gen_C(sd)
            fc = cand_fc(name, ret)
            ic, p, _ = ic_null(fc[:-1], ret[1:len(fc)], sd)
            ics.append(ic)
            ps.append(p)
        ics = np.array(ics)
        ok = float(np.median(ps)) < 0.05 / 3
        print(f"H-{name.upper()}: meanIC={np.mean(ics):+.4f} "
              f"median-p={np.median(ps):.4f} [{'SURVIVES' if ok else 'rejected'}]")


if __name__ == "__main__":
    run()
