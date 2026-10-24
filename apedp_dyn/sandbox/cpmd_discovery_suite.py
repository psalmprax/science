"""CPMD v0.16 — frozen discovery suite H-CTP / H-SER / H-SAC.

Prereg: docs/CPMD_PREREGISTRATION_v016.md (committed BEFORE this file).
Prediction-only stage: IC + shift null + MSE-vs-zero. No trading.
"""
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "apedp_dyn/sandbox")

import numpy as np
from cpmd_synth_trading import one_path

K, M, KK = 20, 5, 20
EPS = 1e-8


def rolling_std(r, w=20):
    return np.array([float(np.std(r[max(0, t - w + 1):t + 1]) + EPS)
                     for t in range(len(r))])


def gen_B(seed, n=3000):
    rng = np.random.RandomState(int(seed))
    hiv = np.zeros(n, dtype=int)
    hiv[0] = 0
    for t in range(1, n):
        hiv[t] = hiv[t - 1] if rng.rand() < 0.97 else 1 - hiv[t - 1]
    sig = np.where(hiv == 0, 0.005, 0.02)
    return sig * rng.normal(0, 1, n)


def forecasts(ret):
    n = len(ret)
    sig = rolling_std(ret)
    z = ret / (sig + EPS)
    f = {k: np.full(n, np.nan) for k in ("ctp", "ser", "sac")}
    Cprev = 0.0
    Vprev = 1.0
    for t in range(K, n - 1):
        zw = z[t - K + 1:t + 1]
        C = float(np.sum(zw) / (EPS + np.sum(np.abs(zw))))
        Q = float(np.sum(zw * np.abs(zw)) / (EPS + np.sum(zw ** 2)))
        rw = ret[t - K + 1:t + 1]
        Vs = float((EPS + np.mean(rw[-M:] ** 2)) / (EPS + np.mean(rw ** 2)))
        Vp = Vprev
        D = max(0.0, float(np.tanh(np.log(Vp / max(Vs, EPS)))))
        f["ctp"][t] = sig[t] * np.sign(C) * (abs(C) - abs(Cprev))
        f["ser"][t] = -sig[t] * Q
        f["sac"][t] = sig[t] * Q * D
        Cprev, Vprev = C, Vs
    return f


def ic_null(fc, actual, seed, nshift=200):
    ok = np.isfinite(fc) & np.isfinite(actual)
    fc, ac = fc[ok], actual[ok]
    if len(fc) < 50:
        return float("nan"), 1.0, float("nan")
    obs = float(np.corrcoef(fc, ac)[0, 1])
    base = float(np.mean(ac ** 2))
    mse0 = float(np.mean((fc - ac) ** 2))
    rng = np.random.RandomState(int(seed))
    cnt = 0
    for _ in range(nshift):
        sh = rng.randint(0, len(fc))
        fs = np.concatenate([fc[sh:], fc[:sh]])
        if float(np.corrcoef(fs, ac)[0, 1]) >= obs:
            cnt += 1
    return obs, float(cnt / nshift), mse0 / base


def stage(tag, seeds, genfn):
    print(f"--- {tag} ---")
    res = {}
    for name in ("ctp", "ser", "sac"):
        ics, ps, mr = [], [], []
        for sd in seeds:
            ret = genfn(sd)
            fc = forecasts(ret)[name]
            # fc[t] predicts ret[t+1]; ic_null masks NaN warmup jointly
            ic, p, m = ic_null(fc[:-1], ret[1:len(fc)], sd)
            ics.append(ic)
            ps.append(p)
            mr.append(m)
        ics = np.array(ics)
        res[name] = (float(np.mean(ics)), float(np.median(ps)), float(np.mean(mr)))
        print(f"H-{name.upper()}: meanIC={np.mean(ics):+.4f} "
              f"median-p={np.median(ps):.3f} MSEratio={np.mean(mr):.3f}")
    return res


def run():
    def A(sd):
        return one_path(sd)[0]
    r1 = stage("Stage 1 discovery gen-A (50000+)", range(50000, 50020), A)
    r2 = stage("Stage 1 discovery gen-B (51000+)", range(51000, 51020), gen_B)
    surv = [k for k in ("ctp", "ser", "sac")
            if r1[k][1] < 0.05 / 3 or r2[k][1] < 0.05 / 3]
    print(f"Survivors to Stage 2: {surv if surv else 'NONE — hypotheses rejected'}")
    for k in surv:
        stage(f"Stage 2 confirmatory H-{k.upper()} gen-A (60000+)",
              range(60000, 60020), A)
        stage(f"Stage 2 confirmatory H-{k.upper()} gen-B (61000+)",
              range(61000, 61020), gen_B)


if __name__ == "__main__":
    run()
