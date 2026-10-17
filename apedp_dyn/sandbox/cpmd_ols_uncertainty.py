"""CPMD v0.11 — OLS uncertainty calibration under correlated noise.

Prereg: docs/CPMD_PREREGISTRATION_v011.md. Point estimator fixed (log-log OLS,
lags 4-12). Five uncertainty methods; 12 conditions x 200 seeds (5000+).
"""
import itertools
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "apedp_dyn/sandbox")

import numpy as np
from cpmd_calibration import gen
from cpmd_effective_slope import LAGS

TRUTH = 0.6  # a=1.2, t0=8, mean lag 8
LX = np.log(LAGS)
XC = LX - LX.mean()
DENOM = float(np.sum(XC ** 2))
BW = 2  # preregistered HAC bandwidth
BL = 3  # preregistered block length
BBOOT_B = 200


def ols_fit(y):
    ly = np.log(y)
    slope = float(np.sum(XC * ly) / DENOM)
    resid = ly - ly.mean() - slope * XC
    return -slope, resid


def ci_ols(y):
    s, r = ols_fit(y)
    n = len(y)
    s2 = float(np.sum(r ** 2)) / (n - 2)
    se = float(np.sqrt(s2 / DENOM))
    return s, s - 1.96 * se, s + 1.96 * se


def ci_hac(y, bw=BW):
    s, r = ols_fit(y)
    len(y)
    w = r * XC
    S = float(np.sum(w ** 2))
    for L in range(1, bw + 1):
        wt = 1.0 - L / (bw + 1.0)
        S += 2.0 * wt * float(np.sum(w[L:] * w[:-L]))
    se = float(np.sqrt(S) / DENOM)
    return s, s - 1.96 * se, s + 1.96 * se


def ci_bboot(y, seed):
    rng = np.random.RandomState(int(seed))
    s0, _ = ols_fit(y)
    n = len(y)
    nb = int(np.ceil(n / BL))
    boots = []
    for _ in range(BBOOT_B):
        idx = np.concatenate([np.arange(st, st + BL) % n
                              for st in rng.randint(0, n, size=nb)])[:n]
        yb = y[idx]
        try:
            sb, _ = ols_fit(yb)
            boots.append(sb)
        except Exception:  # noqa: BLE001, S110 - optional probe may fail silently by design (best-effort stage skip)  # noqa: BLE001 - deliberate catch-all: best-effort probe path reports or skips the failure, never crashes
            pass
    b = np.array(boots)
    if len(b) > 10:
        return s0, float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
    return s0, 9.0, -9.0


def ar1_cov(n, rho, s2=1.0):
    i, j = np.ogrid[:n, :n]
    return s2 * rho ** np.abs(i - j)


def ci_gls(y, cov):
    ly = np.log(y)
    X = np.column_stack([np.ones_like(XC), XC])
    try:
        C = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        return float("nan"), 9.0, -9.0
    XtC = X.T @ C
    try:
        beta = np.linalg.solve(XtC @ X, XtC @ ly)
        V = np.linalg.inv(XtC @ X)
    except np.linalg.LinAlgError:
        return float("nan"), 9.0, -9.0
    resid = ly - X @ beta
    dof = max(len(y) - 2, 1)
    scale = float(resid @ C @ resid) / dof
    se = float(np.sqrt(V[1, 1] * scale))
    return -float(beta[1]), -float(beta[1]) - 1.96 * se, -float(beta[1]) + 1.96 * se


def gen_dep(struct, rho, nl, sd, a=1.2, t0=8.0):
    lags, yclean = gen("power", 0.0, sd, t0=t0, a=a)  # noiseless mean
    mu = yclean
    rng = np.random.RandomState(int(sd) + 777000)
    n = len(mu)
    e = rng.normal(0, nl, n)
    if struct == "ar1":
        for i in range(1, n):
            e[i] = rho * e[i - 1] + np.sqrt(max(1 - rho ** 2, 1e-9)) * e[i]
    elif struct == "cs":
        shared = rng.normal(0, nl * np.sqrt(max(rho, 0)), 1)[0]
        e = shared + rng.normal(0, nl * np.sqrt(max(1 - rho, 1e-9)), n)
    elif struct == "ma1":
        e2 = rng.normal(0, nl, n)
        e = e2 + 0.5 * np.concatenate([[0.0], e2[:-1]])
        e = e * (nl / max(np.std(e), 1e-9))
    return lags, mu * np.exp(e)


def run():
    conds = ([("ar1", r) for r in (0.0, 0.3, 0.5)] + [("cs", 0.3), ("cs", 0.5)]
             + [("ma1", 0.0)])
    print("v0.11: method cov/width per condition (n=200, PASS=coverage>=184)")
    for (struct, rho), nl in itertools.product(conds, (0.1, 0.3)):
        res = {}
        for m in ("OLS", "HAC", "BBOOT", "GLS-O", "GLS-E"):
            res[m] = {"cov": 0, "w": [], "b": [], "n": 0}
        for k in range(200):
            sd = 5000 + k
            _lags, yfull = gen_dep(struct, rho, nl, sd)
            ar = np.arange(2, 14)
            y = yfull[((ar >= 4) & (ar <= 12))]
            n = len(y)
            _s0, _ = ols_fit(y)
            _, r = ols_fit(y)
            s2 = float(np.sum(r ** 2)) / (n - 2)
            outs = {"OLS": ci_ols(y), "HAC": ci_hac(y), "BBOOT": ci_bboot(y, sd),
                    "GLS-O": ci_gls(y, ar1_cov(n, rho if struct == "ar1" else 0.0, s2)),
                    "GLS-E": None}
            # GLS-E: estimate rho from residual autocorrelation
            rr = r - r.mean()
            denom = float(np.sum(rr ** 2))
            rhohat = float(np.sum(rr[1:] * rr[:-1]) / denom) if denom > 0 else 0.0
            rhohat = float(np.clip(rhohat, -0.9, 0.9))
            outs["GLS-E"] = ci_gls(y, ar1_cov(n, rhohat, s2))
            for m, (s, lo, hi) in outs.items():
                if not np.isfinite(s):
                    continue
                res[m]["n"] += 1
                res[m]["b"].append(s - TRUTH)
                if lo <= TRUTH <= hi:
                    res[m]["cov"] += 1
                res[m]["w"].append(hi - lo)
        tag = f"{struct}(rho={rho}) nl={nl}"
        cells = []
        for m in ("OLS", "HAC", "BBOOT", "GLS-O", "GLS-E"):
            d = res[m]
            b = np.array(d["b"])
            ok = "PASS" if (d["cov"] >= 184 and np.median(d["w"]) <= 1.0) else "fail"
            cells.append(f"{m}:{d['cov']}/200 w={np.median(d['w']):.2f} "
                         f"bias={np.mean(b):+.3f} [{ok}]")
        print(tag + " | " + " ".join(cells))


if __name__ == "__main__":
    run()
