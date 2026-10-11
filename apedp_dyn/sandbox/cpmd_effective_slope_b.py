"""CPMD v0.10b — OLS-vs-model advantage hunt (diagnostic only).

Frozen v0.10 gates apply unchanged. Question: does the model-based slope
offer ANY advantage in bias / coverage+width / correlated-noise robustness /
transfer across (alpha, tau0) that OLS lacks? If no: retain OLS, stop.
Seeds 4000+ (disjoint from v0.10 diagnostic 3000-3039). Sealed set unopened.
"""
import itertools
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "apedp_dyn/sandbox")

import numpy as np
from scipy.optimize import curve_fit

from cpmd_calibration import gen
from cpmd_effective_slope import slope_ols, slope_model, LAGS, true_slope

TM = float(np.mean(LAGS))


def ols_ci(lags, y):
    s, se = slope_ols(lags, y)
    return s, s - 1.96 * se, s + 1.96 * se


def model_ci(lags, y, seed, B=50):
    ly = np.log(y)
    pp, _ = curve_fit(lambda t, lA, a, tt: lA - a * np.log(t + tt),
                      lags, ly, p0=(np.log(0.05), 1.0, 8.0), maxfev=50000)
    lA, a, t0 = (float(v) for v in pp)
    rng = np.random.RandomState(int(seed))
    pred = lA - a * np.log(lags + t0)
    resid = ly - pred
    boots = []
    for _ in range(B):
        yb = np.exp(pred + rng.choice(resid, size=len(lags), replace=True))
        try:
            boots.append(slope_model(lags, yb))
        except Exception:
            pass
    b = np.array(boots)
    pt = float(a * TM / (TM + t0))
    if len(b) > 10:
        return pt, float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
    return pt, 9.0, -9.0


def run():
    print("v0.10b: transfer (a,t0) x noise x rho; OLS vs MODEL; seeds 4000+")
    print("cond | OLS bias rmse cov width | MODEL bias rmse cov width | n")
    sd = 4000
    for a, t0, nl, rho in itertools.product((0.8, 1.2, 1.8), (2.0, 8.0, 30.0),
                                            (0.1, 0.3), (0.0, 0.5)):
        truth = float(a * TM / (TM + t0))
        stats = {"ols": [], "mod": []}
        cov = {"ols": 0, "mod": 0}
        wid = {"ols": [], "mod": []}
        n = 0
        for _ in range(20):
            _, yfull = gen("power", nl, sd, t0=t0, a=a, rho=rho)
            ar = np.arange(2, 14)
            y = yfull[((ar >= 4) & (ar <= 12))]
            sd += 1
            try:
                s, lo, hi = ols_ci(LAGS, y)
                stats["ols"].append(s - truth)
                cov["ols"] += (lo <= truth <= hi)
                wid["ols"].append(hi - lo)
            except Exception:
                pass
            try:
                s, lo, hi = model_ci(LAGS, y, seed=sd)
                stats["mod"].append(s - truth)
                cov["mod"] += (lo <= truth <= hi)
                wid["mod"].append(hi - lo)
            except Exception:
                pass
            n += 1
        eo = np.array(stats["ols"])
        em = np.array(stats["mod"])
        print(f"a={a} t0={t0:g} nl={nl} rho={rho}: "
              f"OLS b={np.mean(eo):+.3f} r={np.sqrt(np.mean(eo**2)):.3f} "
              f"c={cov['ols']}/{len(eo)} w={np.median(wid['ols']):.2f} | "
              f"MOD b={np.mean(em):+.3f} r={np.sqrt(np.mean(em**2)):.3f} "
              f"c={cov['mod']}/{len(em)} w={np.median(wid['mod']):.2f} | n={n}")


if __name__ == "__main__":
    run()
