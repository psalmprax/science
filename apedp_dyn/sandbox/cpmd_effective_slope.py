"""CPMD v0.10 — finite-lag effective slope (diagnostic phase).

Estimand: alpha_eff over FIXED lags 4-12 via (a) log-log OLS baseline and
(b) model-based local slope from the fitted shifted power law.
Prereg: docs/CPMD_PREREGISTRATION_v010.md (frozen gates). Sealed set unopened.
"""
import itertools
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "apedp_dyn/sandbox")

import numpy as np
from cpmd_calibration import gen
from scipy.optimize import curve_fit

LAGS = np.arange(4, 13, dtype=float)  # frozen interval


def slope_ols(lags, y):
    lx, ly = np.log(lags), np.log(y)
    A = np.vstack([lx, np.ones_like(lx)]).T
    coef, _, _, _ = np.linalg.lstsq(A, ly, rcond=None)
    slope = float(coef[0])
    intercept = float(coef[1])
    pred = slope * lx + intercept
    s2 = float(np.sum((ly - pred) ** 2)) / (len(lx) - 2)
    se = float(np.sqrt(s2 / np.sum((lx - lx.mean()) ** 2)))
    return -float(slope), se


def slope_model(lags, y):
    pp, _ = curve_fit(lambda t, lA, a, tt: lA - a * np.log(t + tt),
                      lags, np.log(y), p0=(np.log(0.05), 1.0, 8.0),
                      maxfev=50000)
    _lA, a, t0 = (float(v) for v in pp)
    tm = float(np.mean(lags))
    return float(a * tm / (tm + t0))


def true_slope(kind, a=1.2, t0=8.0, b=0.1):
    tm = float(np.mean(LAGS))
    if kind == "power":
        return float(a * tm / (tm + t0))
    if kind == "expo":
        return float(b * tm)
    if kind == "flat":
        return 0.0
    return float("nan")  # regime: no single truth


def run():
    print("v0.10 diagnostic: lags 4-12, seeds 3000+")
    for kind, nl in itertools.product(("power", "expo", "flat", "regime"),
                                      (0.1, 0.3)):
        eo, em, tot = [], [], 0
        for sd in range(3000, 3040):
            _, y = gen(kind, nl, sd)
            yy = y[((np.arange(2, 14) >= 4) & (np.arange(2, 14) <= 12))]
            try:
                s_ols, _ = slope_ols(LAGS, yy)
                eo.append(s_ols)
            except Exception:  # noqa: BLE001, S110 - optional probe may fail silently by design (best-effort stage skip)  # noqa: BLE001 - deliberate catch-all: best-effort probe path reports or skips the failure, never crashes
                pass
            try:
                em.append(slope_model(LAGS, yy))
            except Exception:  # noqa: BLE001, S110 - optional probe may fail silently by design (best-effort stage skip)  # noqa: BLE001 - deliberate catch-all: best-effort probe path reports or skips the failure, never crashes
                pass
            tot += 1
        t = true_slope(kind)
        eo = np.array(eo)
        em = np.array(em)
        line = (f"{kind} nl={nl}: truth={t:.3f} "
                f"OLS bias={np.mean(eo)-t:+.3f} rmse={np.sqrt(np.mean((eo-t)**2)):.3f} "
                f"n={len(eo)}/{tot}")
        if len(em):
            line += (f" | MODEL bias={np.mean(em)-t:+.3f} "
                     f"rmse={np.sqrt(np.mean((em-t)**2)):.3f} n={len(em)}/{tot}")
        else:
            line += " | MODEL n=0 (all failed)"
        print(line)


if __name__ == "__main__":
    run()
