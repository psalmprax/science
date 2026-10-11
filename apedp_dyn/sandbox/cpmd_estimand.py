"""CPMD v0.12 — estimand alignment (Tests A-D).

Prereg: docs/CPMD_PREREGISTRATION_v012.md (frozen before run).
"""
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "apedp_dyn/sandbox")

import numpy as np
from scipy.optimize import curve_fit

from cpmd_calibration import gen
from cpmd_effective_slope import slope_ols, LAGS

TM = float(np.mean(LAGS))
A_TRUE, T0_TRUE = 1.2, 8.0


def pointwise_truth(a=A_TRUE, t0=T0_TRUE, tau=8.0):
    return float(a * tau / (tau + t0))


def finite_window_truth(a=A_TRUE, t0=T0_TRUE):
    lx = np.log(LAGS)
    ly = np.log(0.05 * (LAGS + t0) ** (-a))  # exact noiseless curve
    xc = lx - lx.mean()
    return float(-np.sum(xc * ly) / np.sum(xc ** 2))


def local3(y):
    lx, ly = np.log(LAGS), np.log(y)
    mid = len(lx) // 2
    sl = (ly[mid + 1] - ly[mid - 1]) / (lx[mid + 1] - lx[mid - 1])
    return -float(sl)


def run():
    pt = pointwise_truth()
    fw = finite_window_truth()
    print(f"Test A: pointwise truth={pt:.4f} finite-window truth={pt - fw:+.4f} "
          f"apart -> fw={fw:.4f}")
    for nl in (0.1, 0.3):
        eo_pt, eo_fw, el, em = [], [], [], []
        n = 0
        for k in range(200):
            _, yfull = gen("power", nl, 6000 + k, t0=T0_TRUE, a=A_TRUE)
            ar = np.arange(2, 14)
            y = yfull[((ar >= 4) & (ar <= 12))]
            s, _ = slope_ols(LAGS, y)
            eo_pt.append(s - pt)
            eo_fw.append(s - fw)
            el.append(local3(y) - pt)
            try:
                pp, _ = curve_fit(
                    lambda t, lA, a, tt: lA - a * np.log(t + tt),
                    LAGS, np.log(y), p0=(np.log(0.05), 1.0, 8.0), maxfev=50000)
                em.append(float(pp[1] * TM / (TM + pp[2])) - pt)
            except Exception:
                pass
            n += 1
        for name, e in (("OLS-vs-pointwise", eo_pt), ("OLS-vs-finwin", eo_fw),
                        ("LOCAL3-vs-pointwise", el), ("NLFIT-vs-pointwise", em)):
            e = np.array(e)
            print(f"nl={nl} {name}: bias={np.mean(e):+.4f} "
                  f"rmse={np.sqrt(np.mean(e**2)):.4f} n={len(e)}/{n}")


if __name__ == "__main__":
    run()
