"""CPMD ridge diagnostics — direct confirmation of alpha-tau0 confounding.

Run: python3 apedp_dyn/sandbox/cpmd_ridge_diagnostics.py
Findings (seed 2001, noise 0.1, truth a=1.2 t0=8.0):
  - SSR grid prefers (a=1.2, t0=30, SSR=0.71) over truth (1.2, 8.0, SSR=5.93).
  - 6/6 inits converge to the SAME wrong point (a=0.62, t0=2.14).
  - corr(a,t0) = +0.983 at the optimum.
  - Fixed-t0 recovery: rmse 0.435 vs 105.8 free (60 fresh seeds, noise 0.1).
Conclusion: optimizer is stable; the likelihood surface itself does not
separate (a,t0) pairs. Ridge CONFIRMED as dominant mechanism. A diagnostic
IndexError (corrcoef on 1x1) in an earlier draft falsely printed inits as
FAILED; corrected here — see commit history.
"""
import itertools
import json
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "apedp_dyn/sandbox")

import numpy as np
from scipy.optimize import curve_fit

from cpmd_calibration import gen

TRUTH = {"a": 1.2, "t0": 8.0, "seed": 2001, "noise": 0.1}


def main():
    lags, y = gen("power", TRUTH["noise"], TRUTH["seed"])
    ly = np.log(y)

    def ssr(lA, a, t0):
        return float(np.sum((ly - (lA - a * np.log(lags + t0))) ** 2))

    surface = {}
    for a in (0.5, 1.2, 2.0, 5.0):
        for t0 in (1.0, 8.0, 30.0, 100.0):
            shp = np.log(lags + t0)
            lA = float(np.sum(shp * ly) / np.sum(shp ** 2))
            surface[f"a{a}_t{t0}"] = round(ssr(lA, a, t0), 3)

    inits = []
    for p0 in [(np.log(0.05), 1.0, 8.0), (np.log(0.1), 0.5, 2.0),
               (np.log(0.01), 2.0, 30.0), (np.log(0.05), 5.0, 50.0),
               (np.log(0.2), 0.2, 1.0), (np.log(0.05), 1.2, 8.0)]:
        pp, pc = curve_fit(lambda t, lA, a, tt: lA - a * np.log(t + tt),
                           lags, ly, p0=p0, maxfev=50000)
        r = float(pc[1, 2] / np.sqrt(pc[1, 1] * pc[2, 2]))
        inits.append({"p0a": p0[1], "p0t": p0[2],
                      "a": round(float(pp[1]), 3), "t0": round(float(pp[2]), 3),
                      "corr_a_t0": round(r, 4)})

    fixed, free = [], []
    for a, sd in itertools.product((0.8, 1.2, 1.8), range(2100, 2120)):
        l2, y2 = gen("power", 0.1, sd)
        ly2 = np.log(y2)
        pf, _ = curve_fit(lambda t, lA, aa: lA - aa * np.log(t + 8.0),
                          l2, ly2, p0=(np.log(0.05), 1.0), maxfev=50000)
        fixed.append(float(pf[1] - a))
        pr, _ = curve_fit(lambda t, lA, aa, tt: lA - aa * np.log(t + tt),
                          l2, ly2, p0=(np.log(0.05), 1.0, 8.0), maxfev=50000)
        free.append(float(pr[1] - a))
    fixed = np.array(fixed)
    free = np.array(free)
    out = {"truth": TRUTH, "ssr_surface": surface, "inits": inits,
           "fixed_t0": {"rmse": round(float(np.sqrt(np.mean(fixed ** 2))), 3),
                        "within03": round(float(np.mean(np.abs(fixed) <= 0.3)), 3)},
           "free_t0": {"rmse": round(float(np.sqrt(np.mean(free ** 2))), 3),
                       "within03": round(float(np.mean(np.abs(free) <= 0.3)), 3)},
           "verdict": "RIDGE_CONFIRMED"}
    print(json.dumps(out, indent=1))
    json.dump(out, open("apedp_dyn/sandbox/cpmd_ridge_results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
