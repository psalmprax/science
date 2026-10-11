"""CPMD v0.13 — operating envelope + uncertainty feasibility.

Prereg: docs/CPMD_PREREGISTRATION_v013.md (frozen before run).
"""
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "apedp_dyn/sandbox")

import numpy as np

from cpmd_calibration import gen
from cpmd_effective_slope import LAGS
from cpmd_estimand import finite_window_truth

FW = finite_window_truth()
LX = np.log(LAGS)
XC = LX - LX.mean()
DENOM = float(np.sum(XC ** 2))
T_CRIT = 2.3646  # t_{0.975, df=7}
RBOOT_B = 500


def ols_parts(y):
    ly = np.log(y)
    s = -float(np.sum(XC * ly) / DENOM)
    r = ly - ly.mean() + s * XC
    se = float(np.sqrt(float(np.sum(r ** 2)) / 7 / DENOM))
    return s, r, se


def rboot_ci(y, seed):
    s, r, _ = ols_parts(y)
    rng = np.random.RandomState(int(seed))
    ly = np.log(y)
    pred = ly.mean() - s * XC
    boots = []
    for _ in range(RBOOT_B):
        yb = np.exp(pred + rng.choice(r, size=len(y), replace=True))
        lyb = np.log(yb)
        boots.append(-float(np.sum(XC * lyb) / DENOM))
    b = np.array(boots)
    return s, float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def run_cell(nl, methods, seed0=7000, n=200):
    out = {m: {"cov": 0, "w": [], "e": []} for m in methods}
    for k in range(n):
        _, yf = gen("power", nl, seed0 + k, t0=8.0, a=1.2)
        ar = np.arange(2, 14)
        y = yf[((ar >= 4) & (ar <= 12))]
        s, _, se = ols_parts(y)
        if "OLS-N" in methods:
            out["OLS-N"]["e"].append(s - FW)
            out["OLS-N"]["cov"] += ((s - 1.96 * se) <= FW <= (s + 1.96 * se))
            out["OLS-N"]["w"].append(3.92 * se)
        if "OLS-T" in methods:
            out["OLS-T"]["e"].append(s - FW)
            out["OLS-T"]["cov"] += ((s - T_CRIT * se) <= FW <= (s + T_CRIT * se))
            out["OLS-T"]["w"].append(2 * T_CRIT * se)
        if "RBOOT" in methods:
            sb, lo, hi = rboot_ci(y, seed0 + k)
            out["RBOOT"]["e"].append(sb - FW)
            out["RBOOT"]["cov"] += (lo <= FW <= hi)
            out["RBOOT"]["w"].append(hi - lo)
    return out


def verdict(d):
    e = np.array(d["e"])
    ok = (d["cov"] >= 184 and np.median(d["w"]) <= 1.0
          and abs(np.mean(e)) <= 0.1 and np.sqrt(np.mean(e ** 2)) <= 0.3)
    return ("PASS" if ok else "fail")


def run():
    print("Direction A: envelope (OLS-N), truth=0.5619")
    for nl in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
        d = run_cell(nl, ("OLS-N",))["OLS-N"]
        e = np.array(d["e"])
        print(f"nl={nl}: cov={d['cov']}/200 w={np.median(d['w']):.2f} "
              f"bias={np.mean(e):+.4f} rmse={np.sqrt(np.mean(e**2)):.3f} "
              f"[{verdict(d)}]")
    print("Direction B: high-noise methods")
    for nl in (0.20, 0.25, 0.30):
        r = run_cell(nl, ("OLS-N", "OLS-T", "RBOOT"))
        cells = []
        for m in ("OLS-N", "OLS-T", "RBOOT"):
            d = r[m]
            e = np.array(d["e"])
            cells.append(f"{m}:cov={d['cov']} w={np.median(d['w']):.2f} "
                         f"[{verdict(d)}]")
        print(f"nl={nl} | " + " ".join(cells))


if __name__ == "__main__":
    run()
