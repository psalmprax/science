"""CPMD v0.14 — frozen v0.13 procedure under independent eval, correlated
noise, and transfer. Prereg: docs/CPMD_PREREGISTRATION_v014.md.
"""
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "apedp_dyn/sandbox")

import numpy as np
from cpmd_calibration import gen
from cpmd_effective_slope import LAGS
from cpmd_ols_uncertainty import gen_dep  # generators only

T7 = 2.3646  # t_{.975,df=7}
T5 = 2.5706  # t_{.975,df=5}
T8 = 2.3060  # t_{.975,df=8}


def finwin_truth(lags, kind, a=1.2, t0=8.0, b=0.1):
    lx = np.log(lags)
    if kind == "power":
        ly = np.log(0.05 * (lags + t0) ** (-a))
    elif kind == "expo":
        ly = np.log(0.05 * np.exp(-b * lags))
    else:
        ly = np.full_like(lags, np.log(0.05))
    xc = lx - lx.mean()
    return float(-np.sum(xc * ly) / np.sum(xc ** 2))


def ols_t_cell(lags, tcrit, truth, gen_fn, n=200):
    lx = np.log(lags)
    xc = lx - lx.mean()
    den = float(np.sum(xc ** 2))
    df = len(lags) - 2
    cov = 0
    w, e = [], []
    for k in range(n):
        y = gen_fn(k)
        ly = np.log(y)
        s = -float(np.sum(xc * ly) / den)
        r = ly - ly.mean() + s * xc
        se = float(np.sqrt(float(np.sum(r ** 2)) / df / den))
        lo, hi = s - tcrit * se, s + tcrit * se
        cov += (lo <= truth <= hi)
        w.append(hi - lo)
        e.append(s - truth)
    e = np.array(e)
    ok = (cov >= 184 and np.median(w) <= 1.0 and abs(np.mean(e)) <= 0.1
          and np.sqrt(np.mean(e ** 2)) <= 0.3)
    return (f"cov={cov}/{n} w={np.median(w):.2f} bias={np.mean(e):+.4f} "
            f"rmse={np.sqrt(np.mean(e**2)):.3f} [{'PASS' if ok else 'fail'}]", ok)


def sl(lags, yfull):
    ar = np.arange(2, 2 + len(yfull))
    return yfull[((ar >= lags[0]) & (ar <= lags[-1]))]


def run():
    print("== 1. independent eval (seeds 8000+, base design) ==")
    ok1 = 0
    tot1 = 0
    for i, nl in enumerate((0.05, 0.10, 0.15, 0.20)):
        truth = finwin_truth(LAGS, "power")
        s, ok = ols_t_cell(LAGS, T7, truth,
                           lambda k, nl=nl, off=8000 + i * 200:
                           sl(LAGS, gen("power", nl, off + k)[1]))
        print(f"nl={nl}: {s}")
        ok1 += ok
        tot1 += 1
    print(f"independent eval: {ok1}/{tot1}")
    print("== 2. correlated noise (OLS-T df=7) ==")
    ok2 = 0
    tot2 = 0
    for (struct, rho), nl in list(__import__("itertools").product(
                                  [("ar1", 0.3), ("ar1", 0.5), ("cs", 0.5),
                                   ("ma1", 0.0)], (0.1, 0.2))):
        truth = finwin_truth(LAGS, "power")
        s, ok = ols_t_cell(LAGS, T7, truth,
                           lambda k, st=struct, rh=rho, nl=nl, off=8400:
                           sl(LAGS, gen_dep(st, rh, nl, off + k)[1]))
        print(f"{struct} rho={rho} nl={nl}: {s}")
        ok2 += ok
        tot2 += 1
    print(f"correlated: {ok2}/{tot2}")
    print("== 3. transfer ==")
    ok3 = 0
    tot3 = 0
    L310 = np.arange(3, 11, dtype=float)
    L615 = np.arange(6, 16, dtype=float)
    cases = [("lags3-10/power", L310, T5, "power", False, None),
             ("lags6-15/power", L615, T8, "power", True, None),
             ("lags4-12/expo", LAGS, T7, "expo", False, None),
             ("lags4-12/flat", LAGS, T7, "flat", False, "fp")]
    for name, lags, tc, kind, wide, mode in cases:
        for nl in (0.1, 0.2):
            truth = finwin_truth(lags, kind)
            if mode == "fp":
                fpf = []
                for k in range(200):
                    yf = sl(lags, gen(kind, nl, 8800 + k)[1])
                    lx = np.log(lags)
                    xc = lx - lx.mean()
                    den = float(np.sum(xc ** 2))
                    ly = np.log(yf)
                    s = -float(np.sum(xc * ly) / den)
                    r = ly - ly.mean() + s * xc
                    se = float(np.sqrt(float(np.sum(r ** 2))
                                       / (len(lags) - 2) / den))
                    fpf.append(abs(s) > 0.5 * 2 * tc * se)
                rate = float(np.mean(fpf))
                ok = rate <= 0.05
                print(f"{name} nl={nl}: FP-equiv rate={rate:.3f} "
                      f"[{'PASS' if ok else 'fail'}]")
            else:
                s, ok = ols_t_cell(lags, tc, truth,
                                   lambda k, kd=kind, nl=nl, wd=wide, lg=lags:
                                   sl(lg, gen(kd, nl, 8800 + k, wide=wd)[1]))
                print(f"{name} nl={nl}: {s}")
            ok3 += ok
            tot3 += 1
    print(f"transfer: {ok3}/{tot3}")
    seal = (ok1 == tot1 and ok2 >= 6 and ok3 >= 6)
    print(f"SEALED-SET DECISION: {'OPEN (gates met)' if seal else 'STAYS SHUT'}")


if __name__ == "__main__":
    run()
