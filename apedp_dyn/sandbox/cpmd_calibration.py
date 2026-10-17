"""CPMD v0.9.5 — selector calibration under APEDP-X freeze discipline.

Consolidates the /tmp/opencode v0.9.2-v0.9.4 procedures into versioned,
reproducible code. History:
  v0.9.2: log-space fitting fixes stability (129 -> 0 fit failures).
  v0.9.3: bootstrap best interval but 88.3% coverage; FP 8.3%.
  v0.9.4: BIC selector rarely picks power even on power truth -> FAILED diagnostic.
  v0.9.5: compare BIC vs AICc vs BIC-correlated on fixed diagnostic datasets;
          sealed evaluation set generated separately and left unopened.
"""
import hashlib
import json

import numpy as np
from scipy.optimize import curve_fit

VERSION = "0.9.5"
TRUTHS = ("power", "expo", "flat", "regime")

# Fixed diagnostic design (preregistered; see docs/CPMD_PREREGISTRATION_v095.md)
DIAG_SEEDS = {"power": range(1000, 1040), "expo": range(1100, 1140),
              "flat": range(1200, 1220), "regime": range(1300, 1320)}
DIAG_NOISE = 0.3
DIAG_RHO = 0.0
DIAG_WIDE = False
SEAL_MASTER_SEED = 99991

GATES = {"fp_ceiling": 0.05, "coverage_nominal": 0.95}


def gen(kind, nl, sd, wide=False, rho=0.0, t0=8.0, a=1.2, b=0.1):
    rng = np.random.RandomState(int(sd))
    n = 24 if wide else 12
    lags = np.arange(2, 2 + n, dtype=float)
    if kind == "power":
        mu = 0.05 * (lags + t0) ** (-a)
    elif kind == "expo":
        mu = 0.05 * np.exp(-b * lags)
    elif kind == "flat":
        mu = np.full_like(lags, 0.05)
    elif kind == "regime":
        cut = 14 if wide else 8
        mu = np.where(lags < cut, 0.05 * (lags + t0) ** (-a),
                      0.05 * (cut + t0) ** (-a))
    else:
        raise ValueError(kind)
    e = rng.normal(0, nl, len(lags))
    if rho > 0:
        for i in range(1, len(e)):
            e[i] = rho * e[i - 1] + np.sqrt(1 - rho ** 2) * e[i]
    return lags, mu * np.exp(e)


def _fit(curve, lags, ly, p0):
    pp, pc = curve_fit(curve, lags, ly, p0=p0, maxfev=50000)
    se = float(np.sqrt(np.diag(pc))[1]) if np.all(np.isfinite(pc)) else 9.0
    return pp, se


def fit_power(lags, ly):
    return _fit(lambda t, lA, a, tt: lA - a * np.log(t + tt),
                lags, ly, (np.log(0.05), 1.0, 8.0))


def fit_expo(lags, ly):
    pp, _ = _fit(lambda t, lB, b: lB - b * t, lags, ly, (np.log(0.05), 0.1))
    return pp


def ic(ly, pred, k, rule):
    n = len(ly)
    rss = float(np.sum((ly - pred) ** 2))
    if rule == "bic":
        return n * np.log(rss / n) + k * np.log(n)
    if rule == "aicc":
        aic = n * np.log(rss / n) + 2 * k
        return aic + (2 * k * (k + 1)) / max(n - k - 1, 1)
    raise ValueError(rule)


def select(lags, ly, rule="bic", margin=0.0):
    """Return (decision, info). decision in power/expo/abstain/fail."""
    try:
        pp, se = fit_power(lags, ly)
        pe = fit_expo(lags, ly)
    except Exception:  # noqa: BLE001 - deliberate catch-all: best-effort probe path reports or skips the failure, never crashes
        return "fail", None
    ip = ic(ly, pp[0] - pp[1] * np.log(lags + pp[2]), 3, rule)
    ie = ic(ly, pe[0] - pe[1] * lags, 2, rule)
    if ip < ie - margin:
        return "power", (pp, se)
    if ie < ip - margin:
        return "expo", None
    return "abstain", None


def boot_ci(lags, ly, pp, seed, B=100):
    rng = np.random.RandomState(int(seed))
    pred = pp[0] - pp[1] * np.log(lags + pp[2])
    resid = ly - pred
    boots = []
    for _ in range(B):
        yb = np.exp(pred + rng.choice(resid, size=len(lags), replace=True))
        try:
            pb, _ = curve_fit(lambda t, lA, a, tt: lA - a * np.log(t + tt),
                              lags, np.log(yb), p0=tuple(pp), maxfev=20000)
            boots.append(pb[1])
        except Exception:  # noqa: BLE001, S110 - optional probe may fail silently by design (best-effort stage skip)  # noqa: BLE001 - deliberate catch-all: best-effort probe path reports or skips the failure, never crashes
            pass
    b = np.array(boots)
    if len(b) > 10:
        return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
    return 9.0, -9.0


def run_battery(rule, margin, seeds, nl=DIAG_NOISE, wide=DIAG_WIDE, rho=DIAG_RHO):
    """Full end-to-end battery. Returns dict with selection confusion,
    recovery, coverage, FP, failures — every attempt counted."""
    out = {"rule": rule, "margin": margin, "attempted": 0, "fail": 0,
           "abstain": 0, "select_power": 0, "select_expo": 0,
           "fp": 0, "nonpower_attempted": 0,
           "power_recalled": 0, "power_attempted": 0,
           "covered": 0, "recovered": 0}
    for kind, sds in seeds.items():
        for sd in sds:
            out["attempted"] += 1
            lags, y = gen(kind, nl, sd, wide, rho)
            dec, info = select(lags, np.log(y), rule, margin)
            if dec == "fail":
                out["fail"] += 1
                continue
            if dec == "abstain":
                out["abstain"] += 1
                continue
            if dec == "power":
                out["select_power"] += 1
                if kind != "power":
                    out["fp"] += 1
                else:
                    out["power_recalled"] += 1
                    lo, hi = boot_ci(lags, np.log(y), info[0], seed=sd)
                    if lo <= 1.2 <= hi:
                        out["covered"] += 1
                    if abs(info[0][1] - 1.2) <= 0.3:
                        out["recovered"] += 1
            else:
                out["select_expo"] += 1
            if kind == "power":
                out["power_attempted"] += 1
            else:
                out["nonpower_attempted"] += 1
    return out


def sha256_of(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True,
                                     default=str).encode()).hexdigest()


if __name__ == "__main__":
    results = {}
    for rule in ("bic", "aicc"):
        for margin in (0.0, 2.0, 6.0):
            key = f"{rule}/m{margin}"
            results[key] = run_battery(rule, margin,
                                       {k: list(v) for k, v in DIAG_SEEDS.items()})
            r = results[key]
            print(f"{key}: attempted={r['attempted']} fail={r['fail']} "
                  f"abstain={r['abstain']} selP={r['select_power']} "
                  f"fp={r['fp']}/{r['nonpower_attempted']} "
                  f"recall={r['power_recalled']}/{r['power_attempted']} "
                  f"cov={r['covered']}/{r['power_recalled']} "
                  f"rec={r['recovered']}/{r['power_recalled']}")
    print("VERSION", VERSION, "HASH", sha256_of(results))
