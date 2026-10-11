"""CPMD v0.21 — oracle-calibrated identifiability benchmark.

Prereg: docs/CPMD_PREREGISTRATION_v021.md (exogenous drivers, analytic
bands, code-enforced pre-qualification). Referee scoring /6.
"""
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "apedp_dyn/sandbox")

import numpy as np

from cpmd_discovery_suite import rolling_std, ic_null, EPS

SIG = 0.01
ALPHA = 0.05 / 12
BAND_MED = (0.10, 0.20)


def drivers(seed, rho, n=3000):
    rng = np.random.RandomState(int(seed))
    e1 = rng.normal(0, 1, n)
    e2 = rng.normal(0, 1, n)
    e2 = rho * e1 + np.sqrt(max(1 - rho ** 2, 1e-9)) * e2
    f1 = np.zeros(n)
    f2 = np.zeros(n)
    for t in range(1, n):
        f1[t] = 0.9 * f1[t - 1] + np.sqrt(1 - 0.81) * e1[t]
        f2[t] = 0.9 * f2[t - 1] + np.sqrt(1 - 0.81) * e2[t]
    s1, s2 = float(np.std(f1)), float(np.std(f2))
    return f1 / s1, f2 / s2


def gen_ret(seed, rho, beta, n=3000):
    rng = np.random.RandomState(int(seed) + 555000)
    f1, f2 = drivers(seed, rho, n)
    return beta * f1[:-1] + rng.normal(0, SIG, n - 1), f1, f2


def grammar_fc(name, f1, f2, sig):
    n = len(f1)
    if name == "f1":
        v = f1
    elif name == "f2":
        v = f2
    elif name == "f1+f2":
        v = (f1 + f2) / 2
    elif name == "f1-f2":
        v = (f1 - f2) / 2
    elif name == "tanhf1":
        v = np.tanh(f1)
    elif name == "tanhf2":
        v = np.tanh(f2)
    elif name == "signf1":
        v = np.sign(f1)
    elif name == "signf2":
        v = np.sign(f2)
    elif name == "f1f2":
        v = f1 * f2
    elif name == "f1sq-f2sq":
        v = f1 ** 2 - f2 ** 2
    elif name == "sigf1":
        v = sig * f1
    elif name == "sigf2":
        v = sig * f2
    return v


FORMS = ["f1", "f2", "f1+f2", "f1-f2", "tanhf1", "tanhf2", "signf1",
         "signf2", "f1f2", "f1sq-f2sq", "sigf1", "sigf2"]


def oracle_check(rho, beta):
    ics = []
    for i in range(10):
        r, f1, f2 = gen_ret(96000 + i, rho, beta)
        ok = np.isfinite(f1[:-1]) & np.isfinite(r)
        ics.append(float(np.corrcoef(f1[:-1][ok], r[ok])[0, 1]))
    return float(np.mean(ics))


def discovery(rho, beta, seeds):
    agg = {nm: [] for nm in FORMS}
    paths = []
    for sd in seeds:
        r, f1, f2 = gen_ret(sd, rho, beta)
        sig = rolling_std(np.concatenate([r[:1], r]))
        paths.append((r, f1, f2))
        for nm in FORMS:
            fc = grammar_fc(nm, f1[:-1], f2[:-1], sig[:len(f1) - 1])
            ic, p, _ = ic_null(fc, r, sd)
            agg[nm].append((ic, p))
    rows = [(nm, float(np.mean([a[0] for a in v])),
             float(np.median([a[1] for a in v]))) for nm, v in agg.items()]
    rows.sort(key=lambda x: x[2])
    return rows, paths


def mse_form(nm, paths):
    ms = []
    for r, f1, f2 in paths:
        sig = rolling_std(np.concatenate([r[:1], r]))
        fc = grammar_fc(nm, f1[:-1], f2[:-1], sig[:len(f1) - 1])
        ok = np.isfinite(fc) & np.isfinite(r)
        ms.append(float(np.mean((fc[ok] - r[ok]) ** 2)))
    return np.array(ms)


def referee(rows, paths):
    passed = [x for x in rows if x[2] < ALPHA]
    if not passed:
        return "NO_RECOVERY", passed
    if len(passed) == 1:
        return "UNIQUE_IDENTIFICATION", passed
    m0 = mse_form(passed[0][0], paths)
    m1 = mse_form(passed[1][0], paths)
    d = m0 - m1
    se = float(np.std(d) / np.sqrt(len(d)))
    if abs(float(np.mean(d) / (se + 1e-12))) > 2.093:
        return "UNIQUE_IDENTIFICATION", passed
    return "MULTI_HYPOTHESIS", passed


def run():
    beta = 0.0015
    print("== Gate A: pre-qualification (MED band [0.10,0.20)) ==")
    qual = {}
    for rho in (0.9, 0.5, 0.1):
        ic = oracle_check(rho, beta)
        ok = BAND_MED[0] <= abs(ic) < BAND_MED[1]
        qual[rho] = ok
        print(f"  rho={rho}: oracleIC={ic:+.4f} "
              f"[{'QUALIFIED' if ok else 'VOID (code-enforced)'}]")
    print("== Gates B+C: discovery + referee ==")
    score = 0
    for rho, expect in ((0.1, "unique"), (0.5, "open"), (0.9, "multi")):
        if not qual[rho]:
            print(f"  rho={rho}: VOID — no verdict")
            continue
        rows, paths = discovery(rho, beta, range(97000, 97020))
        top = ", ".join(f"{n}:{i:+.3f}/{p:.3f}" for n, i, p in rows[:3])
        out, passed = referee(rows, paths)
        names = [x[0] for x in passed]
        f1in = any(n in ("f1", "sigf1", "tanhf1", "signf1") for n in names)
        if expect == "unique":
            pts = 2 if (out == "UNIQUE_IDENTIFICATION" and f1in) else (1 if f1in else 0)
        elif expect == "multi":
            pts = 2 if out == "MULTI_HYPOTHESIS" else 0
        else:
            pts = 2 if out == "UNIQUE_IDENTIFICATION" else (1 if out == "MULTI_HYPOTHESIS" and f1in else 0)
        score += pts
        print(f"  rho={rho}: top[{top}] -> {out} survivors={names} [{pts}/2]")
    # null leg
    rows, _ = discovery(0.5, 0.0, range(99000, 99020))
    prom = [n for n, i, p in rows if p < ALPHA]
    out = "NO_RECOVERY" if not prom else "MULTI_HYPOTHESIS"
    pts = 2 if out == "NO_RECOVERY" else 0
    score += pts
    print(f"  null: promotions={prom if prom else 'none'} -> {out} [{pts}/2]")
    print(f"FINAL referee score: {score}/6 "
          f"[{'VALIDATED' if score >= 5 else 'LIMITATION RECORDED'}]")


if __name__ == "__main__":
    run()
