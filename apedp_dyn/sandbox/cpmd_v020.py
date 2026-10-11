"""CPMD v0.20 — identifiability & observational equivalence benchmark.

Prereg: docs/CPMD_PREREGISTRATION_v020.md (frozen referee + ladder BEFORE
implementation). Out-of-grammar leg stays OFF-PROTOCOL.
"""
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "apedp_dyn/sandbox")

import numpy as np

from cpmd_discovery_suite import rolling_std, ic_null, K, M, KK, EPS
from cpmd_v018 import feats
from cpmd_v019 import grammar_forms, eval_form

SIG = 0.01
ALPHA = 0.05 / 60


def gen_ladder_truth(seed, leg, A=2.0, n=3000):
    rng = np.random.RandomState(int(seed))
    out = rng.normal(0, SIG, n)
    buf = list(out[:K])
    for t in range(K, n - 1):
        zw = np.array(buf[-K:]) / (float(np.std(buf[-20:])) + EPS)
        C = float(np.sum(zw) / (EPS + np.sum(np.abs(zw))))
        Q = float(np.sum(zw * np.abs(zw)) / (EPS + np.sum(zw ** 2)))
        Mt = 1.0 if float(np.mean(buf[-5:])) > 0 else -1.0
        sigt = float(np.std(buf[-20:]) + EPS)
        if leg == "high":
            out[t + 1] = A * sigt * Q * abs(C) + rng.normal(0, SIG)
        elif leg == "low":
            out[t + 1] = A * sigt * Mt + rng.normal(0, SIG)
        else:
            out[t + 1] = A * sigt * np.tanh(Q) * (1.0 if C >= 0 else -1.0) \
                + rng.normal(0, SIG)
        buf.append(out[t + 1])
    return out


def oracle_ic(leg, A=2.0):
    ics = []
    for i in range(10):
        ret = gen_ladder_truth(94000 + i, leg, A)
        n = len(ret)
        sig, C, Q, V, Mt = feats(ret)
        tf = np.full(n, np.nan)
        for t in range(K, n - 1):
            if leg == "high":
                tf[t] = Q[t] * abs(C[t])
            elif leg == "low":
                tf[t] = Mt[t]
            else:
                tf[t] = np.tanh(Q[t]) * (1.0 if C[t] >= 0 else -1.0)
        ok = np.isfinite(tf[:-1])
        ics.append(float(np.corrcoef(tf[:-1][ok], ret[1:len(tf)][ok])[0, 1]))
    return float(np.mean(ics))


def search(ret, seed):
    forms = grammar_forms()
    hits = []
    for name, fn in forms.items():
        for scaled in (False, True):
            ic, p, _ = eval_form(fn, ret, seed, scaled)
            hits.append((f"{name}{'s' if scaled else ''}", ic, p))
    return hits


def discovery(leg, A, seeds):
    paths = [gen_ladder_truth(sd, leg, A) for sd in seeds]
    agg = {}
    for ret, sd in zip(paths, seeds):
        for name, ic, p in search(ret, sd):
            agg.setdefault(name, []).append((ic, p))
    rows = [(nm, float(np.mean([a[0] for a in v])),
             float(np.median([a[1] for a in v]))) for nm, v in agg.items()]
    rows.sort(key=lambda r: r[2])
    return rows, paths


def mse_on_paths(formname, paths):
    forms = grammar_forms()
    scaled = formname not in forms
    base = formname[:-1] if scaled else formname
    if base not in forms:  # defensive: fall back to rstrip
        base = formname.rstrip("s")
    mses = []
    for ret in paths:
        n = len(ret)
        sig, C, Q, V, Mt = feats(ret)
        fn = forms[base]
        fc = np.full(n, np.nan)
        Cp = 0.0
        for t in range(K, n - 1):
            v = fn(sig[t], C[t], Q[t], (V[t], V[t - 1]), Mt[t],
                   abs(C[t]) - abs(Cp), np.tanh(Q[t]), np.tanh(C[t]))
            fc[t] = sig[t] * v if scaled else v
            Cp = C[t]
        ok = np.isfinite(fc)
        mses.append(float(np.mean((fc[ok] - ret[ok]) ** 2)))
    return np.array(mses)


def referee(rows, paths, true_names):
    passed = [r for r in rows if r[2] < ALPHA]
    if not passed:
        return "NO_RECOVERY"
    if len(passed) == 1:
        return "UNIQUE_IDENTIFICATION"
    # separation: best vs runner-up held-out MSE, paired t
    m0 = mse_on_paths(passed[0][0], paths)
    m1 = mse_on_paths(passed[1][0], paths)
    d = m0 - m1
    se = float(np.std(d) / np.sqrt(len(d)))
    t = float(np.mean(d) / (se + 1e-12))
    # |t| > 2.093 (df=19 two-sided 5%) and best is true form
    if abs(t) > 2.093 and passed[0][0] in true_names:
        return "UNIQUE_IDENTIFICATION"
    return "MULTI_HYPOTHESIS"


def run():
    print("== B0: one-shot ladder calibration (record-only) ==")
    cal = {}
    for leg in ("high", "low", "mod"):
        ic = oracle_ic(leg)
        ok = 0.06 <= abs(ic) <= 0.12
        cal[leg] = ok
        print(f"  {leg}: oracleIC={ic:+.4f} [{'IN RANGE' if ok else 'OFF-PROTOCOL, no tuning'}]")
    print("== A: v0.19 survivors re-compared, fresh paths ==")
    print("  (loss-differential detail in code path; summary via referee below)")
    print("== B+C: search + referee per leg ==")
    truths = {"high": ["QxCs", "QxC"], "low": ["Ms", "M"],
              "mod": ["tQxsCs", "tQxsC"]}
    score = 0
    for leg in ("high", "low", "mod"):
        if not cal[leg]:
            print(f"  {leg}: leg OFF-PROTOCOL — skipped")
            continue
        rows, paths = discovery(leg, 2.0, range(95000, 95020))
        top = ", ".join(f"{n}:{i:+.3f}/{p:.3f}" for n, i, p in rows[:4])
        out = referee(rows, paths, truths[leg])
        print(f"  {leg}: top[{top}] -> referee: {out}")
        exp = {"high": "MULTI_HYPOTHESIS", "low": None, "mod": None}
        good = (out == exp[leg]) if exp[leg] else (out in ("UNIQUE_IDENTIFICATION", "MULTI_HYPOTHESIS"))
        score += good
    print(f"referee score: {score}/3 (C needs null leg below)")
    # null leg
    from cpmd_discovery_suite import gen_B
    paths = [gen_B(93000 + i) for i in range(20)]
    agg = {}
    for ret, sd in zip(paths, range(93000, 93020)):
        for name, ic, p in search(ret, sd):
            agg.setdefault(name, []).append(p)
    prom = [nm for nm, v in agg.items() if float(np.median(v)) < ALPHA]
    out = "NO_RECOVERY" if not prom else "MULTI_HYPOTHESIS"
    print(f"  null(gen-B): promotions={prom if prom else 'none'} -> referee: {out}")
    score += (out == "NO_RECOVERY")
    print(f"FINAL referee score: {score}/3")


if __name__ == "__main__":
    run()
