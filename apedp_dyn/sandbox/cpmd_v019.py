"""CPMD v0.19 — hidden-mechanism recovery benchmark.

Prereg: docs/CPMD_PREREGISTRATION_v019.md (grammar+budget frozen BEFORE
truths below were written; proposer is blind exhaustive search).
Truths (committed here, never shape the proposer):
  IN:  r = A*sig*Q*|C| + N(0,sig)   |  OUT: r = A*sig*sin(3Q)*M + N(0,sig)
"""
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "apedp_dyn/sandbox")

import numpy as np
from cpmd_discovery_suite import EPS, KK, K, M, ic_null
from cpmd_v018 import feats

SIG = 0.01
ALPHA_GRAMMAR = 0.05 / 60


def gen_truth(seed, which, A, n=3000):
    rng = np.random.RandomState(int(seed))
    base = rng.normal(0, SIG, n)
    out = base.copy()
    buf = list(base[:K])
    for t in range(K, n - 1):
        zw = np.array(buf[-K:]) / (float(np.std(buf[-20:])) + EPS)
        C = float(np.sum(zw) / (EPS + np.sum(np.abs(zw))))
        Q = float(np.sum(zw * np.abs(zw)) / (EPS + np.sum(zw ** 2)))
        rw = np.array(buf[-KK:])
        float((EPS + np.mean(rw[-M:] ** 2)) / (EPS + np.mean(rw ** 2)))
        Mt = 1.0 if float(np.mean(buf[-5:])) > 0 else -1.0
        sigt = float(np.std(buf[-20:]) + EPS)
        if which == "in":
            out[t + 1] = A * sigt * Q * abs(C) + rng.normal(0, SIG)
        else:
            out[t + 1] = A * sigt * np.sin(3 * Q) * Mt + rng.normal(0, SIG)
        buf.append(out[t + 1])
    return out


def grammar_forms():
    # (name, fn(sig,C,Q,V,M,dC,tQ,tC)) — 30 base forms; σ-scaling variants
    # counted inside (each form tested raw AND ×σ). Total ≤ 60.
    return {
        "C": lambda s, C, Q, V, Mm, dC, tQ, tC: C,
        "Q": lambda s, C, Q, V, Mm, dC, tQ, tC: Q,
        "V": lambda s, C, Q, V, Mm, dC, tQ, tC: V[0] - 1,
        "M": lambda s, C, Q, V, Mm, dC, tQ, tC: Mm,
        "aC": lambda s, C, Q, V, Mm, dC, tQ, tC: abs(C),
        "dC": lambda s, C, Q, V, Mm, dC, tQ, tC: dC,
        "tQ": lambda s, C, Q, V, Mm, dC, tQ, tC: tQ,
        "tC": lambda s, C, Q, V, Mm, dC, tQ, tC: tC,
        "QxC": lambda s, C, Q, V, Mm, dC, tQ, tC: Q * abs(C),
        "CxM": lambda s, C, Q, V, Mm, dC, tQ, tC: C * Mm,
        "QxM": lambda s, C, Q, V, Mm, dC, tQ, tC: Q * Mm,
        "sVDM": lambda s, C, Q, V, Mm, dC, tQ, tC: (1.0 if V[0] >= V[1] else -1.0) * Mm,
        "sCxdC": lambda s, C, Q, V, Mm, dC, tQ, tC: (1.0 if C >= 0 else -1.0) * dC,
        "tQxsC": lambda s, C, Q, V, Mm, dC, tQ, tC: tQ * (1.0 if C >= 0 else -1.0),
        "QxCxM": lambda s, C, Q, V, Mm, dC, tQ, tC: Q * abs(C) * Mm,
    }


def eval_form(fn, ret, seed, scaled):
    n = len(ret)
    sig, C, Q, V, Mt = feats(ret)
    fc = np.full(n, np.nan)
    Cprev = 0.0
    for t in range(K, n - 1):
        v = fn(sig[t], C[t], Q[t], (V[t], V[t - 1]), Mt[t],
               abs(C[t]) - abs(Cprev), np.tanh(Q[t]), np.tanh(C[t]))
        fc[t] = (sig[t] * v) if scaled else v
        Cprev = C[t]
    return ic_null(fc[:-1], ret[1:len(fc)], seed)


def score_paths(which, A, seeds):
    paths = [gen_truth(sd, which, A) for sd in seeds]
    forms = grammar_forms()
    rows = []
    for name, fn in forms.items():
        for scaled in (False, True):
            ics, ps = [], []
            for ret, sd in zip(paths, seeds, strict=False):
                ic, p, _ = eval_form(fn, ret, sd, scaled)
                ics.append(ic)
                ps.append(p)
            rows.append((f"{name}{'s' if scaled else ''}", float(np.mean(ics)),
                         float(np.median(ps))))
    rows.sort(key=lambda r: r[2])
    return rows, paths


def run():
    # one-shot amplitude check: SINGLE amplitude A=2.0 per prereg (no search).
    # Recorded whatever it yields; discovery proceeds regardless.
    A_IN = A_OUT = 2.0
    for which, A in (("in", A_IN), ("out", A_OUT)):
        ics = []
        for i in range(10):
            ret = gen_truth(90000 + i, which, A)
            _sig, C, Q, _V, Mt = feats(ret)
            n = len(ret)
            tf = np.full(n, np.nan)
            for t in range(K, n - 1):
                tf[t] = Q[t] * abs(C[t]) if which == "in" else np.sin(3 * Q[t]) * Mt[t]
            ok = np.isfinite(tf[:-1])
            ics.append(float(np.corrcoef(tf[:-1][ok], ret[1:len(tf)][ok])[0, 1]))
        print(f"calib {which} A={A}: oracleIC={np.mean(ics):+.4f} (recorded, unadjusted)")
    print("== discovery (seeds 91000+) ==")
    for which, A in (("in", A_IN), ("out", A_OUT)):
        rows, _ = score_paths(which, A, range(91000, 91020))
        print(f"-- {which}-grammar truth --")
        for name, ic, p in rows[:5]:
            print(f"  {name}: meanIC={ic:+.4f} median-p={p:.4f}")
        Winners = [r for r in rows if r[2] < ALPHA_GRAMMAR]
        print(f"  pass Bonferroni(0.05/60): {[w[0] for w in Winners] if Winners else 'NONE'}")


if __name__ == "__main__":
    run()
