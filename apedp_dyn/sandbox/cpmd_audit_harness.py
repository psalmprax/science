"""CPMD v0.17 — generator-aware discovery audit.

Prereg: docs/CPMD_PREREGISTRATION_v017.md (committed BEFORE this file).
Step 1 oracle bounds, Step 2 harness calibration, Step 3 H-CMD candidate.
"""
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "apedp_dyn/sandbox")

import numpy as np

from cpmd_synth_trading import one_path
from cpmd_discovery_suite import gen_B, rolling_std, ic_null, K, EPS

ALPHA_HARNESS = 0.05 / 2
ALPHA_CMD = 0.025


def oracle_genA(seed):
    ret, _, state = one_path(seed)
    pos = np.zeros(len(ret))
    for t in range(5, len(ret) - 1):
        sig = 1.0 if np.sum(ret[t - 5:t]) > 0 else -1.0
        pos[t] = sig if state[t] == 0 else -sig
    r = ret[6:]
    p = pos[5:-1]
    return p, r


def gen_C(seed, n=3000, sig=0.01):
    rng = np.random.RandomState(int(seed))
    ret = np.zeros(n)
    for t in range(10, n - 1):
        m = float(np.mean(ret[t - 9:t + 1]))
        ret[t + 1] = 0.5 * (1.0 if m > 0 else -1.0) * sig + rng.normal(0, sig)
    return ret


def control_genC(ret):
    n = len(ret)
    sig = rolling_std(ret)
    fc = np.full(n, np.nan)
    for t in range(10, n - 1):
        m = float(np.mean(ret[t - 9:t + 1]))
        fc[t] = (1.0 if m > 0 else -1.0) * sig[t]
    return fc


def hcmd(ret):
    n = len(ret)
    sig = rolling_std(ret)
    z = ret / (sig + EPS)
    fc = np.full(n, np.nan)
    Cprev = 0.0
    for t in range(K, n - 1):
        zw = z[t - K + 1:t + 1]
        C = float(np.sum(zw) / (EPS + np.sum(np.abs(zw))))
        M = 1.0 if float(np.mean(ret[t - 4:t + 1])) > 0 else -1.0
        fc[t] = sig[t] * C * M * (abs(C) - abs(Cprev))
        Cprev = C
    return fc


def eval_fc(fc_fn, genfn, seeds, label):
    ics, ps = [], []
    for sd in seeds:
        ret = genfn(sd)
        fc = fc_fn(ret)
        ic, p, _ = ic_null(fc[:-1], ret[1:len(fc)], sd)
        ics.append(ic)
        ps.append(p)
    ics = np.array(ics)
    print(f"{label}: meanIC={np.mean(ics):+.4f} median-p={np.median(ps):.4f} "
          f"n={len(ics)}")
    return float(np.mean(ics)), float(np.median(ps))


def run():
    print("== Step 1: oracle bounds ==")
    ics = []
    for sd in range(20000, 20030):
        p, r = oracle_genA(sd)
        ok = p != 0
        ics.append(float(np.corrcoef(p[ok], r[ok])[0, 1]))
    ics = np.array(ics)
    print(f"gen-A oracle (true-regime rule): meanIC={np.mean(ics):+.4f}")
    print("gen-B oracle: 0.0000 by construction (symmetric, zero-mean)")
    print("== Step 2: harness calibration ==")
    m1, p1 = eval_fc(control_genC, gen_C, range(70000, 70020),
                     "positive control on gen-C")
    m0, p0 = eval_fc(control_genC, gen_B, range(70500, 70520),
                     "null case (control on gen-B)")
    h_ok = (p1 < ALPHA_HARNESS) and (p0 >= ALPHA_HARNESS)
    print(f"harness: {'FIT' if h_ok else 'UNFIT — stop, no candidate verdict'}")
    if not h_ok:
        return
    print("== Step 3: H-CMD on untouched paths ==")
    ma, pa = eval_fc(hcmd, lambda sd: one_path(sd)[0], range(71000, 71020),
                     "H-CMD gen-A")
    mb, pb = eval_fc(hcmd, gen_B, range(72000, 72020), "H-CMD gen-B")
    if (pa < ALPHA_CMD) or (pb < ALPHA_CMD):
        print("H-CMD: SURVIVES to next evidential stage")
    else:
        print("H-CMD: REJECTED; audit record stands as deliverable")


if __name__ == "__main__":
    run()
