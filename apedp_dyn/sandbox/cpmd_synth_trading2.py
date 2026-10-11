"""CPMD-ST2 — rolling-window sensitivity with phase-split validation.

Prereg: docs/CPMD_PREREGISTRATION_ST2.md (committed BEFORE this file).
Phase 1: seeds 20000+ (ST1 paths). Phase 2: seeds 30000+ (fresh).
"""
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "apedp_dyn/sandbox")

import numpy as np

from cpmd_synth_trading import one_path, ols_slope_width, decay_curve, COST

WINDOWS = (25, 50, 100, 200)


def run_path(seed, windows):
    ret, px, state = one_path(seed)
    W = max(windows)
    burn = 200
    med = {}
    for w in windows:
        thr = []
        for t in range(W, W + burn):
            c = decay_curve(ret[t - w:t])
            s, _ = ols_slope_width(c)
            thr.append(s)
        med[w] = float(np.median(thr))
    t0 = W + burn
    pos = {f"ols{w}": np.zeros(len(ret)) for w in windows}
    pos.update({k: np.zeros(len(ret)) for k in ("mom5", "mom20", "bh", "rev", "rnd")})
    pos["bh"][:] = 1.0
    rr = np.random.RandomState(seed + 1)
    cls = {w: np.zeros(len(ret), dtype=int) for w in windows}
    for t in range(t0, len(ret) - 1):
        m5 = float(np.sum(ret[t - 5:t]))
        m20 = float(np.sum(ret[t - 20:t]))
        sig5 = 1.0 if m5 > 0 else -1.0
        pos["mom5"][t] = sig5
        pos["mom20"][t] = 1.0 if m20 > 0 else -1.0
        pos["rev"][t] = -sig5
        pos["rnd"][t] = 1.0 if rr.rand() < 0.5 else -1.0
        for w in windows:
            c = decay_curve(ret[t - w:t])
            s, wd = ols_slope_width(c)
            est_persist = s < med[w]
            cls[w][t] = 0 if est_persist else 1
            if wd <= 1.0:
                pos[f"ols{w}"][t] = sig5 if est_persist else -sig5
    out = {}
    for k, p in pos.items():
        p = p[t0:-1]
        r = ret[t0 + 1:]
        chg = np.abs(np.diff(np.concatenate([[0.0], p])))
        netrets = p * r - chg * COST
        nav = np.cumsum(netrets)
        dd = float(np.max(np.maximum.accumulate(nav) - nav))
        sharpe = float(np.mean(netrets) / (np.std(netrets) + 1e-12) * np.sqrt(252))
        out[k] = {"sharpe": sharpe, "net": float(np.sum(netrets)), "dd": dd,
                  "turn": float(np.sum(chg))}
    # regime-detection delay per window
    delays = {}
    for w in windows:
        d = []
        t = t0 + 1
        while t < len(ret) - 1:
            if state[t] != state[t - 1]:
                u = t
                while u < len(ret) - 1 and cls[w][u] != state[t]:
                    u += 1
                d.append(u - t if u < len(ret) - 1 else len(ret) - 1 - t)
                t = u + 1
            else:
                t += 1
        delays[w] = float(np.median(d)) if d else float("nan")
    return out, delays


def phase(seeds, rules):
    acc = {k: {"sharpe": [], "net": [], "dd": [], "turn": []} for k in rules}
    dly = {w: [] for w in WINDOWS}
    for sd in seeds:
        o, dl = run_path(sd, WINDOWS)
        for k in rules:
            for m in ("sharpe", "net", "dd", "turn"):
                acc[k][m].append(o[k][m])
        for w in WINDOWS:
            dly[w].append(dl[w])
    return acc, dly


def show(acc, dly, tag):
    print(f"--- {tag} ---")
    for k, d in acc.items():
        print(f"{k}: sharpe={np.mean(d['sharpe']):+.3f} net={np.mean(d['net']):+.4f} "
              f"dd={np.mean(d['dd']):.4f} turn={np.mean(d['turn']):.0f}")
    for w in WINDOWS:
        print(f"delay ols{w}: median {np.nanmedian(dly[w]):.0f} steps")


def run():
    from scipy.stats import wilcoxon
    rules1 = [f"ols{w}" for w in WINDOWS] + ["mom5", "mom20", "bh", "rev", "rnd"]
    acc1, dly1 = phase(range(20000, 20030), rules1)
    show(acc1, dly1, "Phase 1 development (ST1 paths)")
    win = max(WINDOWS, key=lambda w: float(np.mean(acc1[f"ols{w}"]["sharpe"])))
    print(f"Phase-1 Winner: ols{win} "
          f"(sharpe {np.mean(acc1[f'ols{win}']['sharpe']):+.3f}) — FROZEN")
    rules2 = [f"ols{win}", "mom5", "bh"]
    acc2, dly2 = phase(range(30000, 30030), rules2)
    show(acc2, dly2, "Phase 2 independent")
    d = np.array(acc2[f"ols{win}"]["sharpe"]) - np.array(acc2["mom5"]["sharpe"])
    try:
        _, p = wilcoxon(d)
    except Exception:
        p = 1.0
    g1 = float(np.mean(acc2[f"ols{win}"]["sharpe"])) > 0
    g3 = (float(np.mean(acc2[f"ols{win}"]["dd"]))
          < 2 * float(np.mean(acc2["bh"]["dd"])))
    print(f"GATE1 sharpe>0: {g1} | GATE2 beats mom5 p={p:.4f} (<0.05: {p < 0.05}) "
          f"| GATE3 dd<2xBH: {g3}")


if __name__ == "__main__":
    run()
