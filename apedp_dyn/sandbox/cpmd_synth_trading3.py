"""CPMD-ST3 / v0.16 — mechanism discrimination (Tests A-C).

Prereg: docs/CPMD_PREREGISTRATION_ST3.md (committed BEFORE this file).
Primary endpoint: Test A p-value. No estimator changes.
"""
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "apedp_dyn/sandbox")

import numpy as np
from cpmd_synth_trading import decay_curve, ols_slope_width, one_path

COST = 0.0002
HOLD = 5


def frozen_positions(seed, hold=0):
    ret, _, _ = one_path(seed)
    W, burn, t0 = 200, 200, 400
    thr = [ols_slope_width(decay_curve(ret[t - 200:t]))[0]
           for t in range(W, W + burn)]
    med = float(np.median(thr))
    pos = np.zeros(len(ret))
    last_flip = -10 ** 9
    for t in range(t0, len(ret) - 1):
        s, w = ols_slope_width(decay_curve(ret[t - 200:t]))
        m5 = float(np.sum(ret[t - 5:t]))
        sig = 1.0 if m5 > 0 else -1.0
        want = (sig if s < med else -sig) if w <= 1.0 else 0.0
        if hold and want != pos[t - 1] and (t - last_flip) < hold:
            want = pos[t - 1]
        if want != pos[t - 1]:
            last_flip = t
        pos[t] = want
    return ret, pos, t0


def path_stats(ret, pos, t0, fee):
    p = pos[t0:-1]
    r = ret[t0 + 1:]
    gross = p * r
    chg = np.abs(np.diff(np.concatenate([[0.0], p])))
    turn = float(np.sum(chg))  # exposure units; reversal = 2
    netrets = gross - chg * fee
    nav = np.cumsum(netrets)
    dd = float(np.max(np.maximum.accumulate(nav) - nav))
    sharpe = float(np.mean(netrets) / (np.std(netrets) + 1e-12) * np.sqrt(252))
    return {"gross": float(np.sum(gross)), "net": float(np.sum(netrets)),
            "sharpe": sharpe, "dd": dd, "turn": turn}


def run():
    seeds = list(range(20000, 20030))
    data = {sd: frozen_positions(sd) for sd in seeds}
    # ---- Test A: circular-shift null on aggregate gross ----
    obs = sum(path_stats(r, p, t0, 0.0)["gross"] for r, p, t0 in data.values())
    rng = np.random.RandomState(1600)
    null = []
    items = list(data.values())
    for _ in range(1000):
        tot = 0.0
        for r, p, t0 in items:
            pp = p[t0:-1]
            sh = rng.randint(0, len(pp))
            ps = np.concatenate([pp[sh:], pp[:sh]])
            tot += float(np.sum(ps * r[t0 + 1:]))
        null.append(tot)
    null = np.array(null)
    pval = float(np.mean(null >= obs))
    print(f"Test A: observed gross={obs:+.5f} null mean={null.mean():+.5f} "
          f"sd={null.std():.5f} p={pval:.3f} "
          f"[{'EDGE' if pval < 0.05 else 'NO INFORMATION DEMONSTRATED'}]")
    # ---- Test B: fee ladder ----
    print("Test B: fee -> netSharpe / netRet / DD / turnover")
    for fee in (0.0, 0.0001, COST, 0.0005):
        sh, nt, dd, tn = [], [], [], []
        for r, p, t0 in items:
            d = path_stats(r, p, t0, fee)
            sh.append(d["sharpe"])
            nt.append(d["net"])
            dd.append(d["dd"])
            tn.append(d["turn"])
        print(f"  fee={fee*1e4:.0f}bp: sharpe={np.mean(sh):+.3f} net={np.sum(nt):+.4f} "
              f"dd={np.mean(dd):.4f} turn/path={np.mean(tn):.0f}")
    tot_turn = sum(path_stats(r, p, t0, 0.0)["turn"] for r, p, t0 in items)
    be = obs / tot_turn if tot_turn > 0 else float("nan")
    print(f"  break-even cost={be*1e4:.2f}bp (actual 2.00bp; "
          f"obs gross {'+' if obs > 0 else ''}{obs:+.4f} vs cost drag "
          f"{tot_turn * COST:+.4f})")
    # ---- Test C: hold-5, exploratory (ST2 paths) + confirmatory (fresh) ----
    for tag, sds in (("exploratory ST2 paths", seeds),
                     ("confirmatory fresh 40000+", list(range(40000, 40030)))):
        for hold, nm in ((0, "orig"), (HOLD, "hold5")):
            sh, nt, tn = [], [], []
            for sd in sds:
                r, p, t0 = frozen_positions(sd, hold=hold)
                d = path_stats(r, p, t0, COST)
                sh.append(d["sharpe"])
                nt.append(d["net"])
                tn.append(d["turn"])
            print(f"Test C {tag} {nm}: sharpe={np.mean(sh):+.3f} "
                  f"net={np.sum(nt):+.4f} turn/path={np.mean(tn):.0f}")


if __name__ == "__main__":
    run()
