"""CPMD-HLIK — lagged impulse kernel hypothesis test.

Prereg: docs/CPMD_PREREGISTRATION_HLIK.md (frozen W/L/lambda, gated stages).
Stage 1 discovery only; confirmation/trading gated behind it.
"""
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "apedp_dyn/sandbox")

import numpy as np

from cpmd_hdar import gen_panel, trailing_std, panel_ic_null

N = 10
W, L, LAM = 200, 6, 1.0
EPS = 1e-8


def gen_F(seed):
    rng = np.random.RandomState(int(seed))
    n = 3000
    beta = rng.uniform(0.5, 1.5, N)
    slow = np.zeros(N, dtype=bool)
    slow[rng.choice(N, 5, replace=False)] = True
    w = np.array([0.3, 0.15, 0.05])
    F = rng.normal(0, 0.01, n)
    eta = rng.normal(0, 0.005, (n, N))
    R = np.zeros((n, N))
    for t in range(3, n):
        R[t] = beta * F[t] + eta[t]
        R[t, slow] += beta[slow] * sum(w[l] * F[t - 1 - l] for l in range(3))
    return R


def hlik_forecast(R):
    n = len(R)
    sig = trailing_std(R)
    Z = R / (sig + EPS)
    fc = np.full((n, N), np.nan)
    for i in range(N):
        others = [j for j in range(N) if j != i]
        u = np.median(Z[:, others], axis=1)
        for t in range(W, n - 1):
            # design: rows s in [t-W, t-1], target z[i,s+1], cols u[s-l]
            S = np.arange(t - W, t)
            Y = Z[S + 1, i]
            X = np.column_stack([u[S - l] for l in range(L)])
            XtX = X.T @ X + LAM * np.eye(L)
            try:
                bh = np.linalg.solve(XtX, X.T @ Y)
            except np.linalg.LinAlgError:
                continue
            lags = np.array([u[t - l] for l in range(L)])
            fc[t, i] = sig[t, i] * float(bh @ lags)
    return fc


def stage1(tag, seeds, genfn):
    print(f"--- {tag} ---")
    ps = []
    for sd in seeds:
        R = genfn(sd)
        F = hlik_forecast(R)
        _, p = panel_ic_null(F, R, sd)
        ps.append(p)
    print(f"p-values: {[f'{p:.3f}' for p in ps]}")
    return ps


def run():
    from cpmd_hdar import gen_panel as gp
    pD = stage1("gen-D discovery (140000+)",
                range(140000, 140010), lambda sd: gp(sd, True))
    pE = stage1("gen-E discovery (150000+)",
                range(150000, 150010), lambda sd: gp(sd, False))
    d_pass = all(p < 0.05 for p in pD)
    e_abs = all(p >= 0.05 for p in pE)
    print(f"gen-D pass: {d_pass} | gen-E abstains: {e_abs}")
    if not (d_pass and e_abs):
        print("H-LIK: REJECTED at discovery (confirmation + trading closed)")
        return
    print("== confirmation ==")
    qD = stage1("gen-D confirm (160000+)", range(160000, 160010), lambda sd: gp(sd, True))
    qE = stage1("gen-E confirm (170000+)", range(170000, 170010), lambda sd: gp(sd, False))
    qF = stage1("gen-F confirm (180000+)", range(180000, 180010), gen_F)
    f_pass = sum(p < 0.05 for p in qF) >= 8
    if all(p < 0.05 for p in qD) and all(p >= 0.05 for p in qE) and f_pass:
        print("H-LIK: PROCEEDS to Stage 3 trading (new prereg required)")
    else:
        print("H-LIK: REJECTED at confirmation")


if __name__ == "__main__":
    run()
