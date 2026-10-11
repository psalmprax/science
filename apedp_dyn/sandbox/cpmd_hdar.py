"""CPMD-HDAR — delayed asset response hypothesis test.

Prereg: docs/CPMD_PREREGISTRATION_HDAR.md (frozen params, gated stages).
Multi-asset panels only. Prediction first; trading gated behind it.
"""
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "apedp_dyn/sandbox")

import numpy as np

N, L, LAM, C0, GAM = 10, 10, 0.8, 1.0, 1.0
BW, SW, EPS = 100, 20, 1e-8
LAMW = np.array([LAM ** l for l in range(L)])
LAMW /= LAMW.sum()


def gen_panel(seed, delay=True):
    rng = np.random.RandomState(int(seed))
    n = 3000
    beta = rng.uniform(0.5, 1.5, N)
    delta = np.array([0.0] * 5 + [0.5] * 5)
    rng.shuffle(delta)
    if not delay:
        delta = np.zeros(N)
    F = rng.normal(0, 0.01, n)
    eta = rng.normal(0, 0.005, (n, N))
    R = np.zeros((n, N))
    for t in range(1, n):
        R[t] = beta * F[t] + delta * beta * F[t - 1] + eta[t]
    return R, beta, delta


def trailing_std(R, w=SW):
    n = len(R)
    return np.array([[float(np.std(R[max(0, t - w + 1):t + 1, i]) + EPS)
                      for i in range(N)] for t in range(n)])


def hdar_forecast(R):
    n = len(R)
    sig = trailing_std(R)
    Z = R / (sig + EPS)
    m = np.median(Z, axis=1)
    G = np.abs(np.mean(np.tanh(Z / C0), axis=1))
    fc = np.full((n, N), np.nan)
    Ball = np.full((n, N), np.nan)
    for t in range(BW, n - 1):
        Zw, mw = Z[t - BW + 1:t + 1], m[t - BW + 1:t + 1]
        denom = float(np.sum((mw - mw.mean()) ** 2) + EPS)
        Ball[t] = np.array([float(np.sum((mw - mw.mean()) * (Zw[:, i] - Zw[:, i].mean())) / denom)
                            for i in range(N)])
    for t in range(BW, n - 1):
        D = np.zeros(N)
        for i in range(N):
            ehist = np.array([Ball[t - l, i] * m[t - l] - Z[t - l, i]
                              if t - l >= BW else 0.0 for l in range(L)])
            D[i] = float(np.sum(LAMW * ehist))
        fc[t] = sig[t] * G[t] * np.tanh(GAM * D)
    return fc


def panel_ic_null(F, R, seed, nshift=200):
    f, r = F[:-1].ravel(), R[1:].ravel()
    ok = np.isfinite(f) & np.isfinite(r)
    f, r = f[ok], r[ok]
    if len(f) < 500:
        return float("nan"), 1.0
    obs = float(np.corrcoef(f, r)[0, 1])
    rng = np.random.RandomState(int(seed))
    cnt = sum(float(np.corrcoef(np.concatenate([f[s:], f[:s]]), r)[0, 1]) >= obs
              for s in rng.randint(0, len(f), size=nshift))
    return obs, float(cnt / nshift)


def stage12(tag, seeds, delay):
    print(f"--- {tag} ---")
    p1, p2 = [], []
    for sd in seeds:
        R, _, _ = gen_panel(sd, delay)
        F = hdar_forecast(R)
        _, p = panel_ic_null(F, R, sd)
        p1.append(p)
        # Stage 2: residualize on [own past-5, beta*m]
        n = len(R)
        sig = trailing_std(R)
        Z = R / (sig + EPS)
        m = np.median(Z, axis=1)
        ctrl = np.zeros((n, N, 2))
        for t in range(BW, n - 1):
            Zw, mw = Z[t - BW + 1:t + 1], m[t - BW + 1:t + 1]
            b = np.array([float(np.sum((mw - mw.mean()) * (Zw[:, i] - Zw[:, i].mean()))
                                / (np.sum((mw - mw.mean()) ** 2) + EPS))
                          for i in range(N)])
            ctrl[t, :, 0] = np.sum(R[t - 5:t], axis=0)
            ctrl[t, :, 1] = b * m[t]
        f, r = F[:-1].reshape(-1), R[1:].reshape(-1)
        c = ctrl[:-1].reshape(-1, 2)
        ok = np.isfinite(f) & np.isfinite(r) & np.all(np.isfinite(c), axis=1)
        f, r, c = f[ok], r[ok], c[ok]
        X = np.column_stack([c, np.ones(len(c))])
        resid, *_ = np.linalg.lstsq(X, f, rcond=None)
        fr = f - X @ resid
        obs = float(np.corrcoef(fr, r)[0, 1])
        rng = np.random.RandomState(sd + 7)
        cnt = sum(float(np.corrcoef(np.concatenate([fr[s:], fr[:s]]), r)[0, 1]) >= obs
                  for s in rng.randint(0, len(fr), size=nshift))
        p2.append(float(cnt / nshift))
    print(f"Stage1 p-values: {[f'{p:.3f}' for p in p1]}")
    print(f"Stage2 residual p-values: {[f'{p:.3f}' for p in p2]}")
    return p1, p2


def run():
    p1d, p2d = stage12("gen-D discovery (100000+)", range(100000, 100010), True)
    p1e, p2e = stage12("gen-E discovery (110000+)", range(110000, 110010), False)
    d_pass = all(p < 0.05 for p in p1d) and all(p < 0.05 for p in p2d)
    e_abs = all(p >= 0.05 for p in p1e)
    print(f"gen-D stages pass: {d_pass} | gen-E abstains: {e_abs}")
    if not (d_pass and e_abs):
        print("H-DAR: REJECTED at discovery (no confirmation stage)")
        return
    print("== confirmation (fresh 120000+/130000+) ==")
    q1d, q2d = stage12("gen-D confirm", range(120000, 120010), True)
    q1e, _ = stage12("gen-E confirm", range(130000, 130010), False)
    if all(p < 0.05 for p in q1d + q2d) and all(p >= 0.05 for p in q1e):
        print("H-DAR: PROCEEDS to Stage 3 trading (new prereg required)")
    else:
        print("H-DAR: REJECTED at confirmation")


if __name__ == "__main__":
    run()
