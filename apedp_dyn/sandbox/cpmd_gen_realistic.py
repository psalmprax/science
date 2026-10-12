"""gen-R: real-calibrated synthetic market generator.

Calibration targets measured from data/market panels (see
docs/CPMD_GENR_CALIBRATION.md):
  EQUITY: vol .0144, kurt 8.3, sqAC1 .066, xcorr med .30, PC1 40%,
          vol-thirds 1.00/1.30/1.02
  CRYPTO: vol .0387, kurt 18.6, sqAC1 .146, xcorr med .61, PC1 63%,
          vol-thirds 2.00/1.00/0.81 (relative)
Model: 1-factor + GARCH(1,1)-t vols + segment vol multipliers + t shocks.
This is INFRASTRUCTURE (calibration iteration legitimate); no candidate
evaluation happens here.
"""
import numpy as np

PARAMS = {
    # (a,b,c): market / block / idio loadings, a^2+b^2+c^2=1
    "equity": {"n": 14, "vol": 0.0200, "df": 4.3, "alpha": 0.08,
               "abc": (0.50, 0.60, 0.62), "seg": (1.00, 1.30, 1.02), "beta_g": 0.88, "jump": 0.008},
    "crypto": {"n": 15, "vol": 0.0570, "df": 4.2, "alpha": 0.16,
               "abc": (0.75, 0.40, 0.53), "seg": (2.00, 1.00, 0.81), "beta_g": 0.80, "jump": 0.012},
}


def _garch_t(rng, n, alpha, beta_g, df, seed_var):
    vol = np.zeros(n)
    vol[0] = seed_var
    shocks = rng.standard_t(df, n) / np.sqrt(df / (df - 2))
    r = np.zeros(n)
    om = seed_var * max(1 - alpha - beta_g, 0.02)
    for t in range(1, n):
        vol[t] = om + alpha * r[t - 1] ** 2 + beta_g * vol[t - 1]
        r[t] = np.sqrt(max(vol[t], 1e-12)) * shocks[t]
    return r


def gen_R(seed, kind="equity", n=900):
    p = PARAMS[kind]
    rng = np.random.RandomState(int(seed))
    N = p["n"]
    F = _garch_t(rng, n, p["alpha"], p["beta_g"], p["df"], p["vol"] ** 2)
    jp = rng.random(n) < p.get("jump", 0.0)
    F = F + jp * rng.choice([-1.0, 1.0], n) * 5 * p["vol"]
    R = np.zeros((n, N))
    a, b, c = p["abc"]
    nblocks = 3
    Bf = np.array([_garch_t(rng, n, p["alpha"], p["beta_g"], p["df"], p["vol"] ** 2)
                   for _ in range(nblocks)])
    for i in range(N):
        e = _garch_t(rng, n, p["alpha"], p["beta_g"], p["df"], p["vol"] ** 2)
        R[:, i] = a * F + b * Bf[i % nblocks] + c * e
    # NOTE: F,e already scaled by vol inside _garch_t via seed_var; rescale:
    # rebuild cleanly below (kept simple: standardize then scale)
    R = (R - R.mean(0)) / R.std(0) * p["vol"]
    # segment vol multipliers (non-stationarity: regime breaks)
    t = n // 3
    R[:t] *= p["seg"][0]
    R[t:2 * t] *= p["seg"][1]
    R[2 * t:] *= p["seg"][2]
    # renormalize to target vol
    R = (R - R.mean(0)) / R.std(0) * p["vol"]
    return R


def panel_stats(R):
    n, N = R.shape
    kurt = float(np.mean(((R - R.mean(0)) / R.std(0)) ** 4, axis=0).mean())
    sq = R ** 2
    ac1 = float(np.mean([np.corrcoef(sq[:-1, i], sq[1:, i])[0, 1] for i in range(N)]))
    C = np.corrcoef(R.T)
    iu = C[np.triu_indices(N, 1)]
    ev = np.linalg.eigvalsh(C)[::-1]
    v = np.std(R, axis=1)
    t = len(v) // 3
    return {"vol": float(np.mean(np.abs(R))), "kurt": kurt, "sqAC1": ac1,
            "xmed": float(np.median(iu)), "x90": float(np.percentile(iu, 90)),
            "pc1": float(ev[0] / N),
            "seg": (float(v[:t].mean()), float(v[t:2 * t].mean()), float(v[2 * t:].mean()))}


if __name__ == "__main__":
    for kind in ("equity", "crypto"):
        s = panel_stats(gen_R(7, kind))
        print(f"gen-R/{kind}: vol={s['vol']:.4f} kurt={s['kurt']:.1f} "
              f"sqAC1={s['sqAC1']:.3f} xmed={s['xmed']:+.2f} x90={s['x90']:+.2f} "
              f"pc1={s['pc1']:.2f} seg={s['seg'][0]:.4f}/{s['seg'][1]:.4f}/{s['seg'][2]:.4f}")
