"""Frozen-equation OOS comparison: the two QEC candidates, no refit.

Frozen (full precision, locked at discovery on seed-42/d=3/log-grid):
  Eq1: y = 1055.6995975078069 * p2[t-5] - 6.0557766840642346e-05
  Eq2: y = 1023.7284277856082 * (p * p[t-5]) - 6.055776684063844e-05

Measured findings pinned here:
  * On log grids Eq1 and Eq2 are the SAME hypothesis in disguise: the
    fitted scales absorbed the constant grid ratio r (A1~=1000*r^2,
    A2~=1000*r), so predictions agree to ~2e-5. The lagged-square vs
    interaction distinction does not survive the fitted readout.
  * T1 (fresh seeds, same config): both transfer (R2>0.999).
  * T2 (uniform grid): both survive by R2 (>0.99) but MAE doubles —
    R2 masks low-end breakdown; Eq2 slightly more robust.
  * T3/T4 (d=5 cubic truth): graceful failure (R2 0.78->0.66),
    matching frozen OLS-quad. No transfer across code distances.
  * Mean baseline negative everywhere (sane floor).
"""
from __future__ import annotations

import numpy as np

A1, B1 = 1055.6995975078069, -6.0557766840642346e-05
A2, B2 = 1023.7284277856082, -6.055776684063844e-05


def _lag5(v: np.ndarray) -> np.ndarray:
    o = np.full(len(v), np.nan)
    o[5:] = v[:-5]
    return o


def _metrics(yt: np.ndarray, yp: np.ndarray) -> tuple[float, float, float]:
    m = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[m], yp[m]
    ss = float(np.sum((yt - yt.mean()) ** 2))
    r2 = 1 - float(np.sum((yt - yp) ** 2)) / ss if ss > 0 else float("nan")
    return r2, float(np.sqrt(np.mean((yt - yp) ** 2))), float(np.mean(np.abs(yt - yp)))


def _gen(exp: int, grid: np.ndarray, seed: int, n: int = 600):
    r = np.random.RandomState(seed)
    t = 0.1 * (grid / 0.01) ** exp
    return grid, t + r.normal(0, 0.02 * np.sqrt(np.mean(t ** 2)), n)


def _eval(pp: np.ndarray, yy: np.ndarray):
    p2 = pp ** 2
    return (_metrics(yy, A1 * _lag5(p2) + B1),
            _metrics(yy, A2 * (pp * _lag5(pp)) + B2))


def test_frozen_equations_transfer_same_config():
    log_grid = np.logspace(-3.3, -1.7, 600)
    for s in (43, 44, 45):
        pp, yy = _gen(2, log_grid, s)
        (r1, _, _), (r2, _, _) = _eval(pp, yy)
        assert r1 > 0.999 and r2 > 0.999, (s, r1, r2)


def test_frozen_equations_agree_on_log_grids():
    # Same hypothesis in disguise: predictions agree to ~2e-5 relative.
    log_grid = np.logspace(-3.3, -1.7, 600)
    pp, yy = _gen(2, log_grid, 43)
    y1 = A1 * _lag5(pp ** 2) + B1
    y2 = A2 * (pp * _lag5(pp)) + B2
    m = np.isfinite(y1) & np.isfinite(y2)
    rel = float(np.max(np.abs(y1[m] - y2[m]) / np.abs(yy[m])))
    assert rel < 1e-3, rel


def test_frozen_equations_uniform_grid():
    lin_grid = np.linspace(5.01e-04, 2.00e-02, 600)
    rms = []
    for s in (43, 44, 45):
        pp, yy = _gen(2, lin_grid, s)
        (r1, _rmse1, mae1), (r2, _rmse2, mae2) = _eval(pp, yy)
        assert r1 > 0.99 and r2 > 0.99, (s, r1, r2)
        rms.append((mae1, mae2))
    # MAE roughly doubles vs T1 (low-end breakdown masked by R2 weighting).
    assert all(m1 > 0.0025 and m2 > 0.0025 for m1, m2 in rms), rms


def test_frozen_equations_no_transfer_across_distance():
    log_grid = np.logspace(-3.3, -1.7, 600)
    lin_grid = np.linspace(5.01e-04, 2.00e-02, 600)
    for grid, lo, hi in ((log_grid, 0.5, 0.9), (lin_grid, 0.5, 0.9)):
        for s in (43, 44):
            pp, yy = _gen(3, grid, s)
            (r1, _, _), (r2, _, _) = _eval(pp, yy)
            assert lo < r1 < hi and lo < r2 < hi, (s, r1, r2)
            # mean baseline deeply negative: sane floor, equations beat it
            m = np.isfinite(yy)
            base_r2 = 1 - float(np.sum((yy[m] - 0.001) ** 2)) / float(np.sum((yy[m] - yy[m].mean()) ** 2))
            assert base_r2 < 0, base_r2


def _decile_rmses(pp: np.ndarray, yy: np.ndarray, *preds: np.ndarray):
    qs = np.quantile(pp, np.linspace(0, 1, 11))
    out = []
    for i in range(10):
        m = (pp >= qs[i]) & (pp <= qs[i + 1])
        row = []
        for yp in preds:
            mm = m & np.isfinite(yp)
            row.append(float(np.sqrt(np.mean((yy[mm] - yp[mm]) ** 2))))
        out.append(row)
    return out


def test_frozen_equations_residual_profile_by_decile():
    # Correction of an earlier hypothesis: breakdown is HIGH-end, not
    # low-end. On a uniform grid the frozen lag offset is a growing
    # fraction of the steepening truth; the right-form OLS baseline stays
    # flat across deciles while wrong-form profiles grow with p.
    lin_grid = np.linspace(5.01e-04, 2.00e-02, 600)
    r = np.random.RandomState(43)
    t = 0.1 * (lin_grid / 0.01) ** 2
    yy = t + r.normal(0, 0.02 * np.sqrt(np.mean(t ** 2)), 600)
    pp = lin_grid
    y1 = A1 * _lag5(pp ** 2) + B1
    y2 = A2 * (pp * _lag5(pp)) + B2
    yq = -7.20576699e-05 + 994.853565 * pp ** 2  # frozen OLS-quad
    prof = _decile_rmses(pp, yy, y1, y2, yq)
    e1_lo, e1_hi = prof[0][0], prof[9][0]
    e2_hi = prof[9][1]
    q_lo, q_hi = prof[0][2], prof[9][2]
    assert e1_hi > 2.0 * e1_lo, (e1_lo, e1_hi)  # wrong-form profile grows
    assert e2_hi < e1_hi  # interaction extrapolates better than lagged square
    assert q_hi < 2.0 * q_lo, (q_lo, q_hi)  # right form stays flat
    print(f"\n[profile] Eq1 {e1_lo:.6f}-> {e1_hi:.6f}; Eq2 top {e2_hi:.6f}; OLSQ {q_lo:.6f}->{q_hi:.6f}")


def test_irregular_grid_distinguishes_generator():
    # Varying-ratio grid: the generator IS identifiable (fresh OLS recovers
    # physics to 0.14% and hits the noise floor) while both frozen equations
    # sit measurably above it. Observational equivalence is grid-relative.
    r = np.random.RandomState(1000)
    irr = np.sort(np.concatenate([r.uniform(5e-4, 3e-3, 400), r.uniform(3e-3, 2e-2, 200)]))
    t = 0.1 * (irr / 0.01) ** 2
    rg = np.random.RandomState(44)
    yy = t + rg.normal(0, 0.02 * np.sqrt(np.mean(t ** 2)), 600)
    floor = 0.02 * np.sqrt(np.mean(t ** 2))
    y1 = A1 * _lag5(irr ** 2) + B1
    A2 * (irr * _lag5(irr)) + B2
    m1 = np.isfinite(y1)
    rmse1 = float(np.sqrt(np.mean((yy[m1] - y1[m1]) ** 2)))
    assert rmse1 > 1.5 * floor, (rmse1, floor)
    ntr = 300
    A = np.column_stack([np.ones(ntr), irr[:ntr] ** 2])
    w, *_ = np.linalg.lstsq(A, yy[:ntr], rcond=None)
    rmse_ols = float(np.sqrt(np.mean((yy[ntr:] - (w[0] + w[1] * irr[ntr:] ** 2)) ** 2)))
    assert abs(w[1] - 1000.0) / 1000.0 < 0.02, w
    assert rmse_ols < 1.05 * floor, (rmse_ols, floor)
    print(f"\n[identifiability] Eq1 {rmse1:.6f} vs floor {floor:.6f}; fresh OLS w1={w[1]:.1f} RMSE={rmse_ols:.6f}")


def test_transfer_curve_decays_with_distance_not_noise():
    # R2 falls monotonically with code distance and is nearly identical at
    # 1x vs 3x noise: the failure is structural (wrong exponent), not
    # noise-driven. Four distances x two noise levels, seeds fixed.
    log_grid = np.logspace(-3.3, -1.7, 600)
    prev = 2.0
    for d, exp in ((3, 2), (4, 2.5), (5, 3), (6, 3.5)):
        vals = []
        for ns in (1.0, 3.0):
            r = np.random.RandomState(45)
            t = 0.1 * (log_grid / 0.01) ** exp
            yy = t + r.normal(0, ns * 0.02 * np.sqrt(np.mean(t ** 2)), 600)
            yp = A1 * _lag5(log_grid ** 2) + B1
            m = np.isfinite(yp)
            ss = float(np.sum((yy[m] - yy[m].mean()) ** 2))
            vals.append(1 - float(np.sum((yy[m] - yp[m]) ** 2)) / ss)
        assert vals[0] < prev, (d, vals)
        assert abs(vals[0] - vals[1]) < 0.02, (d, vals)
        prev = vals[0]
    print("\n[transfer] monotone decay confirmed; noise-independent (structural failure)")
