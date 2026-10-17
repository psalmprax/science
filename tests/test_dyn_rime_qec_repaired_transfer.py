"""Frozen repaired QEC equation vs new grids, ranges, distances, noise.

Frozen (locked, never refit):
  y = -0.0935793180819152 * (0.015397698640077853*cumsum(p) - 0.00031838423025785684)
      + 1071.268214236806 * (p*p) + 2.6342682758695446e-05
(state_1.0 init 0 over each evaluation frame == cumsum; recomputed per
frame, never carried across datasets.)

Measured (seeds fixed, deterministic numpy evaluation):
  * d=3, fresh log/uniform/irregular grids, extended range [1e-3,5e-2],
    3x noise: R2 0.990-0.995, always above frozen OLS-quad. The quadratic
    contribution holds; range extension holds.
  * d=4..7: monotone decay 0.953/0.817/0.653/0.497. Frozen OLS-quad sits
    slightly ABOVE the repaired equation at every d!=3: the repair's
    state term overfits d=3 specifics a little. No transferable QEC law;
    graceful, ordered degradation instead.
  * Square coefficient 1071.27 vs truth 1000 (7% high, compensates the
    state term): useful structure, not the mechanism itself.
"""
from __future__ import annotations

import numpy as np

from apedp_dyn.sandbox.qec_threshold import generate_qec_threshold_data

S0, B0 = 0.015397698640077853, -0.00031838423025785684
ALPHA, BETA, GAMMA = -0.0935793180819152, 1071.268214236806, 2.6342682758695446e-05


def _repaired(pp: np.ndarray) -> np.ndarray:
    pp = np.asarray(pp, dtype=float)
    return ALPHA * (S0 * np.cumsum(pp) + B0) + BETA * (pp * pp) + GAMMA


def _metrics(yt: np.ndarray, yp: np.ndarray) -> tuple[float, float]:
    m = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[m], yp[m]
    ss = float(np.sum((yt - yt.mean()) ** 2))
    r2 = 1 - float(np.sum((yt - yp) ** 2)) / ss if ss > 0 else float("nan")
    return r2, float(np.sqrt(np.mean((yt - yp) ** 2)))


def _gen(exp: int | float, grid: np.ndarray, seed: int, nscale: float = 1.0):
    r = np.random.RandomState(seed)
    t = 0.1 * (grid / 0.01) ** exp
    n = len(grid)
    return grid, t + r.normal(0, nscale * 0.02 * np.sqrt(np.mean(t ** 2)), n), t


def _frozen_olsq():
    p0, o0 = generate_qec_threshold_data(n_points=600, seed=43)
    w, *_ = np.linalg.lstsq(np.column_stack([np.ones(200), p0[:200] ** 2]), o0[:200], rcond=None)
    return w


def test_repaired_holds_d3_new_grids_ranges_noise():
    wq = _frozen_olsq()
    log = np.logspace(-3.3, -1.7, 600)
    lin = np.linspace(5.01e-04, 2.00e-02, 600)
    r = np.random.RandomState(2000)
    irr = np.sort(np.concatenate([r.uniform(5e-4, 3e-3, 400), r.uniform(3e-3, 2e-2, 200)]))
    ext = np.logspace(-3.0, -1.3, 600)
    cases = [("log-fresh", log, 301), ("log-fresh", log, 302), ("uniform", lin, 303),
             ("uniform", lin, 304), ("irregular", irr, 305), ("irregular", irr, 306),
             ("extended", ext, 307), ("extended", ext, 308), ("noise3x", log, 313)]
    for name, grid, seed in cases:
        ns = 3.0 if name == "noise3x" else 1.0
        pp, yy, _ = _gen(2, grid, seed, nscale=ns)
        r2, _rmse = _metrics(yy, _repaired(pp))
        yo = wq[0] + wq[1] * pp ** 2
        r2o, _ = _metrics(yy, yo)
        assert r2 > 0.985, (name, r2)
        assert r2 > r2o, (name, r2, r2o)  # repair beats frozen OLS on d=3
    print("\n[repaired-d3] holds across fresh/uniform/irregular/extended/noise3x grids")


def test_repaired_decays_ordered_across_distances():
    wq = _frozen_olsq()
    log = np.logspace(-3.3, -1.7, 600)
    prev, _prev_o = 2.0, 2.0
    for d, exp, seed, lo, hi in ((4, 2.5, 309, 0.9, 1.0), (5, 3, 310, 0.7, 0.9),
                                (6, 3.5, 311, 0.5, 0.75), (7, 4, 312, 0.4, 0.6)):
        pp, yy, _ = _gen(exp, log, seed)
        r2, _ = _metrics(yy, _repaired(pp))
        yo = wq[0] + wq[1] * pp ** 2
        r2o, _ = _metrics(yy, yo)
        assert lo < r2 < hi, (d, r2)
        assert r2 < prev, (d, r2, prev)  # monotone decay with distance
        assert r2o > r2, (d, r2, r2o)  # plain quadratic transfers marginally better
        prev, _prev_o = r2, r2o
    print("\n[repaired-transfer] monotone decay d=4..7; OLS-quad above throughout: no transferable law")


def test_fresh_discovery_per_distance_abstains_where_dev_subrange_is_swamped():
    """Fresh per-distance capability, and its measured cause (not grammar).

    Frozen transfer is pinned above. This measures the OTHER axis: run
    discovery on data generated at each code distance.

    Measured: d=3 emits the lag proxy at dev R2 0.35; d=4/5/6 discovery
    ABSTAINS. The cause is not absent signal — the full-frame lag-proxy
    R2 is 0.92/0.86/0.79/0.74 at d=3/4/5/6 — but the generator's
    ABSOLUTE-scale noise (sigma = 2% of full-frame RMS) against a
    chronological dev split that lands on the lowest-p subrange, where the
    signal RMS/noise ratio is 0.94/0.31/0.10/0.034. Below SNR~1 nothing can
    clear the complexity charge, so abstention is the honest output of the
    split-and-noise-scale interaction, NOT grammar reachability.
    """
    import numpy as _np
    import pandas as _pd

    from apedp_dyn.proposers.rime_proposer import RimeProposer
    C, PTH = 0.1, 0.01

    def _gen(d: int, seed: int = 42, n: int = 600):
        e = (d + 1) / 2
        r = _np.random.RandomState(seed)
        p = _np.logspace(-3.3, -1.7, n)
        t = C * (p / PTH) ** e
        return p, t + r.normal(0, 0.02 * _np.sqrt(_np.mean(t ** 2)), n)

    for d in (3, 4, 5, 6):
        e = (d + 1) / 2
        p, y = _gen(d)
        n = len(p)
        ntr = int(0.4 * n)
        sig = C * (p / PTH) ** e
        sigma = 0.02 * _np.sqrt(_np.mean(sig ** 2))
        snr = _np.sqrt(_np.mean(sig[:ntr] ** 2)) / sigma
        # Full frame: discovery always emits, and the winner is an
        # EXPONENT-2 lagged self-product at every distance — the catalogue
        # cannot express p^2.5/p^3/p^3.5, so fresh recovery is wrong-family
        # at d>=4 (this is what the frozen-transfer decay measures).
        df_full = _pd.DataFrame({"p": p, "target": y})
        props = RimeProposer(feature_cols=["p"], target_col="target",
                             max_depth=2, budget=200).fit_and_propose(df_full)
        assert len(props) > 0, (d, snr)
        assert "*" in (props[0].metadata.get("equation") or ""), (d, props[0].metadata.get("equation"))
        # Low-p-only subrange (a chronological dev split): abstains once the
        # absolute-scale noise swamps the local signal (SNR < ~0.15).
        sub = _pd.DataFrame({"p": p[:ntr], "target": y[:ntr]})
        sub_props = RimeProposer(feature_cols=["p"], target_col="target",
                                 max_depth=2, budget=200).fit_and_propose(sub)
        # Measured boundary: emits at SNR 0.94 (d=3, dev R2 0.35), abstains
        # at SNR 0.31/0.10/0.03 (d=4/5/6). Threshold-free statement below.
        if snr > 0.5:
            assert len(sub_props) > 0, (d, snr)
        else:
            assert len(sub_props) == 0, (d, snr, [q.formula for q in sub_props])
        # the full-frame lag proxy is strong at every distance: signal exists
        lag = _np.full(n, _np.nan)
        lag[10:] = p[:-10]
        m = _np.isfinite(lag)
        M = _np.column_stack([lag[m], _np.ones(int(m.sum()))])
        w, *_ = _np.linalg.lstsq(M, y[m], rcond=None)
        r2_full = 1 - float(_np.sum((y[m] - M @ w) ** 2)) / float(_np.sum((y[m] - y[m].mean()) ** 2))
        assert r2_full > 0.7, (d, r2_full)
        print(f"\n[fresh-per-distance] d={d} e={e:.1f} coef={C / PTH ** e:.0f} "
              f"devSNR={snr:.2f} full-frame={len(props)}p lowp-only={len(sub_props)}p "
              f"full-frame-lag-R2={r2_full:.4f}")
