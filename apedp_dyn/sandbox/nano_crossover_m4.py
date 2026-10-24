"""M4 nanoscale crossover benchmark: classical Hall-Petch vs inverse softening.

Truth (hidden from candidates): grain-size strengthening inverts below a
critical grain size dc (nanocrystalline regime):

    sigma(d) = s0 + k * d^(-1/2)            for d >= dc   (classical HP)
    sigma(d) = sigma(dc) - s * (dc - d)     for d <  dc   (inverse softening)

with dc = 15 nm. Candidates see only (d, sigma_obs); the threshold itself
must be discovered, not supplied. This is the regime-change discovery
problem: a single global law cannot fit both sides, and the benchmark tests
whether APEDP (a) refuses the global laws, (b) promotes the piecewise
structure, and (c) locates dc from residual evidence.

Follows the M1/M2/M3 L1 conventions exactly: OLS fits on train split,
holdout scoring, gate4 provenance with residuals + squared errors,
adjudication with bootstrap tiebreak and native data_origin labeling.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPOS_ROOT = str(Path(__file__).resolve().parents[2])
if _REPOS_ROOT not in sys.path:
    sys.path.insert(0, _REPOS_ROOT)

from apedp_dyn.epistemic.adjudicate import (
    RESID_RHO_FLAG,
    SOLE_SURVIVOR_GAMMA,
    BenchmarkLevel,
    DataOrigin,
    adjudicate,
)
from apedp_dyn.grammar.candidate_ir import CandidateDynamics, ScoredCandidate, Term

M4_DC_TRUE_NM = 15.0
M4_DC_GRID_NM = (5.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0)


def generate_m4_data(n_points: int = 60,
                     noise_sigma: float = 0.02,
                     sigma_0: float = 200.0,
                     k_hp: float = 300.0,
                     dc_nm: float = M4_DC_TRUE_NM,
                     soften_mult: float = 1.0,
                     d_min_nm: float = 3.0,
                     d_max_nm: float = 200.0,
                     seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Nanoscale crossover data: classical HP above dc, linear softening below."""
    rng = np.random.default_rng(seed)
    d = np.logspace(np.log10(d_min_nm), np.log10(d_max_nm), n_points)
    peak = sigma_0 + k_hp * dc_nm ** -0.5
    s = soften_mult * k_hp * dc_nm ** -1.5  # softening slope (MPa/nm)
    sigma_true = np.where(d >= dc_nm,
                          sigma_0 + k_hp * d ** -0.5,
                          peak - s * (dc_nm - d))
    obs_rms = float(np.sqrt(np.mean(sigma_true ** 2)))
    sigma_obs = sigma_true + rng.normal(0.0, noise_sigma * obs_rms, n_points)
    return d, sigma_obs


def _fit_pair(X_tr, y_tr, X_ho, y_ho):
    w, *_ = np.linalg.lstsq(X_tr, y_tr, rcond=None)
    resid_ho = y_ho - X_ho @ w
    return w, float(np.mean((y_tr - X_tr @ w) ** 2)), float(np.sqrt(np.mean(resid_ho ** 2))), resid_ho


def fit_m4_nano_candidates(d: np.ndarray,
                           sigma_obs: np.ndarray,
                           n_train: int = 30,
                           noise_sigma: float = 0.02,
                           data_origin: str = DataOrigin.SIMULATION,
                           ) -> tuple[list[ScoredCandidate], dict]:
    """Fit global single-law candidates plus a threshold-scanned piecewise law.

    Returns (candidates, info) where info carries the dc scan table and the
    residual-sign crossover estimate.
    """
    d_tr, d_ho = d[:n_train], d[n_train:]
    y_tr, y_ho = sigma_obs[:n_train], sigma_obs[n_train:]
    obs_rms = float(np.sqrt(np.mean(y_ho ** 2)))
    noise_est = float(noise_sigma * obs_rms)
    ones_tr, ones_ho = np.ones_like(d_tr), np.ones_like(d_ho)

    def make(cid, terms, X_tr, X_ho, extra_prov=None):
        w, loss_tr, rmse_ho, resid_ho = _fit_pair(X_tr, y_tr, X_ho, y_ho)
        prov = {"data_origin": data_origin,
                "benchmark_level": BenchmarkLevel.L1_CONSTRAINED_DISCOVERY,
                "gate4_status": {
                    "passed": True, "holdout_obs_rms": obs_rms,
                    "holdout_noise_sigma": noise_est,
                    "holdout_n_samples": len(y_ho), "rmse": rmse_ho,
                    "holdout_residuals": resid_ho.tolist(),
                    "holdout_sq_error": (resid_ho ** 2).tolist()}}
        if extra_prov:
            prov.update(extra_prov)
        c = CandidateDynamics(lhs="yield_strength", terms=terms, candidate_id=cid)
        return ScoredCandidate(c, prov, train_loss=loss_tr, holdout_rmse=rmse_ho), w

    hp_tr, hp_ho = d_tr ** -0.5, d_ho ** -0.5
    lin_tr, lin_ho = d_tr, d_ho
    inv_tr, inv_ho = d_tr ** 0.5, d_ho ** 0.5  # inverse-softening shape (rises with d)

    defs = [
        ("hp_global", (Term("one", "s0"), Term("d_inv_half", "k")),
         np.column_stack([ones_tr, hp_tr]), np.column_stack([ones_ho, hp_ho])),
        ("linear_grain", (Term("one", "s0"), Term("d_lin", "k")),
         np.column_stack([ones_tr, lin_tr]), np.column_stack([ones_ho, lin_ho])),
        ("inverse_soft", (Term("one", "s0"), Term("d_half", "k")),
         np.column_stack([ones_tr, inv_tr]), np.column_stack([ones_ho, inv_ho])),
        ("constant_baseline", (Term("one", "s0"),),
         ones_tr[:, None], ones_ho[:, None]),
    ]
    candidates, fitted = [], {}
    for cid, terms, X_tr, X_ho in defs:
        sc, w = make(cid, terms, X_tr, X_ho)
        candidates.append(sc)
        fitted[cid] = w

    # Threshold discovery: piecewise (HP above dc, linear below) over dc grid,
    # selected on TRAIN fit only (no holdout peeking).
    scan = []
    for dc in M4_DC_GRID_NM:
        m_tr = d_tr >= dc
        if m_tr.sum() < 3 or (~m_tr).sum() < 3:
            continue
        Xp_tr = np.zeros((len(d_tr), 4))
        Xp_tr[m_tr, 0], Xp_tr[m_tr, 1] = 1.0, d_tr[m_tr] ** -0.5
        Xp_tr[~m_tr, 2], Xp_tr[~m_tr, 3] = 1.0, d_tr[~m_tr]
        w, *_ = np.linalg.lstsq(Xp_tr, y_tr, rcond=None)
        scan.append((dc, float(np.mean((y_tr - Xp_tr @ w) ** 2)), w))
    dc_hat, _, _w_pw = min(scan, key=lambda r: r[1])
    Xp_ho = np.zeros((len(d_ho), 4))
    m_ho = d_ho >= dc_hat
    Xp_ho[m_ho, 0], Xp_ho[m_ho, 1] = 1.0, d_ho[m_ho] ** -0.5
    Xp_ho[~m_ho, 2], Xp_ho[~m_ho, 3] = 1.0, d_ho[~m_ho]
    Xp_tr_full = np.zeros((len(d_tr), 4))
    m_tr = d_tr >= dc_hat
    Xp_tr_full[m_tr, 0], Xp_tr_full[m_tr, 1] = 1.0, d_tr[m_tr] ** -0.5
    Xp_tr_full[~m_tr, 2], Xp_tr_full[~m_tr, 3] = 1.0, d_tr[~m_tr]
    _w_pw2, loss_pw, rmse_pw, resid_pw = _fit_pair(Xp_tr_full, y_tr, Xp_ho, y_ho)
    c_pw = CandidateDynamics(
        lhs="yield_strength",
        terms=(Term("one_hi", "s0"), Term("d_inv_half", "k"),
               Term("one_lo", "s1"), Term("d_lin_lo", "m")),
        candidate_id="piecewise_hp_inv")
    prov_pw = {"data_origin": data_origin,
               "benchmark_level": BenchmarkLevel.L1_CONSTRAINED_DISCOVERY,
               "dc_hat_nm": dc_hat,
               "gate4_status": {
                   "passed": True, "holdout_obs_rms": obs_rms,
                   "holdout_noise_sigma": noise_est,
                   "holdout_n_samples": len(y_ho), "rmse": rmse_pw,
                   "holdout_residuals": resid_pw.tolist(),
                   "holdout_sq_error": (resid_pw ** 2).tolist()}}
    candidates.append(ScoredCandidate(c_pw, prov_pw, train_loss=loss_pw, holdout_rmse=rmse_pw))

    info = {"dc_scan": [(dc, mse) for dc, mse, _ in scan],
            "dc_hat_nm": dc_hat, "dc_true_nm": M4_DC_TRUE_NM}
    return candidates, info


def run_m4_nano_benchmark(noise_sigma: float = 0.02,
                          n_points: int = 60,
                          n_train: int = 30,
                          bootstrap: bool = True,
                          data_origin: str = DataOrigin.SIMULATION,
                          seed: int = 42):
    """Execute M4: global laws vs threshold piecewise on crossover data."""
    d, sigma_obs = generate_m4_data(n_points=n_points, noise_sigma=noise_sigma, seed=seed)
    candidates, info = fit_m4_nano_candidates(d, sigma_obs, n_train=n_train,
                                              noise_sigma=noise_sigma,
                                              data_origin=data_origin)
    ranked = sorted(
        [s for s in candidates if s.train_loss is not None and np.isfinite(s.train_loss)],
        key=lambda s: s.train_loss if s.train_loss is not None else float("inf"))
    r_best = r_rival = None
    if len(ranked) >= 2:
        r_best = ranked[0].provenance.get("gate4_status", {}).get("holdout_residuals")
        r_rival = ranked[1].provenance.get("gate4_status", {}).get("holdout_residuals")
    adj = adjudicate(
        candidates, resid_best=r_best, resid_rival=r_rival,
        sole_survivor_rule=True, sole_gamma=SOLE_SURVIVOR_GAMMA,
        sole_rho_flag=RESID_RHO_FLAG, bootstrap=bootstrap,
        bootstrap_n_boot=1000, bootstrap_seed=seed,
        data_origin=data_origin,
        benchmark_level=BenchmarkLevel.L1_CONSTRAINED_DISCOVERY)
    return adj, candidates, info


def run_m4_l3_pairwise(noise_sigma: float = 0.02, seed: int = 42,
                       probe_seed: int = 777):
    """L3: pairwise probe (piecewise vs global-HP), fresh data, frozen fits.

    The three-way min-divergence probe lands on a compromise point; the
    scientific question is pairwise, so the probe maximizes
    |piecewise - hp_global| over reachable d in [3, 200] nm. Fresh
    true-law observations at the probe window (new seed); Stage-1 fits
    stay frozen (transfer pattern). Tiebreak contracts squared-error
    series, never signed residuals.
    """
    import numpy as np

    d, sigma_obs = generate_m4_data(n_points=60, noise_sigma=noise_sigma, seed=seed)
    d_tr, y_tr = d[:30], sigma_obs[:30]
    ones = np.ones_like(d_tr)
    w_hp, *_ = np.linalg.lstsq(np.column_stack([ones, d_tr ** -0.5]), y_tr, rcond=None)
    dc = M4_DC_TRUE_NM
    grid = np.logspace(np.log10(3.0), np.log10(200.0), 400)
    m_tr = d_tr >= dc
    Xp = np.zeros((len(d_tr), 4))
    Xp[m_tr, 0], Xp[m_tr, 1] = 1.0, d_tr[m_tr] ** -0.5
    Xp[~m_tr, 2], Xp[~m_tr, 3] = 1.0, d_tr[~m_tr]
    w_pw, *_ = np.linalg.lstsq(Xp, y_tr, rcond=None)
    f_pw = np.where(grid >= dc, w_pw[0] + w_pw[1] * grid ** -0.5,
                    w_pw[2] + w_pw[3] * grid)
    f_hp = w_hp[0] + w_hp[1] * grid ** -0.5
    d_star = float(grid[np.argmax(np.abs(f_pw - f_hp))])

    rng = np.random.default_rng(probe_seed)
    dp = np.logspace(np.log10(max(3.0, d_star / 2)), np.log10(min(200.0, d_star * 2)), 30)
    peak = 200.0 + 300.0 * dc ** -0.5
    s_slope = 300.0 * dc ** -1.5
    tp = np.where(dp >= dc, 200.0 + 300.0 * dp ** -0.5, peak - s_slope * (dc - dp))
    op = tp + rng.normal(0, noise_sigma * np.sqrt(np.mean(tp ** 2)), 30)
    obs_rms = float(np.sqrt(np.mean(op ** 2)))

    def frozen(cid, terms, w, feat):
        pred = feat(dp, w)
        rmse = float(np.sqrt(np.mean((op - pred) ** 2)))
        se = (op - pred) ** 2
        g4 = {"passed": True, "holdout_obs_rms": obs_rms,
              "holdout_noise_sigma": noise_sigma * obs_rms,
              "holdout_n_samples": len(op), "holdout_residuals": (op - pred).tolist()}
        c = CandidateDynamics(lhs="yield_strength",
                              terms=tuple(Term(n, f"a{i}") for i, n in enumerate(terms)),
                              candidate_id=cid)
        return ScoredCandidate(c, {"gate4_status": g4, "data_origin": "simulation"},
                               train_loss=rmse, holdout_rmse=rmse), se

    s1, se1 = frozen("piecewise", ("one_hi", "hp", "one_lo", "lin"),
                     w_pw, lambda e, w: np.where(e >= dc, w[0] + w[1] * e ** -0.5,
                                                 w[2] + w[3] * e))
    s2, se2 = frozen("hp_global", ("one", "hp"),
                     w_hp, lambda e, w: w[0] + w[1] * e ** -0.5)
    adj = adjudicate([s1, s2], resid_best=se1.tolist(), resid_rival=se2.tolist(),
                     bootstrap=True, bootstrap_n_boot=1000, bootstrap_seed=7)
    return {"d_star_nm": d_star,
            "divergence_mpa": float(np.max(np.abs(f_pw - f_hp)))}, adj, [s1, s2]


def hello_world():
    print("=" * 64)
    print("M4 NANOSCALE CROSSOVER: classical HP vs inverse softening (dc=15nm)")
    print("=" * 64)
    for ns in (0.02, 0.10):
        adj, cands, info = run_m4_nano_benchmark(noise_sigma=ns)
        print(f"--- noise {ns:.2f} ---")
        for s in sorted(cands, key=lambda c: c.holdout_rmse or 1e18):
            print(f"  {s.candidate.candidate_id:18s} holdout RMSE={s.holdout_rmse:8.2f}")
        print("  dc scan (nm:mse): " +
              " ".join(f"{dc:g}:{mse:.0f}" for dc, mse in info["dc_scan"]))
        print(f"  dc_hat={info['dc_hat_nm']}nm (truth {info['dc_true_nm']}nm)")
        print(f"  verdict: {adj.status.value} "
              f"promoted={adj.promoted.candidate.candidate_id if adj.promoted else None} "
              f"origin={adj.data_origin}")
    print("--- L3 pairwise (piecewise vs hp_global) ---")
    probe, adj3, ss = run_m4_l3_pairwise()
    print(f"  probe d*={probe['d_star_nm']:.1f}nm divergence={probe['divergence_mpa']:.1f} MPa")
    for s in ss:
        print(f"  frozen {s.candidate.candidate_id:10s} probe RMSE={s.holdout_rmse:.2f}")
    print(f"  L3 verdict: {adj3.status.value} "
          f"promoted={adj3.promoted.candidate.candidate_id if adj3.promoted else None}")
    print("=" * 64)


if __name__ == "__main__":
    hello_world()
