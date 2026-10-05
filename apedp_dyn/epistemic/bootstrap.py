"""Paired block-bootstrap tiebreaker for epistemic adjudication (v2).

Motivation (frozen failure): point-estimate ΔRMSE can promote a winner on
noise alignment in near-boundary, high-noise regimes. The tiebreaker re-asks
the would-be UNIQUE_SUPPORT decision with a 95% CI on the paired loss
difference before promotion.

Sign convention: d_t = L_rival,t - L_top,t (per-timepoint squared-error
difference, mean over channels). d_t > 0 → top better at t.

Method: circular block bootstrap (blocks preserve temporal dependence; IID
time-point resampling would understate variance on smooth trajectories).
Block length default floor(sqrt(n)) clamped to [5, 50]; n_boot and seed are
explicit so reruns are bit-reproducible.

Rule (applied ONLY to the would-be UNIQUE branch, i.e. point Δ > τ_epi):
  CI entirely > 0  → top model supported over rival (confirm UNIQUE)
  CI overlaps 0    → MULTI_HYPOTHESIS (apparent margin is noise alignment)
  CI entirely < 0  → MULTI_HYPOTHESIS with rival-favored note (rank revision
                     — promoting the rival instead — is a documented NEXT
                     extension, not this patch)

Scope note (frozen negative control): the STAB/10%/rep2 false promotion is a
SOLE-SURVIVOR promotion (truth holdout_rmse None, no rival pair). This
tiebreaker does not apply there and must not change that cell. A different
hardening (sole-survivor absolute-fit rule) is the next target.
"""
from __future__ import annotations

import numpy as np

N_BOOT_DEFAULT = 2000
CI_DEFAULT = 0.95


def lag1_autocorrelation(resid: np.ndarray) -> float:
    """Centered lag-1 autocorrelation of a residual series.

    Returns nan for degenerate (zero-variance) or empty input; callers
    decide the epistemic meaning (a constant series carries no structure).
    """
    import numpy as _np

    r = _np.asarray(resid, dtype=float).ravel()
    if r.size < 3 or not _np.all(_np.isfinite(r)):
        return float("nan")
    z = r - r.mean()
    denom = float(z @ z)
    if denom <= 0:
        return float("nan")
    return float(z[:-1] @ z[1:] / denom)


def block_bootstrap_ci(
    d: np.ndarray,
    n_boot: int = N_BOOT_DEFAULT,
    block_len: int | None = None,
    ci: float = CI_DEFAULT,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Circular block-bootstrap CI for mean(d). Returns (lo, hi, mean)."""
    d = np.asarray(d, dtype=float).ravel()
    n = len(d)
    if n == 0:
        raise ValueError("empty difference series")
    if not np.all(np.isfinite(d)):
        raise ValueError("non-finite values in difference series")
    if block_len is None:
        block_len = int(max(5, min(50, np.sqrt(n))))
    block_len = int(max(1, min(n, block_len)))
    rng = np.random.default_rng(seed)
    mean = float(np.mean(d))
    # Circular extension so every start index yields a full block.
    ext = np.concatenate([d, d[:block_len]])
    n_blocks = int(np.ceil(n / block_len))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        sample = np.concatenate([ext[s:s + block_len] for s in starts])[:n]
        boots[b] = np.mean(sample)
    alpha = 1.0 - ci
    lo, hi = float(np.quantile(boots, alpha / 2.0)), float(np.quantile(boots, 1.0 - alpha / 2.0))
    return lo, hi, mean


def rmse_bootstrap_ci(se_series, n_boot: int = 1000, seed: int = 0,
                      ci: float = CI_DEFAULT) -> tuple[float, float, float]:
    """Block-bootstrap CI on the holdout RMSE itself (evaluation uncertainty).

    Resamples the per-timepoint SE series with the same circular blocks as
    the tiebreaker; returns (lo, hi, rmse). Diagnostic: quantifies how well
    the evaluation metric is measured, never a promotion input by itself.
    """
    import numpy as _np

    se = _np.asarray(se_series, dtype=float).ravel()
    if se.size == 0 or not _np.all(_np.isfinite(se)):
        raise ValueError("holdout SE series empty/non-finite")
    n = len(se)
    block_len = int(max(5, min(50, _np.sqrt(n))))
    rng = _np.random.default_rng(seed)
    ext = _np.concatenate([se, se[:block_len]])
    n_blocks = int(_np.ceil(n / block_len))
    boots = _np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        sample = _np.concatenate([ext[s:s + block_len] for s in starts])[:n]
        boots[b] = _np.sqrt(max(_np.mean(sample), 0.0))
    rmse = float(_np.sqrt(_np.mean(se)))
    alpha = 1.0 - ci
    return (float(_np.quantile(boots, alpha / 2.0)),
            float(_np.quantile(boots, 1.0 - alpha / 2.0)), rmse)


def holdout_se_series(problem, candidate_ir, param_values: dict[str, float]):
    """Per-timepoint holdout squared-error series for a FITTED candidate.

    Materializes the IR, binds fitted values, simulates the holdout regime,
    returns mean-over-channels SE vs holdout truth at each timepoint.
    Raises RuntimeError if the holdout simulation is invalid (mirrors the
    gate-4 failure that leaves holdout_rmse None).
    """
    import numpy as np

    from apedp_dyn.fitting.parameter_fit import _with_values
    from apedp_dyn.grammar import materialize
    from apedp_dyn.grammar.candidate_ir import SystemCandidate
    from apedp_dyn.simulation import simulate

    if isinstance(candidate_ir, SystemCandidate):
        from apedp_dyn.grammar.coupled import materialize_system

        model0 = materialize_system(problem, candidate_ir)
    else:
        model0 = materialize(problem, candidate_ir)
    model = _with_values(model0, {k: float(v) for k, v in param_values.items()})
    h = problem.holdout
    res = simulate(model, h["t_span"], h["u_of_t"],
                   np.asarray(h["x0"], dtype=float), n_eval=len(h["truth_t"]))
    if not res.success:
        raise RuntimeError(f"holdout simulation invalid: {res.detail}")
    obs = np.asarray(h["truth_y"], dtype=float)
    return np.mean((res.y.T - obs) ** 2, axis=1)


def param_bootstrap_ci(problem, candidate_ir, fitted_values: dict[str, float],
                       n_boot: int = 30, seed: int = 0) -> dict[str, tuple[float, float, float]]:
    """Residual-bootstrap parameter bands (experimental UQ, v1).

    Resamples TRAIN residual vectors (block bootstrap over time, preserving
    temporal dependence), rebuilds noisy-train problems around the FITTED
    trajectory, refits, and collects percentile bands per parameter. Returns
    {param: (lo, hi, original)}. Honors the negative result on Jacobian SEs:
    these bands measure estimator variability under resampled noise, NOT
    coverage of physical truth under misspecification or bad basins. A rep
    whose refit fails returns no sample (recorded count may be < n_boot).
    """
    import dataclasses as _dc

    import numpy as _np

    from apedp_dyn.fitting.parameter_fit import _train_arrays, _with_values, fit_candidate
    from apedp_dyn.grammar import materialize
    from apedp_dyn.grammar.candidate_ir import SystemCandidate

    names = list(fitted_values)
    if isinstance(candidate_ir, SystemCandidate):
        from apedp_dyn.grammar.coupled import materialize_system

        model0 = materialize_system(problem, candidate_ir)
    else:
        model0 = materialize(problem, candidate_ir)
    fitted = _with_values(model0, {k: float(v) for k, v in fitted_values.items()})
    reg, t, Y, _U = _train_arrays(problem)
    from apedp_dyn.simulation import simulate

    res = simulate(fitted, reg["t_span"], reg["u_of_t"],
                   np.asarray(reg["x0"], dtype=float), n_eval=len(t))
    if not res.success:
        raise RuntimeError(f"fitted-model train simulation invalid: {res.detail}")
    resid = Y - res.y.T  # (n, states)
    n = len(t)
    block_len = int(max(5, min(50, _np.sqrt(n))))
    rng = _np.random.default_rng(seed)
    ext = _np.concatenate([resid, resid[:block_len]], axis=0)
    n_blocks = int(_np.ceil(n / block_len))
    samples: dict[str, list[float]] = {k: [] for k in names}
    for _ in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        draw = _np.concatenate([ext[s:s + block_len] for s in starts], axis=0)[:n]
        noisy = dict(reg)
        noisy["truth_y"] = res.y.T + draw
        reprobed = _dc.replace(problem, train=noisy)
        try:
            sc = fit_candidate(reprobed, candidate_ir)
        except (ValueError, TypeError):
            continue
        if sc.fit is None:
            continue
        for k in names:
            samples[k].append(float(sc.fit.param_values[k]))
    out = {}
    for k in names:
        vals = _np.asarray(samples[k])
        if len(vals) < 8:
            raise RuntimeError(f"too few successful refits for {k}: {len(vals)}/{n_boot}")
        out[k] = (float(_np.quantile(vals, 0.025)), float(_np.quantile(vals, 0.975)),
                  float(fitted_values[k]))
    return out


def tiebreak(
    delta_point: float,
    tau_epi: float,
    resid_top: np.ndarray,
    resid_rival: np.ndarray,
    n_boot: int = N_BOOT_DEFAULT,
    seed: int = 0,
) -> dict:
    """Bootstrap tiebreak for a would-be UNIQUE_SUPPORT decision.

    resid_top / resid_rival: per-timepoint squared errors (mean over channels)
    on the SAME holdout record. Returns a dict with lo/hi/mean and the
    recommended branch: 'confirm' | 'multi' | 'rival-favored'.

    Callers must pass *squared* errors. ``adjudicate`` owns that conversion
    (it receives signed residuals for the whiteness guard and squares them
    here), because squared-error series may legitimately be negative for a
    caller deliberately constructing a differential series -- auto-detection
    inside this function would corrupt that case.
    """
    top = np.asarray(resid_top, dtype=float).ravel()
    riv = np.asarray(resid_rival, dtype=float).ravel()
    if top.shape != riv.shape:
        raise ValueError(f"residual shape mismatch: {top.shape} vs {riv.shape}")
    d = riv - top
    lo, hi, mean = block_bootstrap_ci(d, n_boot=n_boot, seed=seed)
    if lo > 0:
        branch = "confirm"
    elif hi < 0:
        branch = "rival-favored"
    else:
        branch = "multi"
    return {"lo": lo, "hi": hi, "mean": mean, "delta_point": float(delta_point),
            "tau_epi": float(tau_epi), "branch": branch, "n": len(d), "n_boot": n_boot}
