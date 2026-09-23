"""APEDP Bayesian experiment designer (Future Work item 1).

Selects the next measurement(s) by posterior-predictive uncertainty, bridging
``apedp.priors`` (SEMF coefficient samples) and ``apedp.jax_pde_solver``
(forward simulations) to ``apedp.scientific_layers.ExperimentDesign``.

Scoring (Gaussian approximation):
    score(x) = Var_post[ y(x) ] / sigma(x)^2  ~  expected information gain
    (EIG = 0.5 * log(1 + Var/sigma^2), monotone in the variance ratio, so the
    variance ratio is used directly as the greedy objective.)

The output is a design *proposal* only -- no experimental data are fabricated.
``closed_loop.py`` consumes the proposal and requires caller-supplied
measurements for the update step.

Typical usage:
    from apedp.experiment_designer import design_from_predictive_samples
    design = design_from_predictive_samples(predictive, xs, n_points=2)
    from apedp.experiment_designer import semf_isotope_design
    design = semf_isotope_design(coeff_samples, Z, N, A, n_points=3)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from apedp.scientific_layers import ExperimentDesign

# ---------------------------------------------------------------------------
# Core: greedy posterior-predictive design
# ---------------------------------------------------------------------------

def predictive_scores(
    predictive: np.ndarray,
    sigma: np.ndarray | float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Variance-ratio scores for posterior-predictive samples.

    Args:
        predictive: Array of shape (n_samples, n_grid) with posterior draws
            of the observable at each candidate point.
        sigma: Measurement std, scalar or shape (n_grid,). Must be finite;
            negative values raise; zeros are floored at 1e-12.

    Returns:
        (scores, spread): score = Var_s[y]/sigma^2 and raw std per grid point.
    """
    pred = np.asarray(predictive, dtype=float)
    if pred.ndim != 2 or pred.shape[0] < 2 or pred.shape[1] < 1:
        raise ValueError("predictive must have shape (n_samples >= 2, n_grid >= 1)")
    if not np.all(np.isfinite(pred)):
        raise ValueError("predictive samples must be finite")
    grid_size = pred.shape[1]
    noise = np.asarray(sigma, dtype=float)
    if noise.ndim == 0:
        noise = np.full(grid_size, float(noise))
    noise = noise.reshape(-1)
    if noise.shape != (grid_size,) or not np.all(np.isfinite(noise)):
        raise ValueError("sigma must be scalar or finite values matching n_grid")
    if np.any(noise < 0.0):
        raise ValueError("measurement uncertainty cannot be negative")
    noise = np.maximum(noise, 1e-12)
    spread = pred.std(axis=0, ddof=1)
    scores = (spread ** 2) / (noise ** 2)
    return scores, spread


def design_from_predictive_samples(
    predictive: np.ndarray,
    xs: np.ndarray,
    *,
    sigma: np.ndarray | float = 1.0,
    feasible_fn: Callable | None = None,
    n_points: int = 1,
    min_distance: float = 0.0,
    objective: str = "posterior variance / measurement variance (EIG proxy)",
) -> ExperimentDesign:
    """Greedily select measurement points maximizing the variance ratio.

    Args:
        predictive: (n_samples, n_grid) posterior-predictive draws.
        xs: Candidate coordinates, shape (n_grid,).
        sigma: Measurement std, scalar or per-point.
        feasible_fn: Optional callable grid -> boolean mask.
        n_points: Batch size (>= 1).
        min_distance: Minimum |x_i - x_j| separation within the batch.
        objective: Label recorded on the design object.

    Returns:
        ExperimentDesign with selected grid coordinates.
    """
    if n_points < 1:
        raise ValueError("n_points must be positive")
    if not np.isfinite(min_distance) or min_distance < 0.0:
        raise ValueError("min_distance must be finite and non-negative")
    grid = np.asarray(xs, dtype=float).reshape(-1)
    pred = np.asarray(predictive, dtype=float)
    if pred.ndim != 2 or pred.shape[1] != grid.size:
        raise ValueError("predictive n_grid must match xs length")
    if grid.size == 0 or not np.all(np.isfinite(grid)):
        raise ValueError("xs must contain finite candidate points")
    scores, spread = predictive_scores(pred, sigma)
    feasible = np.ones(grid.size, dtype=bool)
    if feasible_fn is not None:
        feasible = np.asarray(feasible_fn(grid), dtype=bool).reshape(-1)
        if feasible.shape != (grid.size,):
            raise ValueError("feasible_fn must return one boolean per candidate point")
    masked = np.where(feasible, scores, -np.inf)
    selected: list[int] = []
    for _ in range(min(n_points, int(feasible.sum()))):
        ranked = np.argsort(masked)[::-1]
        choice = next(
            (int(i) for i in ranked
             if np.isfinite(masked[i])
             and all(abs(grid[i] - grid[old]) >= min_distance for old in selected)),
            None,
        )
        if choice is None:
            break
        selected.append(choice)
        masked[choice] = -np.inf
    noise = np.asarray(sigma, dtype=float)
    noise = np.full(grid.size, float(noise)) if noise.ndim == 0 else noise.reshape(-1)
    noise = np.maximum(noise, 1e-12)
    return ExperimentDesign(
        selected_points=[float(grid[i]) for i in selected],
        scores=[float(scores[i]) for i in selected],
        raw_spread=[float(spread[i]) for i in selected],
        candidate_count=pred.shape[0],
        objective=objective,
    )


def design_from_callable(
    predict_fn: Callable[[np.ndarray, Any], np.ndarray],
    param_samples: list,
    xs: np.ndarray,
    **kwargs,
) -> ExperimentDesign:
    """Build the predictive matrix by calling predict_fn(sample, xs) per draw.

    Args:
        predict_fn: Callable (param_sample, xs_1d) -> predictions shape (n_grid,).
        param_samples: Posterior draws over model parameters.
        xs: Candidate coordinates.
        **kwargs: Forwarded to design_from_predictive_samples.

    Returns:
        ExperimentDesign.
    """
    grid = np.asarray(xs, dtype=float).reshape(-1)
    rows = []
    for sample in param_samples:
        row = np.asarray(predict_fn(sample, grid), dtype=float).reshape(-1)
        if row.shape != (grid.size,):
            raise ValueError("predict_fn must return one prediction per grid point")
        rows.append(row)
    if len(rows) < 2:
        raise ValueError("need at least 2 param samples for a variance")
    return design_from_predictive_samples(np.stack(rows), grid, **kwargs)


# ---------------------------------------------------------------------------
# Bridge: SEMF isotope design
# ---------------------------------------------------------------------------

def semf_isotope_design(
    coeff_samples: dict,
    Z: np.ndarray,
    N: np.ndarray,
    A: np.ndarray,
    *,
    sigma: np.ndarray | float = 1.0,
    n_points: int = 1,
    max_draws: int = 200,
) -> ExperimentDesign:
    """Propose the most informative (Z, A) isotopes to measure next.

    Uses ``apedp.priors.semf_binding_energy`` over posterior coefficient draws;
    the grid coordinate is the mass number A (isotopes are discrete, so
    min_distance is not applied -- set n_points <= n_isotopes).

    Args:
        coeff_samples: Dict with vector coeffs (n_samples,) from priors.py.
        Z, N, A: Candidate isotope arrays, shape (n_isotopes,).
        sigma: Measurement std per isotope (MeV).
        n_points: Number of isotopes to propose.
        max_draws: Cap on posterior draws (subsampled deterministically).

    Returns:
        ExperimentDesign over A values. ``selected_points`` are mass numbers;
        pair with the input Z/N/A arrays to recover full isotope labels.
    """
    from apedp.priors import semf_binding_energy  # local import: jax dependency

    Zc, Nc, Ac = (np.asarray(v).reshape(-1) for v in (Z, N, A))
    if not (Zc.shape == Nc.shape == Ac.shape) or Zc.size == 0:
        raise ValueError("Z, N, A must be non-empty arrays of equal length")
    n_iso = Zc.size
    if n_points > n_iso:
        raise ValueError("n_points cannot exceed the candidate isotope count")
    n_avail = int(np.asarray(coeff_samples["a_v"]).size)
    step = max(1, n_avail // max(1, max_draws))
    sub = {k: np.asarray(v).reshape(-1)[::step][:max_draws]
           for k, v in coeff_samples.items() if k in ("a_v", "a_s", "a_c", "a_a")}
    sub["Delta"] = coeff_samples.get("Delta", 1.6)
    if any(sub[k].shape[0] < 2 for k in ("a_v", "a_s", "a_c", "a_a")):
        raise ValueError("need at least 2 coefficient draws")
    pred = np.asarray(
        semf_binding_energy(_j(Zc), _j(Nc), _j(Ac), _jax_coeffs(sub)),
        dtype=float,
    )
    if pred.ndim != 2 or pred.shape[1] != n_iso:
        raise ValueError(f"unexpected SEMF predictive shape {pred.shape}")
    return design_from_predictive_samples(
        pred, Ac.astype(float),
        sigma=sigma, n_points=n_points,
        objective="SEMF posterior variance / measurement variance (EIG proxy)",
    )


def _j(x: np.ndarray):
    """Numpy -> JAX array without a top-level jax import."""
    import jax.numpy as jnp

    return jnp.asarray(np.asarray(x, dtype=float))


def _jax_coeffs(sub: dict) -> dict:
    """Numpy subsample dict -> JAX coefficient dict for priors.py."""
    import jax.numpy as jnp

    out = {k: jnp.asarray(np.asarray(v, dtype=float)) for k, v in sub.items()
           if k in ("a_v", "a_s", "a_c", "a_a")}
    out["Delta"] = float(sub.get("Delta", 1.6))
    return out


# ---------------------------------------------------------------------------
# Test block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Experiment Designer Tests ===\n")

    # Test 1: variance peak is selected (deterministic synthetic draws)
    print("Test 1: peak-variance point selected")
    xs = np.linspace(0.0, 1.0, 11)
    # Two model families disagree most at x = 1.0
    draws = np.stack([xs, xs ** 2, np.zeros_like(xs), xs ** 3])
    design = design_from_predictive_samples(draws, xs, n_points=1)
    print(f"  selected: {design.selected_points} (expect [1.0])")
    assert design.selected_points == [1.0], design.selected_points

    # Test 2: batch + min_distance separation
    print("Test 2: batch separation")
    design2 = design_from_predictive_samples(draws, xs, n_points=2, min_distance=0.5)
    pts = design2.selected_points
    print(f"  selected: {pts}")
    assert len(pts) == 2 and abs(pts[0] - pts[1]) >= 0.5, pts

    # Test 3: sigma weighting flips the choice toward precise points
    print("Test 3: uncertainty weighting")
    sigma = np.where(xs == 1.0, 1e6, 1.0)  # x=1 unmeasurable
    design3 = design_from_predictive_samples(draws, xs, sigma=sigma, n_points=1)
    print(f"  selected: {design3.selected_points} (must not be [1.0])")
    assert design3.selected_points != [1.0], design3.selected_points

    # Test 4: infeasible mask respected
    print("Test 4: feasibility mask")
    design4 = design_from_predictive_samples(
        draws, xs, feasible_fn=lambda g: g < 0.5, n_points=1)
    print(f"  selected: {design4.selected_points}")
    assert design4.selected_points[0] < 0.5, design4.selected_points

    # Test 5: SEMF bridge on 4 candidate isotopes
    print("Test 5: SEMF isotope design")
    import jax

    key = jax.random.PRNGKey(0)
    from apedp.priors import semf_prior_samples

    coeffs = semf_prior_samples(key, n_samples=60)
    # Mixed value types by design: arrays per coefficient, one scalar for Delta.
    np_coeffs: dict[str, Any] = {k: np.asarray(v) for k, v in coeffs.items() if k != "Delta"}
    np_coeffs["Delta"] = float(coeffs["Delta"])
    design5 = semf_isotope_design(
        np_coeffs,
        Z=np.array([26, 28, 50, 82]),
        N=np.array([30, 30, 70, 126]),
        A=np.array([56, 58, 120, 208]),
        n_points=2,
    )
    print(f"  selected A: {design5.selected_points}")
    assert len(design5.selected_points) == 2, design5.selected_points

    # Test 6: validation errors
    print("Test 6: input validation")
    for bad_call, label in [
        (lambda: design_from_predictive_samples(np.ones((1, 5)), xs[:5]), "1 sample"),
        (lambda: design_from_predictive_samples(draws, xs, sigma=-1.0), "neg sigma"),
        (lambda: design_from_predictive_samples(draws, xs, n_points=0), "n_points=0"),
    ]:
        try:
            bad_call()
            raise AssertionError(f"no error for {label}")
        except ValueError:
            print(f"  correctly rejected: {label}")

    print("\n=== All designer tests passed ===")
