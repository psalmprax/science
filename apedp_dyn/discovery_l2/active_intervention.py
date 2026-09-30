"""L3 Active Interventional Synthesis for Resolving Epistemic Ties.

When APEDP issues MULTI_HYPOTHESIS under INTRINSIC_OBSERVATIONAL_NONIDENTIFIABILITY,
passive observational data cannot separate the competing mechanisms.

This module synthesizes the optimal experimental intervention X* that maximizes
the evidential divergence between tied candidate hypotheses:
    X* = argmax_{X in Omega} |f_1(X) - f_2(X)| / sqrt(Var(f_1) + Var(f_2))

Executing an intervention at X* breaks the observational symmetry (e.g. collinearity,
unexcited modes, or gauge equivalence), allowing the epistemic adjudicator to
transition from MULTI_HYPOTHESIS to UNIQUE_SUPPORT.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ProposedIntervention:
    """An active experimental probe designed to break hypothesis degeneracy."""

    probe_point: dict[str, float]
    predicted_divergence: float
    candidate_predictions: dict[str, float]
    rationale: str


def synthesize_discriminating_intervention(
    models: list[tuple[str, Callable[[dict[str, np.ndarray]], np.ndarray]]],
    variable_bounds: dict[str, tuple[float, float]],
    n_grid_points: int = 500,
    seed: int = 42,
) -> ProposedIntervention:
    """Synthesize the optimal probe X* maximizing discrepancy between competing models.

    Args:
        models: list of (model_id, predict_fn) where predict_fn(X_dict) returns predictions.
        variable_bounds: dict mapping variable name to (min_val, max_val) admissible bounds.
        n_grid_points: number of Latin-hypercube or Monte-Carlo candidate probes to evaluate.
        seed: random seed for reproducibility.

    Returns:
        ProposedIntervention containing the optimal point X* and expected divergence.
    """
    if len(models) < 2:
        raise ValueError("At least two competing models are required to synthesize a discriminating probe.")

    rng = np.random.default_rng(seed)
    var_names = list(variable_bounds.keys())

    # Generate candidate interventional points across the admissible domain Omega
    n_vars = len(var_names)
    probes_raw = rng.uniform(0.0, 1.0, size=(n_grid_points, n_vars))
    probes_dict: dict[str, np.ndarray] = {}
    for idx, name in enumerate(var_names):
        lo, hi = variable_bounds[name]
        probes_dict[name] = lo + (hi - lo) * probes_raw[:, idx]

    # Evaluate all models across the candidate intervention points
    preds = {}
    for mod_id, fn in models:
        preds[mod_id] = np.asarray(fn(probes_dict), dtype=float)

    # For the top two rival models, find the point of maximum absolute difference
    mod_a_id, _ = models[0]
    mod_b_id, _ = models[1]
    diff = np.abs(preds[mod_a_id] - preds[mod_b_id])

    best_idx = int(np.argmax(diff))
    max_div = float(diff[best_idx])

    probe_pt = {name: float(probes_dict[name][best_idx]) for name in var_names}
    cand_preds = {mod_id: float(preds[mod_id][best_idx]) for mod_id in preds}

    return ProposedIntervention(
        probe_point=probe_pt,
        predicted_divergence=max_div,
        candidate_predictions=cand_preds,
        rationale=(
            f"Active intervention probe at {probe_pt} maximizes evidential separation "
            f"between {mod_a_id} and {mod_b_id} (divergence = {max_div:.4g})."
        ),
    )


def compute_fisher_information_matrix(
    jacobian_fn: Callable[[np.ndarray], np.ndarray],
    params: np.ndarray,
    noise_sigma: float,
) -> np.ndarray:
    """Compute the Fisher Information Matrix (FIM) for parameter sensitivity:

    FIM = J^T * J / sigma^2
    """
    J = jacobian_fn(params)
    return (J.T @ J) / max(noise_sigma**2, 1e-12)
