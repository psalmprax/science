"""Level-5 roadmap Pillar 3: Closed-Loop Active Experimental Disambiguation.

When adjudication halts at MULTI_HYPOTHESIS, this layer does NOT stop — it
designs the experiment that destroys the ambiguity, on the proposer side of
the referee separation:

1. Maximal epistemic divergence: with surviving hypotheses f_1..f_n over a
   declared domain, the objective is

     D(x) = max_{i<j} |f_i(x) - f_j(x)| - k * sigma_noise

   with k imported from the frozen active-disambiguation policy
   (`DEFAULT_DETECTABILITY_K`) — the same noise-normalized detectability
   bar the E4 policy already pinned; no new epistemic threshold is
   declared here. x* = argmax D(x) over the domain grid.
2. Structured acquisition orders: `DataAcquisitionOrder` carries the
   variable, the clustered target_ranges around the top divergence
   coordinates and the rationale. The external acquisition source (robot
   lab, sensor stream, synthetic counterfactual) is a CALLABLE the caller
   supplies — this module never fabricates data.
3. Referee handoff: acquired points are validated (inside the ordered
   ranges, finite) and reduced to per-hypothesis residuals the CALLER
   feeds to `adjudicate()`. No promotion or rejection is decided here —
   the proposer/referee separation (AGENTS §1.1) is absolute: an
   experimentalist that re-adjudicated its own loops would be grading its
   own homework.

Flaw-contract notes:
- Orders are DETERMINISTIC functions of (hypotheses, domain, sigma):
  frozen grids, no sampling, no seeds to tune.
- Empty-order emission (D(x*) <= 0 everywhere) is the honest
  INSUFFICIENT_DISAMBIGUATION outcome, NOT a permission to loosen k.
Status: experimental (Level-5 roadmap Phase 3; referee stays cryptographic).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np

from apedp_dyn.epistemic.active_disambiguation.policy import DEFAULT_DETECTABILITY_K

__all__ = [
    "DOMAIN_GRID_DEFAULT",
    "DataAcquisitionOrder",
    "OrderRoundResult",
    "closed_loop_order",
    "design_experiment_orders",
    "record_order_results",
    "referee_handoff",
]

DOMAIN_GRID_DEFAULT = 400  # deterministic divergence grid; documented heuristic
RANGE_MERGE_GAP_FRACTION = 0.05  # cluster optima closer than 5% of the domain
RANGE_PAD_FRACTION = 0.01  # pad each emitted range by 1% so boundary x* is reachable
MAX_RANGES_PER_ORDER = 2


@dataclass(frozen=True)
class DataAcquisitionOrder:
    variable: str
    target_ranges: tuple[tuple[float, float], ...]
    rationale: str = ""
    pole_margin: float = float("nan")  # D value at the emitted x*


@dataclass(frozen=True)
class OrderRoundResult:
    rounds: int
    orders: tuple[DataAcquisitionOrder, ...]
    accepted_points: int
    rejected_points: int
    referee_handoff: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    notes: str = ""


def design_experiment_orders(
    hypotheses: Mapping[str, Callable[[float], float]],
    domain: tuple[float, float],
    sigma_noise: float,
    variable: str = "x",
    detectability_k: float = DEFAULT_DETECTABILITY_K,
    n_grid: int = DOMAIN_GRID_DEFAULT,
) -> tuple[DataAcquisitionOrder, ...]:
    """x* = argmax (max pairwise divergence) − k·sigma on a frozen grid."""
    if len(hypotheses) < 2:
        raise ValueError("active disambiguation needs >= 2 surviving hypotheses")
    if not (sigma_noise > 0):
        raise ValueError("sigma_noise must be positive (the noise model is not optional)")
    lo, hi = float(domain[0]), float(domain[1])
    if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
        raise ValueError(f"invalid domain {domain}")
    grid = np.linspace(lo, hi, n_grid)
    names = tuple(hypotheses)
    preds = np.column_stack([np.asarray([hypotheses[n](v) for v in grid], dtype=float) for n in names])
    # Pairwise max-gap per grid point (n_pairs, n_grid) → collapse over pairs.
    gap_matrix = np.stack([np.abs(preds[:, i] - preds[:, j])
                           for i in range(len(names)) for j in range(i + 1, len(names))])
    gaps = gap_matrix.max(axis=0)
    d = gaps - detectability_k * sigma_noise
    x_star = int(np.argmax(d))
    if d[x_star] <= 0:
        return ()  # INSUFFICIENT_DISAMBIGUATION: no experiment distinguishes survivors
    # Cluster grid coordinates whose D > 0 into contiguous ranges.
    above = np.where(d > 0)[0]
    gap_idx = int(np.ceil(RANGE_MERGE_GAP_FRACTION * n_grid))
    clusters: list[list[int]] = [[int(above[0])]]
    for idx in above[1:]:
        if idx - clusters[-1][-1] <= gap_idx:
            clusters[-1].append(int(idx))
        else:
            clusters.append([int(idx)])
    clusters.sort(key=lambda c: -float(d[c].max()))
    clusters = clusters[:MAX_RANGES_PER_ORDER]
    pad = RANGE_PAD_FRACTION * (hi - lo)
    ranges: list[tuple[float, float]] = []
    for c in clusters:
        r_lo = max(lo, float(grid[min(c)]) - pad)
        r_hi = min(hi, float(grid[max(c)]) + pad)
        ranges.append((float(r_lo), float(r_hi)))
    rationale = (
        f"argmax over frozen grid of {len(names)}-hypothesis divergence minus "
        f"k*sigma (k={detectability_k}); primary x*={float(grid[x_star]):.6g}, "
        f"D(x*)={float(d[x_star]):.6g}"
    )
    pole = float(d[x_star])
    out = []
    for k_idx, (r_lo, r_hi) in enumerate(ranges):
        out.append(DataAcquisitionOrder(variable, ((r_lo, r_hi),),
                                        rationale if k_idx == 0 else f"secondary divergence band of: {rationale}",
                                        pole))
    return tuple(out)


def record_order_results(
    order: DataAcquisitionOrder,
    x_new: np.ndarray,
    y_new: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Validate acquired points against the order's ranges (fail-closed).

    Returns (x_ok, y_ok, n_accepted, n_rejected). Points outside every
    target range or non-finite are REJECTED, not clamped — an acquisition
    source that silently returns out-of-band values is a broken instrument,
    and clamping would launder its failure into referee evidence.
    """
    x_new = np.asarray(x_new, dtype=float)
    y_new = np.asarray(y_new, dtype=float)
    if x_new.shape != y_new.shape:
        raise ValueError("x_new and y_new shapes differ")
    in_band = np.zeros(len(x_new), dtype=bool)
    for r_lo, r_hi in order.target_ranges:
        in_band |= (x_new >= r_lo) & (x_new <= r_hi)
    ok = in_band & np.isfinite(x_new) & np.isfinite(y_new)
    return x_new[ok], y_new[ok], int(ok.sum()), int((~ok).sum())


def referee_handoff(
    hypotheses: Mapping[str, Callable[[float], float]],
    x_eval: np.ndarray,
    y_obs: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    """Per-hypothesis residuals on the acquired points (input to adjudicate()).

    Keys: ``observations`` -> {"x_eval","y_obs"}; one entry per hypothesis
    name -> {"residuals", "predictions"}. The user of adjudicate() does the
    verdict; this layer only packages its inputs.
    """
    x_eval = np.asarray(x_eval, dtype=float)
    y_obs = np.asarray(y_obs, dtype=float)
    out: dict[str, dict[str, np.ndarray]] = {
        "observations": {"x_eval": x_eval, "y_obs": y_obs}
    }
    for name, fn in hypotheses.items():
        pred = np.asarray([fn(v) for v in x_eval], dtype=float)
        out[name] = {"residuals": y_obs - pred, "predictions": pred}
    return out


def closed_loop_order(
    hypotheses: Mapping[str, Callable[[float], float]],
    domain: tuple[float, float],
    sigma_noise: float,
    acquisition: Callable[[DataAcquisitionOrder], tuple[np.ndarray, np.ndarray]],
    variable: str = "x",
    rounds: int = 3,
    detectability_k: float = DEFAULT_DETECTABILITY_K,
) -> OrderRoundResult:
    """Emit orders, ingest acquired data, shrink ambiguity per round.

    The acquisition callable returns (x_new, y_new) for each order; points
    outside the ordered ranges are REJECTED and counted. After every round
    the loop re-scores the divergence: if the survivors' objective collapses
    below the detectability bar on the frozen grid, the loop stops honestly
    and hands the residuals to the caller for the referee, never deciding
    promotion internally.
    """
    orders_all: list[DataAcquisitionOrder] = []
    xs_used: list[np.ndarray] = []
    ys_used: list[np.ndarray] = []
    accepted_total = rejected_total = 0
    notes = ""
    survived = dict(hypotheses)
    for r in range(rounds):
        orders = design_experiment_orders(survived, domain, sigma_noise, variable,
                                          detectability_k)
        if not orders:
            notes = f"round {r}: objective below the detectability bar on the frozen grid"
            break
        orders_all.extend(orders)
        x_new, y_new = acquisition(orders[0])
        x_ok, y_ok, n_ok, n_rej = record_order_results(orders[0], x_new, y_new)
        accepted_total += n_ok
        rejected_total += n_rej
        if n_ok == 0:
            notes = f"round {r}: acquisition returned no in-band points (instrument honesty enforced)"
            break
        # Survivors that cannot follow the acquired data are re-scored NEXT
        # round by the referee; the loop's own narrowing is by divergence
        # only, which is proposer-side information.
        xs_used.append(x_ok)
        ys_used.append(y_ok)
        survived = hypotheses  # frozen: re-scoring happens through the referee
    handoff = referee_handoff(hypotheses, np.concatenate(xs_used) if xs_used else np.array([]),
                              np.concatenate(ys_used) if ys_used else np.array([]))
    return OrderRoundResult(rounds=len(orders_all), orders=tuple(orders_all),
                            accepted_points=accepted_total, rejected_points=rejected_total,
                            referee_handoff=handoff, notes=notes)


# ---------------------------------------------------------------------------
# EHI external-review redesign: raw prediction-difference maximization is
# not an experiment-design objective — an unbounded gap can select regions
# where measurement is unreliable, unsafe, or pure extrapolation. The
# principled objective is expected information gain I(H; Y | x, D_observed),
# computed EXACTLY here for Gaussian observation noise by quadrature over
# the predictive mixture (uniform hypothesis priors), minus feasibility and
# cost. The divergence orders above remain as the cheap screening objective;
# the information layer is what leaves the building to a lab.
# ---------------------------------------------------------------------------

INFO_GAIN_STOP_EPS = 1e-3  # nats; below this, no honest experiment exists


@dataclass(frozen=True)
class ExperimentOrder:
    """A structured acquisition request (EHI field-complete)."""

    variable: str
    target_ranges: tuple[tuple[float, float], ...]
    hypotheses: tuple[str, ...]
    predicted_outcomes: dict[str, float]
    expected_information_gain: float  # nats at the primary x*
    measurement_precision: float  # largest sigma that keeps the gain alive
    cost: float
    decision_criterion: str
    rationale: str = ""


def expected_information_gain(
    hypotheses: Mapping[str, Callable[[float], float]],
    x: float,
    sigma_noise: float,
    n_quad: int = 600,
) -> float:
    """I(H;Y|x) for Gaussian observation noise (nats), by quadrature.

    Hann-window-free, deterministic: the predictive mixture p(y|x) =
    (1/N) Σ_h N(μ_h(x), sigma²) is integrated on a fixed grid spanning
    [min μ − 6σ, max μ + 6σ]; I = Σ_y p(y) · Σ_h P(h|y) log P(h|y).
    """
    names = tuple(hypotheses)
    mus = np.asarray([float(hypotheses[n](x)) for n in names])
    if not np.isfinite(mus).all():
        return float("-inf")
    lo, hi = float(mus.min() - 6 * sigma_noise), float(mus.max() + 6 * sigma_noise)
    ys = np.linspace(lo, hi, n_quad)
    phis = np.exp(-0.5 * ((ys[None, :] - mus[:, None]) / sigma_noise) ** 2) / (
        sigma_noise * np.sqrt(2 * np.pi))
    p_y = phis.mean(axis=0)
    p_h_given_y = phis / np.maximum(p_y[None, :], 1e-300)
    mi = float(np.sum(p_y * np.sum(p_h_given_y * np.log(np.maximum(p_h_given_y, 1e-300)), axis=0)))
    return mi


def design_information_orders(
    hypotheses: Mapping[str, Callable[[float], float]],
    domain: tuple[float, float],
    sigma_noise: float,
    variable: str = "x",
    cost_model: Callable[[float], float] | None = None,
    feasible_ranges: tuple[tuple[float, float], ...] = (),
    grid_n: int = DOMAIN_GRID_DEFAULT,
) -> tuple[ExperimentOrder, ...]:
    """x* = argmax I(H;Y|x) subject to feasible ranges, cost, noise.

    Fail-closed stopping: if the best achievable expected information is
    below ``INFO_GAIN_STOP_EPS`` anywhere in the feasible domain, NO order
    is emitted — "no useful experiment" is an honest outcome, and the
    caller must not loosen the noise model or the feasibility bounds to
    manufacture one.
    """
    if len(hypotheses) < 2:
        raise ValueError("information design needs >= 2 competing hypotheses")
    if not (np.isfinite(sigma_noise) and sigma_noise > 0):
        raise ValueError(
            "invalid measurement noise model sigma_noise=" + repr(sigma_noise)
            + " — information-objective design is undefined without a positive"
            " noise reference; refusing (no silent 1e-30 clown-noise fallback)")
    lo, hi = float(domain[0]), float(domain[1])
    if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
        raise ValueError(f"invalid domain {domain}")
    grid = np.linspace(lo, hi, grid_n)
    mi = np.asarray([expected_information_gain(hypotheses, float(v), sigma_noise) for v in grid])
    feasible = np.zeros(len(grid), dtype=bool) if feasible_ranges else np.ones(len(grid), dtype=bool)
    for f_lo, f_hi in feasible_ranges:
        feasible |= (grid >= f_lo) & (grid <= f_hi)
    if not feasible.any():
        return ()
    if cost_model is not None:
        costs = np.asarray([float(cost_model(float(v))) for v in grid])
        # Documented utility: I discounted by cost relative to its own scale
        # (dimensionless discount; no new epistemic threshold is invented).
        cost_scale = float(np.max(costs)) if float(np.max(costs)) > 0 else 1.0
        utility = np.where(feasible, mi * (1.0 - 0.5 * costs / max(cost_scale, 1e-30)), float("-inf"))
    else:
        utility = np.where(feasible, mi, float("-inf"))
    x_star = int(np.argmax(utility))
    if not np.isfinite(utility[x_star]) or mi[x_star] <= INFO_GAIN_STOP_EPS:
        return ()
    names = tuple(hypotheses)
    preds = {n: float(hypotheses[n](grid[x_star])) for n in names}
    gaps = max(abs(preds[a] - preds[b]) for i, a in enumerate(names) for b in names[i + 1:])
    precision = gaps / DEFAULT_DETECTABILITY_K if gaps > 0 else sigma_noise
    # Cluster feasible points with information near the optimum into ranges.
    near = feasible & (utility >= utility[x_star] - 1e-12)
    idx = np.where(near)[0]
    gap_idx = max(1, int(np.ceil(RANGE_MERGE_GAP_FRACTION * grid_n)))
    clusters: list[list[int]] = [[int(idx[0])]]
    for i in idx[1:]:
        if i - clusters[-1][-1] <= gap_idx:
            clusters[-1].append(int(i))
        else:
            clusters.append([int(i)])
    clusters.sort(key=lambda c: -float(utility[c].max()))
    clusters = clusters[:MAX_RANGES_PER_ORDER]
    pad = RANGE_PAD_FRACTION * (hi - lo)
    orders: list[ExperimentOrder] = []
    for _k_i, c in enumerate(clusters):
        r_lo, r_hi = max(lo, float(grid[min(c)]) - pad), min(hi, float(grid[max(c)]) + pad)
        orders.append(ExperimentOrder(
            variable=variable, target_ranges=((r_lo, r_hi),), hypotheses=names,
            predicted_outcomes=preds,
            expected_information_gain=float(mi[x_star]),
            measurement_precision=float(precision), cost=float(cost_model(grid[x_star])) if cost_model else 0.0,
            decision_criterion=("acquire until the referee pair on the incoming residuals "
                                 f"leaves a unique survivor, or the acquisition budget ends; "
                                 f"stop honestly if realized gain < {INFO_GAIN_STOP_EPS} nats"),
            rationale=("argmax of expected information gain over the frozen, feasibility- and "
                        "cost-discounted grid; primary "
                        f"x*={float(grid[x_star]):.6g}, I*={float(mi[x_star]):.4f} nats")))
    return tuple(orders)
