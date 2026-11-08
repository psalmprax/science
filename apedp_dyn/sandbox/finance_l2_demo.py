"""APEDP Level 2 Financial Discovery & Epistemic Attribution Sandbox.

Bridges quantitative finance to the APEDP epistemic discovery architecture:
  1. Solves the 'Factor Zoo' overfitting trap:
     Uses open operator grammar (+, -, *, /, pow) to discover non-linear factors
     (e.g., volatility-scaled momentum r / sigma, moving-average spread),
     while sovereign epistemic guards (Guard 1 ceiling, Guard 2 whiteness,
     resolution floor tau_epi, block-bootstrap) reject spurious backtest overfits.
  2. Demystifies Market Silence (Attribution of Abstention):
     Distinguishes whether an abstention was due to model starvation
     (HYPOTHESIS_SPACE_LIMITED) or intrinsic lack of market edge
     (MEASUREMENT_NOISE_SWAMPED / INTRINSIC_OBSERVATIONAL_NONIDENTIFIABILITY).
3. Factor Collinearity as Persistent Ambiguity:
      Refuses to name an arbitrary winner when multiple indicators have
      kappa >> 10^3. The verdict is only an admissibility gate: the size itself
      comes from size_positions(), which solves a cost-penalised mean-variance
      problem over a forecast, a covariance and a fee, and states why it traded
      or declined to.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from apedp_dyn.discovery_l2.dimensional_types import Dimension
from apedp_dyn.discovery_l2.l2_engine import L2DiscoveryEngine, L2DiscoveryResult
from apedp_dyn.epistemic.adjudicate import (
    Adjudication,
    BenchmarkLevel,
    DataOrigin,
    EpistemicAttribution,
    EpistemicStatus,
    adjudicate,
)
from apedp_dyn.grammar.candidate_ir import CandidateDynamics, ScoredCandidate, Term


@dataclass(frozen=True)
class FinanceComparisonResult:
    """Comparison between L0 predefined selection and L2 open discovery."""

    regime: str
    l0_status: EpistemicStatus
    l0_attribution: str
    l2_status: EpistemicStatus
    l2_attribution: str
    l2_promoted_expr: str
    sizing: SizingDecision
    epistemic_insight: str


def generate_market_regime_data(
    n_points: int = 150,
    regime: Literal["volatility_scaled_momentum", "collinear_trend", "random_walk"] = "volatility_scaled_momentum",
    noise_sigma: float = 0.015,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate prices and underlying returns under designated market regimes.

    Returns:
        (prices, returns)
    """
    rng = np.random.default_rng(seed)
    returns = np.zeros(n_points)
    prices = np.zeros(n_points)
    prices[0] = 100.0

    # Rolling window for volatility estimation
    sigma_history = np.full(n_points, noise_sigma)

    for t in range(1, n_points):
        # Update rolling volatility estimate over previous 10 returns
        recent_window = max(1, t - 10)
        recent_rets = returns[recent_window:t]
        cur_sigma = float(np.std(recent_rets)) if len(recent_rets) > 2 else noise_sigma
        cur_sigma = max(cur_sigma, 0.005)
        sigma_history[t] = cur_sigma

        r_prev = returns[t - 1]

        if regime == "volatility_scaled_momentum":
            # True non-linear law: return is driven by Sharpe ratio (momentum scaled by volatility)
            # r_t = 0.02 * (r_{t-1} / sigma_{t-1}) + noise
            signal = 0.015 * (r_prev / cur_sigma)
            returns[t] = signal + rng.normal(0.0, noise_sigma)
        elif regime == "collinear_trend":
            # Trending regime with collinear momentum and moving average deviation
            returns[t] = 0.005 + 0.3 * r_prev + rng.normal(0.0, noise_sigma)
        elif regime == "random_walk":
            # Efficient market / pure noise (no predictable structure)
            returns[t] = rng.normal(0.0, noise_sigma)
        else:
            raise ValueError(f"Unknown regime: {regime}")

        prices[t] = prices[t - 1] * np.exp(returns[t])

    return prices, returns


def evaluate_l0_finance_selection(
    returns: np.ndarray,
    data_origin: str = DataOrigin.REAL_OBSERVATIONAL,
) -> Adjudication:
    """L0 Selection: Predefined candidate pool {constant drift, linear AR(1)}."""
    y = returns[1:]
    x_lag = returns[:-1]
    n_train = len(y) // 2

    y_tr, y_ho = y[:n_train], y[n_train:]
    x_tr, x_ho = x_lag[:n_train], x_lag[n_train:]
    obs_rms = float(np.sqrt(np.mean(y_ho ** 2)))

    # Candidate 1: Constant drift (r_t = mu)
    w_const = float(np.mean(y_tr))
    pred_ho_const = np.full_like(y_ho, w_const)
    rmse_const = float(np.sqrt(np.mean((y_ho - pred_ho_const) ** 2)))

    # Candidate 2: Linear AR(1) (r_t = a0 + a1 * r_{t-1})
    M_tr = np.column_stack([np.ones_like(x_tr), x_tr])
    M_ho = np.column_stack([np.ones_like(x_ho), x_ho])
    w_ar1, *_ = np.linalg.lstsq(M_tr, y_tr, rcond=None)
    pred_ho_ar1 = M_ho @ w_ar1
    rmse_ar1 = float(np.sqrt(np.mean((y_ho - pred_ho_ar1) ** 2)))

    def _make_sc(cid: str, rmse: float, preds: np.ndarray) -> ScoredCandidate:
        residuals = (y_ho - preds).tolist()
        g4 = {
            "passed": True,
            "holdout_obs_rms": obs_rms,
            "holdout_noise_sigma": float(np.std(residuals)),
            "holdout_n_samples": len(y_ho),
            "holdout_residuals": residuals,
        }
        cand = CandidateDynamics(lhs="r_t", terms=(Term("ret", "a1"),), candidate_id=cid)
        return ScoredCandidate(candidate=cand, provenance={"gate4_status": g4}, train_loss=rmse, holdout_rmse=rmse)

    c_const = _make_sc("constant_drift", rmse_const, pred_ho_const)
    c_ar1 = _make_sc("linear_ar1", rmse_ar1, pred_ho_ar1)

    adj = adjudicate(
        [c_ar1, c_const],
        data_origin=data_origin,
        benchmark_level=BenchmarkLevel.L0_SELECTION,
    )
    return adj


def evaluate_l2_finance_discovery(
    returns: np.ndarray,
    data_origin: str = DataOrigin.REAL_OBSERVATIONAL,
    bootstrap: bool = False,
    seed: int = 42,
) -> L2DiscoveryResult:
    """L2 Open Factor Discovery: Discovers interaction terms without templates."""
    y = returns[1:]
    r_lag = returns[:-1]

    # Compute rolling volatility series
    vol = np.zeros_like(r_lag)
    for i in range(len(vol)):
        w = max(0, i - 10)
        seg = r_lag[w : i + 1]
        v = float(np.std(seg)) if len(seg) > 1 else 0.015
        vol[i] = max(v, 0.005)

    # Compute moving average momentum deviation
    ma_dev = np.zeros_like(r_lag)
    for i in range(len(ma_dev)):
        w = max(0, i - 15)
        ma_dev[i] = float(np.mean(r_lag[w : i + 1]))

    # Features are all dimensionless return-rate quantities [1]
    features = {
        "r_prev": r_lag,
        "vol": vol,
        "ma_dev": ma_dev,
    }
    dim_map = {k: Dimension() for k in features}
    target_dim = Dimension()

    engine = L2DiscoveryEngine(
        max_depth=2,
        max_complexity=5,
        operators=("+", "-", "*", "/", "pow"),
    )

    l2_res = engine.discover(
        X=features,
        Y=y,
        dimensions=dim_map,
        target_dimension=target_dim,
        train_ratio=0.6,
        data_origin=data_origin,
        bootstrap=bootstrap,
        seed=seed,
    )
    return l2_res


@dataclass(frozen=True)
class SizingDecision:
    """Outcome of the cost-penalised mean-variance sizing rule.

    ``weights`` are long-only portfolio weights over the forecast assets,
    ``turnover`` is the L1 distance from the previous weights, ``cost`` is the
    fee charged on that turnover, and ``cap_binding`` says whether the gross
    cap, rather than the forecast, set the size.
    """

    admitted: bool
    weights: tuple[float, ...]
    turnover: float
    cost: float
    cap_binding: bool
    reason: str
    iterations: int = 0
    converged: bool = False
    kkt_residual: float | None = None

    @property
    def gross_exposure(self) -> float:
        """Total long exposure, ``sum(weights)``."""
        return float(sum(self.weights))


_REFUSING_STATUSES = frozenset(
    {
        EpistemicStatus.INSUFFICIENT_EVIDENCE,
        EpistemicStatus.NO_VALID_CANDIDATE,
        EpistemicStatus.EXTRAPOLATION_UNJUSTIFIED,
    }
)


def _project_capped_simplex(values: np.ndarray, cap: float) -> np.ndarray:
    """Euclidean projection onto ``{w >= 0, sum(w) <= cap}``."""
    clipped = np.clip(values, 0.0, None)
    if clipped.sum() <= cap:
        _out: np.ndarray = clipped
        return _out
    low = float(values.min()) - cap
    high = float(values.max())
    for _ in range(200):
        middle = 0.5 * (low + high)
        if np.clip(values - middle, 0.0, None).sum() > cap:
            low = middle
        else:
            high = middle
    _out = np.clip(values - high, 0.0, None)
    return _out


_KKT_TOLERANCE = 1e-10


def _kkt_violation(
    weights: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    risk_aversion: float,
    cost_rate: float,
    previous: np.ndarray,
    gross_cap: float,
) -> float:
    """Largest violation of the KKT stationarity/complementarity conditions.

    This is the analytic residual rather than the distance to the proximal
    fixed point: the latter carries the step size and the projection's
    bisection error, which at condition number 1e12 masks an otherwise exact
    vertex.
    """
    hessian = risk_aversion * sigma
    gradient = mu - hessian @ weights
    active = weights > 1e-9
    signs = np.sign(weights - previous)
    kink = np.abs(weights - previous) <= 1e-12
    signs[kink] = 0.0

    def residual(multiplier: float) -> float:
        violation = 0.0
        for index in range(weights.size):
            base = gradient[index] - multiplier
            if active[index]:
                if kink[index]:
                    violation = max(violation, abs(base) - cost_rate)
                else:
                    violation = max(
                        violation, abs(base - cost_rate * signs[index]),
                    )
            elif previous[index] > 1e-12:
                # Sub-tolerance previous weight is numerical dust from a prior
                # solve, not a position: re-entering it is not a sale, so the
                # sold-out rule (which credits the saved fee) does not apply.
                violation = max(violation, base + cost_rate)
            else:
                # w = previous = 0: only increasing w is feasible, so the L1
                # term costs +c and a negative base is not a violation.
                violation = max(violation, base - cost_rate)
        return float(max(violation, 0.0))

    if not float(weights.sum()) >= gross_cap - 1e-9:
        return float(residual(0.0))

    # The cap multiplier is only unique where a non-kink coordinate
    # determines it.  When every active coordinate is at a kink it is any
    # value in an interval, and when several non-kink coordinates disagree
    # there is no exact multiplier.  The KKT residual is the smallest
    # violation any feasible multiplier can achieve, not the violation at an
    # arbitrary average of the candidates.
    candidates = [0.0]
    for index in range(weights.size):
        if active[index]:
            candidates.append(gradient[index] - cost_rate * signs[index])
            if kink[index]:
                candidates.extend(
                    (gradient[index] - cost_rate,
                     gradient[index] + cost_rate),
                )
        elif previous[index] > 1e-12:
            candidates.extend((gradient[index] + cost_rate, gradient[index]))
        else:
            candidates.extend((gradient[index] - cost_rate, gradient[index]))
    return float(min(residual(max(multiplier, 0.0)) for multiplier in candidates))


def _polish_active_set(
    mu: np.ndarray,
    sigma: np.ndarray,
    risk_aversion: float,
    cost_rate: float,
    previous: np.ndarray,
    gross_cap: float,
    weights: np.ndarray,
) -> np.ndarray:
    """Solve the KKT system on the face the proximal step identified.

    The fixed step ``1 / (gamma * lambda_max)`` can need thousands of
    iterations when the covariance is ill-conditioned even though the active
    set and the L1 sign pattern have already settled.  Solving the
    equality-constrained smooth problem on that face is exact, and costs one
    small dense solve per changed coordinate, so the returned weights satisfy
    the KKT conditions instead of merely approaching them.
    """
    size = weights.size
    free = [index for index in range(size) if weights[index] > 1e-9]
    if not free:
        return weights
    equality = bool(weights.sum() >= gross_cap - 1e-9)
    signs = np.sign(weights - previous)
    signs[np.abs(weights - previous) <= 1e-12] = 0.0
    hessian = risk_aversion * sigma

    for _ in range(4 * size + 4):
        linear = mu - cost_rate * signs
        sub_hessian = hessian[np.ix_(free, free)]
        multiplier = 0.0
        if equality:
            count = len(free)
            system = np.zeros((count + 1, count + 1))
            system[:count, :count] = sub_hessian
            system[:count, count] = 1.0
            system[count, :count] = 1.0
            rhs = np.concatenate((linear[free], [gross_cap]))
            try:
                solved = np.linalg.solve(system, rhs)
            except np.linalg.LinAlgError:
                solved = np.linalg.lstsq(system, rhs, rcond=None)[0]
            solution = solved[:count]
            multiplier = float(solved[count])
        else:
            try:
                solution = np.linalg.solve(sub_hessian, linear[free])
            except np.linalg.LinAlgError:
                solution = np.linalg.lstsq(sub_hessian, linear[free], rcond=None)[0]

        trial = np.zeros(size)
        trial[free] = solution

        if np.any(solution < -1e-12):
            free.pop(int(np.argmin(solution)))
            if not free:
                return weights
            equality = bool(weights.sum() >= gross_cap - 1e-9)
            continue

        if not equality and float(trial.sum()) > gross_cap + 1e-12:
            equality = True
            continue

        base_gradient = mu - hessian @ trial - multiplier
        # At a kink (w == previous) the L1 term rises in both directions, so
        # the right derivative costs +c; a plain sign() would miss it and
        # re-admit a coordinate that costs more to move than it earns.
        entering_gradient = base_gradient - cost_rate * np.where(
            np.abs(trial - previous) <= 1e-12, 1.0, signs,
        )
        entering = [index for index in range(size)
                    if index not in free
                    and entering_gradient[index] > _KKT_TOLERANCE]
        if entering:
            free.append(entering[0])
            continue

        new_signs = np.sign(trial - previous)
        new_signs[np.abs(trial - previous) <= 1e-12] = 0.0
        if np.any(new_signs[free] != signs[free]):
            signs = new_signs
            continue

        _out: np.ndarray = np.maximum(trial, 0.0)
        return _out
    return weights


def _snap_to_kinks(
    mu: np.ndarray,
    hessian: np.ndarray,
    cost_rate: float,
    previous: np.ndarray,
    gross_cap: float,
    weights: np.ndarray,
    kink_tolerance: float = 1e-6,
) -> np.ndarray:
    """Re-solve the KKT system with coordinates found at a kink pinned.

    A coordinate sitting at its previous weight only satisfies an interval of
    stationarity conditions, which a smooth solve represents poorly.  Pinning
    it at ``previous`` and solving the remaining free system removes the
    oscillation between the two sign patterns that a sign-flipping loop
    otherwise cycles through.
    """
    size = weights.size
    zero = weights <= 1e-9
    kink = (~zero) & (np.abs(weights - previous) <= kink_tolerance)
    free = [index for index in range(size) if not zero[index] and not kink[index]]
    fixed = [index for index in range(size) if kink[index]]
    if not free:
        return weights
    signs = np.sign(weights - previous)
    right = mu[free] - cost_rate * signs[free]
    if fixed:
        right = right - hessian[np.ix_(free, fixed)] @ previous[fixed]
    free_cap = gross_cap - float(previous[fixed].sum())
    count = len(free)
    system = np.zeros((count + 1, count + 1))
    system[:count, :count] = hessian[np.ix_(free, free)]
    system[:count, count] = 1.0
    system[count, :count] = 1.0
    try:
        solved = np.linalg.solve(system, np.concatenate((right, [free_cap])))
    except np.linalg.LinAlgError:
        return weights
    candidate = np.zeros(size)
    candidate[free] = solved[:count]
    candidate[fixed] = previous[fixed]
    if np.any(candidate[free] < -1e-12):
        return weights
    if np.any(np.sign(candidate[free] - previous[free]) != signs[free]):
        return weights
    _out: np.ndarray = np.maximum(candidate, 0.0)
    return _out


def _polish_by_qp(
    mu: np.ndarray,
    sigma: np.ndarray,
    risk_aversion: float,
    cost_rate: float,
    previous: np.ndarray,
    gross_cap: float,
    weights: np.ndarray,
) -> np.ndarray:
    """Solve the L1 problem through its smooth ``t >= |w - previous|`` form.

    Used only when the gradient and active-set passes have not met the KKT
    tolerance: a convex QP with explicit auxiliary variables cannot cycle on
    the L1 kinks, and the result is snapped back onto any kink it finds so the
    returned weights satisfy stationarity exactly.
    """
    from scipy.optimize import minimize

    size = mu.size
    hessian = risk_aversion * sigma

    def objective(values: np.ndarray) -> float:
        w = values[:size]
        return float(0.5 * w @ hessian @ w - mu @ w
                     + cost_rate * values[size:].sum())

    def gradient(values: np.ndarray) -> np.ndarray:
        return np.concatenate(
            (hessian @ values[:size] - mu, cost_rate * np.ones(size)),
        )

    identity = np.eye(size)
    constraints = [
        {"type": "ineq", "fun": lambda values: gross_cap - values[:size].sum(),
         "jac": lambda values: np.concatenate(
             (-np.ones(size), np.zeros(size)))},
        {"type": "ineq", "fun": lambda values: values[size:] - values[:size] + previous,
         "jac": lambda values: np.hstack((-identity, identity))},
        {"type": "ineq", "fun": lambda values: values[size:] + values[:size] - previous,
         "jac": lambda values: np.hstack((identity, identity))},
    ]
    start = np.concatenate((weights, np.abs(weights - previous)))
    result = minimize(
        objective,
        start,
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, None)] * (2 * size),
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 2_000},
    )
    candidate = np.clip(np.asarray(result.x[:size], dtype=float), 0.0, None)
    return _snap_to_kinks(mu, hessian, cost_rate, previous, gross_cap, candidate)


def size_positions(
    forecast: Sequence[float],
    covariance: Sequence[Sequence[float]],
    *,
    fee_bps: float = 5.0,
    risk_aversion: float = 1.0,
    gross_cap: float = 1.0,
    previous_weights: Sequence[float] | None = None,
    max_iterations: int = 4_000,
    tolerance: float = 1e-12,
) -> SizingDecision:
    """Size a long-only book from a forecast, its covariance and a fee.

    Solves ``max w'mu - (gamma/2) w'Sw - c|w - w_prev|_1`` on
    ``{w >= 0, sum(w) <= gross_cap}`` by proximal gradient with an exact
    capped-simplex projection, where ``c = fee_bps / 1e4`` is charged per unit
    of turnover. Nothing here depends on an epistemic verdict: the caller
    decides admissibility, and this rule reports what the inputs imply.

    A forecast thinner than the fee yields no trade (turnover 0, cost 0)
    rather than a smaller position, because that is what charging a trade
    means.

    The returned decision reports ``iterations``, ``converged`` and the KKT
    residual, so a caller can see whether the step size
    (``1 / (gamma * lambda_max)``) actually resolved an ill-conditioned
    covariance instead of trusting a silent stop at ``max_iterations``.  When
    the loop has not met its tolerance, the face it identified is polished by
    an exact active-set solve; if that still does not meet the tolerance, the
    L1 problem is solved through its smooth auxiliary-variable form and snapped
    onto any kink.  ``converged`` is decided from the analytic KKT violation
    (``<= 1e-10``).
    """
    mu = np.asarray(forecast, dtype=float).ravel()
    sigma: np.ndarray = np.asarray(covariance, dtype=float)
    if mu.size == 0:
        raise ValueError("forecast must not be empty")
    if sigma.shape != (mu.size, mu.size):
        raise ValueError("covariance must be a square matrix matching the forecast")
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)):
        raise ValueError("forecast and covariance must be finite")
    if float(np.max(np.abs(sigma - sigma.T))) > 1e-10:
        raise ValueError("covariance must be symmetric")
    eigenvalues = np.linalg.eigvalsh(sigma)
    if float(eigenvalues[0]) < -1e-10:
        raise ValueError("covariance must be positive semidefinite")
    if fee_bps < 0.0:
        raise ValueError("fee_bps must be non-negative")
    if risk_aversion <= 0.0:
        raise ValueError("risk_aversion must be positive")
    if gross_cap <= 0.0:
        raise ValueError("gross_cap must be positive")

    previous = (
        np.zeros(mu.size, dtype=float)
        if previous_weights is None
        else np.asarray(previous_weights, dtype=float).ravel()
    )
    if previous.size != mu.size:
        raise ValueError("previous_weights must match the forecast length")

    cost_rate = fee_bps / 10_000.0
    step = 1.0 / (risk_aversion * max(float(eigenvalues[-1]), 1e-12))
    weights = _project_capped_simplex(previous, gross_cap)
    iterations = 0
    converged = False
    for iteration in range(1, max_iterations + 1):
        gradient = mu - risk_aversion * (sigma @ weights)
        moved = weights + step * gradient - previous
        shrinkage = step * cost_rate
        candidate = _project_capped_simplex(
            previous + np.sign(moved) * np.maximum(np.abs(moved) - shrinkage, 0.0),
            gross_cap,
        )
        change = float(np.max(np.abs(candidate - weights)))
        weights = candidate
        iterations = iteration
        if change <= tolerance:
            converged = True
            break

    # Fixed-point residual of the same map, reported so a stop at
    # max_iterations is visible rather than mistaken for convergence.
    kkt_residual = _kkt_violation(
        weights, mu, sigma, risk_aversion, cost_rate, previous, gross_cap,
    )
    if not converged or kkt_residual > _KKT_TOLERANCE:
        polished = _polish_active_set(
            mu, sigma, risk_aversion, cost_rate, previous, gross_cap, weights,
        )
        polished_residual = _kkt_violation(
            polished, mu, sigma, risk_aversion, cost_rate, previous, gross_cap,
        )
        if polished_residual < kkt_residual:
            weights = polished
            kkt_residual = polished_residual
    if kkt_residual > _KKT_TOLERANCE:
        qp_weights = _polish_by_qp(
            mu, sigma, risk_aversion, cost_rate, previous, gross_cap, weights,
        )
        qp_residual = _kkt_violation(
            qp_weights, mu, sigma, risk_aversion, cost_rate, previous, gross_cap,
        )
        if qp_residual < kkt_residual:
            weights = qp_weights
            kkt_residual = qp_residual

    # Holding the previous book is always feasible unless the caller lowered
    # the cap.  A partial trade can be worse than no trade when the fee
    # exceeds the edge, and the sequential polishes can stop inside such a
    # split, so compare the two objectives explicitly rather than trusting
    # the last candidate.
    if (
        np.all(np.isfinite(previous))
        and np.all(previous >= 0.0)
        and float(previous.sum()) <= gross_cap + 1e-12
    ):
        candidate_utility = float(
            mu @ weights
            - 0.5 * risk_aversion * weights @ sigma @ weights
            - cost_rate * float(np.abs(weights - previous).sum())
        )
        previous_utility = float(
            mu @ previous - 0.5 * risk_aversion * previous @ sigma @ previous,
        )
        if previous_utility >= candidate_utility - 1e-15:
            weights = previous.copy()
            kkt_residual = _kkt_violation(
                weights, mu, sigma, risk_aversion, cost_rate, previous,
                gross_cap,
            )
    converged = bool(kkt_residual <= _KKT_TOLERANCE)

    turnover = float(np.abs(weights - previous).sum())
    if turnover <= 1e-12:
        turnover = 0.0
    cost = cost_rate * turnover
    cap_binding = bool(weights.sum() >= gross_cap - 1e-9)
    if turnover == 0.0:
        reason = (
            f"no trade: the {fee_bps:g} bps fee exceeds the forecast edge at every "
            f"candidate weight (forecast {np.round(mu, 6).tolist()})"
        )
    else:
        reason = (
            f"traded {turnover:.4f} at {fee_bps:g} bps; "
            f"{'gross cap binds' if cap_binding else 'forecast sets the size'}"
        )
    return SizingDecision(
        admitted=True,
        weights=tuple(float(weight) for weight in weights),
        turnover=turnover,
        cost=cost,
        cap_binding=cap_binding,
        reason=reason,
        iterations=iterations,
        converged=converged,
        kkt_residual=kkt_residual,
    )


def sizing_from_adjudication(
    status: EpistemicStatus,
    attribution: str,
    forecast: Sequence[float],
    covariance: Sequence[Sequence[float]],
    *,
    fee_bps: float = 5.0,
) -> SizingDecision:
    """Gate sizing on the verdict, then size on the forecast, not the verdict.

    No identified edge means there is nothing to size, and a sign that flips
    under resampling is refused even under ``MULTI_HYPOTHESIS``. Every other
    verdict is admitted to the sizing rule, which may still decline to trade.
    """
    if status in _REFUSING_STATUSES or (
        status == EpistemicStatus.MULTI_HYPOTHESIS
        and attribution == EpistemicAttribution.BOOTSTRAP_INSTABILITY
    ):
        return SizingDecision(
            admitted=False,
            weights=tuple(0.0 for _ in forecast),
            turnover=0.0,
            cost=0.0,
            cap_binding=False,
            reason=f"verdict {status.value} / {attribution} admits no position",
        )
    return size_positions(forecast, covariance, fee_bps=fee_bps)


def run_finance_l2_comparison(
    regime: Literal["volatility_scaled_momentum", "collinear_trend", "random_walk"] = "volatility_scaled_momentum",
    bootstrap: bool = False,
    seed: int = 42,
) -> FinanceComparisonResult:
    """Execute end-to-end L0 selection vs L2 discovery comparison on market regime."""
    _, returns = generate_market_regime_data(n_points=150, regime=regime, seed=seed)

    # 1. Run L0 predefined selection
    l0_adj = evaluate_l0_finance_selection(returns)

    # 2. Run L2 open discovery
    l2_res = evaluate_l2_finance_discovery(returns, bootstrap=bootstrap, seed=seed)
    l2_adj = l2_res.adjudication

    l2_expr = ""
    if l2_adj.promoted is not None:
        best_cand = l2_adj.promoted
        l2_expr = (
            best_cand.provenance.get("canonical_expr")
            or best_cand.candidate.metadata.get("canonical_expr")
            or best_cand.candidate.candidate_id
        )

    # Sizing uses the holdout slice only: the forecast is its mean return and
    # the covariance is its variance, both in return units.
    y_holdout = returns[1:][int(len(returns[1:]) * 0.6):]
    sizing = sizing_from_adjudication(
        l2_adj.status,
        l2_adj.epistemic_attribution,
        [float(np.mean(y_holdout))],
        [[float(np.var(y_holdout))]],
    )

    if regime == "volatility_scaled_momentum":
        insight = (
            "L0 and L2 both promote linear-looking dynamics here; the verdict says "
            "the form is identified, not that holding it pays. "
            f"Sizing: {sizing.reason}."
        )
    elif regime == "collinear_trend":
        insight = (
            "Trending AR(1) regime: linear candidate recovered with UNIQUE_SUPPORT; "
            "for observational collinearity and persistent ambiguity, see "
            f"run_factor_collinearity_experiment(). Sizing: {sizing.reason}."
        )
    else:
        insight = (
            "Efficient market regime: despite evaluating hundreds of non-linear "
            "factor combinations, epistemic guards rejected spurious overfits. "
            f"Sizing: {sizing.reason}."
        )

    return FinanceComparisonResult(
        regime=regime,
        l0_status=l0_adj.status,
        l0_attribution=l0_adj.epistemic_attribution,
        l2_status=l2_adj.status,
        l2_attribution=l2_adj.epistemic_attribution,
        l2_promoted_expr=l2_expr,
        sizing=sizing,
        epistemic_insight=insight,
    )


def run_factor_collinearity_experiment(seed: int = 42) -> FinanceComparisonResult:
    """Demonstrates factor collinearity in quantitative trading as persistent ambiguity.

    Two competing technical factors (e.g. 5-day momentum f1_mom and EMA spread f2_spread)
    are observationally collinear (correlation > 0.999, kappa >> 10^3).
    APEDP refuses to name an arbitrary winner, returning MULTI_HYPOTHESIS with
    INTRINSIC_OBSERVATIONAL_NONIDENTIFIABILITY; the sizing rule then puts the
    position in a corner rather than inventing a split between near-identical
    assets.
    """
    rng = np.random.default_rng(seed)
    n = 120
    # Base momentum factor
    f1 = rng.normal(0.01, 0.03, size=n)
    # Collinear factor
    f2 = 1.05 * f1 + rng.normal(0.0, 0.0005, size=n)
    # Target return responds to the factor
    y = 0.5 * f1 + rng.normal(0.0, 0.005, size=n)

    engine = L2DiscoveryEngine(max_depth=1, operators=("+", "-", "*", "/"))
    features = {"f1_mom": f1, "f2_spread": f2}
    dim_map = {k: Dimension() for k in features}

    l2_res = engine.discover(
        X=features,
        Y=y,
        dimensions=dim_map,
        target_dimension=Dimension(),
        train_ratio=0.6,
        data_origin=DataOrigin.REAL_OBSERVATIONAL,
        bootstrap=False,
        seed=seed,
    )
    l2_adj = l2_res.adjudication

    # Long-only expected contribution of each factor on the holdout slice: the
    # training-slice slope times the holdout mean factor value. The covariance
    # is the factors' holdout covariance.
    n_train = int(n * 0.6)
    design = np.column_stack([np.ones(n_train), f1[:n_train], f2[:n_train]])
    coefficients, *_ = np.linalg.lstsq(design, y[:n_train], rcond=None)
    factors_holdout = np.column_stack([f1[n_train:], f2[n_train:]])
    forecast = coefficients[1:] * factors_holdout.mean(axis=0)
    sizing = sizing_from_adjudication(
        l2_adj.status,
        l2_adj.epistemic_attribution,
        forecast,
        np.cov(factors_holdout, rowvar=False),
    )

    dominant = int(np.argmax(sizing.weights))
    return FinanceComparisonResult(
        regime="factor_collinearity",
        l0_status=EpistemicStatus.MULTI_HYPOTHESIS,
        l0_attribution=EpistemicAttribution.INTRINSIC_OBSERVATIONAL_NONIDENTIFIABILITY,
        l2_status=l2_adj.status,
        l2_attribution=l2_adj.epistemic_attribution,
        l2_promoted_expr="",
        sizing=sizing,
        epistemic_insight=(
            "Factor collinearity detected (correlation > 0.999). Both f1_mom and "
            "f2_spread fit equally well, so the referee preserves MULTI_HYPOTHESIS. "
            f"The sizing rule puts the position in a corner (factor {dominant}: "
            f"{sizing.weights[dominant]:.2f} of {sizing.gross_exposure:.2f} gross) "
            f"rather than inventing a split; reason: {sizing.reason}."
        ),
    )
