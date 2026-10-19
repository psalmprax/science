"""Epistemic Referee contract for continuous ODE and coupled dynamical system validation.

Adheres strictly to AGENTS.md:
- Never inspects development data during hypothesis generation.
- Evaluates candidate ODE rollout on the sealed holdout trajectory.
- Enforces forward trajectory stability, worst-state R^2 floor, and residual whiteness.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np

from apedp_dyn.epistemic.coupled_discovery import CoupledSystemDiscoveryResult
from apedp_dyn.ode.data import OdeTrajectoryData

_UNSTABLE_STATE_MAGNITUDE = 1e12


class EpistemicOdeVerdict(str, Enum):
    """Epistemic taxonomy for continuous dynamical systems."""
    IDENTIFIED_DYNAMICAL_LAW = "IDENTIFIED_DYNAMICAL_LAW"
    PREDICTIVE_WRONG_STRUCTURE = "PREDICTIVE_WRONG_STRUCTURE"
    UNSTABLE_FORWARD_SIMULATION = "UNSTABLE_FORWARD_SIMULATION"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True)
class EpistemicOdeAdjudication:
    verdict: EpistemicOdeVerdict
    mean_holdout_r2: float
    worst_state_r2: float
    is_stable: bool
    rational_justification: str


def _parse_monomial_factors(
    term: str, name_to_index: dict[str, int], state_variable: str
) -> list[tuple[str, int]]:
    """Parse ``x*x^2*y`` into ``[("x", 1), ("x", 2), ("y", 1)]``.

    Raises on any factor that is not a known state with a positive integer
    power (fail closed: no silent reinterpretation of malformed terms).
    """
    factors = [] if term in ("1", "const", "") else term.split("*")
    parsed: list[tuple[str, int]] = []
    for factor in factors:
        base, sep, power = factor.partition("^")
        exponent = 1
        if sep:
            try:
                exponent = int(power)
            except ValueError as exc:
                raise ValueError(
                    f"Term '{term}' in equation '{state_variable}' references "
                    f"non-integer power '{factor}'."
                ) from exc
            if exponent <= 0:
                raise ValueError(
                    f"Term '{term}' in equation '{state_variable}' references "
                    f"non-positive power '{factor}'."
                )
        if base not in name_to_index:
            raise ValueError(
                f"Term '{term}' in equation '{state_variable}' references unknown "
                f"state '{base}' (available: {sorted(name_to_index)})."
            )
        parsed.append((base, exponent))
    return parsed


def _build_rhs_terms(
    result: CoupledSystemDiscoveryResult, name_to_index: dict[str, int]
) -> list[tuple[str, list[tuple[list[tuple[str, int]], float]]]]:
    """Translate discovered equations into evaluable monomial term lists."""
    rhs_terms: list[tuple[str, list[tuple[list[tuple[str, int]], float]]]] = []
    for eq in result.equations:
        if eq.state_variable not in name_to_index:
            raise ValueError(
                f"Discovered equation for state '{eq.state_variable}' has no matching "
                f"holdout trajectory state (available: {sorted(name_to_index)})."
            )
        terms: list[tuple[list[tuple[str, int]], float]] = []
        for term, coeff in zip(eq.active_terms, eq.coefficients, strict=False):
            terms.append(
                (_parse_monomial_factors(term, name_to_index, eq.state_variable), float(coeff))
            )
        rhs_terms.append((eq.state_variable, terms))
    return rhs_terms


def _evaluate_rhs(
    rhs_terms: list[tuple[str, list[tuple[list[tuple[str, int]], float]]]],
    y: np.ndarray,
    name_to_index: dict[str, int],
) -> np.ndarray:
    out = np.zeros(len(rhs_terms), dtype=float)
    for i, (_, terms) in enumerate(rhs_terms):
        acc = 0.0
        for factors, coeff in terms:
            prod = 1.0
            for base, exponent in factors:
                value = y[name_to_index[base]]
                # Squaring is the dominant overflow risk at the rollout's
                # instability boundary; overflow is caught downstream by the
                # non-finite guard in _rk4_rollout.
                with np.errstate(over="ignore", invalid="ignore"):
                    prod *= float(value) ** exponent
            acc += coeff * prod
        out[i] = acc
    return out


def _rk4_rollout(
    rhs_terms: list[tuple[str, list[tuple[list[str], float]]]],
    name_to_index: dict[str, int],
    y0: np.ndarray,
    dt: float,
    n_steps: int,
) -> tuple[np.ndarray, bool]:
    """Fixed-step RK4 rollout of the discovered system from the holdout IC."""
    sim = np.empty((n_steps + 1, len(y0)), dtype=float)
    sim[0] = y0
    y = np.array(y0, dtype=float)
    for k in range(n_steps):
        with np.errstate(over="ignore", invalid="ignore"):
            k1 = _evaluate_rhs(rhs_terms, y, name_to_index)
            k2 = _evaluate_rhs(rhs_terms, y + 0.5 * dt * k1, name_to_index)
            k3 = _evaluate_rhs(rhs_terms, y + 0.5 * dt * k2, name_to_index)
            k4 = _evaluate_rhs(rhs_terms, y + dt * k3, name_to_index)
            y = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if not np.all(np.isfinite(y)) or float(np.max(np.abs(y))) > _UNSTABLE_STATE_MAGNITUDE:
            sim[k + 1:] = np.nan
            return sim, False
        sim[k + 1] = y
    return sim, True


def _r_squared(observed: np.ndarray, simulated: np.ndarray) -> float:
    residual = float(np.sum((observed - simulated) ** 2))
    total = float(np.sum((observed - np.mean(observed)) ** 2))
    if total <= 1e-18:
        return 1.0 if residual <= 1e-18 else float("-inf")
    return 1.0 - residual / total


class OdeReferee:
    """Epistemic referee adjudicating ODE candidate hypotheses.

    The verdict is computed exclusively from a forward RK4 rollout of the
    discovered system on the sealed holdout trajectory. The discovery-side
    in-sample fit statistic is never used as holdout evidence.
    """

    def __init__(
        self,
        min_mean_r2: float = 0.95,
        min_worst_r2: float = 0.85,
    ):
        self._min_mean_r2 = min_mean_r2
        self._min_worst_r2 = min_worst_r2

    def adjudicate_result(
        self,
        result: CoupledSystemDiscoveryResult,
        holdout_data: OdeTrajectoryData | Mapping[str, Sequence[float]],
    ) -> EpistemicOdeAdjudication:
        """Adjudicate a discovered coupled system by forward-integrating it on the sealed holdout.

        ``holdout_data`` must be an :class:`OdeTrajectoryData`; a plain mapping is
        interpreted with unit timestep (documented fallback for callers without a
        time vector). The holdout trajectory must have been held out of discovery.
        """
        if isinstance(holdout_data, OdeTrajectoryData):
            holdout = holdout_data
        elif isinstance(holdout_data, Mapping):
            holdout = OdeTrajectoryData.from_dict(holdout_data, dt=1.0)
        else:
            raise TypeError(
                f"holdout_data must be OdeTrajectoryData or a state mapping, got {type(holdout_data).__name__}"
            )

        states = {name: np.asarray(values, dtype=float) for name, values in holdout.states.items()}
        n_t = len(holdout.t)
        if not result.equations:
            return EpistemicOdeAdjudication(
                verdict=EpistemicOdeVerdict.INSUFFICIENT_EVIDENCE,
                mean_holdout_r2=float("nan"),
                worst_state_r2=float("nan"),
                is_stable=False,
                rational_justification="Discovery result contains no equations to adjudicate.",
            )
        if n_t < 2:
            return EpistemicOdeAdjudication(
                verdict=EpistemicOdeVerdict.INSUFFICIENT_EVIDENCE,
                mean_holdout_r2=float("nan"),
                worst_state_r2=float("nan"),
                is_stable=False,
                rational_justification=f"Holdout trajectory has {n_t} point(s); forward rollout requires >= 2.",
            )

        # The simulation state vector is ordered by the discovered equations, but
        # the holdout supplies states as a mapping whose *insertion order* need
        # not match. Indexing with holdout-dict positions therefore scrambles
        # coupled dynamics whenever the engine emits equations in a different
        # order, raises IndexError when the holdout carries extra variables, and
        # silently certifies half the physics when the discovery result misses
        # holdout vars entirely. Fail closed on any set mismatch, then index y
        # strictly in the discovered equation sequence.
        discovered_states = {eq.state_variable for eq in result.equations}
        holdout_states = set(states)
        if discovered_states != holdout_states:
            missing = sorted(holdout_states - discovered_states)
            extra = sorted(discovered_states - holdout_states)
            return EpistemicOdeAdjudication(
                verdict=EpistemicOdeVerdict.INSUFFICIENT_EVIDENCE,
                mean_holdout_r2=float("nan"),
                worst_state_r2=float("nan"),
                is_stable=False,
                rational_justification=(
                    "Discovered state set does not match the holdout state set; "
                    f"undiscovered holdout states: {missing or 'none'}; "
                    f"equations without a holdout trajectory: {extra or 'none'}. "
                    "No partial system can be certified as a complete dynamical law."
                ),
            )

        # State vector and factor lookups both use the discovered equation order.
        state_to_index = {eq.state_variable: idx for idx, eq in enumerate(result.equations)}
        rhs_terms = _build_rhs_terms(result, state_to_index)
        equation_states = [state for state, _ in rhs_terms]
        y0 = np.array([states[state][0] for state in equation_states], dtype=float)

        sim, stable = _rk4_rollout(rhs_terms, state_to_index, y0, float(holdout.dt), n_t - 1)
        if not stable:
            return EpistemicOdeAdjudication(
                verdict=EpistemicOdeVerdict.UNSTABLE_FORWARD_SIMULATION,
                mean_holdout_r2=float("nan"),
                worst_state_r2=float("nan"),
                is_stable=False,
                rational_justification=(
                    "Forward integration of the discovered system diverged "
                    f"(non-finite or |state| > {_UNSTABLE_STATE_MAGNITUDE:g}) on the holdout trajectory."
                ),
            )

        per_state_r2 = {
            state: _r_squared(states[state], sim[:, j]) for j, state in enumerate(equation_states)
        }
        mean_r2 = float(np.mean(list(per_state_r2.values())))
        worst_r2 = float(np.min(list(per_state_r2.values())))
        metrics = ", ".join(f"{name} R^2={value:.6f}" for name, value in per_state_r2.items())

        if not np.isfinite(mean_r2) or mean_r2 < self._min_mean_r2:
            return EpistemicOdeAdjudication(
                verdict=EpistemicOdeVerdict.PREDICTIVE_WRONG_STRUCTURE,
                mean_holdout_r2=mean_r2,
                worst_state_r2=worst_r2,
                is_stable=True,
                rational_justification=(
                    f"Holdout rollout mean R^2 ({mean_r2:.4f}) below threshold {self._min_mean_r2} "
                    f"[{metrics}]. In-sample kernel fit R^2={result.mean_system_r2:.4f} not used as evidence."
                ),
            )

        if not np.isfinite(worst_r2) or worst_r2 < self._min_worst_r2:
            return EpistemicOdeAdjudication(
                verdict=EpistemicOdeVerdict.PREDICTIVE_WRONG_STRUCTURE,
                mean_holdout_r2=mean_r2,
                worst_state_r2=worst_r2,
                is_stable=True,
                rational_justification=(
                    f"Holdout rollout worst-state R^2 ({worst_r2:.4f}) below threshold {self._min_worst_r2} "
                    f"[{metrics}]. In-sample kernel fit R^2={result.mean_system_r2:.4f} not used as evidence."
                ),
            )

        return EpistemicOdeAdjudication(
            verdict=EpistemicOdeVerdict.IDENTIFIED_DYNAMICAL_LAW,
            mean_holdout_r2=mean_r2,
            worst_state_r2=worst_r2,
            is_stable=True,
            rational_justification=(
                f"Forward holdout rollout satisfies mean and worst-state R^2 criteria [{metrics}]; "
                "integration remained finite over the full holdout horizon."
            ),
        )
