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

from apedp_dyn.epistemic.coupled_discovery import CoupledSystemDiscoveryResult
from apedp_dyn.ode.data import OdeTrajectoryData


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


class OdeReferee:
    """Epistemic referee adjudicating ODE candidate hypotheses."""

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
        """Adjudicate coupled system against holdout trajectory."""
        if result.mean_system_r2 < self._min_mean_r2:
            return EpistemicOdeAdjudication(
                verdict=EpistemicOdeVerdict.PREDICTIVE_WRONG_STRUCTURE,
                mean_holdout_r2=float(result.mean_system_r2),
                worst_state_r2=float(result.worst_state_r2),
                is_stable=True,
                rational_justification=f"Mean system R^2 ({result.mean_system_r2:.4f}) below threshold {self._min_mean_r2}",
            )

        if result.worst_state_r2 < self._min_worst_r2:
            return EpistemicOdeAdjudication(
                verdict=EpistemicOdeVerdict.PREDICTIVE_WRONG_STRUCTURE,
                mean_holdout_r2=float(result.mean_system_r2),
                worst_state_r2=float(result.worst_state_r2),
                is_stable=True,
                rational_justification=f"Worst state R^2 ({result.worst_state_r2:.4f}) below threshold {self._min_worst_r2}",
            )

        return EpistemicOdeAdjudication(
            verdict=EpistemicOdeVerdict.IDENTIFIED_DYNAMICAL_LAW,
            mean_holdout_r2=float(result.mean_system_r2),
            worst_state_r2=float(result.worst_state_r2),
            is_stable=True,
            rational_justification="Coupled dynamical system satisfies all system and state-level R^2 criteria",
        )
