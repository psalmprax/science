"""Identifiability Frontier & Epsilon-Transition Protocol for Mathematical Conjectures.

Maps the empirical resolution boundary where mathematical hypotheses transition:
UNIQUE_SUPPORT -> MULTI_HYPOTHESIS -> UNIDENTIFIABLE
across perturbation scales epsilon in [10^-2, 10^-4, 10^-6, 10^-8, 10^-10, 10^-12, 10^-14, 10^-16, 10^-18]
under controlled floating-point precision and measurement noise.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from apedp.decision.stage_c import (
    CandidateProfile,
    EpistemicState,
    ModelClass,
    StageCDecisionEngine,
)
from apedp.mathematical.mathematical_layer import MathematicalAssuranceEngine, ProofBoundaryStatus

logger = logging.getLogger("apedp.rh_precision")


@dataclass(frozen=True)
class FrontierTransitionPoint:
    """Represents the outcome of a single perturbation scale test."""

    epsilon: float
    noise_sigma: float
    epistemic_state: str
    winner: str | None
    delta_mse: float
    is_resolvable: bool
    proof_status: str


@dataclass(frozen=True)
class IdentifiabilityFrontierReport:
    """Full mapping of the mathematical identifiability frontier."""

    noise_sigma: float
    tested_epsilons: list[float]
    transitions: list[FrontierTransitionPoint]
    unique_support_regime: list[float]
    multi_hypothesis_regime: list[float]
    unidentifiable_regime: list[float]
    critical_resolution_boundary: float
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "noise_sigma": self.noise_sigma,
            "tested_epsilons": self.tested_epsilons,
            "transitions": [
                {
                    "epsilon": t.epsilon,
                    "noise_sigma": t.noise_sigma,
                    "epistemic_state": t.epistemic_state,
                    "winner": t.winner,
                    "delta_mse": t.delta_mse,
                    "is_resolvable": t.is_resolvable,
                    "proof_status": t.proof_status,
                }
                for t in self.transitions
            ],
            "unique_support_regime": self.unique_support_regime,
            "multi_hypothesis_regime": self.multi_hypothesis_regime,
            "unidentifiable_regime": self.unidentifiable_regime,
            "critical_resolution_boundary": self.critical_resolution_boundary,
            "summary": self.summary,
        }


class IdentifiabilityFrontierEngine:
    """Evaluates the mathematical identifiability frontier under controlled perturbation sweeps."""

    DEFAULT_EPSILONS: tuple[float, ...] = (
        1e-2,
        1e-4,
        1e-6,
        1e-8,
        1e-10,
        1e-12,
        1e-14,
        1e-16,
        1e-18,
    )

    def __init__(
        self,
        epsilons: Sequence[float] | None = None,
        floating_point_eps: float = 1e-15,
    ):
        self.epsilons = tuple(epsilons or self.DEFAULT_EPSILONS)
        self.floating_point_eps = floating_point_eps
        self.math_engine = MathematicalAssuranceEngine()
        self.stage_c = StageCDecisionEngine(holdout_margin_ratio=1.15)

    def map_frontier(
        self,
        base_n: np.ndarray,
        noise_sigma: float = 0.0,
        seed: int = 42,
    ) -> IdentifiabilityFrontierReport:
        """Executes the epsilon-transition sweep and identifies phase boundaries."""
        rng = np.random.default_rng(seed)
        transitions: list[FrontierTransitionPoint] = []

        unique_regime: list[float] = []
        multi_regime: list[float] = []
        unidentifiable_regime: list[float] = []

        critical_boundary = self.floating_point_eps

        for eps in self.epsilons:
            # Generate perturbed world: beta_n = 0.5 + eps * sin(n) + noise
            clean_beta = 0.5 + eps * np.sin(base_n)
            noise = rng.normal(0.0, noise_sigma, size=len(base_n)) if noise_sigma > 0 else np.zeros_like(base_n)
            beta_obs = clean_beta + noise

            # 70/30 train/holdout split
            n_tot = len(base_n)
            n_tr = int(n_tot * 0.7)
            ho_y = beta_obs[n_tr:]
            ho_n = base_n[n_tr:]

            # Hypotheses: H0 (0.5), H_eps (0.5 + eps * sin(n))
            pred_h0 = 0.5 * np.ones_like(ho_y)
            pred_heps = 0.5 + eps * np.sin(ho_n)

            mse_h0 = float(np.mean((ho_y - pred_h0) ** 2))
            mse_heps = float(np.mean((ho_y - pred_heps) ** 2))
            delta_mse = abs(mse_h0 - mse_heps)

            cand_h0 = CandidateProfile(
                candidate_id="constant_critical_line",
                expression="0.5",
                model_class=ModelClass.SYMBOLIC,
                math_sound=True,
                dim_status="VALID",
                in_sample_mse=float(np.mean((beta_obs[:n_tr] - 0.5) ** 2)),
                holdout_mse=mse_h0,
                holdout_r2=1.0 if mse_h0 == 0.0 else float(1.0 - mse_h0 / max(np.var(ho_y), 1e-15)),
                complexity=1.0,
                metadata={"holdout_residuals": (ho_y - pred_h0).tolist(), "n_samples": len(ho_y)},
            )
            cand_heps = CandidateProfile(
                candidate_id=f"perturbed_sine_{eps:.0e}",
                expression=f"0.5 + {eps:.1e}*sin(n)",
                model_class=ModelClass.SYMBOLIC,
                math_sound=True,
                dim_status="VALID",
                in_sample_mse=float(np.mean((beta_obs[:n_tr] - (0.5 + eps * np.sin(base_n[:n_tr]))) ** 2)),
                holdout_mse=mse_heps,
                holdout_r2=1.0 if mse_heps == 0.0 else float(1.0 - mse_heps / max(np.var(ho_y), 1e-15)),
                complexity=4.0,
                metadata={"holdout_residuals": (ho_y - pred_heps).tolist(), "n_samples": len(ho_y)},
            )

            # Check mathematical precision resolution
            effective_noise_floor = max(noise_sigma ** 2, self.floating_point_eps)
            is_distinguishable, _, _ = self.math_engine.assess_identifiability_resolution(
                candidate_errors={"H0": mse_h0, "H_eps": mse_heps},
                numerical_resolution=effective_noise_floor,
            )

            # Evaluate Stage C decision
            if not is_distinguishable or eps <= self.floating_point_eps:
                state_str = "UNIDENTIFIABLE"
                winner = None
                unidentifiable_regime.append(eps)
                proof_status = ProofBoundaryStatus.INDISTINGUISHABLE_BELOW_RESOLUTION.value
            else:
                decision = self.stage_c.evaluate([cand_h0, cand_heps])
                state_str = decision.state.value
                winner = decision.winner
                if state_str == EpistemicState.UNIQUE_SUPPORT.value:
                    unique_regime.append(eps)
                else:
                    multi_regime.append(eps)
                proof_status = ProofBoundaryStatus.EMPIRICALLY_SUPPORTED.value

            transitions.append(
                FrontierTransitionPoint(
                    epsilon=eps,
                    noise_sigma=noise_sigma,
                    epistemic_state=state_str,
                    winner=winner,
                    delta_mse=delta_mse,
                    is_resolvable=(state_str == EpistemicState.UNIQUE_SUPPORT.value),
                    proof_status=proof_status,
                )
            )

        # Critical boundary is where UNIQUE ceases to be supported
        if multi_regime:
            critical_boundary = float(max(multi_regime))
        elif unidentifiable_regime:
            critical_boundary = float(max(unidentifiable_regime))

        summary = (
            f"Identifiability Frontier at noise sigma={noise_sigma:.1e}: "
            f"UNIQUE_SUPPORT for eps in [{min(unique_regime) if unique_regime else 0:.1e}, {max(unique_regime) if unique_regime else 0:.1e}]; "
            f"MULTI_HYPOTHESIS for eps in [{min(multi_regime) if multi_regime else 0:.1e}, {max(multi_regime) if multi_regime else 0:.1e}]; "
            f"UNIDENTIFIABLE for eps <= {critical_boundary:.1e}."
        )

        return IdentifiabilityFrontierReport(
            noise_sigma=noise_sigma,
            tested_epsilons=list(self.epsilons),
            transitions=transitions,
            unique_support_regime=unique_regime,
            multi_hypothesis_regime=multi_regime,
            unidentifiable_regime=unidentifiable_regime,
            critical_resolution_boundary=critical_boundary,
            summary=summary,
        )


__all__ = [
    "FrontierTransitionPoint",
    "IdentifiabilityFrontierEngine",
    "IdentifiabilityFrontierReport",
]
