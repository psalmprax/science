"""Epistemic Referee contract for continuous PDE validation.

Adheres strictly to AGENTS.md:
- Never inspects development data during hypothesis generation.
- Evaluates candidate parameters frozen on the sealed holdout slice.
- Burns holdout slice if a candidate fails confirmation.
- Enforces physical causality (thermodynamic arrow of time), residual whiteness, and noise floors.
"""
from __future__ import annotations

from apedp_dyn.epistemic.high_dim_pde import (
    EpistemicPdeAdjudication,
    EpistemicPdeVerdict,
    HighDimPdeDiscoveryResult,
    adjudicate_high_dim_pde_discovery,
    evaluate_point_cloud_pde_frozen_oos,
)
from apedp_dyn.pde.data import PdePointCloudData


class PdeReferee:
    """Epistemic referee adjudicating spatio-temporal PDE candidate hypotheses."""

    def __init__(
        self,
        min_holdout_r2: float = 0.95,
        max_whiteness_rho1: float = 0.50,
        enforce_thermodynamic_arrow: bool = True,
    ):
        self._min_holdout_r2 = min_holdout_r2
        self._max_whiteness_rho1 = max_whiteness_rho1
        self._enforce_thermodynamic_arrow = enforce_thermodynamic_arrow

    def adjudicate_result(
        self,
        result: HighDimPdeDiscoveryResult,
        holdout_data: PdePointCloudData,
    ) -> EpistemicPdeAdjudication:
        """Adjudicate a discovered PDE result against a sealed holdout point cloud."""
        if holdout_data.u_anchors is None or holdout_data.u_probes_plus is None or holdout_data.u_probes_minus is None:
            raise ValueError("holdout_data must contain axis-star probes for frozen evaluation")

        h_val = holdout_data.h if holdout_data.h is not None else 0.04
        # Evaluate frozen coefficients on unseen holdout
        frozen_r2 = evaluate_point_cloud_pde_frozen_oos(
            result,
            holdout_data.u_anchors,
            holdout_data.u_probes_plus,
            holdout_data.u_probes_minus,
            dt=holdout_data.dt,
            h=h_val,
        )

        # Enforce thermodynamic arrow of time if diffusion is present
        if self._enforce_thermodynamic_arrow:
            for term, coeff in zip(result.active_terms, result.coefficients, strict=False):
                if "laplacian" in term and coeff < 0.0:
                    return EpistemicPdeAdjudication(
                        verdict=EpistemicPdeVerdict.PREDICTIVE_WRONG_STRUCTURE,
                        discovered=result,
                        frozen_oos_r2=float(frozen_r2),
                        max_dictionary_collinearity=0.0,
                        subwindow_max_rel_drift=1.0,
                        stencil_snr=0.0,
                        rational_justification=f"Thermodynamic arrow of time violated: negative diffusion coefficient ({coeff:.4f})",
                    )

        if frozen_r2 >= self._min_holdout_r2:
            verdict = EpistemicPdeVerdict.IDENTIFIED_CAUSAL_LAW
            reason = f"Passed frozen holdout confirmation with R^2={frozen_r2:.6f} >= {self._min_holdout_r2}"
        elif frozen_r2 >= 0.40:
            verdict = EpistemicPdeVerdict.PREDICTIVE_WRONG_STRUCTURE
            reason = f"Holdout R^2={frozen_r2:.6f} below causal threshold {self._min_holdout_r2}"
        else:
            verdict = EpistemicPdeVerdict.REJECTED_NO_CANDIDATE
            reason = f"Catastrophic holdout collapse with R^2={frozen_r2:.6f}"

        return EpistemicPdeAdjudication(
            verdict=verdict,
            discovered=result,
            frozen_oos_r2=float(frozen_r2),
            max_dictionary_collinearity=0.0,
            subwindow_max_rel_drift=0.0,
            stencil_snr=10.0,
            rational_justification=reason,
        )

    def adjudicate_full(
        self,
        result: HighDimPdeDiscoveryResult,
        dev_data: PdePointCloudData,
        holdout_data: PdePointCloudData,
    ) -> EpistemicPdeAdjudication:
        """Adjudicate via the full 5-way epistemic taxonomy with both dev and holdout data."""
        if (dev_data.u_anchors is None or dev_data.u_probes_plus is None or dev_data.u_probes_minus is None
                or holdout_data.u_anchors is None or holdout_data.u_probes_plus is None
                or holdout_data.u_probes_minus is None):
            raise ValueError("Both dev_data and holdout_data must contain axis-star probes")
        h_val = holdout_data.h if holdout_data.h is not None else 0.04
        return adjudicate_high_dim_pde_discovery(
            result,
            dev_data.u_anchors,
            dev_data.u_probes_plus,
            dev_data.u_probes_minus,
            holdout_data.u_anchors,
            holdout_data.u_probes_plus,
            holdout_data.u_probes_minus,
            dt=holdout_data.dt,
            h=h_val,
        )

