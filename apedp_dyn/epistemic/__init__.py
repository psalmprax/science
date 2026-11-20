"""Epistemic adjudication: ranking is not promotion."""

from __future__ import annotations

from typing import TYPE_CHECKING

from apedp_dyn._lazy import lazy_exports

if TYPE_CHECKING:  # the names below, for type checkers and readers; nothing is imported at runtime
    from apedp_dyn.epistemic.adjudicate import (
        CAPABILITY_ABSENT,
        REL_FLOOR,
        Adjudication,
        BenchmarkLevel,
        DataOrigin,
        EpistemicAttribution,
        EpistemicContract,
        EpistemicStatus,
        adjudicate,
        bootstrap_available,
        resolution_floor,
        scale_from_observations,
    )
    from apedp_dyn.epistemic.bayesian_active_learning import (
        ActiveLearningSuggestion,
        BayesianActiveLearner,
    )
    from apedp_dyn.epistemic.cross_asset_gate import (
        CrossAssetGateReport,
        CrossAssetGateStatus,
        evaluate_cross_asset_gate,
    )
    from apedp_dyn.epistemic.gate4_provenance import (
        ProposalErrorMetrics,
        ProposalNotEvaluableError,
        gate4_status,
        measure_proposal,
        noise_sigma_from_residuals,
        scored_candidate,
    )
    from apedp_dyn.epistemic.gp_uncertainty import (
        BayesianOptimizer,
        GaussianProcessRegressor,
        GPPrediction,
        RBFKernel,
    )
    from apedp_dyn.epistemic.market_extrapolation import (
        MarketExtrapolationResult,
        MarketRung,
        RungEvaluation,
        evaluate_30_asset_market_ladder,
        evaluate_single_asset_market_ladder,
    )
    from apedp_dyn.epistemic.multiple_testing import (
        MultipleTestingResult,
        adjust_p_values,
    )

# Public name -> the module that defines it. Resolved on first access, so
# importing this package (which importing any of its submodules does) costs
# nothing and pulls in nothing. See apedp_dyn/_lazy.py.
_EXPORTS = {
    "Adjudication": "apedp_dyn.epistemic.adjudicate",
    "BenchmarkLevel": "apedp_dyn.epistemic.adjudicate",
    "CAPABILITY_ABSENT": "apedp_dyn.epistemic.adjudicate",
    "DataOrigin": "apedp_dyn.epistemic.adjudicate",
    "EpistemicAttribution": "apedp_dyn.epistemic.adjudicate",
    "EpistemicContract": "apedp_dyn.epistemic.adjudicate",
    "EpistemicStatus": "apedp_dyn.epistemic.adjudicate",
    "REL_FLOOR": "apedp_dyn.epistemic.adjudicate",
    "adjudicate": "apedp_dyn.epistemic.adjudicate",
    "bootstrap_available": "apedp_dyn.epistemic.adjudicate",
    "resolution_floor": "apedp_dyn.epistemic.adjudicate",
    "scale_from_observations": "apedp_dyn.epistemic.adjudicate",
    "ActiveLearningSuggestion": "apedp_dyn.epistemic.bayesian_active_learning",
    "BayesianActiveLearner": "apedp_dyn.epistemic.bayesian_active_learning",
    "CrossAssetGateReport": "apedp_dyn.epistemic.cross_asset_gate",
    "CrossAssetGateStatus": "apedp_dyn.epistemic.cross_asset_gate",
    "evaluate_cross_asset_gate": "apedp_dyn.epistemic.cross_asset_gate",
    "ProposalErrorMetrics": "apedp_dyn.epistemic.gate4_provenance",
    "ProposalNotEvaluableError": "apedp_dyn.epistemic.gate4_provenance",
    "gate4_status": "apedp_dyn.epistemic.gate4_provenance",
    "measure_proposal": "apedp_dyn.epistemic.gate4_provenance",
    "noise_sigma_from_residuals": "apedp_dyn.epistemic.gate4_provenance",
    "scored_candidate": "apedp_dyn.epistemic.gate4_provenance",
    "BayesianOptimizer": "apedp_dyn.epistemic.gp_uncertainty",
    "GPPrediction": "apedp_dyn.epistemic.gp_uncertainty",
    "GaussianProcessRegressor": "apedp_dyn.epistemic.gp_uncertainty",
    "RBFKernel": "apedp_dyn.epistemic.gp_uncertainty",
    "MarketExtrapolationResult": "apedp_dyn.epistemic.market_extrapolation",
    "MarketRung": "apedp_dyn.epistemic.market_extrapolation",
    "RungEvaluation": "apedp_dyn.epistemic.market_extrapolation",
    "evaluate_30_asset_market_ladder": "apedp_dyn.epistemic.market_extrapolation",
    "evaluate_single_asset_market_ladder": "apedp_dyn.epistemic.market_extrapolation",
    "MultipleTestingResult": "apedp_dyn.epistemic.multiple_testing",
    "adjust_p_values": "apedp_dyn.epistemic.multiple_testing",
}

__all__ = [
    "CAPABILITY_ABSENT",
    "REL_FLOOR",
    "ActiveLearningSuggestion",
    "Adjudication",
    "BayesianActiveLearner",
    "BayesianOptimizer",
    "BenchmarkLevel",
    "CrossAssetGateReport",
    "CrossAssetGateStatus",
    "DataOrigin",
    "EpistemicAttribution",
    "EpistemicContract",
    "EpistemicStatus",
    "GPPrediction",
    "GaussianProcessRegressor",
    "MarketExtrapolationResult",
    "MarketRung",
    "MultipleTestingResult",
    "ProposalErrorMetrics",
    "ProposalNotEvaluableError",
    "RBFKernel",
    "RungEvaluation",
    "adjudicate",
    "adjust_p_values",
    "bootstrap_available",
    "evaluate_30_asset_market_ladder",
    "evaluate_cross_asset_gate",
    "evaluate_single_asset_market_ladder",
    "gate4_status",
    "measure_proposal",
    "noise_sigma_from_residuals",
    "resolution_floor",
    "scale_from_observations",
    "scored_candidate",
]

__getattr__, __dir__ = lazy_exports(__name__, _EXPORTS)
