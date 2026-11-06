"""Epistemic adjudication: ranking is not promotion."""
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
