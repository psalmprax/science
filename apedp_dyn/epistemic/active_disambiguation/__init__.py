"""Active epistemic disambiguation layer (E4). Additive; adjudicate.py untouched."""

from apedp_dyn.epistemic.active_disambiguation.divergence_selection import (
    DivergenceCandidate,
    DivergenceSelection,
    DivergenceTrialResult,
    run_divergence_trial,
    select_divergence_point,
)
from apedp_dyn.epistemic.active_disambiguation.policy import (
    DiscriminationRow,
    HarnessConfig,
    Measurement,
    SelectionOutcome,
    TrialResult,
    compare_policies,
    discrimination_table,
    predict_all,
    run_trial,
    select_measurement,
)

__all__ = [
    "DiscriminationRow",
    "DivergenceCandidate",
    "DivergenceSelection",
    "DivergenceTrialResult",
    "HarnessConfig",
    "Measurement",
    "SelectionOutcome",
    "TrialResult",
    "compare_policies",
    "discrimination_table",
    "predict_all",
    "run_divergence_trial",
    "run_trial",
    "select_divergence_point",
    "select_measurement",
]
