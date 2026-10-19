"""Active epistemic disambiguation layer (E4). Additive; adjudicate.py untouched."""

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
from apedp_dyn.epistemic.active_disambiguation.divergence_selection import (
    DivergenceCandidate,
    DivergenceSelection,
    DivergenceTrialResult,
    run_divergence_trial,
    select_divergence_point,
)

__all__ = [
    "DiscriminationRow",
    "HarnessConfig",
    "Measurement",
    "SelectionOutcome",
    "TrialResult",
    "compare_policies",
    "discrimination_table",
    "predict_all",
    "run_trial",
    "select_measurement",
    "DivergenceCandidate",
    "DivergenceSelection",
    "DivergenceTrialResult",
    "run_divergence_trial",
    "select_divergence_point",
]
