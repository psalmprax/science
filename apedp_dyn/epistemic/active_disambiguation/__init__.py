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
]
