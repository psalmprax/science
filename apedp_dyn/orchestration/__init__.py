"""Orchestration layer: autonomous pre-discovery routing of raw datasets."""

from apedp_dyn.orchestration.dispatch import (
    DEFAULT_RUNNER_FAMILIES,
    DispatchDecision,
    DispatchOutcome,
    RouteDecision,
    dispatch_frame,
)
from apedp_dyn.orchestration.modality_sieve import (
    Modality,
    ModalityVerdict,
    classify_event_stream,
    classify_modality,
    route_engines,
)

__all__ = [
    "DEFAULT_RUNNER_FAMILIES",
    "DispatchDecision",
    "DispatchOutcome",
    "Modality",
    "ModalityVerdict",
    "RouteDecision",
    "classify_event_stream",
    "classify_modality",
    "dispatch_frame",
    "route_engines",
]
