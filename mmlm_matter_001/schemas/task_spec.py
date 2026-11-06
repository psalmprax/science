"""Task records with an explicit public/sealed boundary.

Blueprint §10.2 forbids P1 from seeing the hidden ground-truth topology or
dimensions, and §17 lists hidden-task leakage as a risk with the control
"expose only the declared task specification". The separation is therefore a
type boundary, not a convention: prompt builders accept
:class:`PublicTaskView` only, and :class:`SealedTask` keeps the planted answer
alongside it for the referee side of the run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

TASK_FAMILIES = frozenset({"T1", "T2", "T3"})
SUPPORTED_UNITS = frozenset({"mm"})


@dataclass(frozen=True, slots=True)
class PublicTaskView:
    """Everything a proposer is allowed to see about a task."""

    task_id: str
    family: str
    target_kind: str
    tolerance: float
    input_range_deg: tuple[float, float]
    units: str = "mm"
    envelope_mm: tuple[float, float, float] | None = None
    target_points: tuple[tuple[float, float], ...] = ()
    function_samples: tuple[tuple[float, float], ...] = ()

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("task_id must be non-empty")
        if self.family not in TASK_FAMILIES:
            raise ValueError(f"family must be one of {sorted(TASK_FAMILIES)}, got {self.family!r}")
        if not self.target_kind:
            raise ValueError("target_kind must be non-empty")
        if not (self.tolerance > 0 and math.isfinite(self.tolerance)):
            raise ValueError(f"tolerance must be finite and positive, got {self.tolerance}")
        if self.units not in SUPPORTED_UNITS:
            raise ValueError(f"units must be one of {sorted(SUPPORTED_UNITS)}, got {self.units!r}")
        low, high = self.input_range_deg
        if not (math.isfinite(low) and math.isfinite(high)) or not low < high:
            raise ValueError(f"input_range_deg must be finite and increasing, got {self.input_range_deg}")
        for name in ("target_points", "function_samples"):
            for point in getattr(self, name):
                if len(point) != 2 or not all(math.isfinite(v) for v in point):
                    raise ValueError(f"{name} must hold finite (x, y) pairs, got {point!r}")
        if self.envelope_mm is not None:
            if len(self.envelope_mm) != 3 or not all(
                    math.isfinite(v) and v > 0 for v in self.envelope_mm):
                raise ValueError(
                    f"envelope_mm must be three positive finite extents, got {self.envelope_mm}")


@dataclass(frozen=True, slots=True)
class PublicFeedback:
    """Feedback about this run's own earlier attempts, safe to show a proposer."""

    topology_id: str
    status: str
    error_metric: float | None = None
    credits_used: int = 0

    def __post_init__(self) -> None:
        if not self.topology_id:
            raise ValueError("topology_id must be non-empty")
        if not self.status:
            raise ValueError("status must be non-empty")
        if self.credits_used < 0:
            raise ValueError(f"credits_used must be non-negative, got {self.credits_used}")


@dataclass(frozen=True, slots=True)
class SealedTask:
    """A task plus its planted answer. Never passed to a proposer."""

    public: PublicTaskView
    planted_topology_id: str
    planted_dimensions_mm: dict[str, float]
    hidden_seed: int
    provenance: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.planted_topology_id:
            raise ValueError("planted_topology_id must be non-empty")
        if not self.planted_dimensions_mm:
            raise ValueError("planted_dimensions_mm must be non-empty")
        for name, value in self.planted_dimensions_mm.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"planted dimension {name!r} must be finite and positive")

    @property
    def public_view(self) -> PublicTaskView:
        return self.public
