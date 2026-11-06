"""Preregistered Phase 0 checker thresholds (blueprint §5.2, §5.7, §12.6).

Two numbers decide whether a candidate mechanism passes Phase 0: the Grashof
safety margin (``delta_g``, §5.2) and the angular tolerance a T2 function task
must be met to (§5.7). If either is chosen at the call site, or inherited from
whatever task happens to be open, then it can be moved after seeing a result —
which is exactly what preregistration forbids.

So they live here, as one record that is hashed into the frozen protocol bundle
(:mod:`mmlm_matter_001.protocol.freeze`). A run is then either consistent with
the preregistered thresholds or refuses:

* :meth:`Phase0Preregistration.grashof` runs the §5.2 test with the pinned
  margin, so no caller supplies a margin of its own;
* :meth:`Phase0Preregistration.require_function_tolerance` refuses a task whose
  declared tolerance differs from the pinned one, instead of quietly scoring
  against the task's number;
* :meth:`FrozenProtocol.verify_preregistration
  <mmlm_matter_001.protocol.freeze.FrozenProtocol.verify_preregistration>`
  refuses when a bundle was frozen with different values.

The record is deliberately small: a threshold that is easy to add is a threshold
that is easy to add after the fact.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from mmlm_matter_001.kinematics.cognates import ANGLE_MATCH_DEG, COGNATE_TOLERANCE_MM
from mmlm_matter_001.kinematics.fourbar import (
    DEFAULT_MAX_JUMP_DEG,
    FourBar,
    GrashofResult,
)
from mmlm_matter_001.kinematics.loop_closure import (
    ANGLE_TOLERANCE_DEG,
    RESIDUAL_TOLERANCE_MM,
)
from mmlm_matter_001.kinematics.path_metric import BASE_PATH_SAMPLES
from mmlm_matter_001.models.contract import canonical_json, sha256_text

PREREGISTRATION_VERSION = "1.0"

#: Floating-point equality for a declared tolerance, not a substantive threshold.
_EXACT = 1e-12


class PreregistrationError(RuntimeError):
    """A checker threshold that was not preregistered, or disagrees with it."""


def _require_number(name: str, value: Any, *, minimum: float, strict: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PreregistrationError(f"{name} must be a number, got {value!r}")
    if not math.isfinite(value):
        raise PreregistrationError(f"{name} must be finite, got {value!r}")
    if strict and not value > minimum:
        raise PreregistrationError(f"{name} must be greater than {minimum}, got {value!r}")
    if not strict and not value >= minimum:
        raise PreregistrationError(f"{name} must be at least {minimum}, got {value!r}")
    return float(value)


def _require_int(name: str, value: Any, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PreregistrationError(f"{name} must be an integer, got {value!r}")
    if value < minimum:
        raise PreregistrationError(f"{name} must be at least {minimum}, got {value!r}")
    return value


@dataclass(frozen=True, slots=True)
class Phase0Preregistration:
    """The Phase 0 thresholds, fixed before hidden tasks are opened.

    ``grashof_margin_mm`` is the slack a link set must clear beyond the strict
    Grashof equality; ``function_tolerance_deg`` is the maximum wrapped angular
    error a T2 answer may show.

    The measurement settings are pinned with them, because a threshold is only
    meaningful together with the grid and solver tolerances that produced the
    number it is compared against: a pass at a coarse sample count is a different
    claim from a pass at the pinned one.
    """

    grashof_margin_mm: float
    function_tolerance_deg: float
    rationale: str = ""
    version: str = PREREGISTRATION_VERSION
    path_samples: int = BASE_PATH_SAMPLES
    continuity_max_jump_deg: float = DEFAULT_MAX_JUMP_DEG
    cognate_tolerance_mm: float = COGNATE_TOLERANCE_MM
    cognate_angle_tolerance_deg: float = ANGLE_MATCH_DEG
    loop_angle_tolerance_deg: float = ANGLE_TOLERANCE_DEG
    loop_residual_tolerance_mm: float = RESIDUAL_TOLERANCE_MM

    def __post_init__(self) -> None:
        _require_number(
            "grashof_margin_mm", self.grashof_margin_mm, minimum=0.0, strict=False)
        _require_number(
            "function_tolerance_deg", self.function_tolerance_deg,
            minimum=0.0, strict=True)
        _require_int("path_samples", self.path_samples, minimum=2)
        _require_number(
            "continuity_max_jump_deg", self.continuity_max_jump_deg,
            minimum=0.0, strict=True)
        _require_number(
            "cognate_tolerance_mm", self.cognate_tolerance_mm,
            minimum=0.0, strict=True)
        _require_number(
            "cognate_angle_tolerance_deg", self.cognate_angle_tolerance_deg,
            minimum=0.0, strict=True)
        _require_number(
            "loop_angle_tolerance_deg", self.loop_angle_tolerance_deg,
            minimum=0.0, strict=True)
        _require_number(
            "loop_residual_tolerance_mm", self.loop_residual_tolerance_mm,
            minimum=0.0, strict=True)
        if not isinstance(self.rationale, str):
            raise PreregistrationError(
                f"rationale must be a string, got {type(self.rationale).__name__}")
        if self.version != PREREGISTRATION_VERSION:
            raise PreregistrationError(
                f"preregistration version {self.version!r} is not "
                f"{PREREGISTRATION_VERSION!r}")

    def record(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "grashof_margin_mm": self.grashof_margin_mm,
            "function_tolerance_deg": self.function_tolerance_deg,
            "rationale": self.rationale,
            "path_samples": self.path_samples,
            "continuity_max_jump_deg": self.continuity_max_jump_deg,
            "cognate_tolerance_mm": self.cognate_tolerance_mm,
            "cognate_angle_tolerance_deg": self.cognate_angle_tolerance_deg,
            "loop_angle_tolerance_deg": self.loop_angle_tolerance_deg,
            "loop_residual_tolerance_mm": self.loop_residual_tolerance_mm,
        }

    def effective_settings(self) -> dict[str, Any]:
        """The measurement settings a verdict under this record was produced with."""
        return {
            "path_samples": self.path_samples,
            "continuity_max_jump_deg": self.continuity_max_jump_deg,
            "cognate_tolerance_mm": self.cognate_tolerance_mm,
            "cognate_angle_tolerance_deg": self.cognate_angle_tolerance_deg,
            "loop_angle_tolerance_deg": self.loop_angle_tolerance_deg,
            "loop_residual_tolerance_mm": self.loop_residual_tolerance_mm,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> Phase0Preregistration:
        if not isinstance(record, Mapping):
            raise PreregistrationError(
                f"preregistration record must be a mapping, got {type(record).__name__}")
        expected = {"version", "grashof_margin_mm", "function_tolerance_deg", "rationale"}
        # Measurement settings were added after the first bundles were written;
        # an older record loads with these defaults rather than failing to load.
        optional = {
            "path_samples": BASE_PATH_SAMPLES,
            "continuity_max_jump_deg": DEFAULT_MAX_JUMP_DEG,
            "cognate_tolerance_mm": COGNATE_TOLERANCE_MM,
            "cognate_angle_tolerance_deg": ANGLE_MATCH_DEG,
            "loop_angle_tolerance_deg": ANGLE_TOLERANCE_DEG,
            "loop_residual_tolerance_mm": RESIDUAL_TOLERANCE_MM,
        }
        unexpected = set(record) - expected - set(optional)
        if unexpected:
            raise PreregistrationError(
                f"preregistration record has unexpected fields: {sorted(unexpected)}")
        missing = expected - set(record)
        if missing:
            raise PreregistrationError(
                f"preregistration record is missing fields: {sorted(missing)}")
        return cls(
            grashof_margin_mm=record["grashof_margin_mm"],
            function_tolerance_deg=record["function_tolerance_deg"],
            rationale=record["rationale"],
            version=record["version"],
            **{name: record.get(name, default) for name, default in optional.items()},
        )

    @property
    def digest(self) -> str:
        return sha256_text(canonical_json(self.record()))

    def grashof(self, mechanism: FourBar) -> GrashofResult:
        """Run the §5.2 Grashof test with the pinned margin and nothing else."""
        if not isinstance(mechanism, FourBar):
            raise PreregistrationError(
                f"the Grashof check needs a FourBar, got {type(mechanism).__name__}")
        return mechanism.check_grashof(margin_mm=self.grashof_margin_mm)

    def require_function_tolerance(self, declared_deg: float) -> float:
        """Refuse a T2 task whose tolerance is not the preregistered one."""
        declared = _require_number(
            "declared tolerance", declared_deg, minimum=0.0, strict=True)
        if abs(declared - self.function_tolerance_deg) > _EXACT:
            raise PreregistrationError(
                f"task declares a T2 tolerance of {declared} deg but the "
                f"preregistration pins {self.function_tolerance_deg} deg; a task "
                "cannot re-open a preregistered threshold")
        return self.function_tolerance_deg

    def require_path_samples(self, samples: int) -> int:
        """Refuse a T1 sample count that is not the preregistered one.

        Sampling is part of the measurement: the same tolerance evaluated on a
        coarser grid is a different claim, and a call site that can pick the grid
        can pick the grid its candidate happens to pass on.
        """
        requested = _require_int("path samples", samples, minimum=2)
        if requested != self.path_samples:
            raise PreregistrationError(
                f"path check asks for {requested} samples but the preregistration "
                f"pins {self.path_samples}; sampling is part of the measurement")
        return self.path_samples


__all__ = [
    "PREREGISTRATION_VERSION",
    "Phase0Preregistration",
    "PreregistrationError",
]
