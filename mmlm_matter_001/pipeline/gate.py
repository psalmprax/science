"""Phase 0 gate: the one place where the pinned thresholds decide a candidate.

The preregistered record and the checkers existed side by side without anything
connecting them: a caller could run the checkers with whatever thresholds it
liked, or with none, and the run left no trace. This module is the connection.
It takes a candidate and a task, runs the Grashof test with the pinned margin,
scores the family with the pinned sampling and solver tolerances, asks for the
independent loop-closure second opinion on every pose it accepts, and writes one
verdict row to the audit ledger.

Families:

* **T1** — the ordered path metric, continuity-tracked, refinement-checked and
  loop-closure verified, with the cognate check available on request.
* **T2** — the function metric against the pinned angular tolerance, with the
  same loop-closure second opinion over every pose it scored.
* **T3** — the topology half first (`check_topology`: structure, one degree of
  freedom, link roles, and the inversion the dimensions actually produce), then
  the task's own metric. A topology the checkout cannot verify — the blueprint's
  6-, 8- and 10-link universe — is refused rather than passed on its metadata.

Two refusals are deliberate:

* No preregistration, no verdict. A threshold chosen at the call site is a
  threshold that can move after a result is seen, so the gate raises rather than
  falling back to a default.
* A bundle pinned to other thresholds, or a checkout that no longer implements
  the bundle, refuses before any measurement runs. So does a proposer calling the
  gate directly (AGENTS.md §1.1).

The verdict is a record, not a boolean: it carries the digests it was judged
under, the effective settings, and every reason, so a rejection can be audited
later without re-running anything.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from mmlm_matter_001.evaluation.verdict_ledger import RESERVED_KEYS, VerdictLedger
from mmlm_matter_001.kinematics.boundary import assert_referee_caller
from mmlm_matter_001.kinematics.cognates import check_cognates, roberts_cognates
from mmlm_matter_001.kinematics.fourbar import FourBar, MechanismError
from mmlm_matter_001.kinematics.function_metric import check_function
from mmlm_matter_001.kinematics.path_metric import check_path
from mmlm_matter_001.kinematics.topology_check import check_topology
from mmlm_matter_001.protocol.freeze import FrozenProtocol
from mmlm_matter_001.protocol.preregistration import (
    Phase0Preregistration,
    PreregistrationError,
)
from mmlm_matter_001.schemas import PublicTaskView, TopologyEntry

#: Families this gate can adjudicate.
GATE_FAMILIES = frozenset({"T1", "T2", "T3"})

#: Reasons the gate itself can add, beyond the checkers' own vocabularies.
GATE_REASONS = frozenset({
    "cognates_unavailable",
    "family_not_supported",
    "no_target",
    "tolerance_not_preregistered",
    "topology_missing",
})

#: Reasons that describe what this checkout cannot verify, not a fault in the
#: candidate. A verdict resting only on these is ``unverifiable``: counting it as
#: a failure would bias any pass rate that uses these rows as its denominator.
CAPABILITY_REASONS = frozenset({
    "family_not_supported",
    "topology_unsupported_structure",
})


class GateError(RuntimeError):
    """The gate was asked to adjudicate something it cannot adjudicate."""


@dataclass(frozen=True, slots=True)
class GateVerdict:
    """One adjudicated candidate, in the terms the audit ledger stores."""

    candidate_id: str
    task_id: str
    family: str
    #: ``accepted``, ``refused`` or ``unverifiable`` (a capability gap).
    status: str
    passed: bool
    reasons: tuple[str, ...]
    preregistration_digest: str
    preregistration: Mapping[str, Any]
    effective_settings: Mapping[str, Any]
    protocol_hash: str | None
    samples: int
    grashof: Mapping[str, Any]
    topology: Mapping[str, Any] | None
    path: Mapping[str, Any] | None
    function: Mapping[str, Any] | None
    cognates: Mapping[str, Any] | None
    loop_closure: Mapping[str, Any] | None
    recorded_utc: str

    def record(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "task_id": self.task_id,
            "family": self.family,
            "status": self.status,
            "passed": self.passed,
            "reasons": list(self.reasons),
            "preregistration_digest": self.preregistration_digest,
            "preregistration": dict(self.preregistration),
            "effective_settings": dict(self.effective_settings),
            "protocol_hash": self.protocol_hash,
            "samples": self.samples,
            "grashof": dict(self.grashof),
            "topology": None if self.topology is None else dict(self.topology),
            "path": None if self.path is None else dict(self.path),
            "function": None if self.function is None else dict(self.function),
            "cognates": None if self.cognates is None else dict(self.cognates),
            "loop_closure": (
                None if self.loop_closure is None else dict(self.loop_closure)),
            "recorded_utc": self.recorded_utc,
        }


def _score_path(
    candidate: FourBar,
    task: PublicTaskView,
    preregistration: Phase0Preregistration,
    *,
    branch: str,
    samples: int,
) -> tuple[Mapping[str, Any], Mapping[str, Any], list[str]]:
    path = check_path(
        candidate,
        task,
        branch=branch,
        samples=samples,
        verify_loop_closure=True,
        loop_angle_tolerance_deg=preregistration.loop_angle_tolerance_deg,
        loop_residual_tolerance_mm=preregistration.loop_residual_tolerance_mm,
        continuity=True,
        max_jump_deg=preregistration.continuity_max_jump_deg,
        refine=True,
    )
    loop = {
        "verified": path.loop_closure_verified,
        "checked": path.loop_closure_checked,
        "max_residual_mm": path.max_loop_residual_mm,
    }
    return path.record(), loop, list(path.reasons)


def _score_function(
    candidate: FourBar,
    task: PublicTaskView,
    preregistration: Phase0Preregistration,
    *,
    branch: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any], list[str]]:
    function = check_function(
        candidate,
        task,
        branch=branch,
        verify_loop_closure=True,
        angle_tolerance_deg=preregistration.loop_angle_tolerance_deg,
        residual_tolerance_mm=preregistration.loop_residual_tolerance_mm,
    )
    loop = {
        "verified": function.loop_closure_verified,
        "checked": function.loop_closure_checked,
        "max_residual_mm": function.max_loop_residual_mm,
    }
    return function.record(), loop, list(function.reasons)


def run_gate(
    candidate: FourBar,
    task: PublicTaskView,
    *,
    preregistration: Phase0Preregistration,
    protocol: FrozenProtocol | None = None,
    topology: TopologyEntry | None = None,
    candidate_id: str | None = None,
    branch: str = "open",
    samples: int | None = None,
    verify_cognates: bool = False,
    cognate_input_range_deg: tuple[float, float] | None = None,
    ledger: VerdictLedger | Path | str | None = None,
    recorded_utc: str | None = None,
) -> GateVerdict:
    """Adjudicate one candidate under the pinned Phase 0 protocol.

    ``samples`` may only restate the preregistered count; a different number is
    refused rather than quietly used. For a T3 task, ``topology`` is the entry the
    candidate claims to implement — without it there is nothing to check the
    dimensions against, so the answer is refused. ``verify_cognates`` adds the
    Roberts construction check to a T1 verdict.
    """
    assert_referee_caller("run_gate")
    if not isinstance(candidate, FourBar):
        raise GateError(f"the gate needs a FourBar, got {type(candidate).__name__}")
    if not isinstance(task, PublicTaskView):
        raise GateError(f"the gate needs a PublicTaskView, got {type(task).__name__}")
    if not isinstance(preregistration, Phase0Preregistration):
        raise GateError(
            "the gate refuses to run without a Phase 0 preregistration: a threshold "
            "chosen at the call site can move after a result is seen")
    if protocol is not None:
        if not isinstance(protocol, FrozenProtocol):
            raise GateError(
                f"protocol must be a FrozenProtocol, got {type(protocol).__name__}")
        # Drift and threshold mismatches are refusals, not verdict reasons: the
        # run must not measure anything under a protocol it is not implementing.
        protocol.verify_checkout()
        protocol.verify_preregistration(preregistration)

    pinned_samples = preregistration.require_path_samples(
        preregistration.path_samples if samples is None else samples)
    reasons: list[str] = []
    grashof = preregistration.grashof(candidate)
    if not grashof.is_grashof:
        reasons.append(grashof.reason)

    topology_record: Mapping[str, Any] | None = None
    path_record: Mapping[str, Any] | None = None
    function_record: Mapping[str, Any] | None = None
    cognate_record: Mapping[str, Any] | None = None
    loop_record: Mapping[str, Any] | None = None

    if task.family == "T1":
        path_record, loop_record, extra = _score_path(
            candidate, task, preregistration, branch=branch, samples=pinned_samples)
        reasons.extend(extra)
        if verify_cognates:
            try:
                built = roberts_cognates(candidate, branch=branch)
                cognate_check = check_cognates(
                    built,
                    input_range_deg=cognate_input_range_deg or task.input_range_deg,
                    samples=pinned_samples,
                    tolerance_mm=preregistration.cognate_tolerance_mm,
                    angle_tolerance_deg=preregistration.cognate_angle_tolerance_deg,
                )
            except MechanismError as exc:
                reasons.append("cognates_unavailable")
                cognate_record = {"error": str(exc)}
            else:
                cognate_record = cognate_check.record()
                reasons.extend(cognate_check.reasons)
    elif task.family == "T2":
        try:
            preregistration.require_function_tolerance(task.tolerance)
        except PreregistrationError as exc:
            reasons.append("tolerance_not_preregistered")
            function_record = {"error": str(exc)}
        else:
            function_record, loop_record, extra = _score_function(
                candidate, task, preregistration, branch=branch)
            reasons.extend(extra)
    elif task.family == "T3":
        if topology is None:
            reasons.append("topology_missing")
        else:
            topology_check = check_topology(
                topology, candidate, margin_mm=preregistration.grashof_margin_mm)
            topology_record = topology_check.record()
            reasons.extend(topology_check.reasons)
        # The topology half decides whether the claim is coherent; the task's own
        # metric decides whether the dimensions do the job.
        if task.target_points:
            path_record, loop_record, extra = _score_path(
                candidate, task, preregistration, branch=branch,
                samples=pinned_samples)
            reasons.extend(extra)
        elif task.function_samples:
            function_record, loop_record, extra = _score_function(
                candidate, task, preregistration, branch=branch)
            reasons.extend(extra)
        else:
            reasons.append("no_target")
    else:
        reasons.append("family_not_supported")

    unique = tuple(dict.fromkeys(reasons))
    status = (
        "accepted" if not unique
        else "unverifiable" if set(unique) <= CAPABILITY_REASONS
        else "refused"
    )
    verdict = GateVerdict(
        candidate_id=candidate_id or task.task_id,
        task_id=task.task_id,
        family=task.family,
        status=status,
        passed=status == "accepted",
        reasons=unique,
        preregistration_digest=preregistration.digest,
        preregistration=preregistration.record(),
        effective_settings={
            **preregistration.effective_settings(),
            "samples_used": pinned_samples,
            "refinement": True,
        },
        protocol_hash=None if protocol is None else protocol.protocol_hash,
        samples=pinned_samples,
        grashof=asdict(grashof),
        topology=topology_record,
        path=path_record,
        function=function_record,
        cognates=cognate_record,
        loop_closure=loop_record,
        recorded_utc=recorded_utc
        or dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    )
    if ledger is not None:
        target = ledger if isinstance(ledger, VerdictLedger) else VerdictLedger(ledger)
        payload = {
            key: value
            for key, value in verdict.record().items()
            if key not in RESERVED_KEYS
        }
        target.append(
            candidate_id=verdict.candidate_id,
            record=payload,
            recorded_utc=verdict.recorded_utc,
        )
    return verdict


__all__ = [
    "CAPABILITY_REASONS",
    "GATE_FAMILIES",
    "GATE_REASONS",
    "GateError",
    "GateVerdict",
    "run_gate",
]
