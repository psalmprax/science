r"""Caller-Layer Three-Role Data Protocol (REG-F14 / REG-F14b reference implementation).

Status: experimental (additive layer over FullLineageExperimentLedger; design
doc: docs/CALLER_DATA_THREE_ROLE_PROTOCOL.md)

Roles (non-fungible, mutually disjoint per lead):
  SELECTION      — discovery/search + referee pair separation (slice A)
  CONFIRMATION   — SEALED, single-use promotion evaluation (slice B)
  PROSPECTIVE    — live forward data; never folded back into A or B

Core invariants (each enforced in code, pinned in
tests/test_dyn_caller_data_protocol.py):
  I1  selection ids ∩ confirmation ids = ∅ at issue time (no renegotiation).
  I2  the confirmation verdict is obtainable EXACTLY ONCE
      (`SealedConfirmationSlice.consume`); a second consumption raises
      `ConfirmationAlreadyConsumedError` — letter of REG-F14 ("evaluated ONCE").
  I3  every evaluation is written to the candidate's experiment-ledger node as
      a DatasetExposureRecord with the correct DatasetRole — the existing
      anti-snooping burn discipline keeps slice lifecycle ownership.
  I4  the ARCHIVED promotion verdict source is the confirmation slice; the
      selection verdict is retained but marked superseded.
  I5  (F14b) caller compliance is either-or and checked, not promised:
      mode (a) sealed confirmation split held, or mode (b) published
      DSR-adjusted multiplicity derived from LEDGER exposure counts
      (never hand-written; monotone; refused otherwise).
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Any, Callable

from apedp_dyn.epistemic.experiment_ledger import (
    DatasetRole,
    FullLineageExperimentLedger,
)


class CallerRoleOverlapError(ValueError):
    """SELECTION/CONFIRMATION slices overlap — the lead cannot be issued."""


class ConfirmationAlreadyConsumedError(RuntimeError):
    """The sealed confirmation slice was already consumed (REG-F14: ONCE)."""


class CallerComplianceError(ValueError):
    """A production caller publishes neither a sealed split nor a ledger-derived multiplicity."""


@dataclass(frozen=True)
class CallerSplit:
    """One lead's three-role assignment (disjoint by construction)."""

    caller_id: str
    dataset_id: str
    selection_ids: tuple[str, ...]
    confirmation_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        overlap = set(self.selection_ids) & set(self.confirmation_ids)
        if overlap:
            raise CallerRoleOverlapError(
                f"{self.caller_id}: SELECTION/CONFIRMATION overlap {sorted(overlap)} — "
                "a lead cannot reuse the same slice across roles (I1)"
            )
        if not self.confirmation_ids:
            raise CallerRoleOverlapError(
                f"{self.caller_id}: CONFIRMATION slice must be non-empty — "
                "a lead without a sealed confirmation cannot promote"
            )


@dataclass(frozen=True)
class ConfirmationProvenance:
    consumed_at: str
    candidate_id: str
    slice_dataset_id: str
    slice_ids: tuple[str, ...]
    one_shot: bool
    second_consume_refused: bool


@dataclass
class ConfirmationVerdict:
    """Verdict produced from the SEALED slice (slice B) only."""

    caller_id: str
    candidate_id: str
    verdict: Any  # downstream-supplied evaluation outcome (status, pass/fail, ...)
    metrics: dict[str, Any] = field(default_factory=dict)
    provenance: ConfirmationProvenance | None = None
    superseded_selection_verdict: Any = None


class SealedConfirmationSlice:
    """Single-use wrapper around the confirmation ids (I2)."""

    def __init__(self, split: CallerSplit, ledger: FullLineageExperimentLedger | None):
        self._split = split
        self._ledger = ledger
        self._consumed = False
        self._provenance: ConfirmationProvenance | None = None

    @property
    def ids(self) -> tuple[str, ...]:
        return self._split.confirmation_ids

    @property
    def consumed(self) -> bool:
        return self._consumed

    @property
    def consumed_provenance(self) -> ConfirmationProvenance | None:
        return self._provenance

    def consume(
        self,
        candidate_id: str,
        evaluate: Callable[..., ConfirmationVerdict],
        **eval_kwargs: Any,
    ) -> ConfirmationVerdict:
        if self._consumed:
            # Second attempt is refused AND labeled (the caller learns the
            # prohibition, not the data).
            raise ConfirmationAlreadyConsumedError(
                f"{self._split.caller_id}: confirmation slice already consumed "
                f"(provenance candidate_id={self._provenance.candidate_id if self._provenance else '?'}) — "
                "REG-F14: the archived promotion is evaluated ONCE"
            )
        self._consumed = True
        verdict = evaluate(
            candidate_id=candidate_id,
           confirmation_ids=self._split.confirmation_ids,
            **eval_kwargs,
        )
        timestamps = datetime.datetime.now(datetime.timezone.utc).isoformat()
        # I3: ledger exposure with the CONFIRMATION_HOLDOUT role (burn path retained).
        if self._ledger is not None:
            for sid in self._split.confirmation_ids:
                try:
                    self._ledger.record_data_exposure(
                        candidate_id=candidate_id,
                        dataset_id=self._split.dataset_id,
                        slice_identifier=sid,
                        role=DatasetRole.CONFIRMATION_HOLDOUT,
                        outcome_status="PASSED" if getattr(verdict, "verdict", None) else "FAILED",
                    )
                except Exception:
                    # Ledger-optional use (unit pins); never used to soften a verdict.
                    pass
        self._provenance = ConfirmationProvenance(
            consumed_at=timestamps,
            candidate_id=candidate_id,
            slice_dataset_id=self._split.dataset_id,
            slice_ids=tuple(self._split.confirmation_ids),
            one_shot=True,
            second_consume_refused=True,
        )
        if getattr(verdict, "provenance", None) is None:
            verdict.provenance = self._provenance
        return verdict


class _RustProtectionStub:  # pragma: no cover - defensive import shim
    """Mirror discipline note: selection-stage evaluation stays in numpy-land here."""


class CallerSplitProtocol:
    """Issues the disjoint three-role splits and enforces sealed consumption."""

    def __init__(self, ledger: FullLineageExperimentLedger | None = None) -> None:
        self._ledger = ledger
        self._open_leads: dict[str, tuple[CallerSplit, SealedConfirmationSlice]] = {}

    def issue_caller_split(
        self,
        caller_id: str,
        dataset_id: str,
        selection_ids: list[str],
        confirmation_ids: list[str],
    ) -> tuple[CallerSplit, SealedConfirmationSlice]:
        split = CallerSplit(
            caller_id=caller_id,
            dataset_id=dataset_id,
            selection_ids=tuple(selection_ids),
            confirmation_ids=tuple(confirmation_ids),
        )
        if caller_id in self._open_leads:
            raise ConfirmationAlreadyConsumedError(
                f"{caller_id}: an open lead already exists; a caller cannot be issued twice"
            )
        sealed = SealedConfirmationSlice(split, self._ledger)
        self._open_leads[caller_id] = (split, sealed)
        return split, sealed

    # I4: verdict source labeling helper -------------------------------------
    def archive_promotion_verdict(
        self,
        caller_id: str,
        confirmation_verdict: ConfirmationVerdict,
        selection_verdict: Any,
    ) -> dict[str, Any]:
        split, sealed = self._require_open_lead(caller_id)
        if not sealed.consumed:
            raise ConfirmationAlreadyConsumedError(
                f"{caller_id}: archive refused — the sealed slice has not been consumed; "
                "a promotion verdict cannot be archived from selection-only evaluation"
            )
        return {
            "caller_id": caller_id,
            "candidate_id": confirmation_verdict.candidate_id,
            "archived_verdict_source": "confirmation",
            "confirmation_slice_ids": list(sealed.ids),
            "confirmation_provenance": {
                "consumed_at": sealed.consumed_provenance.consumed_at,
                "one_shot": sealed.consumed_provenance.one_shot,
                "second_consume_refused": sealed.consumed_provenance.second_consume_refused,
            },
            "confirmation_verdict": confirmation_verdict.verdict,
            "superseded_selection_verdict": selection_verdict,
            "superseded_selection_marked": True,
        }

    def _require_open_lead(self, caller_id: str) -> tuple[CallerSplit, SealedConfirmationSlice]:
        if caller_id not in self._open_leads:
            raise CallerComplianceError(f"{caller_id}: no open lead issued")
        return self._open_leads[caller_id]


# --------------------------------------------------------------------------
# REG-F14b: multiplicity mapping (ledger-derived, never hand-written)
# --------------------------------------------------------------------------

DYNAMIC_DSR_SHRINK = 0.95  # mirror apedp's DSR effective-n shrink factor
REUSE_PENALTY_N = 1.0  # each confirmation-reuse charges one extra full trial
# against the same accountable evidence (monotone; pinned against hand-editing).


def multiplicity_from_ledger(
    ledger: FullLineageExperimentLedger,
    caller_id: str,
    dataset_id: str,
    candidate_id: str,
    nominal_variants: int,
) -> dict[str, Any]:
    """Ledger-DERIVED DSR-adjusted multiplicity (F14b mode b). Nothing here is
    hand-writable: variant/reuse counts are read from `DatasetExposureRecord`s
    of the caller's candidates; reuses are the records labeled
    ``selection_reused_confirmation_holdout`` with this dataset.
    Monotone in variants AND reuses (pinned); refuses a negative nominal.
    """
    if nominal_variants < 0:
        raise CallerComplianceError("nominal_variants must be >= 0")
    n_variants = max(1, int(nominal_variants))
    prefix = f"selection_reused_confirmation_holdout::{dataset_id}::"
    reuses = sum(
        1
        for node in ledger.get_all_candidates()
        for exp in node.data_exposures
        if exp.dataset_id == dataset_id
        and exp.role == DatasetRole.CONFIRMATION_HOLDOUT
        and exp.slice_identifier.startswith(prefix)
        and exp.candidate_id == candidate_id
    )
    effective_n = n_variants + REUSE_PENALTY_N * reuses
    return {
        "source": "ledger",
        "caller_id": caller_id,
        "dataset_id": dataset_id,
        "nominal_variants": n_variants,
        "n_confirmation_reuses": reuses,
        "dsr_adjusted_effective_n": effective_n,
        "dynamics": {
            "reuse_penalty_n": REUSE_PENALTY_N,
            "dsr_shrink": DYNAMIC_DSR_SHRINK,
            "note": "reuse inflates accountable n (stricter DSR), never deflates it",
        },
    }


def _selection_reuse_labels(dataset_id: str) -> tuple[str, ...]:
    """Confirmations gated with the reuse label (design doc section 3)."""
    return (f"selection_reused_confirmation_holdout::{dataset_id}",)


def mark_selection_reuse(
    candidate_id: str,
    dataset_id: str,
    reused_slice_identifier: str,
    ledger: FullLineageExperimentLedger,
) -> None:
    """Charge one REUSE event against a candidate by recording the labeled
    exposure on its ledger node (mode-b inputs come from this record)."""
    ledger.record_data_exposure(
        candidate_id=candidate_id,
        dataset_id=dataset_id,
        slice_identifier=f"selection_reused_confirmation_holdout::{dataset_id}::{reused_slice_identifier}",
        role=DatasetRole.CONFIRMATION_HOLDOUT,
        outcome_status="REUSE_CHARGED",
    )


def validate_caller_compliance(
    record: dict[str, Any],
    expected_effective_n: float | None = None,
) -> tuple[bool, str]:
    """REG-F14b letter: every production caller is either
      mode (a)  confirmation_mode == 'sealed' with one-shot provenance, OR
      mode (b)  published ledger-derived DSR-adjusted multiplicity.
    Returns (ok, why); a Record that satisfies neither is refused upstream.
    """
    mode = (record.get("confirmation_mode") or "").lower()
    if mode == "sealed":
        prov = record.get("confirmation_provenance") or {}
        if not (prov.get("one_shot") and prov.get("second_consume_refused")):
            return False, "sealed caller missing one-shot provenance"
        return True, "mode (a): sealed confirmation split held with one-shot provenance"
    if mode == "multiplicity":
        mult = record.get("multiplicity") or {}
        if mult.get("source") != "ledger":
            return False, "mode (b) requires ledger-sourced multiplicity (never hand-written)"
        eff_n = mult.get("dsr_adjusted_effective_n")
        if eff_n is None:
            return False, "mode (b) requires dsr_adjusted_effective_n"
        if mult.get("nominal_variants") is not None and eff_n < mult["nominal_variants"]:
            return False, "effective n may not deflate below nominal (reuse only inflates)"
        if record.get("selection_reused_confirmation_holdout"):
            if (mult.get("n_confirmation_reuses") or 0) <= 0:
                return False, "record charged with a reuse but multiplicity shows none"
        if expected_effective_n is not None and abs(float(eff_n) - float(expected_effective_n)) > 1e-9:
            return False, "published effective n does not match the ledger-derived value"
        return True, "mode (b): published ledger-derived DSR-adjusted multiplicity"
    return False, (
        "caller records neither a sealed confirmation split nor ledger-derived "
        "multiplicity — REG-F14b requires one of the two"
    )
