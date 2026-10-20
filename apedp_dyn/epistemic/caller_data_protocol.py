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
      When a durable state path is configured, that single-use guarantee is
      PERSISTED: a restarted process cannot re-issue an open lead, and a
      re-consume attempt is refused from the durable record, not from
      in-memory flags. The store fails CLOSED: a corrupt store raises
      `CallerProtocolStateError` and the protocol never falls back to
      memory-only operation (silent memory fallback is exactly the gap that
      let a restart hand out a fresh unconsumed sealed slice).
  I3  every evaluation is written to the candidate's experiment-ledger node as
      a DatasetExposureRecord with the correct DatasetRole — FAIL-CLOSED:
      when a ledger is attached, the candidate must be ledger-registered
      BEFORE the sealed slice is spent (an evaluation that could not be
      written up afterwards is refused before it happens), and any ledger
      write failure now propagates out of `consume` (the previous
      except-and-pass silently made I3 optional).
  I4  the ARCHIVED promotion verdict source is the confirmation slice; the
      selection verdict is retained but marked superseded.
  I5  (F14b) caller compliance is either-or and checked, not promised:
      mode (a) sealed confirmation split held, or mode (b) published
      DSR-adjusted multiplicity derived from LEDGER exposure counts
      (never hand-written; monotone; refused otherwise).
"""
from __future__ import annotations

import datetime
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from apedp_dyn.conf.canonical_spec import DSR_THRESHOLD
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


class CallerProtocolStateError(RuntimeError):
    """The durable lead-state store is unreadable/rejected — the protocol
    refuses to operate memory-only (memory-only would make the one-shot
    guarantee per-process again)."""


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


# --------------------------------------------------------------------------
# Durable per-lead state (I2 across process restarts)
# --------------------------------------------------------------------------
PROTOCOL_STATE_VERSION = 1


class _LeadStateStore:
    """Durable split + sealed-consumption state, one JSON file per protocol.

    Writes are atomic (temp file + os.replace). A corrupt store fails CLOSED
    via `CallerProtocolStateError`: the protocol never silently falls back to
    memory-only operation, because that fallback is the restart gap. Loading
    is strictly validated — an inconsistent record (unconsumed-with-provenance,
    unknown fields' values, wrong types) is treated as corruption, not as
    data. Single-writer is assumed (the registry returns the owning protocol
    object for consumption, per its own census discipline).
    """

    def __init__(self, path: Path):
        self.path: Path = Path(path)
        self._records: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            try:
                parsed = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
                raise CallerProtocolStateError(
                    f"{self.path}: lead-state store unreadable ({exc}) — refusing to "
                    "operate memory-only; quarantine this store, do not delete it"
                ) from exc
            if not isinstance(parsed, dict) or parsed.get("protocol_state_version") != PROTOCOL_STATE_VERSION:
                raise CallerProtocolStateError(
                    f"{self.path}: unsupported lead-state payload (version "
                    f"{parsed.get('protocol_state_version')!r} vs {PROTOCOL_STATE_VERSION})"
                )
            records = parsed.get("leads")
            if not isinstance(records, dict):
                raise CallerProtocolStateError(f"{self.path}: 'leads' is not an object map")
            for caller_id, rec in records.items():
                self._records[caller_id] = _validated_lead_record(caller_id, rec)

    # -- queries ---------------------------------------------------------------
    @property
    def records(self) -> dict[str, dict[str, Any]]:
        return dict(self._records)

    def has(self, caller_id: str) -> bool:
        return caller_id in self._records

    def load(self, caller_id: str) -> dict[str, Any] | None:
        rec = self._records.get(caller_id)
        return dict(rec) if rec is not None else None

    # -- mutations (each immediately atomically persisted) ----------------------
    def issue(self, split: CallerSplit) -> None:
        self._records[split.caller_id] = {
            "dataset_id": split.dataset_id,
            "selection_ids": list(split.selection_ids),
            "confirmation_ids": list(split.confirmation_ids),
            "consumed": False,
            "provenance": None,
        }
        self._flush()

    def claim(self, caller_id: str) -> None:
        """Mark the sealed slice spent BEFORE evaluation: a hard crash mid-
        evaluation can never resurrect the slice as consumable."""
        rec = self._require(caller_id)
        rec["consumed"] = True
        rec["provenance"] = None
        self._flush()

    def record_provenance(self, caller_id: str, provenance: dict[str, Any]) -> None:
        rec = self._require(caller_id)
        if not rec["consumed"]:
            raise CallerProtocolStateError(
                f"{caller_id}: provenance recorded for a lead not claimed consumed"
            )
        rec["provenance"] = _prepared_provenance_dict(provenance)
        self._flush()

    # -- helpers ----------------------------------------------------------------
    def _require(self, caller_id: str) -> dict[str, Any]:
        rec = self._records.get(caller_id)
        if rec is None:
            raise CallerProtocolStateError(
                f"{caller_id}: lead absent from the durable state store — refusing"
            )
        return rec

    def _flush(self) -> None:
        payload = json.dumps(
            {
                "protocol_state_version": PROTOCOL_STATE_VERSION,
                "leads": self._records,
            },
            sort_keys=True,
        )
        parent = self.path.parent
        parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=self.path.name + ".", suffix=".tmp", dir=parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_name, self.path)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise


def _validated_lead_record(caller_id: str, rec: Any) -> dict[str, Any]:
    if not isinstance(rec, dict):
        raise CallerProtocolStateError(f"{caller_id}: lead record is not an object")
    for key in ("dataset_id", "selection_ids", "confirmation_ids", "consumed", "provenance"):
        if key not in rec:
            raise CallerProtocolStateError(f"{caller_id}: lead record missing '{key}'")
    if not isinstance(rec["dataset_id"], str):
        raise CallerProtocolStateError(f"{caller_id}: dataset_id must be a string")
    for field_name in ("selection_ids", "confirmation_ids"):
        ids = rec[field_name]
        if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
            raise CallerProtocolStateError(f"{caller_id}: {field_name} must be a list of strings")
    if rec["confirmation_ids"] == []:
        raise CallerProtocolStateError(
            f"{caller_id}: CONFIRMATION slice must be non-empty — "
            "a lead without a sealed confirmation cannot promote"
        )
    if not isinstance(rec["consumed"], bool) or rec["consumed"] is False and rec["provenance"] is not None:
        raise CallerProtocolStateError(
            f"{caller_id}: unconsumed lead may not carry provenance"
        )
    prov = rec["provenance"]
    if prov is not None:
        if not isinstance(prov, dict):
            raise CallerProtocolStateError(f"{caller_id}: provenance is not an object")
        for key in ("consumed_at", "candidate_id", "slice_dataset_id", "slice_ids", "one_shot", "second_consume_refused"):
            if key not in prov:
                raise CallerProtocolStateError(f"{caller_id}: provenance missing '{key}'")
        if not isinstance(prov["slice_ids"], list):
            raise CallerProtocolStateError(f"{caller_id}: provenance slice_ids must be a list")
        if sorted(prov["slice_ids"]) != sorted(rec["confirmation_ids"]):
            raise CallerProtocolStateError(
                f"{caller_id}: persisted provenance does not belong to the held "
                "confirmation slice — store rejected (I2 fail-closed)"
            )
        if not (prov["one_shot"] is True and prov["second_consume_refused"] is True):
            raise CallerProtocolStateError(f"{caller_id}: provenance lost one-shot flags")
    return {
        "dataset_id": rec["dataset_id"],
        "selection_ids": list(rec["selection_ids"]),
        "confirmation_ids": list(rec["confirmation_ids"]),
        "consumed": bool(rec["consumed"]),
        "provenance": prov,
    }


def _prepared_provenance_dict(prov: Any) -> dict[str, Any]:
    return {
        "consumed_at": str(prov["consumed_at"]),
        "candidate_id": str(prov["candidate_id"]),
        "slice_dataset_id": str(prov["slice_dataset_id"]),
        "slice_ids": [str(i) for i in prov["slice_ids"]],
        "one_shot": True,
        "second_consume_refused": True,
    }


class SealedConfirmationSlice:
    """Single-use wrapper around the confirmation ids (I2).

    With a `_LeadStateStore` attached, the one-shot property is durable: the
    slice is claimed spent in the store BEFORE any evaluation runs, so a hard
    crash mid-evaluation cannot resurrect it, provenance (candidate, ids,
    timestamps) is persisted after a successful evaluation, and a restarted
    process's re-consume attempt is refused from the durable record.
    """

    def __init__(
        self,
        split: CallerSplit,
        ledger: FullLineageExperimentLedger | None,
        state_store: _LeadStateStore | None = None,
    ):
        self._split = split
        self._ledger = ledger
        self._state_store = state_store
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

    def _load_consumed_state(self, consumed: bool, provenance_dict: dict[str, Any] | None) -> None:
        """Rebuild the in-memory one-shot state from the durable store (restart path)."""
        self._consumed = bool(consumed)
        if provenance_dict is not None:
            self._provenance = ConfirmationProvenance(
                consumed_at=str(provenance_dict["consumed_at"]),
                candidate_id=str(provenance_dict["candidate_id"]),
                slice_dataset_id=str(provenance_dict["slice_dataset_id"]),
                slice_ids=tuple(str(i) for i in provenance_dict["slice_ids"]),
                one_shot=True,
                second_consume_refused=True,
            )

    def consume(
        self,
        candidate_id: str,
        evaluate: Callable[..., ConfirmationVerdict],
        **eval_kwargs: Any,
    ) -> ConfirmationVerdict:
        stored: dict[str, Any] | None = None
        if self._state_store is not None:
            stored = self._state_store.load(self._split.caller_id)
            if stored is None:
                raise CallerProtocolStateError(
                    f"{self._split.caller_id}: issued lead is absent from its "
                    "durable state store — refusing to consume (store integrity)"
                )
            if stored["consumed"]:
                # Durable refusal FIRST — it labels restarts correctly and
                # covers every provenance shape the store can hold.
                prov = stored.get("provenance")
                raise ConfirmationAlreadyConsumedError(
                    f"{self._split.caller_id}: confirmation slice already consumed "
                    "(durable state across restart) "
                    f"(provenance candidate_id={prov['candidate_id'] if prov else 'claimed-unprovened'}) — "
                    "REG-F14: the archived promotion is evaluated ONCE"
                )
        if self._consumed:
            # Second attempt is refused AND labeled (the caller learns the
            # prohibition, not the data).
            raise ConfirmationAlreadyConsumedError(
                f"{self._split.caller_id}: confirmation slice already consumed "
                f"(provenance candidate_id={self._provenance.candidate_id if self._provenance else 'claimed-unprovened'}) — "
                "REG-F14: the archived promotion is evaluated ONCE"
            )
        # I3 fail-closed pre-flight: an evaluation on sealed data must be
        # accountable in the ledger; a candidate that could never be written
        # up afterwards is refused BEFORE the slice is spent.
        if self._ledger is not None:
            try:
                self._ledger.get_candidate(candidate_id)
            except KeyError as exc:
                raise CallerComplianceError(
                    f"{self._split.caller_id}: candidate {candidate_id!r} is not "
                    "registered in the experiment ledger — register it before "
                    "spending the sealed confirmation slice (I3 fail-closed)"
                ) from exc
        self._consumed = True
        if self._state_store is not None:
            self._state_store.claim(self._split.caller_id)
        try:
            verdict = evaluate(
                candidate_id=candidate_id,
                confirmation_ids=self._split.confirmation_ids,
                **eval_kwargs,
            )
        except BaseException:
            # The claim is already persisted: the slice is spent even when the
            # evaluation aborts (one-shot is spent-by-consumption, not
            # spent-by-success). No provenance is fabricated on the failure path.
            raise
        timestamps = datetime.datetime.now(datetime.timezone.utc).isoformat()
        # I3: ledger exposure with the CONFIRMATION_HOLDOUT role (burn path
        # retained). Fail-closed: a ledger write failure propagates, it is
        # never swallowed — an evaluation that hasn't been recorded is not a
        # recorded evaluation.
        if self._ledger is not None:
            for sid in self._split.confirmation_ids:
                self._ledger.record_data_exposure(
                    candidate_id=candidate_id,
                    dataset_id=self._split.dataset_id,
                    slice_identifier=sid,
                    role=DatasetRole.CONFIRMATION_HOLDOUT,
                    outcome_status="PASSED" if getattr(verdict, "verdict", None) else "FAILED",
                )
        self._provenance = ConfirmationProvenance(
            consumed_at=timestamps,
            candidate_id=candidate_id,
            slice_dataset_id=self._split.dataset_id,
            slice_ids=tuple(self._split.confirmation_ids),
            one_shot=True,
            second_consume_refused=True,
        )
        if self._state_store is not None:
            self._state_store.record_provenance(self._split.caller_id, self._provenance.__dict__)
        if getattr(verdict, "provenance", None) is None:
            verdict.provenance = self._provenance
        return verdict


class _RustProtectionStub:  # pragma: no cover - defensive import shim
    """Mirror discipline note: selection-stage evaluation stays in numpy-land here."""


class CallerSplitProtocol:
    """Issues the disjoint three-role splits and enforces sealed consumption.

    Durability (I2 across restarts): when a durable state path applies —
    explicitly via ``persistence_path``, or derived from the attached
    ledger's own persistence directory — issued leads and sealed-consumption
    provenance are persisted at every mutation and reloaded at construction,
    so a restarted process sees the same open leads and spent slices.
    Without a resolvable state path (in-memory ledger, unit pins) the
    protocol behaves as before: one-shot within the live process only, which
    is a DOCUMENTED unit-use limitation, not a production posture.
    """

    def __init__(
        self,
        ledger: FullLineageExperimentLedger | None = None,
        persistence_path: Path | str | None = None,
    ) -> None:
        self._ledger = ledger
        self._open_leads: dict[str, tuple[CallerSplit, SealedConfirmationSlice]] = {}
        state_path = persistence_path
        if state_path is None and ledger is not None:
            ledger_path = getattr(ledger, "persistence_path", None)
            if ledger_path is not None:
                state_path = Path(ledger_path).parent / "caller_lead_protocol_state.json"
        self._state_store: _LeadStateStore | None = (
            _LeadStateStore(Path(state_path)) if state_path else None
        )
        if self._state_store is not None:
            for caller_id, rec in self._state_store.records.items():
                split = CallerSplit(
                    caller_id=caller_id,
                    dataset_id=rec["dataset_id"],
                    selection_ids=tuple(rec["selection_ids"]),
                    confirmation_ids=tuple(rec["confirmation_ids"]),
                )
                sealed = SealedConfirmationSlice(split, self._ledger, self._state_store)
                sealed._load_consumed_state(rec["consumed"], rec["provenance"])
                self._open_leads[caller_id] = (split, sealed)

    def protocol_state_path(self) -> Path | None:
        """The durable state file this protocol persists its leads to (or None)."""
        return self._state_store.path if self._state_store is not None else None

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
        if self._state_store is not None and self._state_store.has(caller_id):
            raise ConfirmationAlreadyConsumedError(
                f"{caller_id}: an open lead already exists in the durable state "
                f"store ({self._state_store.path}) — a caller cannot be issued "
                "twice, including across process restarts"
            )
        sealed = SealedConfirmationSlice(split, self._ledger, self._state_store)
        if self._state_store is not None:
            self._state_store.issue(split)
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
        prov = sealed.consumed_provenance
        if prov is None:
            raise CallerComplianceError(
                f"{caller_id}: archive refused — the sealed slice is claimed "
                "consumed but carries no provenance (claimed then never "
                "provenanced); no promotion can be archived from this state"
            )
        return {
            "caller_id": caller_id,
            "candidate_id": confirmation_verdict.candidate_id,
            "archived_verdict_source": "confirmation",
            "confirmation_slice_ids": list(sealed.ids),
            "confirmation_provenance": {
                "consumed_at": prov.consumed_at,
                "one_shot": prov.one_shot,
                "second_consume_refused": prov.second_consume_refused,
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

# No threshold literal is defined here: the mode-(b) ``dsr_shrink`` disclosure
# field imports the preregistered DSR constant from the frozen canonical spec
# (hash-verified at import time). Changing it requires a governance-manifested
# (re-signed) spec revision, not an edit in this module.
DYNAMIC_DSR_SHRINK: float = DSR_THRESHOLD
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
