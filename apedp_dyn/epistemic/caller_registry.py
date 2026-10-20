"""Production-caller compliance registry (REG-F14b per-caller wiring).

Register anchor: docs/SYSTEM_FLAWS_AUDIT.md — REG-F14 / REG-F14b.

Census criterion: every ``production``-tagged pipeline that consumes dataset
slices (synthetic or market) and publishes promotion/verdict records. The
census below is the auditable enumeration; each registered caller module now
exposes ``publish_caller_compliance_record`` (thin delegating wrapper), which
is the per-caller WIRING the register demanded.

Wired mode: **mode (b)** — every census caller publishes a
``confirmation_mode = "multiplicity"`` record whose DSR-adjusted effective n
is derived ENTIRELY from ``FullLineageExperimentLedger`` exposure counts
(``multiplicity_from_ledger``; nominal variants = distinct candidates with
selection-family exposures on the caller's dataset; reuse charges via the
``selection_reused_confirmation_holdout`` labels). Nothing is hand-writable:
the registry refuses a published effective n that does not equal the ledger
recomputation, and deflation below nominal is refused upstream by the
protocol validator.

Mode (a) — sealed confirmation splits — is now operationally wired per-caller
through this registry: :func:`issue_sealed_confirmation_lead` issues a lead
over the caller's OWN real slice ids against its registered dataset family
(the census is closed: leads cannot be issued for unknown callers or unknown
datasets), and :func:`publish_sealed_compliance_record` refuses to publish a
sealed record until the confirmation slice has actually been spent (exactly
once, via ``SealedConfirmationSlice.consume``) and the archived verdict
source is the confirmation slice. ``blind_benchmark`` is the first census
caller wired at its own module level (its blind cases are the natural sealed
confirmation family); the other five census callers remain published in mode
(b) by default until a real sealed lead is spent in a live promotion event —
that wiring gap is disclosed in the register, not silently claimed.

Selection-stage exposure discipline: at selection time a caller records its
candidate variants as ``DatasetRole.TRAINING_DEVELOPMENT`` exposures on its
ledger node (additive helper :func:`record_selection_exposure`), which is
what makes the derived multiplicity live instead of nominal.

The register's F14 acceptance letter (end-to-end: pool selected on slice A,
archived head confirmed ONCE on sealed slice B, verdict from B only) stays
pinned in ``tests/test_dyn_caller_data_protocol.py``; this registry plus its
pins (``tests/test_dyn_caller_compliance_wiring.py``) close the per-caller
wiring gate in mode (b).
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Iterable

from apedp_dyn.epistemic.caller_data_protocol import (
    CallerSplit,
    CallerSplitProtocol,
    ConfirmationProvenance,
    ConfirmationVerdict,
    SealedConfirmationSlice,
    multiplicity_from_ledger,
    validate_caller_compliance,
)
from apedp_dyn.epistemic.experiment_ledger import (
    DatasetRole,
    FullLineageExperimentLedger,
)


class CallerRegistryComplianceError(ValueError):
    """The published records do not cover or satisfy the production-caller census."""


@dataclass(frozen=True)
class CallerRegistration:
    """One production caller in the compliance census."""

    caller_id: str
    module: str
    dataset_family: str  # default dataset_id for the caller's selection family
    description: str


# The auditable census (production-tagged data-consuming verdict publishers).
PRODUCTION_CALLERS: tuple[CallerRegistration, ...] = (
    CallerRegistration(
        "dyn_sweep",
        "apedp_dyn.dyn_sweep",
        "dyn_referee_synthetic",
        "frozen DUFF/COUP/STAB four-noise referee sweep (synthetic problem families)",
    ),
    CallerRegistration(
        "multi_asset_signals",
        "apedp_dyn.sandbox.multi_asset_signals_sandbox",
        "multi_asset_signals_panel",
        "multi-asset signal selection sweeps over the market signals panel",
    ),
    CallerRegistration(
        "real_market_l2_sweep",
        "apedp_dyn.sandbox.real_market_l2_sweep",
        "real_market_l2",
        "real-market L2 discovery sweep publishing promotion verdicts",
    ),
    CallerRegistration(
        "blind_benchmark",
        "apedp_dyn.sandbox.blind_benchmark",
        "blind_benchmark_suite",
        "air-gapped blind benchmark runner (cryptographically sealed suites)",
    ),
    CallerRegistration(
        "calibrated_benchmark_suite",
        "apedp_dyn.sandbox.calibrated_benchmark_suite",
        "calibrated_benchmark_tiers",
        "three-tier calibrated benchmark (referee FPR/CAR/VRR, power, transfer)",
    ),
    CallerRegistration(
        "canonical_ledger",
        "apedp_dyn.sandbox.canonical_ledger",
        "canonical_execution_ledger",
        "canonical execution ledger evaluating promotion candidates on fee-modeled fills",
    ),
)

_CALLER_BY_ID = {reg.caller_id: reg for reg in PRODUCTION_CALLERS}


def caller_registration(caller_id: str) -> CallerRegistration:
    """Resolve a census registration; unknown ids are refused (census is closed)."""
    reg = _CALLER_BY_ID.get(caller_id)
    if reg is None:
        raise CallerRegistryComplianceError(
            f"{caller_id}: not in the production-caller census "
            f"({sorted(_CALLER_BY_ID)}) — new data-consuming pipelines must be "
            "registered before they publish verdicts"
        )
    return reg


def record_selection_exposure(
    ledger: FullLineageExperimentLedger,
    caller_id: str,
    candidate_id: str,
    dataset_id: str | None = None,
    slice_identifier: str = "selection_pool",
) -> None:
    """Selection-stage wiring: register the candidate on the ledger node and
    record its TRAINING_DEVELOPMENT exposure in the caller's dataset family.

    This is the additive hook each production caller invokes when a candidate
    variant is scored on selection data — it is what makes the mode-(b)
    multiplicity LIVE (ledger-derived, not nominal).
    """
    reg = caller_registration(caller_id)
    ledger.register_candidate(
        candidate_id=candidate_id,
        proposer_id=caller_id,
        formula=f"caller::{caller_id}",
    )
    ledger.record_data_exposure(
        candidate_id=candidate_id,
        dataset_id=dataset_id or reg.dataset_family,
        slice_identifier=slice_identifier,
        role=DatasetRole.TRAINING_DEVELOPMENT,
        outcome_status="IN_PROGRESS",
    )


def derive_nominal_variants(
    ledger: FullLineageExperimentLedger | None,
    dataset_id: str,
) -> int:
    """n = DISTINCT candidates charged against this caller's selection-family
    exposures (design doc §3: ledger-derived, never hand-written)."""
    if ledger is None:
        return 0
    return len({
        exposure.candidate_id
        for node in ledger.get_all_candidates()
        for exposure in node.data_exposures
        if exposure.dataset_id == dataset_id
        and exposure.role == DatasetRole.TRAINING_DEVELOPMENT
    })


def has_confirmation_reuse(
    ledger: FullLineageExperimentLedger | None,
    dataset_id: str,
    candidate_id: str,
) -> bool:
    """Whether the head candidate carries a ``selection_reused_confirmation_holdout``
    charge on this dataset (the reuse labels are written by
    ``caller_data_protocol.mark_selection_reuse``)."""
    if ledger is None:
        return False
    prefix = f"selection_reused_confirmation_holdout::{dataset_id}::"
    return any(
        exposure.dataset_id == dataset_id
        and exposure.role == DatasetRole.CONFIRMATION_HOLDOUT
        and exposure.slice_identifier.startswith(prefix)
        and exposure.candidate_id == candidate_id
        for node in ledger.get_all_candidates()
        for exposure in node.data_exposures
    )


def publish_compliance_record(
    ledger: FullLineageExperimentLedger | None,
    caller_id: str,
    candidate_id: str,
    dataset_id: str | None = None,
    nominal_variants: int | None = None,
) -> dict[str, Any]:
    """Build + validate the caller's REG-F14b mode-(b) record.

    All quantities are ledger-derived. ``nominal_variants`` defaults to the
    ledger-distinct selection-population count; the reuse charge and both
    multiplicities come from ``multiplicity_from_ledger``. The record is
    validated at publish time — publishing a non-compliant record raises.
    """
    reg = caller_registration(caller_id)
    ds_id = dataset_id or reg.dataset_family
    n_variants = derive_nominal_variants(ledger, ds_id) if nominal_variants is None else int(nominal_variants)
    multiplicity = multiplicity_from_ledger(
        ledger=ledger,
        caller_id=caller_id,
        dataset_id=ds_id,
        candidate_id=candidate_id,
        nominal_variants=n_variants,
    )
    record: dict[str, Any] = {
        "caller_id": caller_id,
        "confirmation_mode": "multiplicity",
        "dataset_id": ds_id,
        "candidate_id": candidate_id,
        "multiplicity": multiplicity,
        "selection_reused_confirmation_holdout": has_confirmation_reuse(ledger, ds_id, candidate_id),
    }
    ok, why = validate_caller_compliance(record)
    if not ok:
        raise CallerRegistryComplianceError(f"{caller_id}: published record refused — {why}")
    return record


# --------------------------------------------------------------------------
# REG-F14 mode (a): operational sealed-confirmation-split wiring.
# --------------------------------------------------------------------------

def issue_sealed_confirmation_lead(
    ledger: FullLineageExperimentLedger | None,
    caller_id: str,
    selection_ids: list[str],
    confirmation_ids: list[str],
    dataset_id: str | None = None,
    protocol: CallerSplitProtocol | None = None,
) -> tuple[CallerSplitProtocol, CallerSplit, SealedConfirmationSlice]:
    """Issue the caller's own real three-role lead (mode (a) wiring).

    Census discipline enforced here:
      * the caller must be a registered production caller (closed census);
      * the split must be issued against the caller's REGISTERED dataset
        family (``dataset_id`` defaults to it and must match exactly) — a
        caller cannot claim a split over a dataset outside its census row;
      * the protocol instance is returned, so the same object that issued
        the lead must be used for consumption and archiving (no re-issuance
        under a fresh protocol: a caller cannot be issued twice per lead).
    """
    reg = caller_registration(caller_id)
    ds_id = dataset_id or reg.dataset_family
    if ds_id != reg.dataset_family:
        raise CallerRegistryComplianceError(
            f"{caller_id}: sealed lead dataset {ds_id!r} is not this caller's "
            f"registered dataset family {reg.dataset_family!r} — the census is closed"
        )
    protocol = protocol or CallerSplitProtocol(ledger)
    split, sealed = protocol.issue_caller_split(
        caller_id=caller_id,
        dataset_id=ds_id,
        selection_ids=list(selection_ids),
        confirmation_ids=list(confirmation_ids),
    )
    return protocol, split, sealed


def publish_sealed_compliance_record(
    ledger: FullLineageExperimentLedger | None,
    caller_id: str,
    candidate_id: str,
    protocol: CallerSplitProtocol,
    sealed: SealedConfirmationSlice,
    confirmation_verdict: ConfirmationVerdict,
    selection_verdict: Any = None,
) -> dict[str, Any]:
    """Build + validate this caller's REG-F14b mode-(a) record.

    Refuses to publish (fail-closed) unless EVERY letter of F14's sealed path
    actually happened on the objects passed in:
      * the sealed confirmation slice was CONSUMED (exactly once — an
        unconsumed slice means the only evaluation so far was on selection
        data, which may never be archived as the promoted claim);
      * the verdict provenance carries one-shot / second-consume-refused
        flags and belongs to this sealed slice's ids and dataset family;
      * the archived record labels the CONFIRMATION slice as the verdict
        source and marks the selection verdict superseded.
    """
    reg = caller_registration(caller_id)
    if not sealed.consumed:
        raise CallerRegistryComplianceError(
            f"{caller_id}: mode-(a) record refused — the sealed confirmation "
            "slice has NOT been consumed; a promotion verdict cannot be "
            "archived from selection-only evaluation (REG-F14: evaluated ONCE)"
        )
    prov: ConfirmationProvenance | None = confirmation_verdict.provenance
    if prov is None:
        raise CallerRegistryComplianceError(
            f"{caller_id}: confirmation verdict carries no one-shot provenance"
        )
    if not tuple(prov.slice_ids) == tuple(sealed.ids):
        raise CallerRegistryComplianceError(
            f"{caller_id}: verdict provenance does not belong to the sealed "
            "confirmation slice held"
        )
    if prov.slice_dataset_id != reg.dataset_family:
        raise CallerRegistryComplianceError(
            f"{caller_id}: sealed slice dataset {prov.slice_dataset_id!r} is not "
            f"this caller's registered dataset family {reg.dataset_family!r}"
        )
    archived = protocol.archive_promotion_verdict(
        caller_id=caller_id,
        confirmation_verdict=confirmation_verdict,
        selection_verdict=selection_verdict,
    )
    if archived["candidate_id"] != candidate_id:
        raise CallerRegistryComplianceError(
            f"{caller_id}: archived head is {archived['candidate_id']!r}, not {candidate_id!r}"
        )
    record: dict[str, Any] = {
        "caller_id": caller_id,
        "confirmation_mode": "sealed",
        "dataset_id": reg.dataset_family,
        "candidate_id": candidate_id,
        "confirmation_slice_ids": list(archived["confirmation_slice_ids"]),
        "confirmation_provenance": {
            "consumed_at": archived["confirmation_provenance"]["consumed_at"],
            "one_shot": archived["confirmation_provenance"]["one_shot"],
            "second_consume_refused": archived["confirmation_provenance"]["second_consume_refused"],
            "candidate_id": candidate_id,
            "slice_ids": list(archived["confirmation_slice_ids"]),
        },
        "archived_verdict_source": archived["archived_verdict_source"],
        "superseded_selection_verdict_marked": archived["superseded_selection_marked"],
        "confirmation_verdict": archived["confirmation_verdict"],
    }
    ok, why = validate_caller_compliance(record)
    if not ok:
        raise CallerRegistryComplianceError(f"{caller_id}: published sealed record refused — {why}")
    return record


def caller_ids() -> tuple[str, ...]:
    """The census ids (stable order for pins and reports)."""
    return tuple(_CALLER_BY_ID)


def validate_census_compliance(records: Iterable[dict[str, Any]]) -> dict[str, tuple[bool, str]]:
    """Validate a *set* of published records against the WHOLE census.

    Refuses: a census caller with no published record, an unknown caller id,
    and any record that fails :func:`validate_caller_compliance`. Returns the
    per-caller (ok, why) verdicts — the wiring gate the register described.
    """
    by_id: dict[str, dict[str, Any]] = {}
    for rec in records:
        cid = rec.get("caller_id")
        if cid in by_id:
            raise CallerRegistryComplianceError(f"duplicate published record for caller {cid}")
        if cid not in _CALLER_BY_ID:
            raise CallerRegistryComplianceError(
                f"published record for UNKNOWN caller {cid} — census is closed; register the caller first"
            )
        by_id[cid] = rec
    verdicts: dict[str, tuple[bool, str]] = {}
    for reg in PRODUCTION_CALLERS:
        rec = by_id.get(reg.caller_id)
        if rec is None:
            verdicts[reg.caller_id] = (
                False,
                f"no published compliance record — REG-F14b requires every production "
                f"caller to certify mode (a) or (b)",
            )
            continue
        ok, why = validate_caller_compliance(rec)
        verdicts[reg.caller_id] = (ok, f"{why} (dataset family {rec.get('dataset_id', '?')})")
    return verdicts


def census_is_compliant(records: Iterable[dict[str, Any]]) -> bool:
    """True iff every census caller has a validating record."""
    return all(ok for ok, _ in validate_census_compliance(records).values())


def module_publisher(registration: CallerRegistration):
    """Import-time helper for pins: fetch the wired publisher from the caller module."""
    mod = import_module(registration.module)
    publisher = getattr(mod, "publish_caller_compliance_record", None)
    if publisher is None:
        raise CallerRegistryComplianceError(
            f"{registration.module}: census caller is not wired — "
            f"missing publish_caller_compliance_record (per-caller wiring gate)"
        )
    return publisher
