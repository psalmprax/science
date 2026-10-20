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
protocol validator. Mode (a) — sealed confirmation splits — remains the
per-caller operational upgrade (the ``CallerSplitProtocol`` reference path)
and is already accepted by the same validator; the census wires every caller
into mode (b) because no production pipeline currently holds an operational
sealed confirmation split.

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
