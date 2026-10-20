"""Pins for the REG-F14b per-caller wiring (production-caller census).

Register anchor: docs/SYSTEM_FLAWS_AUDIT.md — REG-F14 / REG-F14b (per-caller
wiring gate, mode (b) ledger-derived multiplicity across the census; mode (a)
sealed splits remain the optional upgrade validated by the same validator).
Design + reference implementation: docs/CALLER_DATA_THREE_ROLE_PROTOCOL.md,
apedp_dyn/epistemic/caller_data_protocol.py; the wiring census lives in
apedp_dyn/epistemic/caller_registry.py.
"""

from __future__ import annotations

import pytest

from apedp_dyn.epistemic.caller_data_protocol import (
    mark_selection_reuse,
    validate_caller_compliance,
)
from apedp_dyn.epistemic.caller_registry import (
    PRODUCTION_CALLERS,
    CallerRegistryComplianceError,
    CallerRegistration,
    census_is_compliant,
    module_publisher,
    validate_census_compliance,
)
from apedp_dyn.epistemic.experiment_ledger import FullLineageExperimentLedger

SELECTION_DATASET = "multi_asset_signals_panel"


def _populated_ledger(n_candidates: int = 3,
                      caller_id: str = "multi_asset_signals",
                      dataset_id: str = SELECTION_DATASET) -> FullLineageExperimentLedger:
    """A fresh in-memory ledger with n candidates scored on the caller's SELECTION pool."""
    ledger = FullLineageExperimentLedger()
    from apedp_dyn.epistemic.caller_registry import record_selection_exposure

    for i in range(n_candidates):
        record_selection_exposure(
            ledger, caller_id=caller_id,
            candidate_id=f"variant_{i}", dataset_id=dataset_id,
            slice_identifier="slice_A",
        )
    return ledger


def test_census_is_closed_and_nonempty():
    ids = tuple(reg.caller_id for reg in PRODUCTION_CALLERS)
    assert len(PRODUCTION_CALLERS) >= 5
    assert len(set(ids)) == len(ids)
    assert "multi_asset_signals" in ids and "dyn_sweep" in ids


def test_every_census_module_is_wired():
    """The per-caller wiring gate: each census module must publish."""
    for reg in PRODUCTION_CALLERS:
        publisher = module_publisher(reg)
        assert callable(publisher), reg.caller_id


def test_every_census_record_validates_in_mode_b():
    for reg in PRODUCTION_CALLERS:
        publisher = module_publisher(reg)
        record = publisher(
            ledger=_populated_ledger(caller_id=reg.caller_id, dataset_id=reg.dataset_family),
            candidate_id="variant_0",
        )
        assert record["confirmation_mode"] == "multiplicity"
        assert record["dataset_id"] == reg.dataset_family
        ok, why = validate_caller_compliance(record)
        assert ok, (reg.caller_id, why)
        # Ledger-derived, not hand-written: effective n equals the
        # distinct-candidate count derived from the exposure records.
        assert record["multiplicity"]["source"] == "ledger"
        assert record["multiplicity"]["nominal_variants"] == 3
        assert record["multiplicity"]["dsr_adjusted_effective_n"] == 3


def test_census_gate_refuses_missing_caller():
    records = [module_publisher(reg)(ledger=_populated_ledger(), candidate_id="variant_0")
               for reg in PRODUCTION_CALLERS[:-1]]  # drop the last census caller
    assert not census_is_compliant(records)
    verdicts = validate_census_compliance(records)
    dropped = PRODUCTION_CALLERS[-1].caller_id
    assert verdicts[dropped][0] is False
    assert "no published compliance record" in verdicts[dropped][1]
    # The fully-published census passes the same gate.
    ok = [module_publisher(reg)(ledger=_populated_ledger(), candidate_id="variant_0")
          for reg in PRODUCTION_CALLERS]
    assert census_is_compliant(ok)


def test_census_gate_refuses_unknown_caller_and_duplicates():
    with pytest.raises(CallerRegistryComplianceError):
        module_publisher(
            CallerRegistration("rogue_caller", "apedp_dyn.conf.paths",
                               "rogue_dataset", "not in the census")
        )
    good = module_publisher(PRODUCTION_CALLERS[0])(
        ledger=_populated_ledger(), candidate_id="variant_0")
    bogus = dict(good, caller_id="rogue_caller")
    with pytest.raises(CallerRegistryComplianceError):
        validate_census_compliance([bogus])
    with pytest.raises(CallerRegistryComplianceError):
        validate_census_compliance([good, good])


def test_reuse_charge_flows_into_published_multiplicity():
    ledger = _populated_ledger()
    head = "variant_0"
    mark_selection_reuse(
        candidate_id=head, dataset_id=SELECTION_DATASET,
        reused_slice_identifier="slice_A", ledger=ledger,
    )
    record = module_publisher(
        next(r for r in PRODUCTION_CALLERS if r.caller_id == "multi_asset_signals")
    )(ledger=ledger, candidate_id=head)
    assert record["selection_reused_confirmation_holdout"] is True
    # Reuse only inflates the accountable n (never hand-edited):
    assert record["multiplicity"]["n_confirmation_reuses"] == 1
    assert record["multiplicity"]["dsr_adjusted_effective_n"] == 4
    # ...and the validator matches the recomputation exactly.
    ok, why = validate_caller_compliance(record, expected_effective_n=4)
    assert ok, why


def test_hand_written_multiplicity_refused():
    hand_written = {
        "caller_id": "multi_asset_signals",
        "confirmation_mode": "multiplicity",
        "dataset_id": SELECTION_DATASET,
        "candidate_id": "variant_0",
        "multiplicity": {"source": "self_reported", "dsr_adjusted_effective_n": 1},
    }
    ok, why = validate_caller_compliance(hand_written)
    assert not ok
    assert "ledger-sourced" in why


def test_end_to_end_wiring_letter():
    """Register letter (wiring form): pool scored on slice A drives the
    published record; the archived head's multiplicity is exactly the ledger
    recomputation — selection verdicts can no longer hide multiplicity."""
    ledger = _populated_ledger(n_candidates=5)
    record = module_publisher(
        next(r for r in PRODUCTION_CALLERS if r.caller_id == "multi_asset_signals")
    )(ledger=ledger, candidate_id="variant_4")
    assert record["confirmation_mode"] == "multiplicity"
    assert record["multiplicity"]["nominal_variants"] == 5
    assert record["multiplicity"]["n_confirmation_reuses"] == 0
    assert record["selection_reused_confirmation_holdout"] is False
    assert census_is_compliant([
        module_publisher(reg)(ledger=ledger, candidate_id="variant_4")
        for reg in PRODUCTION_CALLERS
    ])


# --------------------------------------------------------------------------
# REG-F14 mode (a): operational sealed-split wiring (blind_benchmark first).
# --------------------------------------------------------------------------


def _blind_suite_cases() -> list:
    from apedp_dyn.sandbox.blind_benchmark import BlindCase

    return [
        BlindCase(case_id=f"BLIND-fpr-collar-{i}", domain="wire-test",
                  warranted=warranted, run=_never_promote, blind=True,
                  population="unsafe" if not warranted else "valid")
        for i, warranted in [(1, False), (2, True), (3, False)]
    ]


def _never_promote():
    return {"promoted": False, "abstained": True}


def _selection_case_ids() -> list[str]:
    return [f"sel_case_{i}" for i in range(3)]


def test_mode_a_sealed_split_end_to_end_operational():
    """First census caller wired for mode (a) over its OWN real slice ids:
    selection pool scored on A → blind confirmation slice consumed ONCE →
    verdict archived from the sealed slice (selection superseded) → the
    published sealed record validates."""
    from apedp_dyn.sandbox.blind_benchmark import (
        evaluate_confirmation_cases,
        open_sealed_confirmation_lead,
        publish_sealed_compliance_record,
    )

    suite_cases = _blind_suite_cases()
    conf_ids = [c.case_id for c in suite_cases]
    ledger = FullLineageExperimentLedger()

    # Selection-stage scoring of the archived head on the SELECTION family
    # (this is what makes the multiplicity mode live if the split weren't held).
    from apedp_dyn.epistemic.caller_registry import record_selection_exposure

    for sid in _selection_case_ids():
        record_selection_exposure(
            ledger, caller_id="blind_benchmark", candidate_id="variant_A",
            dataset_id="blind_benchmark_suite", slice_identifier=sid,
        )

    protocol, split, sealed = open_sealed_confirmation_lead(
        ledger=ledger,
        selection_case_ids=_selection_case_ids(),
        confirmation_case_ids=conf_ids,
    )
    assert split.selection_ids == tuple(_selection_case_ids())
    assert sealed.ids == tuple(conf_ids)
    assert set(split.selection_ids) & set(sealed.ids) == set()  # disjoint by construction

    # Registry discipline: a suite cannot hold two open sealed leads on the
    # same owning protocol (the protocol refuses re-issuance per caller).
    with pytest.raises(Exception):
        open_sealed_confirmation_lead(
            ledger=ledger,
            selection_case_ids=_selection_case_ids(),
            confirmation_case_ids=conf_ids,
            protocol=protocol,
        )

    # Consume the sealed confirmation slice EXACTLY ONCE (evaluate = suite runs).
    verdict = sealed.consume(
        candidate_id="variant_A",
        evaluate=evaluate_confirmation_cases,
        cases=suite_cases,
    )
    assert verdict.verdict == "PASSED"  # no blind case promoted-unwarranted
    assert verdict.provenance.one_shot is True
    assert verdict.provenance.slice_ids == tuple(conf_ids)

    record = publish_sealed_compliance_record(
        ledger=ledger, candidate_id="variant_A", protocol=protocol, sealed=sealed,
        confirmation_verdict=verdict, selection_verdict={"sharpie": 0.9},
    )
    assert record["confirmation_mode"] == "sealed"
    assert record["archived_verdict_source"] == "confirmation"
    assert record["confirmation_slice_ids"] == conf_ids
    assert record["superseded_selection_verdict_marked"] is True
    ok, why = validate_caller_compliance(record)
    assert ok, why

    # The spent slice cannot be re-watched by a second candidate (REG-F14: ONCE).
    with pytest.raises(Exception):
        sealed.consume(
            candidate_id="variant_B", evaluate=evaluate_confirmation_cases, cases=suite_cases,
        )


def test_mode_a_refused_before_consumption():
    """Fail-closed publication: a sealed record cannot be published while the
    confirmation slice is UNconsumed (selection-only evaluation may never be
    archived as the promoted claim)."""
    from apedp_dyn.epistemic.caller_data_protocol import ConfirmationVerdict
    from apedp_dyn.sandbox.blind_benchmark import (
        open_sealed_confirmation_lead, publish_sealed_compliance_record,
    )

    ledger = FullLineageExperimentLedger()
    protocol, split, sealed = open_sealed_confirmation_lead(
        ledger=ledger,
        selection_case_ids=_selection_case_ids(),
        confirmation_case_ids=[c.case_id for c in _blind_suite_cases()],
    )
    unconsumed_verdict = ConfirmationVerdict(
        caller_id="blind_benchmark", candidate_id="variant_A",
        verdict="PASSED",
    )
    with pytest.raises(CallerRegistryComplianceError):
        publish_sealed_compliance_record(
            ledger=ledger, candidate_id="variant_A", protocol=protocol, sealed=sealed,
            confirmation_verdict=unconsumed_verdict,
        )


def test_mode_a_record_refused_with_foreign_provenance():
    """A verdict whose provenance does not belong to the sealed slice held is
    refused at publication (no rebadging another slice's consumption as sealed)."""
    from apedp_dyn.epistemic.caller_data_protocol import ConfirmationProvenance, ConfirmationVerdict
    from apedp_dyn.sandbox.blind_benchmark import (
        evaluate_confirmation_cases,
        open_sealed_confirmation_lead, publish_sealed_compliance_record,
    )

    suite_cases = _blind_suite_cases()
    conf_ids = [c.case_id for c in suite_cases]
    ledger = FullLineageExperimentLedger()
    # I3 fail-closed: the consuming candidate must be ledger-registered
    # BEFORE the sealed slice can be spent on it.
    ledger.register_candidate("variant_A", proposer_id="blind_benchmark", formula="caller::blind_benchmark")
    protocol, split, sealed = open_sealed_confirmation_lead(
        ledger=ledger,
        selection_case_ids=_selection_case_ids(),
        confirmation_case_ids=conf_ids,
    )
    verdict = sealed.consume(
        candidate_id="variant_A", evaluate=evaluate_confirmation_cases, cases=suite_cases,
    )
    # Sabotage the provenance id list → publication must refuse.
    foreign = ConfirmationProvenance(
        consumed_at=verdict.provenance.consumed_at,
        candidate_id="variant_A",
        slice_dataset_id=verdict.provenance.slice_dataset_id,
        slice_ids=tuple(["some-other-slice"]),
        one_shot=True, second_consume_refused=True,
    )
    with pytest.raises(CallerRegistryComplianceError):
        publish_sealed_compliance_record(
            ledger=ledger, candidate_id="variant_A", protocol=protocol, sealed=sealed,
            confirmation_verdict=ConfirmationVerdict(
                caller_id="blind_benchmark", candidate_id="variant_A",
                verdict=verdict.verdict, metrics=verdict.metrics, provenance=foreign,
            ),
        )


def test_confirmation_evaluation_refuses_unknown_and_foreign_ids():
    from apedp_dyn.sandbox.blind_benchmark import evaluate_confirmation_cases

    suite_cases = _blind_suite_cases()
    conf_ids = [c.case_id for c in suite_cases]
    with pytest.raises(ValueError):
        evaluate_confirmation_cases("variant_A", conf_ids[:-1] + ["NOT-REGISTERED"], suite_cases)
    # Frozen-referee guard: the evaluation path re-runs verify_referee() first.
    verdict = evaluate_confirmation_cases("variant_A", conf_ids, suite_cases)
    assert verdict.verdict == "PASSED"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
