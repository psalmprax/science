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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
