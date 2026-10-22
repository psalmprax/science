"""Pins for the Caller-Layer Three-Role Data Protocol (REG-F14/F14b).

Acceptance criteria (docs/CALLER_DATA_THREE_ROLE_PROTOCOL.md):
  F1  END-TO-END (the register's letter): a candidate pool is scored on slice
      A (selection); the archived head is evaluated EXACTLY ONCE on untouched
      sealed slice B; the recorded verdict carries B ids with one-shot
      provenance; A ∩ B = ∅; a second consume raises.
  F2  Role issuance refuses overlapping selection/confirmation slices, and a
      lead cannot be issued twice for the same caller.
  F3  Verdict source labeling: the archived promotion record names
      `confirmation` as the archived verdict source and marks the selection
      verdict superseded; archiving without consuming the sealed slice raises.
  F4  Ledger integration: confirmation evaluations land as
      DatasetRole.CONFIRMATION_HOLDOUT exposure records.
  F5  Mode (a) compliance validates; neither mode refuses with a named reason.
  F6  Mode (b): multiplicity is ledger-derived (source label checked),
      effective n >= nominal, and a reuse-charged record must show reuses > 0.
  F7  Multiplicity is monotone in variants and reuses and equals the
      ledger-derived value (no hand-tuning).
  F8  Durability: issued leads and sealed-slice consumption survive process
      restarts via the protocol's durable state store — a restarted process
      cannot re-issue an open lead, cannot re-consume a spent slice, and a
      corrupt store fails CLOSED (no silent memory-only fallback).
  F9  Fail-closed I3: with a ledger attached, consume() refuses an
      UNREGISTERED candidate BEFORE the sealed slice is spent, and ledger
      write failures after evaluation propagate (never swallowed).
  F10 No hardcoded threshold literal: the mode-(b) ``dsr_shrink`` disclosure
      is the preregistered DSR constant imported from the canonical spec.
"""

from __future__ import annotations

import json

import pytest

from apedp_dyn.conf.canonical_spec import DSR_THRESHOLD
from apedp_dyn.epistemic.caller_data_protocol import (
    DYNAMIC_DSR_SHRINK,
    CallerComplianceError,
    CallerProtocolStateError,
    CallerRoleOverlapError,
    CallerSplitProtocol,
    ConfirmationAlreadyConsumedError,
    ConfirmationVerdict,
    mark_selection_reuse,
    multiplicity_from_ledger,
    validate_caller_compliance,
)
from apedp_dyn.epistemic.experiment_ledger import (
    DatasetRole,
    FullLineageExperimentLedger,
)


def _verdict_factory(status="UNIQUE_SUPPORT"):
    def _evaluate(candidate_id: str, confirmation_ids, **kw):
        return ConfirmationVerdict(
            caller_id=kw.get("caller_id"),
            candidate_id=candidate_id,
            verdict=status,
            metrics={"n_b": len(confirmation_ids)},
        )
    return _evaluate


def test_end_to_end_pool_on_A_confirmed_once_on_sealed_B():  # F1
    A = [f"a_slice_{i:02d}" for i in range(10)]
    B = [f"b_slice_{i:02d}" for i in range(4)]
    assert not (set(A) & set(B))  # register: "no shared slice"
    ledger = FullLineageExperimentLedger(persistence_path=None)
    proto = CallerSplitProtocol(ledger=ledger)
    _split, sealed = proto.issue_caller_split(
        caller_id="dyn_caller_01", dataset_id="ds_leak_demo",
        selection_ids=A, confirmation_ids=B)

    # A candidate POOL is scored on slice A (selection); head = carved-off best.
    pool_scores = {"hyp_a": 0.31, "hyp_b": 0.12, "hyp_c": 0.55}
    head_by_selection = max(pool_scores, key=pool_scores.get)
    # I3 fail-closed: the archived head MUST be ledger-registered before the
    # sealed slice may be spent on it.
    ledger.register_candidate(head_by_selection, proposer_id="demo", formula="y=x")

    v = sealed.consume(head_by_selection, _verdict_factory(), caller_id="dyn_caller_01")
    assert v.verdict == "UNIQUE_SUPPORT"
    assert v.provenance.one_shot is True
    assert v.provenance.slice_ids == tuple(B)  # verdict provenance = slice B only
    assert sealed.consumed is True

    with pytest.raises(ConfirmationAlreadyConsumedError):
        sealed.consume("hyp_other", _verdict_factory())  # second consumption refused

    rec = proto.archive_promotion_verdict("dyn_caller_01", v, selection_verdict=0.55)
    assert rec["archived_verdict_source"] == "confirmation"
    assert rec["confirmation_slice_ids"] == B
    assert rec["superseded_selection_marked"] is True


def test_overlapping_roles_refused_at_issue():  # F2
    proto = CallerSplitProtocol()
    with pytest.raises(CallerRoleOverlapError) as ei:
        proto.issue_caller_split("c", "ds", ["x", "shared"], ["shared", "y"])
    assert "shared" in str(ei.value)
    proto.issue_caller_split("c", "ds", ["a1", "a2"], ["b1"])  # ok
    with pytest.raises(ConfirmationAlreadyConsumedError):
        proto.issue_caller_split("c", "ds", ["c1"], ["d1"])  # double issue refused


def test_archive_requires_consumed_sealed_slice():  # F3 (negative half)
    proto = CallerSplitProtocol()
    _split, _sealed = proto.issue_caller_split("c2", "ds", ["a"], ["b"])
    dummy = ConfirmationVerdict(caller_id="c2", candidate_id="h", verdict=None)
    with pytest.raises(ConfirmationAlreadyConsumedError):
        proto.archive_promotion_verdict("c2", dummy, selection_verdict="x")


def test_ledger_exposure_roles_written():  # F4
    ledger = FullLineageExperimentLedger(persistence_path=None)
    ledger.register_candidate("hyp_a", proposer_id="p", formula="y=x")
    proto = CallerSplitProtocol(ledger=ledger)
    _, sealed = proto.issue_caller_split("c3", "dsF4", ["a"], ["bF4_1", "bF4_2"])
    sealed.consume("hyp_a", _verdict_factory())
    node = ledger.get_candidate("hyp_a")
    roles = [e.role for e in node.data_exposures]
    assert all(r == DatasetRole.CONFIRMATION_HOLDOUT for r in roles)
    assert {e.slice_identifier for e in node.data_exposures} == {"bF4_1", "bF4_2"}


def test_mode_a_compliance_and_unclassified_refusal():  # F5
    sealed_record = {
        "confirmation_mode": "sealed",
        "confirmation_provenance": {"one_shot": True, "second_consume_refused": True},
    }
    ok, why = validate_caller_compliance(sealed_record)
    assert ok and "mode (a)" in why
    bad = {"confirmation_mode": "sealed", "confirmation_provenance": {"one_shot": False}}
    ok2, why2 = validate_caller_compliance(bad)
    assert not ok2 and "provenance" in why2
    ok3, why3 = validate_caller_compliance({})
    assert not ok3 and "neither" in why3


def test_mode_b_requires_ledger_source_and_reuse_charge():  # F6
    ledger = FullLineageExperimentLedger(persistence_path=None)
    ledger.register_candidate("h1", proposer_id="p", formula="y=x")
    mult = multiplicity_from_ledger(ledger, "c4", "dsF6", "h1", nominal_variants=3)
    ok, why = validate_caller_compliance(
        {"confirmation_mode": "multiplicity", "multiplicity": mult})
    assert ok and "mode (b)" in why
    # hand-written multiplicity is refused
    forged = {"source": "hand", "dsr_adjusted_effective_n": 1.0}
    ok2, why2 = validate_caller_compliance(
        {"confirmation_mode": "multiplicity", "multiplicity": forged})
    assert not ok2 and "ledger-sourced" in why2
    # reuse without multiplicity reflection is refused
    mark_selection_reuse("h1", "dsF6", "bF6_1", ledger)
    mult2 = multiplicity_from_ledger(ledger, "c4", "dsF6", "h1", nominal_variants=3)
    assert mult2["n_confirmation_reuses"] == 1
    ok3, why3 = validate_caller_compliance({
        "confirmation_mode": "multiplicity",
        "multiplicity": {"source": "ledger", "dsr_adjusted_effective_n": 3.0,
                         "nominal_variants": 3},
        "selection_reused_confirmation_holdout": True,
    })
    assert not ok3 and "reuse" in why3


def test_multiplicity_monotone_and_matches_ledger():  # F7
    ledger = FullLineageExperimentLedger(persistence_path=None)
    ledger.register_candidate("h1", proposer_id="p", formula="y=x")
    m0 = multiplicity_from_ledger(ledger, "c5", "dsF7", "h1", nominal_variants=2)
    assert m0["dsr_adjusted_effective_n"] == 2.0  # no reuse -> nominal
    mark_selection_reuse("h1", "dsF7", "b1", ledger)
    m1 = multiplicity_from_ledger(ledger, "c5", "dsF7", "h1", nominal_variants=2)
    assert m1["dsr_adjusted_effective_n"] > m0["dsr_adjusted_effective_n"]  # reuse inflates
    mBigger = multiplicity_from_ledger(ledger, "c5", "dsF7", "h1", nominal_variants=4)
    assert mBigger["dsr_adjusted_effective_n"] > m1["dsr_adjusted_effective_n"]  # variants inflate


# --------------------------------------------------------------------------
# F8: durable sealed-slice consumption (restart-proof one-shot)
# --------------------------------------------------------------------------

def _ready_ledger(tmp_path, filename="experiment_ledger.jsonl"):
    """A ledger with a real on-disk persistence path (production shape)."""
    from apedp_dyn.epistemic.experiment_ledger import FullLineageExperimentLedger as LL

    return LL(persistence_path=tmp_path / filename)


def test_sealed_consumption_survives_process_restart(tmp_path):  # F8
    ledger = _ready_ledger(tmp_path)
    ledger.register_candidate("hyp_head", proposer_id="p", formula="y=x")

    # "Process A": issue the lead and spend the sealed slice.
    proto_a = CallerSplitProtocol(ledger=ledger, persistence_path=tmp_path / "caller_lead_protocol_state.json")
    proto_a.issue_caller_split("dyn_caller_09", "dsF8", ["a1", "a2"], ["b1", "b2"])
    sealed_a = proto_a._require_open_lead("dyn_caller_09")[1]
    v = sealed_a.consume("hyp_head", _verdict_factory())
    assert v.provenance.one_shot is True

    # "Process B": a FRESH protocol (same durable state path) must see the
    # spent slice — no fresh unconsumed sealed slice can be handed out.
    proto_b = CallerSplitProtocol(ledger=ledger, persistence_path=tmp_path / "caller_lead_protocol_state.json")
    with pytest.raises(ConfirmationAlreadyConsumedError):
        proto_b.issue_caller_split("dyn_caller_09", "dsF8", ["a1", "a2"], ["b1", "b2"])
    sealed_b = proto_b._require_open_lead("dyn_caller_09")[1]
    assert sealed_b.consumed is True
    with pytest.raises(ConfirmationAlreadyConsumedError) as ei:
        sealed_b.consume("hyp_head", _verdict_factory())
    assert "durable state across restart" in str(ei.value)
    assert sealed_b.consumed_provenance is not None
    assert sealed_b.consumed_provenance.candidate_id == "hyp_head"
    assert sealed_b.consumed_provenance.slice_ids == ("b1", "b2")

    # The spent slice remains archivable after the restart (provenance intact).
    rec = proto_b.archive_promotion_verdict("dyn_caller_09", v, selection_verdict=None)
    assert rec["archived_verdict_source"] == "confirmation"


def test_restart_refuses_open_lead_reissue_and_keeps_it_open(tmp_path):  # F8
    # "Process A" issues an open (unconsumed) lead, then dies.
    ledger = _ready_ledger(tmp_path)
    proto_a = CallerSplitProtocol(ledger=ledger, persistence_path=tmp_path / "state.json")
    proto_a.issue_caller_split("c_restart", "dsF8b", ["a1"], ["b1"])

    # "Process B" must not be able to mint a fresh unconsumed sealed slice.
    proto_b = CallerSplitProtocol(ledger=ledger, persistence_path=tmp_path / "state.json")
    with pytest.raises(ConfirmationAlreadyConsumedError):
        proto_b.issue_caller_split("c_restart", "dsF8b", ["a1"], ["b1"])
    sealed_b = proto_b._require_open_lead("c_restart")[1]
    assert sealed_b.consumed is False  # still unconsumed — but not re-issuable


def test_aborted_evaluation_leaves_the_slice_spent_after_restart(tmp_path):  # F8
    ledger = _ready_ledger(tmp_path)
    ledger.register_candidate("hyp_x", proposer_id="p", formula="y=x")
    proto_a = CallerSplitProtocol(ledger=ledger, persistence_path=tmp_path / "state.json")
    proto_a.issue_caller_split("c_crash", "dsF8c", ["a1"], ["b1"])
    sealed = proto_a._require_open_lead("c_crash")[1]

    def _exploding_evaluate(**kw):
        raise RuntimeError("crash mid-evaluation")

    with pytest.raises(RuntimeError):
        sealed.consume("hyp_x", _exploding_evaluate)
    assert sealed.consumed is True  # consumed-by-claim, no fabricated provenance
    assert sealed.consumed_provenance is None

    proto_b = CallerSplitProtocol(ledger=ledger, persistence_path=tmp_path / "state.json")
    sealed_b = proto_b._require_open_lead("c_crash")[1]
    with pytest.raises(ConfirmationAlreadyConsumedError) as ei:
        sealed_b.consume("hyp_x", _verdict_factory())
    assert "claimed-unprovened" in str(ei.value)
    # The claimed-but-unprovenanced slice cannot archive a promotion either.
    with pytest.raises(CallerComplianceError):
        proto_b.archive_promotion_verdict("c_crash", ConfirmationVerdict("c_crash", "hyp_x", None), None)


def test_corrupt_state_store_fails_closed(tmp_path):  # F8 (fail-closed)
    from apedp_dyn.epistemic.caller_data_protocol import PROTOCOL_STATE_VERSION
    store = tmp_path / "state.json"
    store.write_text("{not json", encoding="utf-8")
    with pytest.raises(CallerProtocolStateError):
        CallerSplitProtocol(ledger=_ready_ledger(tmp_path), persistence_path=store)

    # Corrupt CONTENT that parses is refused too (wrong version marker).
    store2 = tmp_path / "state_wrong_version.json"
    store2.write_text(json.dumps(
        {"protocol_state_version": PROTOCOL_STATE_VERSION + 1, "leads": {}}), encoding="utf-8")
    with pytest.raises(CallerProtocolStateError):
        CallerSplitProtocol(persistence_path=store2)

    # A tampered record (provenance claiming ids outside the held slice) is
    # rejected at load, not trusted.
    store3 = tmp_path / "state_tampered.json"
    store3.write_text(json.dumps({
        "protocol_state_version": PROTOCOL_STATE_VERSION,
        "leads": {"rogue": {"dataset_id": "ds", "selection_ids": ["a"],
                            "confirmation_ids": ["b"], "consumed": True,
                            "provenance": {"consumed_at": "t", "candidate_id": "x",
                                           "slice_dataset_id": "ds",
                                           "slice_ids": ["other"],
                                           "one_shot": True, "second_consume_refused": True}}},
    }), encoding="utf-8")
    with pytest.raises(CallerProtocolStateError):
        CallerSplitProtocol(persistence_path=store3)


def test_unregistered_candidate_refused_before_slice_is_spent(tmp_path):  # F9
    ledger = _ready_ledger(tmp_path)  # candidate NOT registered
    proto = CallerSplitProtocol(ledger=ledger, persistence_path=tmp_path / "state.json")
    _, sealed = proto.issue_caller_split("c_f9", "dsF9", ["a1"], ["b1"])
    with pytest.raises(CallerComplianceError) as ei:
        sealed.consume("hyp_ghost", _verdict_factory())
    assert "not registered" in str(ei.value)
    assert sealed.consumed is False  # the slice was NOT spent by the refusal
    # After registering, the same slice still evaluates exactly once.
    ledger.register_candidate("hyp_late", proposer_id="p", formula="y=x")
    v = sealed.consume("hyp_late", _verdict_factory())
    assert v.verdict == "UNIQUE_SUPPORT"


def test_ledger_write_failure_propagates_from_consume():  # F9 (no silent swallow)
    class _ExplodingLedger:
        def get_candidate(self, candidate_id):
            return object()  # pre-flight passes

        def record_data_exposure(self, **kw):
            raise OSError("ledger disk failure")

    proto = CallerSplitProtocol(ledger=_ExplodingLedger())
    _, sealed = proto.issue_caller_split("c_f9b", "dsF9b", ["a1"], ["b1"])
    with pytest.raises(OSError):
        sealed.consume("hyp_reg", _verdict_factory())
    assert sealed.consumed is True  # spent by claim; never silently "recorded"


def test_dsr_shrink_imported_from_canonical_spec_not_hardcoded():  # F10
    from apedp_dyn.epistemic.caller_data_protocol import (
        multiplicity_from_ledger as mfl,
    )

    assert DYNAMIC_DSR_SHRINK == DSR_THRESHOLD  # the preregistered constant
    ledger = FullLineageExperimentLedger(persistence_path=None)
    ledger.register_candidate("h1", proposer_id="p", formula="y=x")
    m = mfl(ledger, "c", "dsF10", "h1", nominal_variants=1)
    assert m["dynamics"]["dsr_shrink"] == DSR_THRESHOLD
