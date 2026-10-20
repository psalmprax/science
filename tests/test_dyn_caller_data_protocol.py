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
"""

from __future__ import annotations

import pytest

from apedp_dyn.epistemic.caller_data_protocol import (
    CallerComplianceError,
    CallerRoleOverlapError,
    CallerSplitProtocol,
    ConfirmationAlreadyConsumedError,
    ConfirmationVerdict,
    multiplicity_from_ledger,
    mark_selection_reuse,
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
    split, sealed = proto.issue_caller_split(
        caller_id="dyn_caller_01", dataset_id="ds_leak_demo",
        selection_ids=A, confirmation_ids=B)

    # A candidate POOL is scored on slice A (selection); head = carved-off best.
    pool_scores = {"hyp_a": 0.31, "hyp_b": 0.12, "hyp_c": 0.55}
    head_by_selection = max(pool_scores, key=pool_scores.get)

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
    split, sealed = proto.issue_caller_split("c2", "ds", ["a"], ["b"])
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
