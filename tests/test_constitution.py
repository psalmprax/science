"""Tests for the epistemic constitution and the controlled self-improvement layer.

PROVENANCE: written alongside the re-authored ``apedp/constitution.py`` and
``apedp/self_improvement.py``, which existed only as orphaned bytecode until
this session. Both modules were verified verdict-identical to the bytecode
across their full distinguishable input space before these tests were
written -- ``tools/constitution_equivalence.py`` (1027/1027) and
``tools/self_improvement_equivalence.py`` (24/24).

These tests are not that verification. They pin the behaviour so a future
edit to either module is caught, and they cover the properties the
equivalence harness cannot: that the gate has no override path, that the
ledger retains rejected proposals, and that the reserved third input to the
gatekeeper decision is visibly unwired rather than quietly in use.
"""
from __future__ import annotations

import dataclasses

import pytest

from apedp.constitution import (
    ConstitutionalAuditor,
    ConstitutionalVerdict,
    InvariantCheck,
    InvariantID,
)
from apedp.self_improvement import (
    DiagnosticSignal,
    FailureDetector,
    GatekeeperDecision,
    ImprovementLedger,
    ImprovementLevel,
    IndependentGatekeeper,
    ModificationProposal,
)


def _proposal(**kw) -> ModificationProposal:
    base = {
        "proposal_id": "P-1", "level": ImprovementLevel.LEVEL_1_PARAMETER,
        "target_component": "coupling_threshold", "modifications": {"x": 0.1},
        "hypothesis": "lowering the threshold helps",
        "rationale": "observed 3 misses",
        "author": "tester", "timestamp": "2026-01-01T00:00:00+00:00",
    }
    base.update(kw)
    return ModificationProposal(**base)


class TestTheTenInvariants:
    def test_there_are_exactly_ten_and_each_is_described(self) -> None:
        assert len(InvariantID) == 10
        for inv in InvariantID:
            assert inv.value.startswith("INV-"), inv
            assert len(inv.value) > 20, f"{inv.name} has no description"

    def test_a_clean_proposal_is_compliant(self) -> None:
        v = ConstitutionalAuditor().audit(
            proposal_dict={"supports_ambiguity_preservation": True},
            regression_passed=True,
            provenance_trace={"author": "a", "timestamp": "t"},
        )
        assert v.compliant is True
        assert v.rejection_reasons == []
        assert len(v.checks) == 10
        assert all(c.satisfied for c in v.checks)

    @pytest.mark.parametrize("flag,expected", [
        ("classified_contradiction_as_valid", "INV-1"),
        ("unresolved_treated_as_valid", "INV-2"),
        ("selected_by_train_fit_alone", "INV-3"),
        ("modifies_benchmark_data", "INV-4"),
        ("retrospective_criterion_change", "INV-5"),
    ])
    def test_each_violating_flag_is_rejected(self, flag: str, expected: str) -> None:
        v = ConstitutionalAuditor().audit(
            proposal_dict={flag: True, "supports_ambiguity_preservation": True},
            regression_passed=True,
            provenance_trace={"author": "a", "timestamp": "t"},
        )
        assert v.compliant is False
        assert any(r.startswith(expected) for r in v.rejection_reasons), (
            f"{flag} should trip {expected}, got {v.rejection_reasons}")

    def test_regression_failure_is_a_constitutional_violation(self) -> None:
        v = ConstitutionalAuditor().audit(
            proposal_dict={"supports_ambiguity_preservation": True},
            regression_passed=False,
            provenance_trace={"author": "a", "timestamp": "t"},
        )
        assert v.compliant is False
        assert any("INV-7" in r for r in v.rejection_reasons)

    def test_missing_provenance_is_a_violation(self) -> None:
        v = ConstitutionalAuditor().audit(
            proposal_dict={"supports_ambiguity_preservation": True},
            regression_passed=True,
            provenance_trace={"author": "a"},  # no timestamp
        )
        assert v.compliant is False
        assert any("INV-10" in r for r in v.rejection_reasons)

    def test_compliant_is_derived_from_the_rejection_list(self) -> None:
        """The two must not be able to disagree.

        `compliant` is computed as `len(rejections) == 0`, not as
        `all(c.satisfied)`. If it were the latter, a check appended without
        a matching rejection would report success while recording a
        violation.
        """
        v = ConstitutionalAuditor().audit(
            proposal_dict={"unresolved_treated_as_valid": True},
            regression_passed=True,
            provenance_trace={"author": "a", "timestamp": "t"},
        )
        assert v.compliant == (len(v.rejection_reasons) == 0)
        assert v.compliant is False

    def test_the_digest_covers_the_decision_not_the_whole_proposal(self) -> None:
        """Two proposals reaching the same verdict must hash the same.

        That is what makes the sha256 a tamper-evident record of the
        *decision* rather than of arbitrary proposal text.
        """
        a = ConstitutionalAuditor()
        kw = {"regression_passed": True,
              "provenance_trace": {"author": "a", "timestamp": "t"}}
        v1 = a.audit(proposal_dict={"supports_ambiguity_preservation": True,
                                    "rationale": "one"}, **kw)
        v2 = a.audit(proposal_dict={"supports_ambiguity_preservation": True,
                                    "rationale": "a completely different reason",
                                    "modifications": {"z": 9}}, **kw)
        assert v1.sha256 == v2.sha256

        v3 = a.audit(proposal_dict={"supports_ambiguity_preservation": True,
                                    "unresolved_treated_as_valid": True}, **kw)
        assert v3.sha256 != v1.sha256, "a different decision must hash differently"

    def test_verdict_is_json_serialisable(self) -> None:
        import json

        v = ConstitutionalAuditor().audit(
            proposal_dict={"supports_ambiguity_preservation": True},
            regression_passed=True,
            provenance_trace={"author": "a", "timestamp": "t"})
        blob = json.loads(json.dumps(v.to_dict()))
        assert blob["compliant"] is True
        assert len(blob["checks"]) == 10
        assert blob["checks"][0]["description"].startswith("INV-")

    def test_dataclass_shapes_are_stable(self) -> None:
        assert [f.name for f in dataclasses.fields(InvariantCheck)] == [
            "invariant_id", "satisfied", "reason", "metadata"]
        assert [f.name for f in dataclasses.fields(ConstitutionalVerdict)] == [
            "compliant", "checks", "sha256", "rejection_reasons"]


class TestTheGateHasNoOverride:
    def test_a_regression_failure_discards_a_constitutionally_clean_proposal(
        self,
    ) -> None:
        d = IndependentGatekeeper.evaluate_and_decide(_proposal(), None, False)
        assert d.accepted is False
        assert d.status == "DISCARDED"
        assert "FAIL" in d.regression_summary

    def test_a_constitutional_violation_discards_a_clean_regression_run(self) -> None:
        p = _proposal(modifies_benchmark_data=True)
        d = IndependentGatekeeper.evaluate_and_decide(p, None, True)
        assert d.accepted is False
        assert d.status == "DISCARDED"

    def test_the_proposal_type_cannot_currently_satisfy_inv_9(self) -> None:
        """KNOWN DEFECT, pinned rather than fixed.

        The gate is inert. ``ConstitutionalAuditor`` reads seven boolean
        flags off the proposal; ``ModificationProposal`` can express only
        two of them, so the other five always take their default. One of
        those defaults is the violating value -- ``supports_ambiguity_preservation``
        defaults to False, which trips INV-9 -- and so every proposal is
        discarded and the self-improvement architecture never promotes
        anything.

        This is not a reconstruction artifact. The equivalence harness
        compares this behaviour against the orphaned bytecode across every
        reachable input and finds it identical
        (``tools/self_improvement_equivalence.py``, 24/24), so the original
        module discards for the same reason.

        The gate is not *supposed* to be unconditionally open -- INV-9
        exists precisely so a proposal that forces a winner is refused. The
        defect is the mismatch between the two types: the constitution
        demands a property the proposal has no field to declare, so
        compliance is unreachable rather than merely uncommon.

        Two repairs, both design changes rather than bug fixes, and neither
        taken here:
        - add the five missing fields to ModificationProposal, so a
          proposal can state its own compliance; or
        - have the auditor treat an absent key as "not asserted" and
          record it as unverified rather than violated, which is the
          treatment every other check in this codebase received in 227eede
          and 8c0be24 -- and the one that keeps a check that never ran from
          reading as a check that failed.

        Whichever is chosen, the assertions below stop holding and are
        meant to be the thing that fails loudly on that day.
        """
        d = IndependentGatekeeper.evaluate_and_decide(_proposal(), None, True)
        assert d.accepted is False
        assert d.status == "DISCARDED"
        assert d.constitutional_verdict.rejection_reasons == [
            "INV-9 violated: system forces a winner when evidence is ambiguous."
        ], d.constitutional_verdict.rejection_reasons

    def test_the_decision_records_its_own_provenance(self) -> None:
        d = IndependentGatekeeper.evaluate_and_decide(_proposal(), None, True)
        assert len(d.provenance_hash) == 64
        assert d.provenance_hash != ""
        assert d.timestamp

    def test_the_reserved_third_input_is_visibly_unwired(self) -> None:
        """`target_metric_deltas` is populated by nothing.

        The gate decides on constitution plus regression alone. This asserts
        the field is empty rather than populated from some unstated source,
        so the day it is wired up this fails and the change is deliberate.
        """
        d = IndependentGatekeeper.evaluate_and_decide(_proposal(), None, True)
        assert d.target_metric_deltas == {}

    def test_decision_to_dict_round_trips(self) -> None:
        import json

        d = IndependentGatekeeper.evaluate_and_decide(_proposal(), None, True)
        blob = json.loads(json.dumps(d.to_dict()))
        assert blob["accepted"] is False  # see the pinned INV-9 defect above
        assert blob["constitutional_verdict"]["compliant"] is False
        assert blob["proposal_id"] == "P-1"

    def test_gatekeeper_dataclass_shape_is_stable(self) -> None:
        assert [f.name for f in dataclasses.fields(GatekeeperDecision)] == [
            "proposal_id", "accepted", "status", "constitutional_verdict",
            "target_metric_deltas", "regression_summary", "provenance_hash",
            "timestamp"]


class TestTheLedgerRetainsFailures:
    def test_a_discarded_proposal_is_still_recorded(self) -> None:
        """INV-8. An improvement process that discards its rejected attempts
        cannot learn which directions have already been tried."""
        ledger = ImprovementLedger()
        p = _proposal()
        d = IndependentGatekeeper.evaluate_and_decide(p, None, False)
        ledger.record_decision(p, d)
        rec = ledger.to_list()
        assert len(rec) == 1
        assert rec[0]["decision"]["accepted"] is False
        assert rec[0]["entry_hash"] == d.provenance_hash

    def test_the_ledger_is_append_only_across_mixed_outcomes(self) -> None:
        """Retention is the property under test, not the accept/reject mix.

        Every proposal is currently discarded by the pinned INV-9 defect,
        so this cannot assert a mix. What INV-8 actually requires is that a
        rejected decision is retained rather than dropped, which is checked
        here against two rejections that differ in *why*.
        """
        ledger = ImprovementLedger()
        for i, reg in enumerate([True, False]):
            p = _proposal(proposal_id=f"P-{i}")
            ledger.record_decision(p, IndependentGatekeeper.evaluate_and_decide(p, None, reg))
        rec = ledger.to_list()
        assert len(rec) == 2
        assert [r["proposal"]["id"] for r in rec] == ["P-0", "P-1"]
        # Both were discarded, for different recorded reasons.
        assert rec[0]["decision"]["regression_summary"] == "PASS: All frozen benchmarks intact"
        assert rec[1]["decision"]["regression_summary"] == "FAIL: Regression detected"
        assert rec[0]["entry_hash"] != rec[1]["entry_hash"]

    def test_an_empty_ledger_is_empty_not_missing(self) -> None:
        assert ImprovementLedger().to_list() == []


class TestFailureDetector:
    def test_no_history_means_no_signals(self) -> None:
        assert FailureDetector.detect_weaknesses([]) == []

    def test_an_unresolvable_transcendental_rate_over_20pc_is_flagged(self) -> None:
        h = [{"candidates": [{"dim_status": "UNRESOLVED"}] * 3}]
        sig = FailureDetector.detect_weaknesses(h)
        assert [s.signal_type for s in sig] == ["HIGH_UNRESOLVED_TRANSCENDENTAL_RATE"]
        assert sig[0].severity == "HIGH"
        assert sig[0].suggested_level is ImprovementLevel.LEVEL_3_PIPELINE
        assert sig[0].evidence_metrics["unresolved_ratio"] == 1.0

    def test_the_rate_rule_does_not_fire_below_the_threshold(self) -> None:
        """1 in 10 is 10%, under the 20% bar. A rule that fires here is a
        rule that has stopped discriminating."""
        h = [{"candidates": [{"dim_status": "UNRESOLVED"}]
              + [{"dim_status": "VALID"}] * 9}]
        assert FailureDetector.detect_weaknesses(h) == []

    def test_scalar_fit_with_near_ood_divergence_is_critical(self) -> None:
        h = [{"candidates": [{"train_mse": 0.01, "near_ood_mse": 500.0}]}]
        sig = FailureDetector.detect_weaknesses(h)
        assert [s.signal_type for s in sig] == ["SCALAR_FIT_NEAR_OOD_DIVERGENCE"]
        assert sig[0].severity == "CRITICAL"
        assert sig[0].suggested_level is ImprovementLevel.LEVEL_4_SCIENTIFIC_METHOD

    def test_the_divergence_rule_needs_both_halves(self) -> None:
        """Good train fit with good near-OOD, or bad fit entirely, is fine.

        A rule keyed on either half alone would fire on every well-behaved
        candidate.
        """
        assert FailureDetector.detect_weaknesses(
            [{"candidates": [{"train_mse": 0.01, "near_ood_mse": 0.02}]}]) == []
        assert FailureDetector.detect_weaknesses(
            [{"candidates": [{"train_mse": 50.0, "near_ood_mse": 500.0}]}]) == []

    def test_a_coupling_ratio_far_below_its_threshold_is_flagged(self) -> None:
        h = [{"coupling_residual_ratio": 0.10}]  # default threshold 0.15
        sig = FailureDetector.detect_weaknesses(h)
        assert [s.signal_type for s in sig] == ["PROVISIONAL_COUPLING_THRESHOLD_MISS"]
        assert sig[0].suggested_level is ImprovementLevel.LEVEL_1_PARAMETER
        assert sig[0].evidence_metrics["threshold"] == 0.15

    def test_an_above_threshold_ratio_is_not_flagged(self) -> None:
        assert FailureDetector.detect_weaknesses(
            [{"coupling_residual_ratio": 0.10, "coupling_residual_threshold": 0.05}]) == []

    def test_a_missing_ratio_is_not_a_zero_ratio(self) -> None:
        """Absent evidence must not read as a passing measurement.

        `None` is skipped rather than coerced to 0.0, which would land below
        the 0.06 floor and be indistinguishable from a genuinely tiny
        residual.
        """
        assert FailureDetector.detect_weaknesses([{}]) == []
        assert FailureDetector.detect_weaknesses(
            [{"coupling_residual_ratio": 0.01}]) == []

    def test_multiple_rules_can_fire_in_one_pass(self) -> None:
        h = [{"candidates": [{"dim_status": "UNRESOLVED"},
                             {"train_mse": 0.0, "near_ood_mse": 1e9}],
              "coupling_residual_ratio": 0.08}]
        types = [s.signal_type for s in FailureDetector.detect_weaknesses(h)]
        assert len(types) == 3, types

    def test_signal_shape_is_stable(self) -> None:
        assert [f.name for f in dataclasses.fields(DiagnosticSignal)] == [
            "signal_type", "severity", "affected_component", "description",
            "evidence_metrics", "suggested_level"]
