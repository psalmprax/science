"""An abstention has to say which one it was.

``EpistemicState`` says the synthesizer withheld endorsement. It cannot say
why, so for a long time the adversarial gate OR'd three unrelated conditions
together and stamped one fixed sentence on the result. Every abstention in the
external-60 round read ``"Residual autocorrelation or discontinuity indicates
out-of-grammar phenomena"`` whether the cause was a shock front, a
misspecified smooth family, or nothing at all.

That last one is the expensive conflation. The gate fed its criteria the
``-1.0`` sentinel that stands in for "there is no top model", so a run in
which every template failed to converge produced a record claiming the data
was chaotic -- an infrastructure failure reported as a scientific finding,
from a model that does not exist.

These tests pin the codes, the priority when several fire, and the fact that
the reported reason cannot disagree with the decision that produced it.
"""
from __future__ import annotations

import numpy as np
import pytest

from apedp.decision.stage_c import (
    ABSTENTION_REASON_TEXT,
    AbstentionReason,
    EpistemicState,
)
from apedp.model_synthesis import (
    _ABSTENTION_PRIORITY,
    _abstention_rationale,
    _classify_adversarial,
)


def _classify(**kw):
    """Defaults describe a clean, well-fit, non-shock run -- one that clears."""
    base = {
        "n_candidates": 1, "has_shock": False, "top_r2": 0.99,
        "top_autocorr": 0.01,
    }
    base.update(kw)
    return _classify_adversarial(**base)


class TestEnumIsWellFormed:
    """The vocabulary itself, before any behaviour is attributed to it."""

    def test_every_member_has_text(self) -> None:
        """A code with no prose is a code no report can show."""
        missing = set(AbstentionReason) - set(ABSTENTION_REASON_TEXT)
        assert not missing, f"reasons without text: {missing}"

    def test_no_duplicate_codes(self) -> None:
        """``str`` enum: a collision would alias two causes to one record."""
        values = [r.value for r in AbstentionReason]
        assert len(values) == len(set(values))

    def test_every_priority_member_is_a_real_reason(self) -> None:
        """A typo in the priority tuple would silently never fire."""
        assert set(_ABSTENTION_PRIORITY) <= set(AbstentionReason)

    def test_priority_covers_every_reason_the_gate_can_produce(self) -> None:
        """No gate output may be unrankable.

        ``_classify_adversarial`` returns ``ordered[0]``, so a reason missing
        from the priority tuple cannot be reported as a primary even though
        the gate fired on it.
        """
        producible = {
            AbstentionReason.NO_CANDIDATE_FITTED,
            AbstentionReason.SHOCK_DISCONTINUITY,
            AbstentionReason.RESIDUAL_AUTOCORRELATION,
            AbstentionReason.LOW_VALIDATION_R2,
        }
        assert producible <= set(_ABSTENTION_PRIORITY)

    def test_str_enum_compares_to_its_code(self) -> None:
        """Consumers serialize to JSON; the value must be the wire format."""
        assert AbstentionReason.SHOCK_DISCONTINUITY == "SHOCK_DISCONTINUITY"
        assert (
            json_roundtrip(AbstentionReason.SHOCK_DISCONTINUITY)
            == "SHOCK_DISCONTINUITY"
        )


def json_roundtrip(reason: AbstentionReason) -> str:
    import json

    return json.loads(json.dumps({"r": reason}))["r"]


class TestCausesStayDistinct:
    """Each gate condition maps to its own code, not to a shared bucket."""

    def test_clean_run_clears(self) -> None:
        primary, all_fired = _classify()
        assert primary is None
        assert all_fired == ()

    def test_shock_front_is_its_own_cause(self) -> None:
        primary, _ = _classify(has_shock=True, top_r2=0.90)
        assert primary is AbstentionReason.SHOCK_DISCONTINUITY

    def test_autocorrelated_residuals_are_their_own_cause(self) -> None:
        primary, _ = _classify(top_autocorr=0.60, top_r2=0.97)
        assert primary is AbstentionReason.RESIDUAL_AUTOCORRELATION

    def test_low_r2_is_the_catch_all(self) -> None:
        primary, _ = _classify(top_r2=0.50)
        assert primary is AbstentionReason.LOW_VALIDATION_R2

    def test_empty_candidate_set_is_not_low_r2(self) -> None:
        """The regression this whole change exists to prevent.

        ``n_candidates=0`` arrives with the ``-1.0``/``0.0`` sentinels that
        stand in for a missing model. Reading those as a measured R-squared
        of -1.0 and an autocorrelation of 0.0 attributes a chaotic-data
        finding to an optimizer failure.
        """
        primary, all_fired = _classify(
            n_candidates=0, top_r2=-1.0, top_autocorr=0.0,
        )
        assert primary is AbstentionReason.NO_CANDIDATE_FITTED
        assert AbstentionReason.LOW_VALIDATION_R2 not in all_fired

    def test_empty_candidates_suppress_every_downstream_cause(self) -> None:
        """Nothing to autocorrelate and no R-squared to be low."""
        _, all_fired = _classify(
            n_candidates=0, has_shock=True, top_r2=-1.0, top_autocorr=0.99,
        )
        assert all_fired == (AbstentionReason.NO_CANDIDATE_FITTED,)

    def test_the_four_causes_are_four_distinct_codes(self) -> None:
        """The whole point: a bucket of four would pass every other test."""
        seen = {
            _classify(n_candidates=0, top_r2=-1.0)[0],
            _classify(has_shock=True, top_r2=0.90)[0],
            _classify(top_autocorr=0.60, top_r2=0.97)[0],
            _classify(top_r2=0.50)[0],
        }
        assert len(seen) == 4


class TestPriority:
    """When the OR fires more than one criterion, which cause is reported."""

    def test_specific_cause_outranks_the_catch_all(self) -> None:
        primary, all_fired = _classify(has_shock=True, top_r2=0.10)
        assert primary is AbstentionReason.SHOCK_DISCONTINUITY
        assert AbstentionReason.LOW_VALIDATION_R2 in all_fired

    def test_every_fired_cause_is_retained(self) -> None:
        """Reporting only the primary would hide that the fit was also bad."""
        _, all_fired = _classify(has_shock=True, top_autocorr=0.60, top_r2=0.10)
        assert set(all_fired) == {
            AbstentionReason.SHOCK_DISCONTINUITY,
            AbstentionReason.RESIDUAL_AUTOCORRELATION,
            AbstentionReason.LOW_VALIDATION_R2,
        }

    def test_all_fired_is_ordered_by_specificity(self) -> None:
        primary, all_fired = _classify(
            has_shock=True, top_autocorr=0.60, top_r2=0.10,
        )
        assert all_fired == tuple(
            r for r in _ABSTENTION_PRIORITY if r in set(all_fired)
        )
        assert all_fired[0] == primary

    def test_priority_is_total_and_deterministic(self) -> None:
        """A set's iteration order must not decide what gets reported."""
        first = _classify(has_shock=True, top_autocorr=0.60, top_r2=0.10)
        for _ in range(20):
            assert _classify(
                has_shock=True, top_autocorr=0.60, top_r2=0.10,
            ) == first


class TestBoundaryConditions:
    """Thresholds are the gate; the tests pin both sides of each one."""

    @pytest.mark.parametrize(
        "r2,abstains",
        [(0.9599, True), (0.96, False)],
    )
    def test_shock_r2_threshold_is_strict_below(self, r2, abstains) -> None:
        """The shock arm is ``< 0.96``. Exactly 0.96 clears that arm."""
        assert (_classify(has_shock=True, top_r2=r2)[0] is not None) is abstains

    @pytest.mark.parametrize(
        "autocorr,abstains",
        [(0.4501, True), (0.45, False)],
    )
    def test_autocorr_threshold_is_strict_above(self, autocorr, abstains) -> None:
        assert (
            _classify(top_autocorr=autocorr, top_r2=0.97)[0] is not None
        ) is abstains

    @pytest.mark.parametrize(
        "r2,abstains",
        [(0.9899, True), (0.99, False)],
    )
    def test_autocorr_r2_gate_is_strict_below(self, r2, abstains) -> None:
        assert (
            _classify(top_autocorr=0.60, top_r2=r2)[0] is not None
        ) is abstains

    @pytest.mark.parametrize(
        "r2,abstains",
        [(0.8499, True), (0.85, False)],
    )
    def test_low_r2_threshold_is_strict_below(self, r2, abstains) -> None:
        assert (_classify(top_r2=r2)[0] is not None) is abstains

    def test_shock_alone_never_abstains(self) -> None:
        """A shock front the fit handles is not a reason to withhold."""
        assert _classify(has_shock=True, top_r2=0.99)[0] is None

    def test_autocorrelation_alone_never_abstains(self) -> None:
        """Autocorrelation with a good R-squared is caught by Stage C.

        That arm carries a second condition precisely because residuals can be
        mildly autocorrelated on a genuinely good fit; without the R-squared
        conjunct it would abstain on data it fits well.
        """
        assert _classify(top_autocorr=0.60, top_r2=0.999)[0] is None

    def test_a_perfect_fit_never_abstains(self) -> None:
        assert _classify(top_r2=1.0, top_autocorr=0.0)[0] is None


class TestRationaleFollowsTheCodes:
    """Prose is generated, so it cannot describe a different cause."""

    def test_rationale_names_every_fired_code(self) -> None:
        _, all_fired = _classify(has_shock=True, top_r2=0.10)
        text = _abstention_rationale(all_fired)
        for r in all_fired:
            assert r.value in text, f"{r.value} missing from rationale"

    def test_rationale_includes_the_own_text_of_each_code(self) -> None:
        _, all_fired = _classify(n_candidates=0, top_r2=-1.0)
        text = _abstention_rationale(all_fired)
        assert ABSTENTION_REASON_TEXT[all_fired[0]] in text

    def test_rationale_is_marked_as_formal_abstention(self) -> None:
        """Existing consumers match on this prefix; it must survive."""
        text = _abstention_rationale((AbstentionReason.LOW_VALIDATION_R2,))
        assert text.startswith("Formal abstention")

    def test_distinct_causes_produce_distinct_prose(self) -> None:
        """The old bug in one assertion: one string for every cause."""
        causes = [
            (AbstentionReason.NO_CANDIDATE_FITTED,),
            (AbstentionReason.SHOCK_DISCONTINUITY,),
            (AbstentionReason.RESIDUAL_AUTOCORRELATION,),
            (AbstentionReason.LOW_VALIDATION_R2,),
        ]
        texts = {_abstention_rationale(c) for c in causes}
        assert len(texts) == len(causes)

    def test_empty_reason_tuple_has_no_standing_text(self) -> None:
        """Clearing the gate must not emit a rationale.

        A cleared model has no reason, and an empty-tuple rationale reading
        "Formal abstention []:" would be a lie in the diagnostics of every
        model that was actually endorsed.
        """
        assert _abstention_rationale(()) == ""

    def test_rationale_is_stable_across_calls(self) -> None:
        args = (AbstentionReason.SHOCK_DISCONTINUITY, AbstentionReason.LOW_VALIDATION_R2)
        assert _abstention_rationale(args) == _abstention_rationale(args)


@pytest.mark.slow
class TestGateAgreesWithItsOwnClassifier:
    """End-to-end through synthesize(), on the record a consumer reads.

    Marked slow: each case runs a real synthesis (~90s for the class). The
    classifier is pinned exhaustively by the fast tests above; what these add
    is the wiring -- that the field and the diagnostics keys are actually
    populated on a real record, which only a real run can show.
    """

    @staticmethod
    def _engine():
        from apedp.model_synthesis import AutonomousModelSynthesizer

        return AutonomousModelSynthesizer()

    def test_underspecified_data_carries_its_own_code(self) -> None:
        """N too small is a different finding from a bad fit."""
        n = 5
        x = np.linspace(0.0, 1.0, n)
        y = 2.0 * x + 1.0
        out = self._engine().synthesize(x, y, x, y)
        assert out[0].epistemic_state is EpistemicState.INSUFFICIENT_EVIDENCE
        assert (
            out[0].abstention_reason is AbstentionReason.UNDERSPECIFIED_DATA
        ), out[0].abstention_reason

    def test_reason_reaches_diagnostics_under_both_keys(self) -> None:
        """The benchmark serializes diagnostics, not the dataclass."""
        n = 5
        x = np.linspace(0.0, 1.0, n)
        y = 2.0 * x + 1.0
        m = self._engine().synthesize(x, y, x, y)[0]
        assert (
            m.diagnostics["abstention_reason"]
            == AbstentionReason.UNDERSPECIFIED_DATA.value
        )
        assert m.diagnostics["abstention_reasons"] == [
            AbstentionReason.UNDERSPECIFIED_DATA.value
        ]
        assert AbstentionReason.UNDERSPECIFIED_DATA.value in m.diagnostics["rationale"]

    def test_field_and_diagnostics_cannot_disagree(self) -> None:
        """Two copies of one fact is a race; this pins them together."""
        n = 5
        x = np.linspace(0.0, 1.0, n)
        y = 2.0 * x + 1.0
        m = self._engine().synthesize(x, y, x, y)[0]
        assert m.diagnostics["abstention_reason"] == m.abstention_reason.value
        assert m.diagnostics["abstention_reasons"] == [
            m.abstention_reason.value
        ]

    def test_a_cleared_model_has_no_abstention_reason(self) -> None:
        """``None`` means endorsed. A placeholder here would be a lie."""
        x = np.linspace(0.0, 10.0, 60)
        y = 3.0 * np.exp(-0.25 * x) + 0.5
        m = self._engine().synthesize(x, y, x, y)[0]
        assert m.abstention_reason is None, (
            f"clean exponential decay abstained: {m.abstention_reason}"
        )
        assert "abstention_reason" not in m.diagnostics

    def test_any_abstention_carries_a_code(self) -> None:
        """A model flagged not-ok must never have a null reason.

        This is the invariant that makes the field usable for filtering: any
        record a consumer treats as abstained has to name its cause.
        """
        x = np.linspace(0.0, 10.0, 60)
        for y in (
            3.0 * np.exp(-0.25 * x) + 0.5,   # clean
            np.random.default_rng(0).normal(size=60),  # noise
            np.where(x < 5.0, 0.0, 10.0),    # shock front
        ):
            for m in self._engine().synthesize(x, y, x, y):
                withheld = m.epistemic_state in (
                    EpistemicState.INSUFFICIENT_EVIDENCE,
                    EpistemicState.NO_VALID_CANDIDATE,
                )
                if withheld:
                    assert m.abstention_reason is not None, (
                        f"abstained with no reason code: {m.expression_str}"
                    )
                    assert m.abstention_reason in set(AbstentionReason)
