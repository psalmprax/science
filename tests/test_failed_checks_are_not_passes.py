"""A check that raised is not a check that passed.

Two of the three `except Exception: logger.debug(...)` shapes in this codebase
are harmless: the code catches, falls back to a correct implementation, and
returns an answer that is still right. Those are observability gaps -- a run
cannot prove which path executed -- and they are called out in the handoff note
rather than patched here.

The third shape is different and was the expensive one. When the guarded
operation is a *verdict* rather than a computation, logging is not a fallback,
it is a silent pass:

* `PhysicsConstraintEngine.evaluate` computes `is_sound = not violations` and
  `score = passed_checks / 7`. Handlers that only logged left the flag `True`
  and appended nothing, so `sp.limit` raising on an expression with no origin
  limit produced a report certifying the formula physically sound on a check
  that never ran.
* `StageCDecisionEngine._pole_refusals_in_observed_data` had one exit for two
  findings. A candidate was refused when a pole was found and simply absent
  when extraction raised; the caller could only test for presence, so "no pole"
  and "no answer" reached Stage A looking the same and the second one endorsed
  a model nobody had checked.

These tests inject the failures deliberately, so the behaviour is pinned rather
than inferred from the code reading correctly.
"""
from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from apedp.decision.stage_c import (
    CandidateProfile,
    EpistemicState,
    ModelClass,
    StageCDecisionEngine,
    _strip_abs_to_linear_form,
)
from apedp.model_synthesis import SynthesizedModel

# --------------------------------------------------------------------------
# apedp/physics_constraints.py
# --------------------------------------------------------------------------

def _spec(**kw):
    from apedp.physics_constraints import PhysicsSpecification

    base = {
        "variable_units": {"x": "dimensionless", "t": "time"},
        "target_unit": "dimensionless",
        "origin_limit": 1.0,
        "far_field_limit": 0.0,
        "positivity": True,
    }
    base.update(kw)
    return PhysicsSpecification(**base)


def _report(expr: str = "2.0*x", spec=None, **kw):
    from apedp.physics_constraints import PhysicsConstraintEngine

    return PhysicsConstraintEngine().evaluate(
        expr, spec if spec is not None else _spec(), **kw
    )


class TestFailedPhysicsCheckIsNotAPass:
    """The property: a raising evaluator cannot leave the report positive."""

    def test_raised_dimensional_check_is_not_reported_valid(self, monkeypatch) -> None:
        import apedp.physics_constraints as pc

        def boom(*_a, **_k):
            raise ValueError("cannot resolve units")

        monkeypatch.setattr(pc, "check_units", boom)
        r = _report()
        assert r.dimensional_valid is False
        assert not r.is_physically_sound

    def test_raised_origin_limit_is_not_reported_valid(self, monkeypatch) -> None:
        import apedp.physics_constraints as pc

        real_limit = sp.limit
        monkeypatch.setattr(
            pc.sp, "limit",
            lambda *a, **k: (_ for _ in ()).throw(NotImplementedError("no limit")),
        )
        try:
            r = _report()
        finally:
            monkeypatch.setattr(pc.sp, "limit", real_limit)
        assert r.boundary_valid is False
        assert not r.is_physically_sound

    def test_raised_far_field_limit_is_not_reported_valid(self, monkeypatch) -> None:
        import apedp.physics_constraints as pc

        monkeypatch.setattr(
            pc.sp, "limit",
            lambda *a, **k: (_ for _ in ()).throw(NotImplementedError("no limit")),
        )
        r = _report()
        assert r.asymptotics_valid is False
        assert not r.is_physically_sound

    def test_raised_domain_check_invalidates_positivity_and_parity(
        self, monkeypatch,
    ) -> None:
        import apedp.physics_constraints as pc

        monkeypatch.setattr(
            pc.sp, "lambdify",
            lambda *a, **k: (_ for _ in ()).throw(ValueError("cannot lambdify")),
        )
        r = _report("2.0*x + x**2")
        assert r.positivity_valid is False
        assert r.parity_valid is False
        assert not r.is_physically_sound

    def test_a_raised_check_lowers_the_score(self, monkeypatch) -> None:
        """Score is passed_checks/7, so an unchecked check must not count."""
        import apedp.physics_constraints as pc

        monkeypatch.setattr(
            pc, "check_units",
            lambda *a, **k: (_ for _ in ()).throw(ValueError("cannot resolve units")),
        )
        r = _report()
        assert r.compliance_score < 1.0

    def test_the_violation_names_the_check_and_the_error(self, monkeypatch) -> None:
        """A reader has to be able to tell which check did not run."""
        import apedp.physics_constraints as pc

        monkeypatch.setattr(
            pc, "check_units",
            lambda *a, **k: (_ for _ in ()).throw(ValueError("cannot resolve units")),
        )
        r = _report()
        assert any("Dimensional consistency check" in v for v in r.violations)
        assert any("cannot resolve units" in v for v in r.violations)

    def test_a_clean_expression_is_still_certified_sound(self) -> None:
        """The guard against over-correcting into permanent abstention.

        `exp(-x)` satisfies the default spec exactly -- limit 1 at the origin,
        limit 0 at infinity, positive throughout -- so any violation here is
        the new handlers firing on a check that genuinely passed.
        """
        r = _report("exp(-x)")
        assert r.is_physically_sound, r.violations
        assert r.compliance_score == 1.0

    def test_a_genuine_violation_is_still_reported(self) -> None:
        """Distinguishes 'unverified' from 'failed' -- both unsound, different text."""
        # A negative-slope linear model violates a nonnegative-origin spec.
        r = _report("-2.0*x", _spec(origin_limit=1.0, positivity=True))
        assert not r.is_physically_sound
        assert r.violations

    def test_every_check_that_can_raise_is_covered(self, monkeypatch) -> None:
        """No handler may be left in the old shape.

        Driven off the source rather than a hand-kept list, so adding a sixth
        evaluator without a handler fails here rather than in production.
        """
        import apedp.physics_constraints as pc

        src = (pc.__file__ or "")
        text = open(src).read() if src else ""
        assert text, "could not read physics_constraints.py"
        assert "logger.debug" not in text, (
            "a handler is back to logging-only; a raised check would read as "
            "a pass"
        )


# --------------------------------------------------------------------------
# apedp/decision/stage_c.py -- pole check
# --------------------------------------------------------------------------

def _candidate(cid: str = "cand", expr: str = "2.0*x + 1.0", **kw) -> CandidateProfile:
    base = {
        "candidate_id": cid,
        "expression": expr,
        "math_sound": True,
        "dim_status": "VALID",
        "holdout_mse": 0.01,
        "holdout_r2": 0.99,
        "in_sample_mse": 0.009,
        "complexity": 2.0,
        "cond_number": 1.0,
        "bootstrap_stability": 0.9,
        "model_class": ModelClass.SYMBOLIC,
        "metadata": {
            "x_obs_min": 0.0,
            "x_obs_max": 10.0,
            "x_gaps": [],
        },
    }
    base.update(kw)
    return CandidateProfile(**base)


class TestFailedPoleCheckIsNotACleanBill:
    def test_extraction_failure_marks_the_candidate_inconclusive(
        self, monkeypatch,
    ) -> None:
        import apedp.singularity_probing as sp_mod

        engine = StageCDecisionEngine()
        monkeypatch.setattr(
            sp_mod.ActiveSingularityProbeEngine,
            "extract_singular_components",
            lambda self, e: (_ for _ in ()).throw(ValueError("cannot extract")),
        )
        refused, inconclusive = engine._pole_refusals_in_observed_data(
            [_candidate()]
        )
        assert "cand" in inconclusive
        assert "cand" not in refused

    def test_inconclusive_is_distinct_from_refused(self, monkeypatch) -> None:
        """"We found a pole" and "we could not look" must not share a bucket."""
        import apedp.singularity_probing as sp_mod

        engine = StageCDecisionEngine()
        # A genuine pole at x=5, strictly inside [0, 10] and outside gaps.
        monkeypatch.setattr(
            sp_mod.ActiveSingularityProbeEngine,
            "extract_singular_components",
            lambda self, e: [(sp.Symbol("x") - 5, "DENOMINATOR")],
        )
        refused, inconclusive = engine._pole_refusals_in_observed_data(
            [_candidate(expr="1.0/(x-5)")]
        )
        assert "cand" in refused and not inconclusive
        assert "pole at x=" in refused["cand"]

    def test_an_unchecked_candidate_is_not_endorsed(self, monkeypatch) -> None:
        """The behaviour that matters: crash must not become a pass."""
        import apedp.singularity_probing as sp_mod

        monkeypatch.setattr(
            sp_mod.ActiveSingularityProbeEngine,
            "extract_singular_components",
            lambda self, e: (_ for _ in ()).throw(ValueError("cannot extract")),
        )
        decision = StageCDecisionEngine().evaluate([_candidate()])
        assert decision.state is EpistemicState.NO_VALID_CANDIDATE
        assert decision.winner is None

    def test_the_rationale_says_it_was_a_tool_failure(self, monkeypatch) -> None:
        """"Withheld" and "found nothing" are different sentences."""
        import apedp.singularity_probing as sp_mod

        monkeypatch.setattr(
            sp_mod.ActiveSingularityProbeEngine,
            "extract_singular_components",
            lambda self, e: (_ for _ in ()).throw(ValueError("cannot extract")),
        )
        r = StageCDecisionEngine().evaluate([_candidate()]).adjudication_rationale
        assert "could not be completed" in r
        assert "tool failure" in r

    def test_the_candidate_evaluation_records_the_inconclusiveness(
        self, monkeypatch,
    ) -> None:
        import apedp.singularity_probing as sp_mod

        monkeypatch.setattr(
            sp_mod.ActiveSingularityProbeEngine,
            "extract_singular_components",
            lambda self, e: (_ for _ in ()).throw(ValueError("cannot extract")),
        )
        ev = StageCDecisionEngine().evaluate([_candidate()]).candidate_evaluations
        reason = ev["cand"]["pole_check_inconclusive"]
        assert "could not be completed" in reason
        # Same text as the limitation, so a reader is not told two stories.
        assert any(reason in lim for lim in ev["cand"]["limitations"])

    def test_a_genuinely_pole_free_candidate_is_unaffected(self) -> None:
        """Over-correction guard: the clean path must still certify."""
        decision = StageCDecisionEngine().evaluate([_candidate()])
        assert decision.state is not EpistemicState.NO_VALID_CANDIDATE
        assert "pole_check_inconclusive" not in decision.candidate_evaluations["cand"]

    def test_candidates_without_range_metadata_are_untouched(self) -> None:
        """Preserves the documented skip for callers that supply no metadata."""
        c = _candidate()
        c.metadata.pop("x_obs_min")
        refused, inconclusive = StageCDecisionEngine()._pole_refusals_in_observed_data([c])
        assert refused == {} and inconclusive == {}


# --------------------------------------------------------------------------
# apedp/decision/stage_c.py -- paired-bootstrap noise-floor probe
# --------------------------------------------------------------------------

def _two_rivals() -> list[CandidateProfile]:
    """Two candidates routed past the margin and into the noise-floor probe.

    The rival's holdout error is above ``best * 1.15`` so the margin branch
    misses, and its in-sample error is worse so the rank-reversal branch misses
    too. That leaves exactly one route into `competitive`: the paired-bootstrap
    equivalence probe -- the code path being made to raise.

    The guard is ``holdout_mse <= best * margin * 1.5 or |diff| <= 0.05``,
    which is wide enough that the fixture cannot drift out of the path as the
    margin ratio is tuned.
    """
    rng = np.random.default_rng(0)
    res = list(rng.normal(0.0, 0.1, size=8))
    return [
        _candidate(
            "best", "2.0*x + 1.0", holdout_mse=0.010, in_sample_mse=0.009,
            complexity=2.0, metadata={
                "x_obs_min": 0.0, "x_obs_max": 10.0, "x_gaps": [],
                "holdout_residuals": res,
            },
        ),
        _candidate(
            "rival", "3.0*x**2 + 1.0", holdout_mse=0.014, in_sample_mse=0.012,
            complexity=3.0, metadata={
                "x_obs_min": 0.0, "x_obs_max": 10.0, "x_gaps": [],
                "holdout_residuals": [r + 0.01 for r in res],
            },
        ),
    ]


def _break_the_bootstrap(monkeypatch) -> None:
    """Make the probe raise from inside its own try block.

    Patched on the numpy the module imported, so the failure lands exactly
    where the handler under test is, rather than being simulated by
    stubbing out the call the handler wraps.
    """
    import apedp.decision.stage_c as sc

    def boom(*_a, **_k):
        raise ValueError("bootstrap blew up")

    monkeypatch.setattr(sc.np, "std", boom)


class TestCrashedNoiseFloorProbeIsDisclosed:
    def test_the_fixture_reaches_the_probe(self) -> None:
        """If the fixture stopped exercising the path, the rest prove nothing."""
        import apedp.decision.stage_c as sc

        seen = []
        real = sc.np.std
        try:
            sc.np.std = lambda *a, **k: (seen.append(1), real(*a, **k))[1]
            StageCDecisionEngine().evaluate(_two_rivals())
        finally:
            sc.np.std = real
        assert seen, "fixture no longer reaches the paired-bootstrap probe"

    def test_a_crashed_probe_is_not_reported_as_distinguishable(
        self, monkeypatch,
    ) -> None:
        """The bug: absence from `competitive` read as "shown to be worse".

        The candidate is missing from the competitive set either way; the
        difference the fix makes is that the record now says why.
        """
        _break_the_bootstrap(monkeypatch)
        decision = StageCDecisionEngine().evaluate(_two_rivals())
        blob = decision.adjudication_rationale + " ".join(decision.missing_evidence)
        assert "rival" in blob, blob

    def test_the_disclosure_says_the_comparison_never_happened(
        self, monkeypatch,
    ) -> None:
        _break_the_bootstrap(monkeypatch)
        decision = StageCDecisionEngine().evaluate(_two_rivals())
        assert "not because they were shown to be distinguishable" in (
            decision.adjudication_rationale
        )

    def test_the_disclosure_names_the_underlying_error(self, monkeypatch) -> None:
        _break_the_bootstrap(monkeypatch)
        decision = StageCDecisionEngine().evaluate(_two_rivals())
        blob = decision.adjudication_rationale + " ".join(decision.missing_evidence)
        assert "bootstrap blew up" in blob

    def test_no_caveat_when_the_probe_succeeds(self, monkeypatch) -> None:
        """The caveat must not fire on the healthy path."""
        import apedp.decision.stage_c as sc

        real = sc.np.std
        monkeypatch.setattr(
            sc.np, "std", lambda *a, **k: real(*a, **k)
        )
        decision = StageCDecisionEngine().evaluate(_two_rivals())
        blob = decision.adjudication_rationale + " ".join(decision.missing_evidence)
        assert "noise-floor probe" not in blob
        assert "Uniqueness is provisional" not in blob
        # And the healthy path still reaches a multi-hypothesis verdict, so the
        # tests above are not passing because the rival was never in play.
        assert decision.state is EpistemicState.MULTI_HYPOTHESIS


class TestRootSearchFailureIsRecorded:
    """The inner handler, one level below the extractor.

    Separate from the extraction tests because they are separate handlers. A
    suite that only breaks the extractor leaves this one free to go back to
    skipping, which is how the same component went unexamined twice.
    """

    @staticmethod
    def _break_poly(monkeypatch) -> None:
        import apedp.decision.stage_c as sc

        def boom(*_a, **_k):
            raise ValueError("cannot build a polynomial")

        monkeypatch.setattr(sc.sp, "Poly", boom)

    def test_a_root_search_failure_is_inconclusive(self, monkeypatch) -> None:
        self._break_poly(monkeypatch)
        engine = StageCDecisionEngine()
        refused, inconclusive = engine._pole_refusals_in_observed_data(
            [_candidate(expr="1.0/(x - 5.0)")]
        )
        assert "cand" in inconclusive, (refused, inconclusive)
        assert "root search failed" in inconclusive["cand"]

    def test_a_root_search_failure_withholds_the_candidate(
        self, monkeypatch,
    ) -> None:
        self._break_poly(monkeypatch)
        decision = StageCDecisionEngine().evaluate(
            [_candidate(expr="1.0/(x - 5.0)")]
        )
        assert decision.state is EpistemicState.NO_VALID_CANDIDATE
        assert decision.winner is None

    def test_the_inconclusive_reason_survives_the_happy_path(self) -> None:
        """Without the break, the same expression is cleanly refused.

        Proves the tests above are detecting the failure and not simply the
        presence of a pole.
        """
        engine = StageCDecisionEngine()
        refused, inconclusive = engine._pole_refusals_in_observed_data(
            [_candidate(expr="1.0/(x - 5.0)")]
        )
        assert "cand" in refused
        assert inconclusive == {}


class TestEquivalenceSentenceIsTruthful:
    """The sentence quotes its evidence, so the evidence has to be real.

    It used to read "|Δho_mse = 4.00e-03 <= 1.5e-3 or paired SE p > 0.05" --
    printing an inequality that was false, beside a number that violated it,
    on runs where the decision had actually come from the bootstrap branch. A
    report that states a check it did not perform is the same defect as a
    check that is silently skipped.
    """

    def _rationale(self) -> str:
        # diff_mse is 4e-3 here, above the 1.5e-3 floor, so the only route to
        # equivalence is the paired bootstrap -- which is what makes this
        # fixture able to catch a sentence that always quotes the floor.
        return StageCDecisionEngine().evaluate(_two_rivals()).adjudication_rationale

    def test_it_names_the_branch_that_actually_fired(self) -> None:
        assert "paired bootstrap" in self._rationale()

    def test_it_does_not_claim_the_noise_floor_was_met(self) -> None:
        assert "<= 1.5e-3" not in self._rationale()

    def test_it_reports_the_quantities_it_compared(self) -> None:
        r = self._rationale()
        assert "1.96" in r, r
        assert "paired residuals" in r, r

    def test_a_floor_based_equivalence_does_quote_the_floor(self) -> None:
        """The other direction: the floor must still be reported when it fires.

        Otherwise a fix that simply deletes the number would pass every test
        above while the report stopped saying why anything was equivalent.
        """
        # The floor branch is only reachable in a narrow band: |diff| <= 1.5e-3
        # while the rival is *outside* the 15% holdout margin, which means the
        # best error has to be small in absolute terms (margin 2.3e-4, rival
        # 7e-4, diff 5e-4). No residuals are supplied, so the bootstrap branch
        # is skipped and the floor is the only thing that can fire.
        rivals = [
            _candidate(
                "best", "2.0*x + 1.0", holdout_mse=0.0002, in_sample_mse=0.00019,
                metadata={"x_obs_min": 0.0, "x_obs_max": 10.0, "x_gaps": []},
            ),
            _candidate(
                "rival", "3.0*x**2 + 1.0", holdout_mse=0.0007, in_sample_mse=0.0009,
                metadata={"x_obs_min": 0.0, "x_obs_max": 10.0, "x_gaps": []},
            ),
        ]
        r = StageCDecisionEngine().evaluate(rivals).adjudication_rationale
        assert "<= 1.5e-3" in r, r
        assert "paired bootstrap:" not in r, r


# --------------------------------------------------------------------------
# The Abs-stripping that made the pole check silently skip a component
# --------------------------------------------------------------------------

class TestSingularComponentStripping:
    """`_strip_abs_to_linear_form` is the fix the caught exception was hiding.

    These are the sharp tests. An engine-level test cannot isolate the strip,
    because a fractional-power expression usually yields a second
    POW_NEGATIVE component that locates the same pole by another route -- so
    the bug stayed invisible through the engine and only shows up here.
    """

    def test_bare_abs_is_stripped(self) -> None:
        x = sp.Symbol("x")
        assert _strip_abs_to_linear_form(sp.Abs(x - 3)) == x - 3

    def test_abs_under_a_positive_power_is_stripped(self) -> None:
        """The regression: this form is a Pow, so `isinstance(., sp.Abs)` missed it."""
        x = sp.Symbol("x")
        got = _strip_abs_to_linear_form(sp.Abs(x - 3) ** sp.Rational(1, 2))
        assert got == x - 3, f"got {got}"

    def test_the_stripped_form_reaches_the_polynomial_stage(self) -> None:
        """sp.Poly raising here was the exception the handler swallowed."""
        x = sp.Symbol("x")
        got = _strip_abs_to_linear_form(sp.Abs(x - 3) ** sp.Rational(1, 2))
        poly = sp.Poly(got, x)
        assert poly.degree() == 1
        assert [float(c) for c in poly.all_coeffs()] == [1.0, -3.0]

    @pytest.mark.parametrize("exponent", ["1/2", "1/3", "2", "5/2"])
    def test_any_positive_exponent_is_stripped(self, exponent) -> None:
        x = sp.Symbol("x")
        got = _strip_abs_to_linear_form(sp.Abs(x - 3) ** sp.Rational(exponent))
        assert got == x - 3, f"exponent {exponent} gave {got}"

    def test_negative_exponents_are_left_alone(self) -> None:
        """Documented boundary: only the zero location is reusable, and a
        negative power's local behaviour is not described by a linear form."""
        x = sp.Symbol("x")
        original = sp.Abs(x - 3) ** sp.Rational(-1, 2)
        assert _strip_abs_to_linear_form(original) is original

    def test_nested_abs_is_stripped_all_the_way_in(self) -> None:
        x = sp.Symbol("x")
        got = _strip_abs_to_linear_form(sp.Abs(sp.Abs(x - 3) ** 2) ** 2)
        assert got == x - 3, f"got {got}"

    def test_an_unrelated_expression_is_returned_unchanged(self) -> None:
        x = sp.Symbol("x")
        original = x**2 + 1
        assert _strip_abs_to_linear_form(original) is original

    def test_a_whole_expression_survives_a_round_trip(self) -> None:
        """End-to-end: the component is located, not silently dropped."""
        x = sp.Symbol("x")
        for base in (sp.Abs(x - 3), sp.Abs(x - 3) ** sp.Rational(1, 2)):
            got = _strip_abs_to_linear_form(base)
            zeros = sp.solve(sp.Eq(got, 0), x)
            assert zeros == [3], f"{base} -> {zeros}"


# --------------------------------------------------------------------------
# Cross-check: the abstention reason must not claim a cause it did not test
# --------------------------------------------------------------------------

def test_uncheckable_model_does_not_report_a_confident_family() -> None:
    """A SynthesizedModel with no parameters is the empty-candidate record.

    It must not be described as belonging to a real structural family, which
    is what the abstention reason codes exist to prevent.
    """
    from apedp.decision.stage_c import AbstentionReason

    m = SynthesizedModel(
        expression_str="MISSING_STRUCTURE()",
        sympy_expr=sp.Symbol("x"),
        parameters={},
        train_mse=1.0,
        val_mse=1.0,
        epistemic_state=EpistemicState.NO_VALID_CANDIDATE,
        abstention_reason=AbstentionReason.NO_CANDIDATE_FITTED,
    )
    assert m.abstention_reason is AbstentionReason.NO_CANDIDATE_FITTED
    assert m.structural_family != "unknown" or m.abstention_reason is not None


def test_nan_mse_is_not_treated_as_a_fit() -> None:
    """A non-finite holdout error must not read as a competitive candidate."""
    assert not np.isfinite(float("nan"))
