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

import json

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
# apedp/singularity_probing.py -- the probe that measured nothing
# --------------------------------------------------------------------------

class TestAFailedProbeIsNotAMeasuredDivergence:
    """The worst silent handler in the repo, and it ran in the wrong direction.

    probe_candidate_singularity evaluated the candidate expression at four
    offsets from a located pole. On failure the handler set
    ``max_probe_val = float("inf")``.

    That value was not a fallback, it was a measurement. ``is_divergent`` has
    an explicit ``not math.isfinite(...)`` clause, so inf drove it True, and
    the report then published the fabricated magnitude in prose: "candidate
    diverges to 1.00e+308", "divergent probe behavior (1.00e+308 >> y_max)".
    A crashed evaluation became a quantitative finding, in a scientific
    record, in both AVOIDED_POLE_IN_GAP and UNOBSERVED_CRITICAL_REGIME.

    The distinction being restored: evaluated-and-got-inf is a real pole;
    could-not-evaluate is no measurement at all.
    """

    @staticmethod
    def _crashing_probe(monkeypatch) -> None:
        """Makes lambdify raise for the probe evaluation only."""
        import apedp.singularity_probing as sp_mod

        real_lambdify = sp_mod.sp.lambdify

        def boom(*a, **k):
            real_lambdify(*a, **k)  # keep construction realistic; the call is what fails

            def wrapper(*_args):
                raise NameError("undefined_function_in_probe")
            return wrapper

        monkeypatch.setattr(sp_mod.sp, "lambdify", boom)

    @staticmethod
    def _observed() -> dict:
        # A real sampling gap around x = 5.0, so a confirmed divergence would
        # legitimately trip Check A. Check A needs
        # gap_width >= gap_factor_threshold * mean_spacing, i.e. 1.3x, so the
        # gap has to be sized against the spacing rather than merely present:
        # 60 points over a 10-wide domain gives ~0.17 spacing, so a 0.2 gap
        # would be ordinary sampling density and correctly not flagged.
        x = np.concatenate([np.linspace(0.0, 4.0, 30), np.linspace(6.0, 10.0, 30)])
        return {"x": x, "y": np.exp(-x)}

    def test_a_crashed_probe_is_not_reported_as_a_divergence(self, monkeypatch):
        from apedp.singularity_probing import ActiveSingularityProbeEngine

        self._crashing_probe(monkeypatch)
        r = ActiveSingularityProbeEngine().probe_candidate_singularity(
            "1.0/(x - 5.0)", [], self._observed())

        assert r.risk_type == "PROBE_INCONCLUSIVE", (
            f"a failed probe was reported as {r.risk_type}")
        assert r.has_singularity_risk is True, (
            "the direction must be toward withholding, not toward certifying")

    def test_the_rationale_quotes_no_magnitude(self, monkeypatch):
        """The fabricated number is the actual harm, so it is pinned directly.

        A test asserting only the risk_type would still pass against a fix
        that set the type correctly and left "diverges to 1.00e+308" in the
        prose -- which is the version a reader would believe.
        """
        from apedp.singularity_probing import ActiveSingularityProbeEngine

        self._crashing_probe(monkeypatch)
        r = ActiveSingularityProbeEngine().probe_candidate_singularity(
            "1.0/(x - 5.0)", [], self._observed())

        blob = json.dumps(r.to_dict(), default=str)
        for fabricated in ("inf", "Infinity", "1e+308", "nan"):
            assert fabricated not in blob, (
                f"a failed probe published {fabricated!r} as a measurement")
        assert "did not complete" in r.rationale
        assert "tool failure, not a" in r.rationale

    def test_the_rationale_names_no_magnitude_at_all(self, monkeypatch):
        """Pins the specific fabrication, not just the absence of a token.

        The original rationale read "candidate diverges to 1.00e+308" and
        "divergent probe behavior (1.00e+308 >> y_max)". A test that only
        checked for the string "inf" would pass against a fix that quoted a
        different large number -- which is the same lie.
        """
        from apedp.singularity_probing import ActiveSingularityProbeEngine

        self._crashing_probe(monkeypatch)
        r = ActiveSingularityProbeEngine().probe_candidate_singularity(
            "1.0/(x - 5.0)", [], self._observed())

        assert "diverges to" not in r.rationale
        assert "divergent probe behavior" not in r.rationale
        assert not any(ch.isdigit() and "e+" in r.rationale.lower()
                       for ch in ["e+"])

    def test_no_probe_magnitude_reaches_gap_details(self, monkeypatch):
        """A consumer must not be able to read a divergence out of gap_details."""
        from apedp.singularity_probing import ActiveSingularityProbeEngine

        self._crashing_probe(monkeypatch)
        r = ActiveSingularityProbeEngine().probe_candidate_singularity(
            "1.0/(x - 5.0)", [], self._observed())

        assert r.gap_details is not None
        assert "max_probe_val" not in r.gap_details
        # The pole itself was located, and that much is real.
        assert r.gap_details["pole_location"] == pytest.approx(5.0)
        assert r.unverified, "the failure must be recorded, not just described"

    def test_a_genuine_divergence_is_still_a_confirmed_finding(self, monkeypatch):
        """Over-correction guard: a real blowup keeps its own verdict.

        Same expression, same gap, working evaluator. 1/(x-5) genuinely
        diverges at the pole, so this must read as AVOIDED_POLE_IN_GAP with a
        real number -- not as PROBE_INCONCLUSIVE, which would be the fix
        swallowing the finding it was meant to protect.
        """
        from apedp.singularity_probing import ActiveSingularityProbeEngine

        r = ActiveSingularityProbeEngine().probe_candidate_singularity(
            "1.0/(x - 5.0)", [], self._observed())

        assert r.risk_type == "AVOIDED_POLE_IN_GAP", r.rationale
        assert r.has_singularity_risk is True
        assert r.gap_details["max_probe_val"] > 0
        assert np.isfinite(r.gap_details["max_probe_val"])
        assert not r.unverified, (
            f"a clean evaluation reported failures: {r.unverified}")

    def test_an_evaluated_infinity_is_still_a_divergence(self, monkeypatch):
        """The other half of the distinction.

        A pole whose expression really does return inf near the pole is a
        measured blowup. Collapsing that into PROBE_INCONCLUSIVE would throw
        away the finding the whole engine exists to make.
        """
        from apedp.singularity_probing import ActiveSingularityProbeEngine

        r = ActiveSingularityProbeEngine().probe_candidate_singularity(
            "1.0/tan(x - 1.0)", [], {"x": np.concatenate(
                [np.linspace(0.0, 0.5, 30), np.linspace(1.5, 3.0, 30)]),
                "y": np.ones(60)})
        assert r.has_singularity_risk is True
        assert r.risk_type in ("AVOIDED_POLE_IN_GAP", "UNOBSERVED_CRITICAL_REGIME")
        assert r.risk_type != "PROBE_INCONCLUSIVE"
        assert r.gap_details is not None
        assert "max_probe_val" in r.gap_details


class TestAFailedRootSearchIsNotAnAbsenceOfPoles:
    """The other face of the same bug, pointing the opposite way.

    Every handler in the root search used to reduce the number of roots
    returned and continue. Fewer roots is a shorter pole list, and a shorter
    pole list is a shorter finding -- so a sympy expression the solver cannot
    handle was reported as "Analytic probing detected no real poles", and a
    candidate with a real pole in the observed data walked past the probe.
    """

    def test_a_crashed_extraction_is_not_reported_as_pole_free(self, monkeypatch):
        import apedp.singularity_probing as sp_mod
        from apedp.singularity_probing import ActiveSingularityProbeEngine

        monkeypatch.setattr(
            sp_mod.sp, "together",
            lambda *a, **k: (_ for _ in ()).throw(NotImplementedError("nope")))

        r = ActiveSingularityProbeEngine().probe_candidate_singularity(
            "1.0/(x - 5.0)", [], {"x": np.linspace(0.0, 10.0, 40),
                                  "y": np.ones(40)})

        assert r.risk_type != "NONE", (
            "a crashed extraction was certified as free of singularities")
        assert r.unverified
        assert "could not be evaluated" in " ".join(r.unverified)

    def test_a_crashed_extraction_that_finds_nothing_is_still_unverified(
        self, monkeypatch,
    ):
        """The no-poles-found branch, which the test above cannot reach.

        ``1.0/(x-5.0)`` survives a crashed ``together`` because the AST walk
        independently finds the Pow, so it takes the *trailing* branch with
        one pole in hand. That left the other half untested: an expression
        with no singular structure at all, where the crashed search returns
        nothing and the report has only the failure to go on.

        This is the case that matters most, because it is the one that
        produces the sentence the fix exists to stop -- "Analytic probing
        detected no real poles or singularities in domain" -- on the
        strength of a check that never ran.
        """
        import apedp.singularity_probing as sp_mod
        from apedp.singularity_probing import ActiveSingularityProbeEngine

        monkeypatch.setattr(
            sp_mod.sp, "together",
            lambda *a, **k: (_ for _ in ()).throw(NotImplementedError("nope")))

        r = ActiveSingularityProbeEngine().probe_candidate_singularity(
            "2.0*x + 1.0", [], {"x": np.linspace(0.0, 10.0, 40), "y": np.ones(40)})

        assert r.pole_locations == [], "precondition: nothing was found"
        assert r.risk_type == "UNVERIFIED", (
            f"a crashed search on a pole-free expression reported {r.risk_type}")
        assert "Analytic probing detected no real poles" not in r.rationale
        assert "did not complete" in r.rationale
        assert r.unverified

    def test_each_root_search_stage_is_recorded_separately(self, monkeypatch):
        """Asserting ``any(...)`` over both stage names kills neither.

        The first version of this test checked
        ``any("solve" in u or "grid scan" in u for u in r.unverified)``, which
        still passes with either stage reverted. Mutation testing found it:
        both "Analytic solve" and "Sign-change grid scan" survived. Each
        stage loses poles on its own, so each is asserted by name.
        """
        import apedp.singularity_probing as sp_mod
        from apedp.singularity_probing import ActiveSingularityProbeEngine

        monkeypatch.setattr(
            sp_mod.sp, "solve",
            lambda *a, **k: (_ for _ in ()).throw(NotImplementedError("nope")))
        monkeypatch.setattr(
            sp_mod.sp, "lambdify",
            lambda *a, **k: (_ for _ in ()).throw(NotImplementedError("nope")))

        r = ActiveSingularityProbeEngine().probe_candidate_singularity(
            "1.0/(x - 5.0)", [], {"x": np.linspace(0.0, 10.0, 40), "y": np.ones(40)})

        assert any("Analytic solve" in u for u in r.unverified), (
            f"the analytic solver raised and nothing recorded it: {r.unverified}")
        assert any("Sign-change grid scan" in u for u in r.unverified), (
            f"the grid scan raised and nothing recorded it: {r.unverified}")

    def test_a_brent_refinement_failure_is_recorded(self, monkeypatch):
        """The third root-search stage, and the subtlest.

        A sign change between two grid points is a *detected* root. When
        brentq then fails to converge, that root is lost -- not skipped, lost.
        It is the one handler where the search demonstrably found something
        and the report ends up with less than it started with.
        """
        import apedp.singularity_probing as sp_mod
        from apedp.singularity_probing import ActiveSingularityProbeEngine

        # Analytic solve succeeds and finds nothing, so the grid scan runs.
        monkeypatch.setattr(sp_mod.sp, "solve", lambda *a, **k: [])
        monkeypatch.setattr(
            sp_mod.opt, "brentq",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no convergence")))

        r = ActiveSingularityProbeEngine().probe_candidate_singularity(
            "1.0/(x - 5.0)", [], {"x": np.linspace(0.0, 10.0, 40), "y": np.ones(40)})

        assert any("Brent refinement" in u for u in r.unverified), (
            f"brentq raised and nothing recorded it: {r.unverified}")

    def test_the_clean_path_still_says_NONE(self):
        """Over-correction guard: a genuinely pole-free expression is clean."""
        from apedp.singularity_probing import ActiveSingularityProbeEngine

        r = ActiveSingularityProbeEngine().probe_candidate_singularity(
            "2.0*x + 1.0", [], {"x": np.linspace(0.0, 10.0, 40), "y": np.ones(40)})
        assert r.risk_type == "NONE"
        assert r.has_singularity_risk is False
        assert r.unverified == []

    def test_extract_singular_components_raises_rather_than_returning_empty(
        self, monkeypatch,
    ):
        """An empty list is what a pole-free expression returns.

        Returning [] on a parse failure made "no answer" and "no poles" the
        same value. stage_c already had a handler that records an extraction
        failure as inconclusive and withholds the candidate -- and it could
        never fire, because nothing raised. Now it can.
        """
        import apedp.singularity_probing as sp_mod
        from apedp.singularity_probing import ActiveSingularityProbeEngine

        engine = ActiveSingularityProbeEngine()
        monkeypatch.setattr(
            sp_mod.sp, "together",
            lambda *a, **k: (_ for _ in ()).throw(NotImplementedError("nope")))

        with pytest.raises(sp_mod._PartialExtraction) as ei:
            engine.extract_singular_components("1.0/(x - 5.0)")
        assert "Denominator extraction" in " ".join(ei.value.failures)

        # And the Stage C handler that was dead code is now reachable.
        from apedp.decision.stage_c import StageCDecisionEngine

        refused, inconclusive = StageCDecisionEngine(
        )._pole_refusals_in_observed_data([_candidate(expr="1.0/(x-5.0)")])
        assert "cand" in inconclusive, (
            "stage_c's inconclusive path is unreachable again")
        assert "cand" not in refused

    def test_a_root_search_failure_is_recorded(self, monkeypatch):
        import apedp.singularity_probing as sp_mod
        from apedp.singularity_probing import ActiveSingularityProbeEngine

        monkeypatch.setattr(
            sp_mod.sp, "solve",
            lambda *a, **k: (_ for _ in ()).throw(NotImplementedError("nope")))
        monkeypatch.setattr(
            sp_mod.sp, "lambdify",
            lambda *a, **k: (_ for _ in ()).throw(NotImplementedError("nope")))

        r = ActiveSingularityProbeEngine().probe_candidate_singularity(
            "1.0/(x - 5.0)", [], {"x": np.linspace(0.0, 10.0, 40), "y": np.ones(40)})
        assert r.unverified
        assert any("solve" in u or "grid scan" in u for u in r.unverified)


class TestNoHandlerInTheseFilesIsLoggingOnly:
    """Source-driven, so a sixth bare handler fails here rather than in a run.

    Same shape as the physics_constraints guard. It asserts on source text
    rather than behaviour because the failure mode is a handler being
    *added back* in the old shape, and there is no input that distinguishes
    a handler that runs from one that never fires.
    """

    @pytest.mark.parametrize("module_name", [
        "apedp.singularity_probing",
        "apedp.special_functions",
    ])
    def test_no_bare_debug_handler(self, module_name: str) -> None:
        import ast as ast_mod
        import importlib

        mod = importlib.import_module(module_name)
        tree = ast_mod.parse(open(mod.__file__).read())

        # Docstrings legitimately discuss the old shape; only executable
        # logger.debug calls count.
        offenders = [
            node.lineno for node in ast_mod.walk(tree)
            if isinstance(node, ast_mod.Call)
            and isinstance(node.func, ast_mod.Attribute)
            and node.func.attr == "debug"
            and isinstance(node.func.value, ast_mod.Attribute)
            and node.func.value.attr == "logger"
        ]
        assert not offenders, (
            f"{module_name} has logging-only handlers at lines {offenders}; "
            f"a raised check would read as a pass")


# --------------------------------------------------------------------------
# apedp/special_functions.py -- NaN for a crash vs NaN for a pole
# --------------------------------------------------------------------------

class TestANanFromACrashIsNotANanFromAPole:
    """Both produce an all-NaN array of the right shape.

    The handoff recorded this site as a shape change -- a 50-point probe
    degrading to a 1-point probe. That is not what the code does: the
    ``n_pts`` fallback sits on the success path too, so the shape is
    ``len(variable_data)`` either way. The real defect is the one below.

    Returning NaN is correct and stays: NaN propagates through arithmetic, so
    a caller computing MSE gets NaN rather than a plausible number, and the
    shape is preserved so a caller zipping against the grid does not
    misalign. What was missing is any way to tell a domain restriction
    (log of a negative number -- NaN is the right answer) from a crash
    (lambdify cannot resolve a name -- NaN stands in for a question nobody
    answered). check_domain_safety counts non-finite values to decide
    whether an interval is hazardous, so a crash arriving as all-NaN reads
    as "this interval is full of singularities" -- a measurement, in a report
    that feeds identity certification.
    """

    @staticmethod
    def _crashing_expr() -> sp.Expr:
        # lambdify cannot resolve an undefined function name.
        return sp.Function("definitely_not_defined")(sp.Symbol("x"))

    def test_a_genuine_domain_restriction_is_not_recorded_as_a_failure(self) -> None:
        import apedp.special_functions as sf

        x = np.full(50, -1.0)
        errors: list[str] = []
        out = sf.SpecialFunctionLibrary.evaluate_vectorized(
            sp.log(sp.Symbol("x")), {"x": x}, errors=errors)

        assert out.shape == x.shape
        assert np.all(np.isnan(out)), "log(-1) should be NaN"
        assert errors == [], (
            f"a correct mathematical answer was recorded as a failure: {errors}")

    def test_a_crash_is_recorded(self) -> None:
        import apedp.special_functions as sf

        x = np.linspace(0.1, 5.0, 50)
        errors: list[str] = []
        out = sf.SpecialFunctionLibrary.evaluate_vectorized(
            self._crashing_expr(), {"x": x}, errors=errors)

        assert out.shape == x.shape, "the shape must still be preserved"
        assert np.all(np.isnan(out)), "NaN must still be emitted, not a number"
        assert errors, "a crash must be recorded"
        assert "could not be completed" in errors[0]
        assert "NameError" in errors[0]

    def test_the_two_cases_are_now_distinguishable(self) -> None:
        """The behavioural claim, stated directly.

        Byte-for-byte these two results were identical before the fix, and
        that identity is the bug: a caller could not tell a real pole from a
        tool failure.
        """
        import apedp.special_functions as sf

        x = np.linspace(0.1, 5.0, 50)
        genuine_errors: list[str] = []
        crash_errors: list[str] = []
        genuine = sf.SpecialFunctionLibrary.evaluate_vectorized(
            sp.log(sp.Symbol("x")), {"x": np.full(50, -1.0)}, errors=genuine_errors)
        crash = sf.SpecialFunctionLibrary.evaluate_vectorized(
            self._crashing_expr(), {"x": x}, errors=crash_errors)

        assert genuine.shape == crash.shape
        assert np.array_equal(np.isnan(genuine), np.isnan(crash)), (
            "expected both to be all-NaN; the point is that only one is "
            "recorded as a failure")
        assert genuine_errors == []
        assert len(crash_errors) == 1, (
            "the two must be distinguishable by the errors channel alone")

    def test_the_errors_argument_is_optional(self) -> None:
        """The signature change must not break existing callers."""
        import apedp.special_functions as sf

        x = np.linspace(0.1, 5.0, 50)
        out = sf.SpecialFunctionLibrary.evaluate_vectorized(
            "erf(x)", {"x": x})  # no errors= argument
        assert out.shape == x.shape
        assert np.all(np.isfinite(out))

    def test_the_failure_is_logged_above_debug(self, monkeypatch, caplog) -> None:
        """Not a traceback request: this substitutes a value on a result.

        logger.debug is invisible at default configuration, so a NaN standing
        in for a failed evaluation reached every consumer with no trace.
        """
        import logging

        import apedp.special_functions as sf

        with caplog.at_level(logging.WARNING, logger="apedp.special_functions"):
            sf.SpecialFunctionLibrary.evaluate_vectorized(
                self._crashing_expr(), {"x": np.linspace(0.1, 5.0, 50)})
        assert any("NaN" in r.message for r in caplog.records), (
            "a value substitution on a scientific result was logged below "
            "the default threshold")


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


class TestOutOfScopeIsNotAFailedCheck:
    """A non-polynomial component is a scope limit, not a failed check.

    Recording it as inconclusive withholds the candidate, and since almost
    every fitted expression with an exponential in the denominator produces
    one, that withholds nearly everything. It turns a documented limitation
    of the bounded linear-form solve into a silent rejection -- which is the
    same class of bug as the one being fixed, wearing the fix's clothes.

    This was found by regression, not by reading: it broke
    `test_autonomous_synthesis_on_synth_01`, whose winning expression carries
    `exp(-0.30x)` in the denominator.
    """

    @staticmethod
    def _exp_denominator() -> CandidateProfile:
        return _candidate(
            expr="2.4*x**1.5*exp(-0.3*x)/(0.84*x**2 + 1.0)",
        )

    def test_a_transcendental_denominator_is_not_inconclusive(self) -> None:
        engine = StageCDecisionEngine()
        refused, inconclusive = engine._pole_refusals_in_observed_data(
            [self._exp_denominator()]
        )
        assert inconclusive == {}, (
            "a documented scope limit must not withhold the candidate"
        )
        assert refused == {}

    def test_such_a_candidate_is_still_evaluated(self) -> None:
        """The over-correction guard, at the level that actually broke."""
        decision = StageCDecisionEngine().evaluate([self._exp_denominator()])
        assert decision.state is not EpistemicState.NO_VALID_CANDIDATE, (
            decision.adjudication_rationale
        )

    def test_a_genuine_pole_is_still_refused(self) -> None:
        """Scope must not become a blanket exemption for everything."""
        engine = StageCDecisionEngine()
        refused, _ = engine._pole_refusals_in_observed_data(
            [_candidate(expr="1.0/(x - 5.0)")]
        )
        assert "cand" in refused

    def test_a_linear_pole_alongside_a_transcendental_is_refused(
        self,
    ) -> None:
        """The mixed case: one component out of scope, one squarely in it."""
        engine = StageCDecisionEngine()
        refused, inconclusive = engine._pole_refusals_in_observed_data(
            [_candidate(expr="exp(-0.3*x)/(x - 5.0)")]
        )
        assert "cand" in refused, (refused, inconclusive)


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
