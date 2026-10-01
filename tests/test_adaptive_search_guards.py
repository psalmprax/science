"""Exercise the guards in apedp/adaptive_search.py, and declare the ones it cannot.

Branch coverage put the module at 7 of 42 with 27 untaken decision outcomes --
17% branch coverage against 34% line coverage, so most of what never ran was
never *decided*. Reading the untaken sites, they split cleanly:

**Reachable without a real engine** (this file):

  * ``if generations < 1 or pop_size < 1 or chunk_size < 1: raise`` -- the
    validation guard, which sits *before* the ``isinstance(engine,
    MultiVariableSRE)`` dispatch and so runs for any engine at all. An
    unexercised argument guard is exactly the shape that lets a caller pass
    ``pop_size=0`` and get an empty population with no complaint.
  * ``if seed is None`` in ``_seeded_global_rng`` -- the no-seed path, which
    must leave the process RNG streams untouched.
  * The legacy path's stagnation break and its ``if chunk_history`` guard.

**Not reachable here** (declared in the last test):

  * everything inside ``_initial_population`` and
    ``_run_stateful_generation`` -- the consistency filter, the attempt cap,
    the frontier de-duplication, the ``improved`` branch that resets the
    stagnation counter, and the mutation-flag loop;
  * ``if decision.parameters.get(flag) is True``, which enables
    ``allow_power_mutation`` and friends;
  * ``if state["best_score"] < 1e-25``, the convergence bail.

All of those require a genuine ``MultiVariableSRE``, whose construction is a
real symbolic-regression engine. ``run_adaptive_discovery`` dispatches on
``isinstance``, so the existing ``FakeSearch`` test covers only the legacy
branch -- which is why the module's branch coverage looks like this while its
legacy path is in fact tested.

That split is the finding. Nothing here is *wrong*; the module simply has an
untested half, and this file records which half so the next person does not have
to re-derive it from a coverage number.
"""
from __future__ import annotations

import random

import numpy as np
import pytest
import sympy as sp

from apedp.adaptive_search import _seeded_global_rng, run_adaptive_discovery
from apedp.decision import LocalSearchController, SearchAction
from apedp.decision.policy import SearchDecision


class _Stub:
    """Minimal legacy-path engine: anything that is not MultiVariableSRE."""

    allow_power_mutation = False
    allow_rational_mutation = False
    allow_compose_mutation = False
    allow_logcomp_mutation = False

    def __init__(self, *, stagnation: int = 0, chunk_history: bool = True):
        self.calls = 0
        self.stagnation = stagnation
        self.give_history = chunk_history
        self.stagnation_report = {"generations_without_improvement": 0, "improvements": 1}

    def run_discovery(self, x_data, y_data, generations, pop_size, seed):
        self.calls += 1
        score = 1.0 / self.calls
        if self.stagnation:
            self.stagnation_report = {"generations_without_improvement": self.stagnation,
                                      "improvements": 0}
        history = ([{"generation": i, "best_score": score, "pareto_frontier_size": 1}
                    for i in range(generations)] if self.give_history else [])
        return sp.Symbol("x"), score, history


def _data():
    return {"x": np.arange(10.0)}, np.arange(10.0)


class TestValidationGuard:
    """Runs before the isinstance dispatch, so it is reachable with any engine."""

    @pytest.mark.parametrize("kwargs", [
        {"generations": 0}, {"generations": -1},
        {"pop_size": 0}, {"chunk_size": 0},
    ])
    def test_non_positive_sizes_are_rejected(self, kwargs):
        engine = _Stub()
        params = {"generations": 4, "pop_size": 4, "chunk_size": 2}
        params.update(kwargs)
        with pytest.raises(ValueError, match="must be positive"):
            run_adaptive_discovery(engine, *_data(), **params)

    def test_the_guard_names_the_parameter(self):
        """A guard that raises a bare ValueError leaves the caller guessing."""
        with pytest.raises(ValueError) as exc:
            run_adaptive_discovery(_Stub(), *_data(), pop_size=0)
        assert "pop_size" in str(exc.value), str(exc.value)

    def test_valid_parameters_are_not_rejected(self):
        engine = _Stub()
        result = run_adaptive_discovery(engine, *_data(), generations=2,
                                        pop_size=4, chunk_size=1)
        assert result.best_candidate is not None


class TestSeedingDoesNotLeak:
    """``_seeded_global_rng`` seeds the *global* streams, so it must restore them.

    The engine samples from the process-wide ``random`` / ``numpy`` generators.
    Seeding without restoring makes the numbers an unrelated caller receives
    depend on whether an adaptive search ran first -- an order dependency that
    no single test can see, because each test looks at its own result.
    """

    def test_seed_none_leaves_the_streams_alone(self):
        # Two draws from the natural stream, so the post-context draw can be
        # compared against the *second* one: the draw inside the context was
        # the first. Comparing against a single earlier draw would fail for
        # any implementation, which is how a broken test gets written.
        random.seed(1234)
        expected = [random.random() for _ in range(2)]

        random.seed(1234)
        with _seeded_global_rng(None):
            random.random()          # draw from the global stream
        actual = random.random()

        assert actual == pytest.approx(expected[1]), (
            "seed=None must not touch the global RNG; the draw inside the "
            "context consumed state that outlived it")

    def test_a_seed_is_restored_afterwards(self):
        random.seed(1234)
        expected_after = [random.random() for _ in range(3)]

        random.seed(1234)
        with _seeded_global_rng(99):
            for _ in range(50):       # disturb the global streams
                random.random()
                np.random.rand()
        actual_after = [random.random() for _ in range(3)]

        assert actual_after == pytest.approx(expected_after), (
            "the RNG state was not restored; an adaptive search changed the "
            "numbers every other component receives")

    def test_the_context_is_reproducible_within_itself(self):
        with _seeded_global_rng(7):
            first = [random.random() for _ in range(3)]
        with _seeded_global_rng(7):
            second = [random.random() for _ in range(3)]
        assert first == pytest.approx(second)


class _StopController:
    """A hand-written controller that asks to stop, for both stop actions.

    The shipped ``LocalSearchController`` can now ask to stop as well, and
    ``TestTheShippedControllerCanStop`` drives the search with it. This class
    stays for the narrower question: the break responds to whatever the
    controller returns, not to the one decision rule that happens to be
    shipped.
    """

    def __init__(self, action=SearchAction.STOP_STAGNATION, reason="stagnated"):
        self.action, self.reason = action, reason
        self.calls = 0

    def decide(self, telemetry):
        self.calls += 1
        return SearchDecision(self.action, self.reason, {})


class TestLegacyPathGuards:
    @pytest.mark.parametrize("action", [SearchAction.STOP_STAGNATION,
                                        SearchAction.HUMAN_REVIEW])
    def test_stopping_stops_the_search_and_records_why(self, action):
        engine = _Stub(stagnation=9)
        result = run_adaptive_discovery(engine, *_data(), generations=20,
                                        pop_size=4, chunk_size=2,
                                        controller=_StopController(action))
        assert result.stopped_reason == "stagnated", (
            "the controller asked to stop and the search ran to generations "
            "anyway; the break is never reached")
        assert engine.calls == 1, (
            f"the break did not stop the loop: {engine.calls} chunks ran")

    def test_a_mild_stall_widens_the_grammar_instead_of_stopping(self):
        """Rung 2 of the escalation ladder, with the shipped controller.

        A stall just past the limit gets exactly one chance to be answered by
        a wider search space. Stopping here would be wrong in the other
        direction: it is the behaviour that made the stop branch untestable
        while also leaving a genuinely stalled search running.
        """
        engine = _Stub(stagnation=3)
        result = run_adaptive_discovery(engine, *_data(), generations=6,
                                        pop_size=4, chunk_size=2,
                                        controller=LocalSearchController(
                                            stagnation_limit=2))
        assert result.stopped_reason is None
        assert engine.calls == 3, engine.calls
        assert {a["action"] for a in result.actions} == {
            SearchAction.EXPAND_GRAMMAR.value}

    def test_a_healthy_search_keeps_going_and_reports_no_reason(self):
        engine = _Stub()
        result = run_adaptive_discovery(engine, *_data(), generations=6,
                                        pop_size=4, chunk_size=2)
        assert result.stopped_reason is None
        assert engine.calls == 3, engine.calls

    def test_empty_history_does_not_crash_the_legacy_path(self):
        """``if chunk_history:`` -- the guard for an engine returning nothing."""
        engine = _Stub(chunk_history=False)
        result = run_adaptive_discovery(engine, *_data(), generations=4,
                                        pop_size=4, chunk_size=2)
        assert result.best_candidate is not None
        assert result.actions == [], (
            "no chunk history means nothing to hand the controller, so no "
            "action should have been recorded")

    def test_the_best_score_is_the_minimum_across_chunks(self):
        """``if score < best_score:`` -- the incumbent-tracking branch."""
        engine = _Stub()
        result = run_adaptive_discovery(engine, *_data(), generations=6,
                                        pop_size=4, chunk_size=2)
        # The stub's score falls as 1/calls, so the last chunk is the best.
        assert result.best_score == pytest.approx(1.0 / 3)


class TestTheMultiVariablePathIsDeclaredUntested:
    """State the gap rather than let a coverage number imply otherwise."""

    def test_the_multi_variable_path_is_not_covered_by_the_legacy_tests(self):
        """The existing FakeSearch test cannot reach it, by construction.

        ``run_adaptive_discovery`` dispatches on ``isinstance(engine,
        MultiVariableSRE)``, so a stub that is not one always takes the legacy
        branch. That is why the module reads at 17% branch coverage while its
        legacy path is genuinely tested.
        """
        from apedp.discovery import MultiVariableSRE

        assert not isinstance(_Stub(), MultiVariableSRE)
        engine = _Stub()
        result = run_adaptive_discovery(engine, *_data(), generations=2, pop_size=2)
        assert result.stateful is False, (
            "the stub took the stateful path; the dispatch assumption this "
            "module's coverage gap rests on has changed")

    def test_a_permanently_stalled_search_is_stopped_by_the_shipped_controller(self):
        """The gap this file used to declare, now closed.

        ``LocalSearchController`` previously could only return EXPAND_GRAMMAR,
        RUN_VALIDATION, INCREASE_EXPLORATION, or CONTINUE, which left the stop
        branch in ``_run_legacy_chunked`` as dead code under the default
        controller: a search that had plainly stagnated went on spending
        generations and nothing ever populated ``stopped_reason``. The
        controller now escalates to STOP_STAGNATION, and the default is what
        this drives -- no stand-in.
        """
        engine = _Stub(stagnation=9)
        result = run_adaptive_discovery(engine, *_data(), generations=20,
                                        pop_size=4, chunk_size=2,
                                        controller=LocalSearchController(
                                            stagnation_limit=2))
        assert result.actions[-1]["action"] == SearchAction.STOP_STAGNATION.value
        assert result.stopped_reason, (
            "the controller stopped the search but did not say why")
        assert engine.calls == 1, (
            f"the break did not stop the loop: {engine.calls} chunks ran")

    @pytest.mark.parametrize("score", [float("inf"), float("nan")])
    def test_an_unusable_incumbent_score_is_handed_to_a_human(self, score):
        """A missing or NaN score must not read as a healthy search.

        ``nan > 1e-3`` is False, so before the finite-score guard existed a
        NaN fell past the residual-bar branch to a plain CONTINUE -- the one
        answer that means "nothing is wrong here".
        """
        decision = LocalSearchController().decide({
            "generations_without_improvement": 0,
            "pareto_frontier_size": 0,
            "generation": 3,
            "final_normalized_score": score,
        })
        assert decision.action is SearchAction.HUMAN_REVIEW, decision
        assert "finite" in decision.reason, decision.reason

    def test_every_rung_of_the_ladder_is_reachable(self):
        """The action set is a claim about the policy; check it is true.

        The old version of this test asserted the opposite -- that
        STOP_STAGNATION and HUMAN_REVIEW were *never* emitted -- which is the
        same shape of test pointed at the bug instead of the fix.

        INCREASE_EXPLOITATION is excluded deliberately: no rung of the local
        ladder produces it, and until one does it is a client-facing value
        that only ``JevCompatibleBackend`` can emit. Listing it here as
        expected would be the kind of assertion that passes by accident.
        """
        emitted = set()
        for limit in (2, 5):
            for stagnation in (0, 2, 3, 9, 100):
                for frontier in (0, 1, 25, 200):
                    for generation in (0, 1, 5):
                        for score in (1.0, 1e-4, float("inf")):
                            emitted.add(LocalSearchController(
                                stagnation_limit=limit).decide({
                                    "generations_without_improvement": stagnation,
                                    "pareto_frontier_size": frontier,
                                    "generation": generation,
                                    "final_normalized_score": score,
                                }).action)
        assert emitted == set(SearchAction) - {
            SearchAction.INCREASE_EXPLOITATION}, (
            "emitted", sorted(a.name for a in emitted), "unreached",
            sorted(a.name for a in set(SearchAction) - emitted))

    def test_the_untested_guards_are_named_where_they_can_be_found(self):
        """A guard that silently gains coverage is good; one that silently
        gains a *change* without anyone knowing the path ran is how a real
        regression would land. The list here is the one to delete entries from
        as paths get covered."""
        import ast
        import inspect

        import apedp.adaptive_search as mod

        tree = ast.parse(inspect.getsource(mod))
        decision_lines = {
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, (ast.If, ast.While, ast.IfExp))
        }
        # The guards this module cannot reach without a real engine, by source
        # line. If a refactor moves them, this fails rather than going stale.
        source_lines = inspect.getsource(mod).splitlines()
        expected = {
            "if seed is None:",                              # _seeded_global_rng
            "if improved:",                                   # stagnation counter
            "if generations < 1",                             # validation
            "if decision.parameters.get(flag) is True:",      # mutation flags
            'if state["best_score"] < 1e-25:',                # convergence bail
            "if str(candidate) not in elite_text:",           # frontier dedup
        }
        for guard in expected:
            linenos = [i + 1 for i, line in enumerate(source_lines)
                       if guard in line]
            assert linenos, f"guard vanished from the source: {guard!r}"
            assert decision_lines.intersection(linenos), (
                f"{guard!r} is no longer a decision point at {linenos}; the "
                "AST walk or the shape of the guard changed")