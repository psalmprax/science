"""The agentic variation operator: the bench, the session, the proposer, with a scripted stand-in for the model.

What these show is that the loop works as described: residual hints reach the model, its actions
are executed and nothing else is, budgets end a session, and the search loses nothing when the
model is absent or useless. Whether a real model recovers more laws this way is not tested here.
"""
from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import pytest
import sympy as sp

from apedp_dyn.grammar.proposer_interface import CandidateProposal, ProposerBase, StrategyArchetype
from apedp_dyn.proposers import agentic_variation as av


def _rows(noise: float = 0.0, n: int = 300, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """y = 2 x0 + 3 x0 x1 on three inputs, the third of no consequence."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.5, 3.0, (n, 3))
    y = 2.0 * X[:, 0] + 3.0 * X[:, 0] * X[:, 1] + noise * rng.standard_normal(n)
    return X, y


def _frame(noise: float = 0.0) -> pd.DataFrame:
    X, y = _rows(noise)
    return pd.DataFrame({"mass": X[:, 0], "speed": X[:, 1], "colour": X[:, 2], "target": y})


class ReadsHints:
    """A stand-in for a model that follows the residual: it adds the term the bench says is missing."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if "No data is given" in prompt:
            return "I do not know this law."
        if "c0*x0 + c1*x0*x1 | error" in prompt and "exact fit" in prompt:
            return "THINK: nothing is left in the residual\nSUBMIT: c0*x0 + c1*x0*x1"
        if "residual follows x0*x1" in prompt:
            return "THINK: the residual follows the product\nFIT: c0*x0 + c1*x0*x1"
        return "THINK: start with the simplest\nFIT: c0*x0 + c1"


class Stub(ProposerBase):
    """A search that returns fixed formulas and remembers what it was seeded with."""

    def __init__(self, formulas: Sequence[str]) -> None:
        self.formulas = list(formulas)
        self.seeded: list[list[str]] = []

    @property
    def proposer_id(self) -> str:
        return "stub_search"

    def fit_and_propose(self, train_data: Any, target_col: str | None = None, guesses: Sequence[str] | None = None,
                        **kwargs: Any) -> Sequence[CandidateProposal]:
        self.seeded.append(list(guesses or []))
        return tuple(CandidateProposal(self.proposer_id, f"stub:{i}", formula, StrategyArchetype.TACTICAL, False)
                     for i, formula in enumerate(self.formulas))


# --- the bench -------------------------------------------------------------------------------------

def test_a_wrong_form_is_told_what_its_residual_still_follows():
    bench = av.Workbench(*_rows())
    trial = bench.fit("c0*x0 + c1")
    assert isinstance(trial, av.Trial) and trial.error > 0.1 and trial.source == "agent"
    # data minus fit: positive where the fit is too low, so it rises with the input whose term is missing
    assert trial.hints[0][0] == "x1" and trial.hints[0][1] > 0.9
    assert dict(trial.hints)["x0*x1"] > 0.5
    assert "largest error at x0=" in trial.worst and "residual follows x0*x1" in trial.line()


def test_the_right_form_fits_exactly_and_leaves_no_hint():
    bench = av.Workbench(*_rows())
    trial = bench.fit("c0*x0 + c1*x0*x1")
    assert isinstance(trial, av.Trial) and trial.error < av.EXACT and trial.hints == [] and "exact fit" in trial.line()
    numbers = sorted(float(v) for v in trial.fitted.atoms(sp.Float))
    assert numbers == pytest.approx([2.0, 3.0])


@pytest.mark.parametrize("text", ["__import__('os').system('true')", "x0 + open('f')", "x9 + 1", "lambda: 1", "x0; x1", ""])
def test_text_the_whitelist_does_not_accept_is_never_evaluated(text):
    bench = av.Workbench(*_rows())
    result = bench.fit(text)
    assert isinstance(result, str) and result.startswith("not usable") and bench.rejected == 1 and not bench.trials


def test_an_expression_without_an_input_is_not_a_law():
    bench = av.Workbench(*_rows())
    assert bench.fit("c0 + c1") == "not usable: the expression uses no input" and bench.rejected == 1


def test_an_expression_that_carries_its_numbers_is_recorded_as_it_is():
    bench = av.Workbench(*_rows())
    x0, x1 = sp.symbols("x0 x1")
    trial = bench.admit(2.0 * x0 + 3.0 * x0 * x1, "search")
    assert trial is not None and trial.error < av.EXACT and trial.source == "search"
    assert set(sp.sympify(trial.skeleton).free_symbols) == {x0, x1, sp.Symbol("c0"), sp.Symbol("c1")}
    assert bench.admit(sp.Float(3.0), "search") is None and bench.admit(sp.Symbol("z") + x0, "search") is None


def test_the_better_fit_of_a_structure_is_the_one_kept():
    bench = av.Workbench(*_rows())
    x0, x1 = sp.symbols("x0 x1")
    poor = bench.admit(1.0 * x0 + 1.0 * x0 * x1, "search")
    assert poor is not None and poor.error > 0.1
    better = bench.fit("c0*x0 + c1*x0*x1")
    assert isinstance(better, av.Trial) and better.error < av.EXACT and len(bench.trials) == 1
    again = bench.admit(1.0 * x0 + 1.0 * x0 * x1, "search")
    assert again is not None and again.error < av.EXACT          # the worse fit does not replace the better


def test_the_probe_features_drop_what_only_repeats_another():
    X = np.column_stack([np.linspace(1, 2, 50), np.linspace(1, 2, 50) ** 3, np.random.default_rng(0).normal(size=50)])
    names = list(av.probe_features(X))
    assert "x0" in names and "x1" not in names and "x2" in names      # x1 is a monotone function of x0
    assert "U-shape in x0" in names and "x0*x2" in names


def test_a_u_shaped_residual_is_named_as_one():
    rng = np.random.default_rng(1)
    X = rng.uniform(-2.0, 2.0, (400, 2))
    hints = av.residual_hints(av.probe_features(X), X[:, 0] ** 2 - np.mean(X[:, 0] ** 2))
    assert hints[0][0] == "U-shape in x0" and hints[0][1] > 0.8
    assert av.residual_hints(av.probe_features(X), rng.normal(size=400)) == []
    assert av.residual_hints(av.probe_features(X), np.full(400, np.nan)) == []


def test_an_expression_has_to_earn_its_cost():
    X, y = _rows(noise=0.3)
    bench = av.Workbench(X, y)
    for text in ("c0*x0 + c1*x0*x1", "c0*x0 + c1*x0*x1 + c2*x2 + c3*x2**2 + c4*sin(x2)", "c0*x0", "c0*x1 + c1"):
        assert isinstance(bench.fit(text), av.Trial)
    ranked, earned, shortlist = bench.ranked(), bench.earned(), bench.shortlist()
    assert ranked[0].skeleton.count("x2") >= 3 and ranked[0].cost > ranked[1].cost   # the best fit on noisy rows is the elaborate one
    assert ranked[0].error > (1.0 - av.GAIN) * ranked[1].error                       # ... by less than an expression has to gain
    assert [t.skeleton for t in earned] == ["c0*x0 + c1*x0*x1", "c0*x0"]             # the law leads; the cheaper, worse one follows
    assert [t.skeleton for t in shortlist[:2]] == [t.skeleton for t in earned] and len(shortlist) == len(ranked)
    assert "fits no better than a cheaper one" in bench.lineage().splitlines()[2]
    assert "fits no better than a cheaper one" not in bench.lineage().splitlines()[0]


def test_an_exact_fit_always_earns_its_cost_and_nothing_dearer_does_after_it():
    bench = av.Workbench(*_rows())
    for text in ("c0*x0", "c0*x0 + c1*x0*x1", "c0*x0 + c1*x0*x1 + c2*sin(x2)"):
        assert isinstance(bench.fit(text), av.Trial)
    assert [t.skeleton for t in bench.earned()] == ["c0*x0 + c1*x0*x1", "c0*x0"]


def test_notes_are_found_by_their_key_words():
    assert "x/(c + x)" in av.knowledge("saturation of the rate")
    assert "c*xi*xj" in av.knowledge("interaction")
    assert av.knowledge("quantum gravity").startswith("No note on that. Topics: residual, interaction")
    assert av.complexity(sp.sympify("c0*x0 + c1*x0*x1")) == av.complexity(sp.sympify("2.0*x0 + 3.0*x0*x1"))


# --- reading a reply -----------------------------------------------------------------------------------

def test_only_action_lines_are_read_from_a_reply():
    reply = ("Sure! Here is my plan.\n"
             "THINK: the residual grows with x1\n"
             "- fit: `c0*x0 + c1*x1`\n"
             "import os; os.remove('x')\n"
             "Know = saturation\n"
             "SUBMIT: y = c0*x0 ; c0*x1.\n")
    assert av.read_actions(reply) == [("THINK", "the residual grows with x1"), ("FIT", "c0*x0 + c1*x1"),
                                      ("KNOW", "saturation"), ("SUBMIT", "y = c0*x0 ; c0*x1.")]
    assert av._expressions("y = c0*x0 ; c0*x1.") == ["c0*x0", "c0*x1"]
    assert av.read_actions("I think the answer is x0 + x1.") == []


# --- a session -----------------------------------------------------------------------------------------

def test_a_model_that_reads_the_hints_repairs_its_first_guess_and_submits():
    bench, model = av.Workbench(*_rows()), ReadsHints()
    session = av.run_session(model, bench, steps=6)
    assert session.stopped == "submitted" and session.steps == 3 and session.fits == 2
    best = bench.ranked()[0]
    assert best.skeleton == "c0*x0 + c1*x0*x1" and best.error < av.EXACT and best.submitted
    assert session.submitted == [best.key]
    assert "residual follows x0*x1" in model.prompts[1]            # the hint was in front of the model ...
    assert "exact fit" in model.prompts[2]                         # ... and so was the result of its repair
    assert "Steps left: 5" in model.prompts[1] and len(session.transcript) == 3


def test_a_session_ends_when_its_budget_is_spent():
    tried = iter(f"c0*x0**{k} + c1" for k in range(2, 40))
    bench = av.Workbench(*_rows())
    session = av.run_session(lambda prompt: f"FIT: {next(tried)}", bench, steps=4, max_fits=18)
    assert session.stopped == "budget spent" and session.steps == 4 and session.fits == 4
    bench = av.Workbench(*_rows())
    many = "\n".join(f"FIT: c0*x0**{k} + c1" for k in range(2, 12))
    session = av.run_session(lambda prompt: many, bench, steps=9, max_fits=5)
    assert session.fits == 5 and len(bench.trials) == 3            # three per reply, and no more than the budget


def test_a_model_that_only_talks_ends_its_own_session():
    bench = av.Workbench(*_rows())
    session = av.run_session(lambda prompt: "I believe the law is linear in x0.", bench, steps=6)
    assert session.stopped == "two replies without an action" and session.steps == 2 and session.unusable_replies == 2
    assert not bench.trials


def test_a_model_that_cannot_be_reached_ends_the_session_and_nothing_else():
    def unreachable(prompt: str) -> str:
        raise ConnectionError("no route")

    bench = av.Workbench(*_rows())
    session = av.run_session(unreachable, bench)
    assert session.stopped.startswith("model unavailable: ConnectionError") and session.steps == 0 and not bench.trials


def test_a_session_does_not_start_a_step_after_its_deadline():
    bench = av.Workbench(*_rows())
    session = av.run_session(ReadsHints(), bench, deadline=time.monotonic() - 1.0)
    assert session.stopped == "out of time" and session.steps == 0


def test_a_note_is_returned_to_the_model_that_asks_for_it():
    replies = iter(["KNOW: saturation", "SUBMIT: c0*x0/(c1 + x0)"])
    seen: list[str] = []

    def model(prompt: str) -> str:
        seen.append(prompt)
        return next(replies)

    session = av.run_session(model, av.Workbench(*_rows()), steps=4)
    assert session.notes_read == 1 and "KNOW saturation -> " in seen[1] and "x/(c + x)" in seen[1]
    assert session.stopped == "submitted"


# --- the proposer --------------------------------------------------------------------------------------

def test_the_agent_alone_proposes_the_law_in_the_frames_own_names():
    proposer = av.AgenticVariationProposer(ReadsHints(), rounds=1)
    proposals = proposer.fit_and_propose(_frame(), target_col="target")
    law = sp.sympify(proposals[0].formula)
    assert {str(s) for s in law.free_symbols} == {"mass", "speed"}
    assert sp.simplify(law - sp.sympify("2.0*mass + 3.0*mass*speed")).xreplace(
        {n: sp.Float(round(float(n), 6)) for n in law.atoms(sp.Float)}) == 0
    assert proposals[0].proposer_id == "agentic_variation" and proposals[0].metadata["found_by"] == "agent"
    assert proposals[0].metadata["submitted_by_agent"] and not proposals[0].metadata["recited_before_data"]
    assert proposals[0].metadata["selection_pool_size"] == 2 == proposer.last_run["structures_tried"]
    assert proposals[0].parameters["dev_relative_rmse"] < av.EXACT
    frame = _frame()
    assert np.allclose(proposals[0].compute_score(frame), frame["target"])
    assert proposer.last_run["sessions"][0]["stopped"] == "submitted" and len(proposer.last_run["transcript_sha256"]) == 64


def test_the_agent_repairs_what_the_search_found_and_the_search_is_seeded_with_the_repair():
    stub = Stub(["1.9*mass + 4.0", "mass*speed"])
    proposer = av.AgenticVariationProposer(ReadsHints(), lambda budget: stub, rounds=2, budget_seconds=10.0)
    proposals = proposer.fit_and_propose(_frame(), target_col="target", guesses=["mass"])
    assert stub.seeded[0] == ["mass"]                                          # round one: the path's own guess
    assert any("mass*speed" in seed and "+" in seed for seed in stub.seeded[1])  # round two: the agent's repair
    assert proposals[0].metadata["found_by"] == "agent" and proposals[0].parameters["dev_relative_rmse"] < av.EXACT
    sources = {p.metadata["found_by"] for p in proposals}
    assert sources == {"agent", "search", "seed"}
    assert proposer.last_run["search_runs"] == [
        {"round": 0, "proposals": 2, "seeded_with": 1, "proposer_id": "stub_search"},
        {"round": 1, "proposals": 2, "seeded_with": 3, "proposer_id": "stub_search"}]
    assert all(p.metadata["selection_pool_size"] == len(proposals) for p in proposals)
    # every structure handed on is recorded with who first wrote it, so a recovery can be traced
    assert proposer.last_run["returned"] == [{"formula": p.formula, "source": p.metadata["found_by"]} for p in proposals]


def test_without_a_usable_model_the_searchs_expressions_are_still_returned():
    def unreachable(prompt: str) -> str:
        raise TimeoutError("slow")

    for client in (None, unreachable, lambda prompt: "Let me think about this."):
        stub = Stub(["2.0*mass + 3.0*mass*speed", "mass*speed"])
        proposer = av.AgenticVariationProposer(client, lambda budget, stub=stub: stub, rounds=1)
        proposals = proposer.fit_and_propose(_frame(), target_col="target")
        assert [p.metadata["found_by"] for p in proposals] == ["search", "search"]
        assert sp.simplify(sp.sympify(proposals[0].formula) - sp.sympify("2.0*mass + 3.0*mass*speed")) == 0
        assert proposer.last_run["by_agent"] == 0


def test_a_search_whose_formulas_cannot_be_read_is_returned_as_it_was():
    stub = Stub(["mass +* speed", "unknown_name*2"])
    proposals = av.AgenticVariationProposer(None, lambda budget: stub, rounds=1).fit_and_propose(_frame(), target_col="target")
    assert [p.proposer_id for p in proposals] == ["stub_search", "stub_search"]


def test_a_search_formula_that_cannot_be_evaluated_is_a_failed_trial_and_not_a_crash():
    stub = Stub(["mystery(mass) + speed", "2.0*mass + 3.0*mass*speed"])
    proposer = av.AgenticVariationProposer(None, lambda budget: stub, rounds=1)
    proposals = proposer.fit_and_propose(_frame(), target_col="target")
    assert len(proposals) == 1 and proposals[0].parameters["dev_relative_rmse"] < av.EXACT
    assert proposals[0].metadata["selection_pool_size"] == 2        # the one that failed was tried, and is counted


def test_a_law_the_model_states_before_seeing_data_is_marked_as_recited():
    class Recites(ReadsHints):
        def __call__(self, prompt: str) -> str:
            if "No data is given" in prompt:
                self.prompts.append(prompt)
                return "y = c0*x0 + c1*x0*x1"
            return super().__call__(prompt)

    proposals = av.AgenticVariationProposer(Recites(), rounds=1).fit_and_propose(_frame(), target_col="target")
    assert proposals[0].metadata["recited_before_data"] is True
    assert not any(p.metadata["recited_before_data"] for p in proposals[1:])
    unprobed = av.AgenticVariationProposer(Recites(), rounds=1, probe=False).fit_and_propose(_frame(), target_col="target")
    assert not unprobed[0].metadata["recited_before_data"]


def test_the_search_gets_its_share_of_the_time_in_every_round():
    budgets: list[float] = []

    def search(budget: float) -> ProposerBase | None:
        budgets.append(budget)
        return None if len(budgets) == 2 else Stub(["mass"])

    av.AgenticVariationProposer(None, search, rounds=3, budget_seconds=240.0).fit_and_propose(_frame(), target_col="target")
    assert budgets == [80.0, 80.0, 80.0]


# --- the stage -----------------------------------------------------------------------------------------

def test_the_agentic_stage_wraps_the_default_searches_and_keeps_the_fit_guard_outermost():
    from apedp_dyn.discovery_l2 import agentic_stage
    from apedp_dyn.proposers.fit_guard import FitGuardProposer

    stage, text = agentic_stage.agentic_search_stage(None, 120.0, cores=1, rounds=3)
    assert isinstance(stage, FitGuardProposer) and text == "PySR and an agent taking turns, 3 rounds, 120 s of search"
    _, text = agentic_stage.agentic_search_stage(None, 240.0, cores=8)
    assert text.startswith("four searches and an agent taking turns, 2 rounds")


def test_the_agentic_stage_runs_through_the_refined_path_with_a_scripted_model():
    from apedp_dyn.discovery_l2 import refined_search
    from apedp_dyn.proposers.fit_guard import FitGuardProposer

    X, y = _rows(noise=0.02)
    pool, _ = _rows(n=400, seed=7)
    rng = np.random.default_rng(8)

    def measure(rows: np.ndarray) -> np.ndarray:
        out: np.ndarray = 2.0 * pool[rows, 0] + 3.0 * pool[rows, 0] * pool[rows, 1] + 0.02 * rng.standard_normal(len(rows))
        return out

    stage = FitGuardProposer(av.AgenticVariationProposer(ReadsHints(), rounds=1, target_col="__target__"))
    result = refined_search.discover_refined({f"x{i}": X[:, i] for i in range(3)}, y, pool, measure, proposer=stage,
                                             noise="absolute", polynomial_library=False)
    assert result.law is not None and result.action in refined_search.REPORTED
    law = sp.sympify(result.law)
    assert {str(s) for s in law.free_symbols} == {"x0", "x1"}
    coefficients = sorted(float(c) for c in sp.Poly(law, *sp.symbols("x0 x1")).coeffs())
    assert coefficients == pytest.approx([2.0, 3.0], abs=0.05)
