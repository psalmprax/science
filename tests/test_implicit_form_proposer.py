"""The proposer of polynomial relations between the inputs and the target (experimental, not in force)."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import pytest
import sympy as sp

from apedp_dyn.proposers import implicit_form as im
from apedp_dyn.proposers.fit_guard import FitGuardProposer

ROWS = 300


def _table(law: str, ranges: dict[str, tuple[float, float]], noise: float, seed: int = 0, rows: int = ROWS,
           widen: float = 1.0) -> tuple[pd.DataFrame, np.ndarray]:
    """A table of the law with noise as a share of its spread, and the law without noise on the same rows."""
    rng = np.random.default_rng([seed, rows, round(noise * 1e4)])
    columns = {name: rng.uniform(lo, lo + (hi - lo) * widen, rows) for name, (lo, hi) in ranges.items()}
    clean = np.asarray(sp.lambdify(list(columns), sp.sympify(law), "numpy")(*columns.values()), dtype=float)
    return pd.DataFrame({**columns, "__target__": clean + noise * float(np.std(clean)) * rng.standard_normal(rows)}), clean


def _r2(clean: np.ndarray, predicted: np.ndarray) -> float:
    return float(1.0 - np.sum((clean - predicted) ** 2) / np.sum((clean - clean.mean()) ** 2))


def _proposed(frame: pd.DataFrame, **options: Any) -> tuple[list[Any], im.ImplicitFormProposer]:
    proposer = im.ImplicitFormProposer(**options)
    return list(proposer.fit_and_propose(frame, target_col="__target__")), proposer


#: the law, its inputs' ranges, and its own relation as exponents of (inputs..., target)
LAWS: dict[str, tuple[str, dict[str, tuple[float, float]], set[tuple[int, ...]]]] = {
    "saturation": ("2.0*p/(1 + 0.8*p)", {"p": (0.1, 10.0)}, {(0, 1), (1, 1), (1, 0)}),
    "lens": ("u*v/(u + v)", {"u": (0.5, 5.0), "v": (0.5, 5.0)}, {(1, 0, 1), (0, 1, 1), (1, 1, 0)}),
    "energy": ("sqrt(p**2 + m**2)", {"p": (0.2, 3.0), "m": (0.2, 3.0)}, {(0, 0, 2), (2, 0, 0), (0, 2, 0)}),
    "three halves power": ("0.36*d**1.5", {"d": (0.4, 30.0)}, {(0, 2), (3, 0)}),
    "time dilation": ("3.0*(sqrt(1 - v**2) - 1)", {"v": (0.1, 0.9)}, {(0, 2), (0, 1), (2, 0)}),
    "line": ("0.5*D", {"D": (0.03, 2.0)}, {(0, 1), (1, 0)}),
}


@pytest.mark.parametrize("name", list(LAWS))
def test_on_rows_without_noise_a_law_s_own_relation_is_proposed_first_and_solved_for_the_target(name: str) -> None:
    law, ranges, own = LAWS[name]
    proposals, proposer = _proposed(_table(law, ranges, 0.0)[0])
    assert 1 <= len(proposals) <= im.PROPOSALS and proposer.last_run["ran"] and proposer.last_run["proposed"] == len(proposals)
    first = proposals[0]
    assert {tuple(e) for e in first.metadata["support"]} == own and first.metadata["rank"] == 1
    fresh, clean = _table(law, ranges, 0.0, seed=7)
    assert _r2(clean, first.compute_score(fresh)) >= 1 - 1e-9
    assert sp.sympify(first.formula).free_symbols <= {sp.Symbol(n) for n in ranges}     # the target is solved for
    values = [np.asarray(p.compute_score(fresh), dtype=float) for p in proposals]
    assert all(float(np.max(np.abs(a - b))) > 0 for i, a in enumerate(values) for b in values[:i])      # no formula twice


@pytest.mark.parametrize("name", list(LAWS))
@pytest.mark.parametrize("noise", [0.01, 0.05])
def test_with_noise_a_law_s_own_relation_is_among_the_first_three(name: str, noise: float) -> None:
    # one of each number of terms leads the list, so longer relations that fit a chance bend cannot push the law out
    law, ranges, own = LAWS[name]
    for seed in range(3):
        proposals, _ = _proposed(_table(law, ranges, noise, seed=seed)[0])
        place = next(i for i, p in enumerate(proposals) if {tuple(e) for e in p.metadata["support"]} == own)
        assert place < 3, (seed, [p.metadata["support"] for p in proposals[:3]])
        fresh, clean = _table(law, ranges, 0.0, seed=7)
        assert _r2(clean, proposals[place].compute_score(fresh)) >= 0.999


def test_a_perfect_square_does_not_lead_a_noisy_line() -> None:
    # (y - a*D)**2 = 0 halves the first-order distance and led every honest relation before the ranking was changed
    for noise in (0.01, 0.05, 0.2):
        proposals, _ = _proposed(_table("0.5*D", {"D": (0.03, 2.0)}, noise)[0])
        assert [[0, 1], [1, 0]] in [p.metadata["support"] for p in proposals[:3]]
        assert not any("sqrt(D**2)" in str(p.formula) for p in proposals)       # y**2 = c*D**2 is the line again
        assert [[0, 2], [1, 1], [2, 0]] not in [p.metadata["support"] for p in proposals]      # the square itself
    # at 1 % noise seventeen longer relations stand above the line in the one list (it is 18th there): it is proposed
    # because the best relation of each number of terms comes first
    proposals, _ = _proposed(_table("0.5*D", {"D": (0.03, 2.0)}, 0.01)[0])
    line = next(p for p in proposals if p.metadata["support"] == [[0, 1], [1, 0]])
    assert line.metadata["rank"] > im.PROPOSALS and proposals.index(line) < 3


def test_a_formula_with_a_pole_among_the_rows_is_not_proposed() -> None:
    # the law multiplied through by (1 - 0.8*p) fitted these rows better than the law; its pole sits at p = 1.25
    frame, _ = _table("2.0*p/(1 + 0.8*p)", {"p": (0.1, 10.0)}, 0.01)
    p = sp.Symbol("p")
    for proposal in _proposed(frame)[0]:
        below = sp.denom(sp.together(sp.sympify(proposal.formula)))
        values = np.asarray(sp.lambdify([p], below, "numpy")(frame["p"].to_numpy()), dtype=complex) * np.ones(len(frame))
        assert np.all(np.abs(values.imag) == 0) and float(values.real.min()) * float(values.real.max()) > 0, proposal.formula
    assert im._solved([(0, 1), (1, 0), (1, 1)], np.array([1.0, -1.0, -0.5]), np.column_stack([np.ones(3), [1.0, 2.5, 4.0], [1.0, 2.5, 4.0]])) is None


def test_the_units_of_the_columns_do_not_change_what_is_proposed() -> None:
    frame, _ = _table("2.0*p/(1 + 0.8*p)", {"p": (0.1, 10.0)}, 0.01)
    scaled = pd.DataFrame({"p": frame["p"] * 1e6, "__target__": frame["__target__"] * 1e-9})
    plain, other = _proposed(frame)[0], _proposed(scaled)[0]
    assert [p.metadata["support"] for p in plain[:5]] == [p.metadata["support"] for p in other[:5]]
    np.testing.assert_allclose(other[0].compute_score(scaled), 1e-9 * plain[0].compute_score(frame), rtol=1e-6)


def test_it_proposes_candidates_and_decides_nothing() -> None:
    proposals, proposer = _proposed(_table("u*v/(u + v)", {"u": (0.5, 5.0), "v": (0.5, 5.0)}, 0.01)[0])
    assert proposer.proposer_id == "implicit_form" and len(proposals) == im.PROPOSALS
    assert {type(p).__name__ for p in proposals} == {"CandidateProposal"} and len({p.candidate_id for p in proposals}) == im.PROPOSALS
    assert not any(hasattr(p, name) for p in proposals for name in ("accepted", "verdict", "passed", "reported"))
    assert not hasattr(proposer, "decide") and not hasattr(proposer, "accept")
    guarded = FitGuardProposer(im.ImplicitFormProposer(), target_col="__target__")     # as a search stage is given to the route
    frame = _table("u*v/(u + v)", {"u": (0.5, 5.0), "v": (0.5, 5.0)}, 0.01)[0]
    assert 1 <= len(guarded.fit_and_propose(frame, target_col="__target__", guesses=["u + v"])) <= im.PROPOSALS


@pytest.mark.parametrize("law, ranges", [("2.0*exp(-0.7*x)", {"x": (0.2, 4.0)}), ("sin(1.3*x) + 2.0", {"x": (0.2, 4.0)}),
                                         ("-7.0*log(x) - 2.0", {"x": (1.0, 6.0)})])
def test_a_law_it_cannot_write_still_gets_approximations_and_they_fail_outside_the_rows(law: str, ranges: dict[str, Any]) -> None:
    # this is the trap: the proposer cannot say "none of these"; the referee has to
    proposals, _ = _proposed(_table(law, ranges, 0.01)[0])
    assert proposals
    inside, clean_in = _table(law, ranges, 0.0, seed=5)
    outside, clean_out = _table(law, ranges, 0.0, seed=6, widen=2.0)
    first = proposals[0]
    with np.errstate(all="ignore"):
        assert _r2(clean_in, first.compute_score(inside)) >= 0.99
        beyond = np.asarray(first.compute_score(outside), dtype=float)
    assert not np.all(np.isfinite(beyond)) or _r2(clean_out, beyond) < 0.999


def test_the_setting_is_lowered_with_the_number_of_inputs_to_keep_to_the_budget() -> None:
    assert [im.setting(d, 700) for d in (1, 2, 3, 4, 6, 9)] == [(3, 4), (3, 4), (3, 3), (2, 4), (2, 3), (2, 2)]
    for inputs in range(1, 10):
        degree, terms = im.setting(inputs, 700) or (0, 0)
        assert sum(math.comb(len(im.monomials(inputs, degree)), k) for k in range(2, terms + 1)) <= im.MAX_SUPPORTS
    assert im.setting(1, 5) == (3, 3) and im.setting(1, 3) is None                # a set of k terms needs more than k + 1 rows
    assert im.setting(40, 700) is None                                            # too many inputs for any setting
    assert len(im.monomials(1, 3)) == 9 and (0, 3) not in im.monomials(1, 3) and im.monomials(1, 3)[-1] == (0, 0)
    assert all(e[-1] <= im.DEGREE_IN_Y for e in im.monomials(3, 3))


def test_sets_that_say_nothing_new_are_not_fitted() -> None:
    assert im._usable([(0, 1), (1, 0)]) and im._usable([(0, 2), (3, 0)])
    assert not im._usable([(1, 0), (2, 0)])                 # no target
    assert not im._usable([(0, 1), (0, 2)])                 # no input
    assert not im._usable([(1, 1), (2, 0)])                 # x*(y - c*x): the set without the factor is fitted on its own
    assert not im._usable([(0, 2), (2, 0)])                 # y**2 = c*x**2


def test_tables_it_cannot_read_give_no_proposals_and_say_why() -> None:
    proposer = im.ImplicitFormProposer()
    assert proposer.fit_and_propose(pd.DataFrame({"__target__": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}), target_col="__target__") == []
    assert proposer.fit_and_propose(pd.DataFrame({"x": [1.0, 2.0, 3.0], "__target__": [1.0, 2.0, 3.0]}), target_col="__target__") == []
    assert "too few rows" in proposer.last_run["why_not"] and proposer.last_run["ran"] is False
    assert proposer.fit_and_propose(pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]}), target_col="__target__") == []
    frame, _ = _table("2.0*p/(1 + 0.8*p)", {"p": (0.1, 10.0)}, 0.0)
    frame.loc[3, "p"], frame.loc[8, "__target__"] = np.nan, np.inf              # rows that are not finite are left out
    frame["zero"] = 0.0                                                         # a column of zeros has no monomial to fit
    proposals = proposer.fit_and_propose(frame, target_col="__target__")
    assert {tuple(e) for e in proposals[0].metadata["support"]} == {(0, 0, 1), (1, 0, 1), (1, 0, 0)}


def test_a_large_table_is_fitted_on_the_same_sample_every_time() -> None:
    frame, _ = _table("u*v/(u + v)", {"u": (0.5, 5.0), "v": (0.5, 5.0)}, 0.01, rows=1200)
    first, proposer = _proposed(frame)
    again, _ = _proposed(frame)
    assert proposer.last_run["rows_fitted"] == im.SEARCH_ROWS and [p.formula for p in first] == [p.formula for p in again]
    assert proposer.last_run["degree"] == 3 and proposer.last_run["max_terms"] == 4
    assert 0 < proposer.last_run["relations_fitted"] <= im.MAX_SUPPORTS
    assert len(first) <= proposer.last_run["written_out"] <= im.ATTEMPTS
    # the best of each number of terms comes first, whatever its place in the one list
    assert sorted(p.metadata["terms"] for p in first[:3]) == [2, 3, 4] and first[0].metadata["rank"] == 1
