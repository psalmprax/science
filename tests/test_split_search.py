"""A table searched by its parts: the parts' laws are put together in the shape read, refitted and proposed; nothing more."""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import sympy as sp

from apedp_dyn.grammar.proposer_interface import ProposerBase
from apedp_dyn.proposers import split_search, table_shape
from apedp_dyn.proposers.composed import formula_proposal

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ("{x}", "{x}**2", "sqrt({x} + 1)", "log(1 + {x})", "sin({x})")


class _Templates(ProposerBase):
    """A search of one input that tries a few forms with a scale and an offset, best fit first."""

    def __init__(self, seconds: float, seen: list[float]) -> None:
        seen.append(seconds)

    @property
    def proposer_id(self) -> str:
        return "templates"

    def fit_and_propose(self, train_data, target_col=None, **kwargs):
        names = [c for c in train_data.columns if c != target_col]
        assert len(names) == 1, "these tests split down to one input"
        x, y = train_data[names[0]].to_numpy(dtype=float), train_data[target_col].to_numpy(dtype=float)
        scored = []
        for template in TEMPLATES:
            text = template.format(x=names[0])
            value = sp.lambdify(sp.Symbol(names[0]), sp.sympify(text), "numpy")(x)
            (a, b), *_ = np.linalg.lstsq(np.column_stack([value, np.ones(len(x))]), y, rcond=None)
            scored.append((float(np.mean((a * value + b - y) ** 2)), f"{float(a)!r}*{text} + {float(b)!r}"))
        return [formula_proposal(self.proposer_id, text, names) for _, text in sorted(scored)]


class _Fails(ProposerBase):
    def __init__(self, seconds: float) -> None:
        pass

    @property
    def proposer_id(self) -> str:
        return "fails"

    def fit_and_propose(self, train_data, target_col=None, **kwargs):
        raise RuntimeError("no search today")


def _frame(law, inputs: int, rows: int = 400, noise: float = 0.0, seed: int = 5):
    rng = np.random.default_rng(seed)
    X = rng.uniform(1.0, 5.0, (rows, inputs))
    clean = law(*X.T)
    frame = pd.DataFrame(X, columns=[f"x{i}" for i in range(inputs)])
    frame["target"] = clean + noise * float(np.sqrt(np.mean(clean**2))) * rng.standard_normal(rows)
    return frame, clean


def _r2(proposal, frame, clean) -> float:
    return 1.0 - float(np.sum((proposal.compute_score(frame) - clean) ** 2)) / float(np.sum((clean - clean.mean()) ** 2))


def test_the_parts_laws_are_put_together_and_the_whole_is_the_law():
    seen: list[float] = []
    frame, clean = _frame(lambda a, b, c: 2.0 * a**2 * np.log(1.0 + b) + 0.5 * np.sin(c), 3)
    proposer = split_search.SplitSearchProposer(lambda seconds: _Templates(seconds, seen), seconds=30.0)
    proposals = proposer.fit_and_propose(frame, target_col="target")
    assert 1 <= len(proposals) <= split_search.MAX_PROPOSALS and len({p.formula for p in proposals}) == len(proposals)
    assert all(p.proposer_id == "table_split" for p in proposals)
    assert _r2(proposals[0], frame, clean) > 1.0 - 1e-8
    expr = sp.sympify(proposals[0].formula)
    assert expr.free_symbols == {sp.Symbol("x0"), sp.Symbol("x1"), sp.Symbol("x2")}
    assert {str(f.func) for f in expr.atoms(sp.Function)} == {"log", "sin"}
    run = proposer.last_run
    assert run["parts"] == 3 and run["widest"] == 1 and run["proposed"] == len(proposals) and len(run["part_laws"]) == 3
    assert seen == [10.0, 10.0, 10.0]                       # the seconds are shared out among the parts


def test_noisy_rows_still_give_an_accurate_whole_because_its_constants_are_refitted_on_them():
    frame, clean = _frame(lambda a, b: (1.0 + a) * np.sqrt(b + 1.0), 2, noise=0.02)
    proposer = split_search.SplitSearchProposer(lambda seconds: _Templates(seconds, []))
    proposals = proposer.fit_and_propose(frame, target_col="target")
    assert proposals and _r2(proposals[0], frame, clean) > 0.999


def test_a_table_that_does_not_split_gets_no_proposal():
    frame, _ = _frame(lambda a, b: np.sin(0.7 * a * b), 2, noise=0.01)
    proposer = split_search.SplitSearchProposer(lambda seconds: _Templates(seconds, []))
    assert proposer.fit_and_propose(frame, target_col="target") == ()
    assert proposer.last_run["parts"] == 1 and proposer.last_run["proposed"] == 0


def test_a_part_whose_search_fails_is_stood_in_for_and_it_is_said():
    frame, clean = _frame(lambda a, b: (1.0 + a) * np.sqrt(b + 1.0), 2)
    proposer = split_search.SplitSearchProposer(_Fails)
    proposals = proposer.fit_and_propose(frame, target_col="target")
    assert len(proposals) == 1 and _r2(proposals[0], frame, clean) > 0.99
    assert all(note.get("stand_in") and "no search today" in note["error"] for note in proposer.last_run["part_laws"])


def test_the_stand_in_is_a_polynomial_in_each_input_of_the_part():
    rng = np.random.default_rng(0)
    X = rng.uniform(1.0, 5.0, (60, 2))
    text = split_search.stand_in(["u", "v"], X, 1.0 + 2.0 * X[:, 0] - 0.5 * X[:, 1] ** 2)
    value = sp.lambdify([sp.Symbol("u"), sp.Symbol("v")], sp.sympify(text), "numpy")(*X.T)
    assert np.allclose(value, 1.0 + 2.0 * X[:, 0] - 0.5 * X[:, 1] ** 2)


def test_it_reads_the_table_once_with_the_reader_it_is_given():
    calls = []

    def reader(X, y):
        calls.append(X.shape)
        return table_shape.read(X, y)

    frame, _ = _frame(lambda a, b: (1.0 + a) * np.sqrt(b + 1.0), 2)
    split_search.SplitSearchProposer(lambda seconds: _Templates(seconds, []), reader=reader).fit_and_propose(frame, target_col="target")
    assert calls == [(400, 2)]


def test_nothing_in_the_package_builds_it():
    users = subprocess.run(["git", "grep", "-l", "split_search", "--", "apedp_dyn"], cwd=ROOT, capture_output=True, text=True).stdout.split()
    assert users == ["apedp_dyn/proposers/split_search.py"]
