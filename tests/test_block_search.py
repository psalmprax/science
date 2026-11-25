"""The block search of one input: the right law of one or two blocks is among what it returns, and it decides nothing."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import sympy as sp

from apedp_dyn.grammar.proposer_interface import ProposerBase
from apedp_dyn.proposers import block_search as bs
from apedp_dyn.proposers.composed import formula_proposal

X = np.random.default_rng(0).uniform(1.0, 5.0, 400)


def _exact(law: str, truth: str, x: np.ndarray = X) -> bool:
    """The law reproduces the truth to rounding once its own constants are as returned."""
    s = sp.Symbol("x")
    got = np.asarray(sp.lambdify(s, sp.sympify(law), "numpy")(x), dtype=float)
    want = np.asarray(sp.lambdify(s, sp.sympify(truth), "numpy")(x), dtype=float)
    return bool(np.sqrt(np.mean((got - want) ** 2)) <= 1e-8 * (np.std(want) + 1e-12))


@pytest.mark.parametrize(("truth", "blocks"), [
    ("2.9*exp(-0.3*x)", 1),
    ("0.45*sqrt(x + 2.9)", 1),
    ("0.7*sin(0.45*x) - 0.45/x", 2),
    ("x*exp(-0.7*x)", 2),
    ("sqrt(x + 0.45)/(log(3.7*x + 1) + 0.45)", 2),
])
def test_the_law_is_returned_with_the_blocks_it_has(truth: str, blocks: int) -> None:
    y = np.asarray(sp.lambdify(sp.Symbol("x"), sp.sympify(truth), "numpy")(X), dtype=float)
    laws = bs.search(X, y, "x")
    assert laws == sorted(laws, key=lambda item: item[0])          # best fit first
    assert any(n == blocks and _exact(law, truth) for _, law, n in laws)


def test_a_law_is_also_returned_without_its_additive_constant() -> None:
    y = 2.9 * np.exp(-0.3 * X)
    laws = [law for _, law, n in bs.search(X, y, "x") if n == 1]
    bare = [law for law in laws if not sp.sympify(law).is_Add]
    assert any(_exact(law, "2.9*exp(-0.3*x)") for law in bare)


def test_degenerate_blocks_are_not_returned() -> None:
    near = ("leaf", next(i for i, b in enumerate(bs.BLOCKS) if b.name == "cos"), 0.001)   # cos(0.001 x): a parabola
    assert bs.degenerate(near, 0.0, 1.0, X, X**2)
    honest = ("leaf", next(i for i, b in enumerate(bs.BLOCKS) if b.name == "cos"), 0.7)
    assert not bs.degenerate(honest, 0.0, 1.0, X, np.cos(0.7 * X))
    assert bs.degenerate(honest, 1.0e6, 1.0, X, np.cos(0.7 * X))      # a huge constant cancelling


def test_every_returned_law_parses_and_is_finite() -> None:
    y = np.tanh(0.3 * X) * np.log1p(0.45 * X)
    for _, law, _ in bs.search(X, y, "x"):
        values = np.asarray(sp.lambdify(sp.Symbol("x"), sp.sympify(law), "numpy")(X), dtype=float)
        assert np.all(np.isfinite(values))


def test_the_proposer_searches_one_input_only_and_ranks_nothing_as_accepted() -> None:
    frame = pd.DataFrame({"x0": X, "target": 1.3 / X})
    proposals = bs.BlockSearchProposer(seconds=10).fit_and_propose(frame, target_col="target")
    assert proposals and all(p.proposer_id == "block_search" for p in proposals)
    assert not hasattr(proposals[0], "accepted")
    wide = pd.DataFrame({"x0": X, "x1": X[::-1], "target": X})
    assert bs.BlockSearchProposer().fit_and_propose(wide, target_col="target") == ()


class _Fixed(ProposerBase):
    def __init__(self, seconds: float, seen: list[float]) -> None:
        seen.append(seconds)

    @property
    def proposer_id(self) -> str:
        return "fixed"

    def fit_and_propose(self, train_data, target_col=None, **kwargs):
        return [formula_proposal(self.proposer_id, "x0**2", ["x0"])]


def test_blocks_first_gives_the_other_search_only_the_seconds_left_and_puts_the_best_fit_first() -> None:
    frame = pd.DataFrame({"x0": X, "target": 1.3 / X})
    seen: list[float] = []
    proposer = bs.BlocksFirst(60.0, lambda s: _Fixed(s, seen))
    merged = proposer.fit_and_propose(frame, target_col="target")
    assert seen and seen[0] <= 60.0 - proposer.last_run["block_seconds"] + 0.01
    assert any(p.formula == "x0**2" for p in merged)
    errors = [float(np.mean((p.compute_score(frame) - frame["target"]) ** 2)) for p in merged]
    assert errors[0] == min(errors)
