"""The memory's proposals with the work counted in fits: the same on any machine, and the timed one untouched."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from apedp_dyn.proposers import structure_memory as timed
from apedp_dyn.proposers import structure_memory_counted as counted


def _memory() -> timed.Memory:
    memory = timed.Memory()
    for law in ("2.3*x0/(x0 + 1.7) - 0.7*exp(-0.45*x1)", "1.2*sqrt(x0) + 0.4*x1**2", "3.1*log(x0 + 2.0)*x1",
                "0.8*exp(-0.3*x0) + 1.5/(x1 + 0.6)"):
        memory.learn(law)
    return memory


def _table(rows: int = 20) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    X = rng.uniform(0.5, 3.0, (rows, 2))
    return X, 1.9 * X[:, 0] / (X[:, 0] + 0.8) + 0.6 * np.exp(-0.7 * X[:, 1])


def test_the_growth_stops_at_a_number_of_fits_and_says_so() -> None:
    X, y = _table()
    _best, _motifs, whole = counted.grow_counted(_memory(), X, y, max_fits=10 ** 9)
    best, _motifs, cut = counted.grow_counted(_memory(), X, y, max_fits=40)
    assert cut["stopped"] == "out of fits" and cut["max_fits"] == 40
    assert cut["fits"] < whole["fits"]                      # the count is what stopped it, and it did less work
    assert best                                             # what was grown before the stop is still returned


def test_the_same_table_gives_the_same_laws_twice() -> None:
    X, y = _table()
    first, motifs, report = counted.grow_counted(_memory(), X, y, max_fits=300)
    second, _motifs, again = counted.grow_counted(_memory(), X, y, max_fits=300)
    assert [(g.steps, g.theta) for g in first] == [(g.steps, g.theta) for g in second]
    assert report == again
    assert [str(timed.written(g, ["x0", "x1"], motifs)) for g in first]


def test_with_no_limit_reached_the_counted_growth_is_the_timed_growth() -> None:
    """Only the stopping rule differs: where neither stops early, the two grow the same laws from the same fits."""
    X, y = _table(12)
    grown, _motifs, report = counted.grow_counted(_memory(), X, y, max_fits=10 ** 9)
    reference, _motifs, expected = timed.grow(_memory(), X, y, seconds=1e9)
    assert [(g.steps, g.theta) for g in grown] == [(g.steps, g.theta) for g in reference]
    assert report["fits"] == expected["fits"] and report["stopped"] == expected["stopped"]


def test_the_remembered_whole_laws_stop_at_a_number_of_fits() -> None:
    X, y = _table()
    everything, whole = counted.remembered_whole_counted(_memory(), ["x0", "x1"], X, y, max_fits=10 ** 9)
    some, cut = counted.remembered_whole_counted(_memory(), ["x0", "x1"], X, y, max_fits=5)
    assert whole["stopped"] == "every remembered law tried" and cut["stopped"] == "out of fits"
    assert cut["fits"] < whole["fits"] and len(some) <= len(everything)
    reference = timed.remembered_whole(_memory(), ["x0", "x1"], X, y, seconds=1e9)
    assert [str(law) for _bic, law in everything] == [str(law) for _bic, law in reference]


def test_the_proposer_gives_the_same_proposals_twice_and_finds_a_remembered_form() -> None:
    X, y = _table()
    frame = pd.DataFrame({"x0": X[:, 0], "x1": X[:, 1], "target": y})
    proposer = counted.CountedMemoryProposer(_memory(), max_proposals=3, grow_fits=600, whole_fits=200)
    first = [str(p.formula) for p in proposer.fit_and_propose(frame)]
    run = dict(proposer.last_run)
    second = [str(p.formula) for p in proposer.fit_and_propose(frame)]
    assert first and first == second
    assert run["max_fits"] == 600 and run["whole"]["max_fits"] == 200
    assert proposer.proposer_id == "structure_memory_counted"
    # the table's law has the shape of the first law remembered: it is among the proposals, to rounding
    fits = []
    for text in first:
        import sympy as sp

        fn = sp.lambdify([sp.Symbol("x0"), sp.Symbol("x1")], sp.sympify(text), "numpy")
        fits.append(float(np.max(np.abs(fn(X[:, 0], X[:, 1]) - y))))
    assert min(fits) < 1e-4


def test_nothing_in_force_uses_the_counted_memory() -> None:
    """It is a proposal: the reading and the routes keep the timed growth their frozen plans name.

    One module of the package builds on it: the memory's list in a learned order (``ranked_memory.py``), itself a
    proposal that nothing in the package builds. So nothing in force reaches the counted memory through it either.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    sources = {path.relative_to(root).as_posix(): path.read_text(encoding="utf-8")
               for path in (root / "apedp_dyn").rglob("*.py")}
    users = sorted(name for name, text in sources.items()
                   if "structure_memory_counted" in text and not name.endswith("/structure_memory_counted.py"))
    assert users == ["apedp_dyn/proposers/ranked_memory.py"]
    assert [name for name, text in sources.items()
            if "ranked_memory" in text and not name.endswith("/ranked_memory.py")] == []


@pytest.mark.parametrize("name", ["GROW_FITS", "WHOLE_FITS"])
def test_the_limits_are_whole_numbers_of_fits(name: str) -> None:
    assert isinstance(getattr(counted, name), int) and getattr(counted, name) > 0
