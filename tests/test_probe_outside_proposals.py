"""The kill tests of four outside proposals (development; ``scripts/probe_outside_proposals.py``)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import probe_outside_proposals as probes

RECORD = ROOT / "benchmarks" / "outside_proposals" / "dev_probes.json"


def test_the_implicit_search_finds_a_rational_law_s_own_form_on_exact_rows() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(0.1, 10.0, 17)
    ranked = probes.implicit_search(p[:, None], 2.0 * p / (1 + 0.8 * p), 2, 4)
    assert set(ranked[0][1]) == {(1, 1), (0, 1), (1, 0)}                       # q*p, q and p: Langmuir's law as a relation
    coefficients = dict(zip(ranked[0][1], ranked[0][2] / ranked[0][2][list(ranked[0][1]).index((0, 1))], strict=True))
    assert coefficients[(1, 1)] == pytest.approx(0.8, rel=1e-6) and coefficients[(1, 0)] == pytest.approx(-2.0, rel=1e-6)
    solved = probes.solve_y(ranked[0][1], ranked[0][2], np.array([[1.0], [5.0]]), np.array([1.0, 2.0]))
    assert solved == pytest.approx([2.0 / 1.8, 10.0 / 5.0], rel=1e-6)


def test_the_implicit_probe_says_what_it_does_with_a_law_that_is_not_algebraic() -> None:
    rows = probes.probe_implicit(seeds=3, laws=["control: exponential 2*exp(-0.7*x)", "Langmuir a*p/(1 + b*p)"], sizes=[20])
    control = next(r for r in rows if r["law"].startswith("control") and r["noise"] == 0.0)
    assert control["own_form_first"] is None and control["accurate_outside"] == 0      # a relation that holds inside the range only
    own = next(r for r in rows if r["law"].startswith("Langmuir") and r["noise"] == 0.0)
    assert own["own_form_first"] == own["draws"] == 3 and own["accurate_outside"] == 3


def test_the_real_tables_are_the_development_ones_and_kepler_s_form_is_first() -> None:
    rows = {row["table"]: row for row in probes.probe_real()}
    assert set(rows) == {"kepler_solar_system", "langmuir_table_ix", "leavitt"}
    assert rows["kepler_solar_system"]["own_form_rank"] == 1 and rows["kepler_solar_system"]["rows"] == 8
    assert rows["langmuir_table_ix"]["own_form_rank"] > 1 and rows["leavitt"]["own_form_rank"] > 1


def test_the_gradient_statistic_is_zero_for_a_composition_and_not_for_another_subset() -> None:
    got = probes.probe_gradient(rows=300, laws=["sin(x1*x2)*x3 + x4"], noises=(0.0,))[0]
    assert got["compositions"] == 2 and got["others"] == 8 and got["told_apart"] is True
    assert got["largest_for_a_composition"] < 0.01 < got["smallest_for_another"]


def test_the_weak_form_finds_a_true_equation_on_a_dense_even_table_and_counts_every_true_one() -> None:
    assert frozenset({(0, 0), (2, 0)}) in probes.true_equations("sin(2.0*z + 1.0)")          # f'' + 4 f = 0
    assert frozenset({(0, 0), (1, 0), (1, 1)}) in probes.true_equations("(z + 1)^-2")        # (z + 1) f' + 2 f = 0
    rows = probes.probe_weak(seeds=2, functions=["sin(2.0*z + 1.0)", "(z + 1)^-2"], sizes=(500,))
    even = next(r for r in rows if r.get("positions") == "evenly spaced" and r["noise"] == 0.0)
    assert even["a_true_equation_first"] == even["draws"] == 4


def test_the_weak_euler_lagrange_form_has_a_plus_and_gives_the_pendulum_s_lagrangian() -> None:
    got = probes.lagrangian_trial("pendulum: qdd = -2.25*sin(q)", 2000, 0.0, 0)
    assert got["first"] and got["weak_error"] < 1e-3                # L = qd^2/2 + 2.25*cos(q): the ratio is +4.5, not -4.5


def test_the_record_is_the_probes_and_says_it_is_development() -> None:
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    assert record["development_not_a_result"] is True and record["tag"] == probes.TAG
    assert set(record) >= {"implicit", "real", "gradient", "weak", "lagrangian"}
    # Ranks far down a list of several hundred relations can move by a place between machines (a relation and its
    # multiple by an input tie to rounding), so the record is checked for what does not: the tables, their rows, and
    # Kepler's form at the head.
    now = {row["table"]: row for row in probes.probe_real()}
    then = {row["table"]: row for row in record["real"]}
    assert {k: (v["rows"], v["supports_ranked"]) for k, v in then.items()} == {k: (v["rows"], v["supports_ranked"]) for k, v in now.items()}
    assert then["kepler_solar_system"]["own_form_rank"] == 1 and then["langmuir_table_ix"]["own_form_rank"] > 30 < then["leavitt"]["own_form_rank"]
    assert abs(then["langmuir_table_ix"]["own_form_rank"] - now["langmuir_table_ix"]["own_form_rank"]) <= 3
    assert len(record["gradient"]) == 30 and len(record["implicit"]) == 54 and len(record["lagrangian"]) == 18
