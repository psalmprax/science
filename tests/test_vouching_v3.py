"""The last checks, third version: the second version, and no law with a pole among the rows (development only)."""
from __future__ import annotations

import numpy as np
import sympy as sp

from apedp_dyn.discovery_l2 import vouching_v2, vouching_v3

x0, x1 = sp.symbols("x0 x1")
POLE_LAW = "3.7/((x0 + 3.7)*(sin(2.9*x1) + 0.7))"


def _rows(n: int = 700, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(1.0, 5.0, (n, 2))


def test_it_finds_denominators_and_logarithm_arguments() -> None:
    parts = vouching_v3.singular_parts(sp.sympify("x0/(x1 - 1) + exp(1/(x0 - 3)) + log(x0 - 2) + sqrt(x1)"))
    assert set(map(str, parts)) == {"x1 - 1", "x0 - 3", "x0 - 2"}
    assert vouching_v3.singular_parts(sp.sympify("2.5*x0 + 3")) == []


def test_a_pole_among_the_rows_is_found_and_one_outside_them_is_not() -> None:
    X = _rows()
    assert vouching_v3.pole_among(POLE_LAW, ["x0", "x1"], X) == "sin(2.9*x1) + 0.7"
    assert vouching_v3.pole_among("x0/(x1 + 0.5)", ["x0", "x1"], X) is None          # its pole is at x1 = -0.5
    assert vouching_v3.pole_among("x0/(x1 - 6)", ["x0", "x1"], X) is None            # beyond the rows' range
    assert vouching_v3.pole_among("x0/(x1 - 3)", ["x0", "x1"], X) == "x1 - 3"


def test_it_withholds_what_the_second_version_passes_with_a_pole_and_changes_nothing_else() -> None:
    X = _rows()
    rng = np.random.default_rng(1)
    clean = 3.7 / ((X[:, 0] + 3.7) * (np.sin(2.9 * X[:, 1]) + 0.7))
    keep = np.abs(np.sin(2.9 * X[:, 1]) + 0.7) > 0.05                                  # rows the data happened to give
    X, clean = X[keep], clean[keep]
    y = clean + 0.001 * float(np.sqrt(np.mean(clean ** 2))) * rng.standard_normal(len(clean))
    second = vouching_v2.vouch(POLE_LAW, [], ["x0", "x1"], X[:500], y[:500], X[500:])
    third = vouching_v3.after_second(second, ["x0", "x1"], X[:500], X[500:])
    assert third.law is None and "pole" in third.reason and third.record["pole_rule"] == "sin(2.9*x1) + 0.7"
    # without a pole the third version is the second
    smooth = 0.8 * X[:, 0] + 1.9 / (X[:, 0] + 0.6)
    ys = smooth + 0.01 * float(np.sqrt(np.mean(smooth ** 2))) * rng.standard_normal(len(smooth))
    law = "0.8*x0 + 1.9/(x0 + 0.6)"
    a = vouching_v2.vouch(law, [], ["x0", "x1"], X[:500], ys[:500], X[500:])
    b = vouching_v3.vouch(law, [], ["x0", "x1"], X[:500], ys[:500], X[500:])
    assert (a.law, a.reason, a.switched) == (b.law, b.reason, b.switched) and b.record["pole_rule"] == "no pole among the rows"


def test_a_withheld_law_stays_withheld() -> None:
    nothing = vouching_v2.Vouched(None, "every candidate failed a test of its residuals")
    assert vouching_v3.after_second(nothing, ["x0"], _rows()[:, :1]) is nothing
