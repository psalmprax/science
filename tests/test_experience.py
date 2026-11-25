"""Pins for the store of solved cases: what a table looks like, a law without its constants, and the nearest cases."""
from __future__ import annotations

import numpy as np

from apedp_dyn.proposers import experience


def _table(law, inputs: int = 2, seed: int = 0, rows: int = 300):
    rng = np.random.default_rng(seed)
    X = rng.uniform(1.0, 5.0, (rows, inputs))
    return X, law(X)


def test_a_signature_is_rank_statistics_and_ignores_input_order_and_units() -> None:
    X, y = _table(lambda X: 1.3 * X[:, 0] / (X[:, 1] + 0.7))
    signature = experience.signature(X, y)
    assert signature.shape == (experience.PER_INPUT * experience.MAX_INPUTS + experience.SPREAD_FEATURES,)
    assert np.allclose(experience.signature(X[:, ::-1], y), signature, atol=1e-4)          # the inputs in another order
    assert np.allclose(experience.signature(1000.0 * X, 0.01 * y), signature, atol=1e-4)   # other units
    assert not np.allclose(experience.signature(X, -y), signature, atol=1e-2)               # another table


def test_a_law_s_fitted_constants_become_free_constants_and_whole_numbers_stay() -> None:
    assert experience.skeleton("1.3*x0/(sqrt(x0 + 2.3) + 3.7)") == "c0*x0/(c1 + sqrt(c2 + x0))"
    assert experience.skeleton("-1.0*x0**3") == "c0*x0**3"
    assert experience.skeleton("x0*x1") == "x0*x1"


def test_the_nearest_cases_have_as_many_inputs_and_the_most_similar_table_first() -> None:
    store = experience.Experience()
    laws = {"rise": lambda X: 2.0 * X[:, 0] + 0.5 * X[:, 1], "fall": lambda X: 3.0 / X[:, 0] - X[:, 1],
            "wave": lambda X: np.sin(3.0 * X[:, 0]) * X[:, 1]}
    for name, law in laws.items():
        X, y = _table(law, seed=1)
        store.add(name, name, X, y)
    X1, y1 = _table(lambda X: np.exp(-X[:, 0]), inputs=1, seed=2)
    store.add("one input", "exp(-1.0*x0)", X1, y1)
    X, y = _table(lambda X: 1.7 * X[:, 0] + 0.9 * X[:, 1], seed=3)             # another rising table, other constants
    near = store.nearest(X, y, k=2)
    assert near[0].key == "rise" and len(near) == 2 and all(case.inputs == 2 for case in near)
    assert store.nearest(X, y, k=5, exclude=["rise"])[0].key != "rise"
    assert [case.key for case in store.nearest(X1, y1, k=5)] == ["one input"]
    assert experience.Experience().nearest(X, y) == []


def test_the_store_comes_back_the_same_and_keeps_no_rows() -> None:
    store = experience.Experience()
    X, y = _table(lambda X: X[:, 0] * X[:, 1])
    case = store.add("k", "1.0*x0*x1", X, y)
    again = experience.Experience.from_dict(store.as_dict())
    assert again.cases == [case] and set(store.as_dict()["cases"][0]) == {"key", "law", "inputs", "signature"}
