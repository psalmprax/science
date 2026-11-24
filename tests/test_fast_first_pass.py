"""The native first pass returns what the Python reference returns, and nothing changes where it is not used."""
from __future__ import annotations

import math

import numpy as np
import pytest

from apedp_dyn.discovery_l2 import confirmed_stop, fast_first_pass, main_search
from apedp_dyn.discovery_l2 import exhaustive_search as es
from apedp_dyn.discovery_l2.guided_search import GuidedSearch

native = pytest.mark.skipif(fast_first_pass.kernel() is None, reason="the native kernel is not installed in this environment")


@pytest.fixture(scope="module")
def guided() -> GuidedSearch:
    return GuidedSearch(main_search.size_prior)


def _rows(kind: str, rows: int = 120, inputs: int = 1, noise: float = 0.0, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.uniform(1.0, 5.0, (rows, inputs))
    a = X[:, 0]
    b = X[:, -1]
    y = {"simple": 3.0 * np.sqrt(a) - 2.0,                      # one term: the search stops early when there is noise
         "sum": 2.0 * np.log(a) + 0.7 * b ** 2 + 1.0,           # a sum of two terms: only the full search finds it
         "outside": np.tanh(a) / (1.0 + b),                     # not in the enumeration
         "noise": rng.standard_normal(rows),
         "large": 1e6 * a ** 2 + 3e9}[kind]
    return X, y + noise * float(np.sqrt(np.mean(y ** 2))) * rng.standard_normal(rows)


# ---------------------------------------------------------------- the expression table
def test_the_table_is_the_enumeration(guided) -> None:
    nodes, _nats, _order = guided.order(2, 4)
    table = fast_first_pass.table_of(nodes)
    assert fast_first_pass.table_of(nodes) is table                     # built once per enumeration
    assert len(table.ops) == len(nodes) and list(table.complexity) == [c for _node, c in nodes]
    for i, (node, _c) in enumerate(nodes):
        assert table.ops[i] == fast_first_pass.OPS[node[0]]
        if node[0] == "x":
            assert table.left[i] == node[1] and table.right[i] == -1
            continue
        assert nodes[table.left[i]][0] == node[1] and table.left[i] < i  # a child comes before its parent
        if len(node) == 3:
            assert nodes[table.right[i]][0] == node[2] and table.right[i] < i
        else:
            assert table.right[i] == -1


def test_the_operation_codes_cover_the_reference_grammar() -> None:
    assert set(fast_first_pass.OPS) == {"x", *es.UNARY, *es.BINARY}
    assert sorted(fast_first_pass.OPS.values()) == list(range(12))


# ---------------------------------------------------------------- without the kernel nothing changes
def test_without_the_kernel_the_reference_answers(guided, monkeypatch) -> None:
    monkeypatch.setattr(fast_first_pass, "kernel", lambda: None)
    X, y = _rows("simple", noise=0.01)
    assert fast_first_pass.search_confirmed(guided, X, y, ["x0"], 5) == confirmed_stop.search_confirmed(guided, X, y, ["x0"], 5)


def test_rows_the_kernel_is_not_given_go_to_the_reference(guided, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(fast_first_pass, "_search_confirmed", lambda *a, **k: calls.append(1))
    X, y = _rows("simple")
    tiny = fast_first_pass.search_confirmed(guided, X[:3], y[:3], ["x0"], 3)          # fewer than four rows
    assert calls == [] and tiny == confirmed_stop.search_confirmed(guided, X[:3], y[:3], ["x0"], 3)


def test_the_main_search_uses_it_only_inside_the_block(guided) -> None:
    original = main_search.search_confirmed
    assert original is confirmed_stop.search_confirmed
    with fast_first_pass.enabled():
        inside = main_search.search_confirmed
        assert inside is not original
        X, y = _rows("simple", noise=0.01)
        assert inside(guided, X, y, ["x0"], 5) == original(guided, X, y, ["x0"], 5)
    assert main_search.search_confirmed is original
    with pytest.raises(RuntimeError), fast_first_pass.enabled():
        raise RuntimeError("anything")
    assert main_search.search_confirmed is original         # restored after an error too


def test_no_bound_is_claimed_near_the_floor_or_for_an_ill_conditioned_fit() -> None:
    doubt = fast_first_pass._doubt(np.array([1.0, 1e-18, 1.0, 1.0]), np.array([10.0, 10.0, 1e15, np.nan]), total=4.0, floor=1e-20)
    assert np.isfinite(doubt[0]) and doubt[0] < 1e-9
    assert np.isinf(doubt[1]) and np.isinf(doubt[2]) and np.isinf(doubt[3])


def test_no_bound_is_claimed_for_a_column_two_libraries_cannot_agree_on() -> None:
    rss, condition = np.ones(4), np.full(4, 10.0)
    doubt = fast_first_pass._doubt(rss, condition, 4.0, 1e-20, np.array([0.0, 1e-13, 1e-2, np.inf]))
    assert doubt[0] == fast_first_pass._doubt(rss, condition, 4.0, 1e-20)[0]
    assert doubt[0] < doubt[1] < 1e-10 and np.isinf(doubt[2]) and np.isinf(doubt[3])


# ---------------------------------------------------------------- with the kernel: the same result
CASES = [("simple", 1, 0.0, 6), ("simple", 1, 0.01, 6), ("sum", 2, 0.0, 4), ("sum", 2, 0.01, 4), ("outside", 2, 0.02, 4),
         ("noise", 1, 0.0, 6), ("large", 1, 0.0, 6), ("large", 1, 0.001, 6), ("outside", 1, 0.0, 6)]


@native
@pytest.mark.parametrize("kind,inputs,noise,size", CASES)
def test_the_native_first_pass_returns_the_reference_result(guided, kind: str, inputs: int, noise: float, size: int) -> None:
    X, y = _rows(kind, inputs=inputs, noise=noise, seed=3)
    names = [f"x{i}" for i in range(inputs)]
    reference = confirmed_stop.search_confirmed(guided, X, y, names, size)
    assert fast_first_pass.search_confirmed(guided, X, y, names, size, check=True) == reference
    assert math.isfinite(reference.best.description_length)


@native
def test_every_path_of_the_reference_is_exercised(guided) -> None:
    early = confirmed_stop.search_confirmed(guided, *_rows("simple", noise=0.01, seed=3), ["x0"], 6)
    full = confirmed_stop.search_confirmed(guided, *_rows("sum", inputs=2, noise=0.01, seed=3), ["x0", "x1"], 4)
    assert early.stopped_early and early.evaluated < early.candidates
    assert not full.stopped_early and full.residual_structure == 1.0 and len(full.best.terms) == 2


@native
def test_other_stopping_settings_are_passed_on(guided) -> None:
    X, y = _rows("simple", noise=0.01, seed=5)
    for patience, floor in ((1.0, 10), (5.0, 1000), (2.5, 0)):
        reference = confirmed_stop.search_confirmed(guided, X, y, ["x0"], 6, patience=patience, floor=floor)
        assert fast_first_pass.search_confirmed(guided, X, y, ["x0"], 6, patience=patience, floor=floor, check=True) == reference


@native
def test_a_difference_from_the_reference_is_raised_when_checking(guided, monkeypatch) -> None:
    X, y = _rows("simple", noise=0.01)
    right = confirmed_stop.search_confirmed(guided, X, y, ["x0"], 5)
    wrong = type(right)(right.best, right.evaluated + 1, right.candidates, right.stopped_early, right.residual_structure)
    monkeypatch.setattr(fast_first_pass, "_search_confirmed", lambda *a, **k: wrong)
    with pytest.raises(fast_first_pass.FirstPassMismatchError):
        fast_first_pass.search_confirmed(guided, X, y, ["x0"], 5, check=True)
    assert fast_first_pass.search_confirmed(guided, X, y, ["x0"], 5) == wrong          # unchecked, it is returned as it is


@native
def test_the_kernel_refuses_a_table_it_cannot_run() -> None:
    kernel = fast_first_pass.kernel()
    x = np.ascontiguousarray(np.linspace(1.0, 2.0, 8).reshape(-1, 1))
    y = np.ascontiguousarray(x[:, 0] * 2.0)
    nn = np.ascontiguousarray(np.array([[(i + 1) % 8] for i in range(8)], dtype=np.uint32))
    ops, left, right = np.array([1, 0], np.uint8), np.array([1, 0], np.int32), np.array([-1, -1], np.int32)
    with pytest.raises(ValueError):                                     # a child after its parent
        kernel.first_pass_singles(ops, left, right, x, y, nn, 0.0)
    with pytest.raises(ValueError):                                     # a column the rows do not have
        kernel.first_pass_singles(np.array([0], np.uint8), np.array([4], np.int32), np.array([-1], np.int32), x, y, nn, 0.0)
    rss, _slope, _intercept, _structure, _condition, eval_error, status = kernel.first_pass_singles(
        np.array([0, 9], np.uint8), np.array([0, 0], np.int32), np.array([-1, 0], np.int32), x, y, nn, 0.0)
    assert status.tolist() == [fast_first_pass.FITTED, fast_first_pass.RANK_DEFICIENT] and rss[0] < 1e-25   # x0 - x0 is constant
    assert eval_error[0] == 0.0 and np.isinf(eval_error[1])


@native
def test_the_sine_of_a_huge_number_is_left_to_the_reference(guided) -> None:
    """The case that separated the two on another processor: ``sin(exp(exp(x)))`` on a table of noise."""
    kernel = fast_first_pass.kernel()
    x = np.ascontiguousarray(np.linspace(1.0, 5.0, 50).reshape(-1, 1))
    y = np.ascontiguousarray(np.random.default_rng(0).standard_normal(50))
    nn = np.ascontiguousarray(np.array([[(i + 1) % 50, (i - 1) % 50] for i in range(50)], dtype=np.uint32))
    ops = np.array([0, 2, 2, 4, 4], np.uint8)                       # x0, exp, exp(exp), sin(exp(exp)), sin(x0)
    left, right = np.array([0, 0, 1, 2, 0], np.int32), np.full(5, -1, np.int32)
    rss, _slope, _intercept, _structure, condition, eval_error, status = kernel.first_pass_singles(ops, left, right, x, y, nn, 1e-20)
    assert status[3] == fast_first_pass.FITTED and eval_error[3] > 1e30 and eval_error[4] < 1e-12
    doubt = fast_first_pass._doubt(rss, condition, float(np.sum((y - y.mean()) ** 2)), 1e-20, eval_error)
    assert np.isinf(doubt[3]) and np.isfinite(doubt[4]) and np.isfinite(doubt[0])
    for seed in range(4):                                             # and on tables of noise the two still agree
        X, noise = _rows("noise", rows=200, inputs=2, seed=seed)
        assert fast_first_pass.search_confirmed(guided, X, noise, ["x0", "x1"], 4, check=True)
