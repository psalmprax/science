"""Exhaustive search with description-length scoring (apedp_dyn/discovery_l2/exhaustive_search.py)."""
from __future__ import annotations

import numpy as np
import sympy as sp

from apedp_dyn.discovery_l2 import exhaustive_search as es
from apedp_dyn.discovery_l2 import work_queue as wq

RNG = np.random.default_rng(0)
X = RNG.uniform(1, 5, (400, 2))
NAMES = ["x0", "x1"]


def _same_function(model: str, truth) -> bool:
    fn = sp.lambdify(sp.symbols("x0 x1"), sp.sympify(model), "numpy")
    grid = np.random.default_rng(9).uniform(1, 5, (200, 2))
    return bool(np.allclose(fn(*grid.T), truth(grid), rtol=0.02))


def test_the_enumeration_is_deterministic_and_free_of_rescaled_duplicates():
    first = es.enumerate_expressions(2, 4)
    assert first == es.enumerate_expressions(2, 4)
    strings = [es.to_string(node, NAMES) for node, _ in first]
    assert "x0" in strings and "((x0) * (x1))" in strings
    assert "((x0) + (x0))" not in strings  # 2*x0 is x0 up to scale
    assert all(1 <= c <= 4 for _, c in first)
    probes = es._probes(2)
    keys = [es._fingerprint(es.evaluate(node, probes)) for node, _ in first]
    assert None not in keys and len(set(keys)) == len(keys)


def test_a_single_term_law_is_found_with_its_coefficients():
    y = 2.5 * np.exp(-X[:, 0]) * X[:, 1] + 0.3
    best = es.search(X, y * (1 + 0.005 * RNG.standard_normal(len(y))), NAMES, 5)[0]
    assert _same_function(es.model_string(best), lambda g: 2.5 * np.exp(-g[:, 0]) * g[:, 1] + 0.3)


def test_a_sum_whose_terms_fit_badly_alone_is_found():
    y = 3 * np.sqrt(X[:, 0]) + 2 * X[:, 1]
    best = es.search(X, y * (1 + 0.005 * RNG.standard_normal(len(y))), NAMES, 4)[0]
    assert _same_function(es.model_string(best), lambda g: 3 * np.sqrt(g[:, 0]) + 2 * g[:, 1])


def test_description_length_prefers_the_simpler_of_two_equal_fits():
    assert es.description_length(1.0, 100, 3, 2, 10) < es.description_length(1.0, 100, 5, 2, 10)
    assert es.description_length(1.0, 100, 3, 2, 10) < es.description_length(1.0, 100, 3, 3, 10)
    assert es.description_length(0.5, 100, 3, 2, 10) < es.description_length(1.0, 100, 3, 2, 10)


def test_slices_cover_every_candidate_once_and_merge_to_the_single_process_answer():
    y = X[:, 0] / X[:, 1] + 0.01 * RNG.standard_normal(len(X))
    whole = es.score_slice(X, y, NAMES, 4, keep=10**6)
    parts = [es.score_slice(X, y, NAMES, 4, part=k, parts=3, keep=10**6) for k in range(3)]
    assert sorted(r["index"] for p in parts for r in p) == sorted(r["index"] for r in whole)
    merged = es.merge(parts, X, y, NAMES, 4)
    single = es.merge([whole], X, y, NAMES, 4)
    assert [(s.expression, s.description_length) for s in merged[:20]] == [(s.expression, s.description_length) for s in single[:20]]


def test_the_work_queue_carries_the_slices(tmp_path):
    y = X[:, 0] * X[:, 1]
    data = tmp_path / "d.npz"
    np.savez(data, X=X, y=y)
    jobs = [{"job_id": f"part-{k}", "data": str(data), "names": NAMES, "max_complexity": 4, "part": k, "parts": 4}
            for k in range(4)]
    queue = wq.WorkQueue.create(tmp_path / "q", jobs)
    assert wq.run_worker(queue.root, es.slice_job, "w") == 4
    results = wq.collect(queue.root)
    best = es.merge([r["best"] for r in results], X, y, NAMES, 4)[0]
    assert best.expression == "((x0) * (x1))"
