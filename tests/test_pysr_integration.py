"""The real PySR through the proposer and the main search path. Needs PySR and Julia.

Skipped unless ``APEDP_RUN_PYSR=1``; `.github/workflows/main-search-smoke.yml` runs it.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
import sympy as sp

from apedp_dyn.discovery_l2 import main_search
from apedp_dyn.proposers.pysr_proposer import PySRProposer

pytestmark = pytest.mark.skipif(os.environ.get("APEDP_RUN_PYSR") != "1", reason="needs PySR and Julia; set APEDP_RUN_PYSR=1")
BUDGET = float(os.environ.get("APEDP_PYSR_BUDGET_SECONDS", "60"))


def _same_shape(law: str, truth: str) -> bool:
    grid = np.random.default_rng(9).uniform(1.0, 5.0, (200, 2))
    symbols = [sp.Symbol("a"), sp.Symbol("b")]
    u = sp.lambdify(symbols, sp.sympify(law), "numpy")(*grid.T) * np.ones(200)
    v = sp.lambdify(symbols, sp.sympify(truth), "numpy")(*grid.T)
    return bool(abs(np.corrcoef(u, v)[0, 1]) > 0.9999)


def test_real_pysr_proposes_in_feature_names_and_accepts_a_seed():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"a": rng.uniform(1, 5, 300), "b": rng.uniform(1, 5, 300)})
    frame["target"] = frame["a"] * np.exp(-0.37 * frame["b"])
    proposals = PySRProposer(budget_seconds=BUDGET).fit_and_propose(frame, guesses=["a/b"])
    assert proposals and all(p.proposer_id == "pysr_proposer" for p in proposals)
    assert all(sp.sympify(p.formula).free_symbols <= {sp.Symbol("a"), sp.Symbol("b")} for p in proposals)
    best = min(proposals, key=lambda p: p.parameters["dev_loss"])
    assert np.allclose(best.compute_score(frame), frame["target"], rtol=0.05)
    print("PYSR_PROPOSALS", [(p.formula, p.parameters["dev_loss"]) for p in proposals])


def test_real_pysr_through_the_main_search_path():
    rng = np.random.default_rng(1)
    truth = "a*exp(-0.37*b)"
    fn = sp.lambdify([sp.Symbol("a"), sp.Symbol("b")], sp.sympify(truth), "numpy")
    X = rng.uniform(1.0, 5.0, (300, 2))
    pool = rng.uniform(1.0, 5.0, (600, 2))
    y = fn(*X.T) * (1 + 0.01 * rng.standard_normal(300))
    noise = np.random.default_rng(2)
    result = main_search.discover({"a": X[:, 0], "b": X[:, 1]}, y, pool,
                                  lambda rows: fn(*pool[rows].T) * (1 + 0.01 * noise.standard_normal(len(rows))),
                                  proposer=PySRProposer(budget_seconds=BUDGET), always_search=True)
    print("MAIN_SEARCH", result.action, result.law, result.found_by, result.search, result.adequacy)
    print("MAIN_SEARCH_CANDIDATES", [c["formula"][:60] for c in result.candidates])
    assert result.search["ran"] and result.search["proposals"] >= 1
    assert result.action in ("confirmed", "equivalent_class", "unchallenged")
    assert result.law is not None and _same_shape(result.law, truth)
