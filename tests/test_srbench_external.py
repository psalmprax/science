"""The outside-engine harness (scripts/run_srbench_external.py), with stand-in engines.

PySR and PhySO are not dependencies of the package; their adapters are exercised
on GitHub Actions by the workflow's smoke job. These tests cover what does not
need them: that the protocol is the baseline's, the job list, and the summary.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_srbench_baseline as base
import run_srbench_external as ext


def test_rows_and_scoring_are_the_baselines_own():
    assert ext.base is base
    for name in ("make_split", "symbolic_recovery", "score_run", "run_one", "shard"):
        assert getattr(ext.base, name) is getattr(base, name)
    assert set(ext.NOISE_LEVELS) <= set(base.NOISE_LEVELS)  # every cell has a baseline cell to compare with
    assert ext.TIME_LIMIT_S > ext.SEARCH_BUDGET_S


def test_jobs_cover_every_problem_once_per_engine_and_noise():
    jobs = ext.all_jobs()
    assert len(jobs) == 133 * len(ext.NOISE_LEVELS) * len(ext.ENGINES)
    assert len({(j["problem"], j["noise"], j["engine"]) for j in jobs}) == len(jobs)


def test_the_smoke_law_is_scored_through_the_shared_scorer(monkeypatch):
    x0, x1 = sp.symbols("x0 x1")

    def finds_it(split, entry, budget):
        model = 1.0 * x0 * x1 + 1.0 * x0
        return {"model": model, "pred": split["X_test"][:, 0] * split["X_test"][:, 1] + split["X_test"][:, 0]}

    def misses_it(split, entry, budget):
        return {"model": 3.1 * x0, "pred": 3.1 * split["X_test"][:, 0]}

    monkeypatch.setitem(ext.RUNNERS, "pysr", finds_it)
    monkeypatch.setitem(ext.RUNNERS, "physo", misses_it)
    rows = {r["engine"]: r for r in ext.smoke(budget=1.0)}
    assert rows["exhaustive_mdl"]["symbolic"], "the in-house engine runs for real here and must find x0*x1 + x0"
    assert rows["pysr"]["symbolic"] and rows["pysr"]["accuracy_solution"]
    assert not rows["physo"]["symbolic"] and not rows["physo"]["accuracy_solution"]


def test_predictions_broadcast_a_constant_model():
    out = ext._predict(sp.Float(2.0), ["x0"], np.ones((5, 1)))
    assert out.shape == (5,) and np.all(out == 2.0)


def test_the_summary_counts_timeouts_and_errors():
    rows = [{"engine": "pysr", "family": "feynman", "noise": 0.0, "status": s, "seconds": 1.0, "symbolic": sym,
             "accuracy_solution": acc} for s, sym, acc in (("ok", True, True), ("timeout", None, False), ("error", None, False))]
    cell = ext.summarise(rows)["pysr|feynman|0.0"]
    assert (cell["n"], cell["symbolic"], cell["timeouts"], cell["errors"]) == (3, 1, 1, 1)


def test_the_frozen_files_match_the_recorded_digest():
    import json

    assert json.loads(ext.DIGEST.read_text())["sha256"] == ext.frozen_digests()
