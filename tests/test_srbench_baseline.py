"""The SRBench baseline harness (preregistration/PREREG-SRBENCH-BASELINE-2026-11.md).

Fast tests only: the manifest, the scoring, the split, sharding, isolation of a
run. Tests that need the downloaded data skip when the cache is absent.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fetch_srbench_ground_truth as srb
import run_srbench_baseline as bench

from apedp_dyn.discovery_l2.dimensional_types import Dimension

PROBLEMS = bench.load_manifest()
X0, X1 = sp.symbols("x0 x1")
GRID = np.random.default_rng(0).uniform(1, 5, (400, 2))


def _have_data(entry) -> bool:
    path = srb.CACHE / f"{entry['name']}.tsv.gz"
    return path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == entry["data"]["sha256"]


# ------------------------------------------------------------------ the manifest
def test_the_manifest_is_srbenchs_ground_truth_track():
    families = [p["family"] for p in PROBLEMS]
    assert families.count("feynman") == 119 and families.count("strogatz") == 14
    for p in PROBLEMS:
        assert len(p["data"]["sha256"]) == 64 and p["data"]["size"] > 0
        assert p["data"]["url"].startswith("https://media.githubusercontent.com/media/EpistasisLab/pmlb/")


@pytest.mark.parametrize("entry", PROBLEMS, ids=lambda p: p["name"])
def test_every_formula_parses_over_its_own_variables(entry):
    expr = bench.truth_expr(entry)
    assert {str(s) for s in expr.free_symbols} <= {f"x{i}" for i in range(len(entry["features"]))}
    assert not expr.atoms(sp.core.function.AppliedUndef), "an unknown function name slipped through"
    if entry["family"] == "feynman":
        assert set(entry["ranges"]) == set(entry["features"])


def test_units_parse_in_the_ai_feynman_basis():
    assert srb.parse_units("Electric field, m^(-1) * V^1") == {"l": "-1", "i": "1"}
    assert srb.parse_units("Mass, kg^1") == {"m": "1"}
    assert srb.parse_units("f, Dimensionless") == {}


def _dimension(expr, dims):
    if expr.is_Symbol:
        return dims[str(expr)]
    if expr.is_number:
        return Dimension()
    if expr.is_Mul:
        out = Dimension()
        for arg in expr.args:
            out = out * _dimension(arg, dims)
        return out
    if expr.is_Pow:
        return _dimension(expr.base, dims) ** Fraction(str(sp.nsimplify(expr.exp)))
    if expr.is_Add:
        found = {_dimension(arg, dims) for arg in expr.args}
        assert len(found) == 1, f"terms of {expr} differ in dimension"
        return found.pop()
    for arg in expr.args:
        assert _dimension(arg, dims) == Dimension(), f"{type(expr).__name__} of a dimensional argument"
    return Dimension()


def test_feynman_units_are_consistent_except_the_one_the_preregistration_names():
    inconsistent = []
    for entry in (p for p in PROBLEMS if p["family"] == "feynman"):
        dims = {f"x{i}": Dimension(**entry["units"][n]) for i, n in enumerate(entry["features"])}
        try:
            ok = _dimension(bench.truth_expr(entry), dims) == Dimension(**entry["target_units"])
        except AssertionError:
            ok = False
        if not ok:
            inconsistent.append(entry["name"])
    assert inconsistent == ["feynman_I_43_16"]


def test_formulas_reproduce_their_data_where_the_data_is_cached():
    checked = 0
    for entry in PROBLEMS:
        if not _have_data(entry):
            continue
        frame = srb.load_dataset(entry).sample(100, random_state=0)
        X = frame[entry["features"]].to_numpy(float)
        fn = sp.lambdify([sp.Symbol(f"x{i}") for i in range(len(entry["features"]))], bench.truth_expr(entry),
                         "numpy", dummify=True)
        value = np.asarray(fn(*X.T), float) * np.ones(len(X))
        y = frame["target"].to_numpy(float)
        assert np.max(np.abs(value - y)) <= 1e-6 * np.max(np.abs(y)), entry["name"]
        checked += 1
    if checked == 0:
        pytest.skip("no SRBench data cached (python scripts/fetch_srbench_ground_truth.py data)")


def test_a_download_that_does_not_match_its_pointer_is_refused(tmp_path, monkeypatch):
    entry = {"name": "fake", "data": {"sha256": "0" * 64, "size": 3, "url": "https://example.invalid/x"}}
    monkeypatch.setattr(srb, "_get", lambda url: b"abc")
    with pytest.raises(ValueError, match="manifest says"):
        srb.fetch_dataset(entry, tmp_path)
    assert not (tmp_path / "fake.tsv.gz").exists()


# ------------------------------------------------------------------ scoring
@pytest.mark.parametrize("model", [1.0008 * X0 * X1 - 0.0068, X0 * X1 + 3.7e-8, 2.5 * X1 * X0])
def test_the_truth_up_to_constants_is_recovered(model):
    assert bench.symbolic_recovery(model, X0 * X1, ["x0", "x1"], GRID)["recovered"]


@pytest.mark.parametrize("model", [
    -0.0149 - 0.0018 * X0**2 + 1.0011 * X0 * X1 + 0.01 * X0,  # a superset: extra terms may not be zeroed
    1.05 * X0 * X1 ** sp.Float(1.02),                       # a wrong exponent is a different law
    2.3 * X0 + 0.7 * X1**2,
    None,
])
def test_anything_else_is_not(model):
    assert not bench.symbolic_recovery(model, X0 * X1, ["x0", "x1"], GRID)["recovered"]


def test_a_close_approximation_is_not_a_recovery():
    poly = 0.3 + 0.6 * X0 - 0.05 * X0**2 + 0.002 * X0**3
    result = bench.symbolic_recovery(poly, sp.sqrt(X0), ["x0", "x1"], GRID)
    assert not result["recovered"] and result["rel_rmse"] < 1e-2


def test_formula_strings_parse_or_are_refused():
    assert bench.parse_formula("1.0000*x0*x1 + 0.2*const") == 1.0 * X0 * X1 + 0.2
    assert bench.parse_formula("-6.4992 * ((x2 - 1) / (x2^2)) + 1.6046") is not None
    assert bench.parse_formula("+1 * ((sq(x0))) +2") == X0**2 + 2
    for implicit in ("(0.667*x^1) * y' + (-1.000) * y = 0", "+1 * (x3[t-2]) + 1", None, ""):
        assert bench.parse_formula(implicit) is None


# ------------------------------------------------------------------ rows, shards, isolation
def test_the_split_is_seeded_disjoint_and_noisy_only_where_it_should_be():
    import pandas as pd

    entry = {"name": "toy", "features": ["a", "b"]}
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"a": rng.uniform(1, 2, 2000), "b": rng.uniform(1, 2, 2000)})
    frame["target"] = frame.a * frame.b
    clean = bench.make_split(entry, frame, 0.0)
    noisy = bench.make_split(entry, frame, 0.1)
    assert clean["X_train"].shape == (1000, 2) and clean["X_test"].shape == (500, 2)
    assert np.array_equal(clean["X_train"], noisy["X_train"]) and np.array_equal(clean["y_test"], noisy["y_test"])
    assert not np.array_equal(clean["y_train"], noisy["y_train"])
    assert np.allclose(clean["y_train"], clean["X_train"][:, 0] * clean["X_train"][:, 1])
    small = bench.make_split(entry, frame.head(400), 0.0)
    assert len(small["y_train"]) == 300 and len(small["y_test"]) == 100


def test_shards_partition_the_jobs_by_problem():
    jobs = bench.all_jobs()
    assert len(jobs) == 133 * 4 * 4
    shards = [bench.shard(jobs, k, 32) for k in range(32)]
    keys = [tuple(sorted(j.items())) for s in shards for j in s]
    assert len(keys) == len(set(keys)) == len(jobs)
    for s in shards:
        for problem in {j["problem"] for j in s}:
            assert sum(j["problem"] == problem for j in s) == 16  # every run of a problem in one shard


def _fake_entry(tmp_path, monkeypatch):
    import pandas as pd

    entry = {"name": "toy", "family": "feynman", "features": ["a", "b"], "formula": "a*b",
             "units": {"a": {}, "b": {}}, "target_units": {}}
    frame = pd.DataFrame({"a": np.linspace(1, 2, 2000), "b": np.linspace(2, 3, 2000)})
    frame["target"] = frame.a * frame.b
    monkeypatch.setattr(bench.srb, "load_dataset", lambda e: frame)
    return {"toy": entry}


def test_a_run_that_overruns_is_killed_and_counted(tmp_path, monkeypatch):
    entries = _fake_entry(tmp_path, monkeypatch)
    monkeypatch.setitem(bench.RUNNERS, "rime", lambda split, entry: time.sleep(30))
    started = time.perf_counter()
    row = bench.run_one({"problem": "toy", "noise": 0.0, "seed": 0, "engine": "rime"}, entries, time_limit=1.0)
    assert row["status"] == "timeout" and time.perf_counter() - started < 10


def test_a_crash_is_an_outcome_not_an_abort(tmp_path, monkeypatch):
    entries = _fake_entry(tmp_path, monkeypatch)

    def boom(split, entry):
        raise RuntimeError("engine fell over")

    monkeypatch.setitem(bench.RUNNERS, "rime", boom)
    row = bench.run_one({"problem": "toy", "noise": 0.0, "seed": 0, "engine": "rime"}, entries, time_limit=30)
    assert row["status"] == "error" and "engine fell over" in row["error"]


def test_an_engine_that_finds_the_law_is_scored_as_recovering_it(tmp_path, monkeypatch):
    entries = _fake_entry(tmp_path, monkeypatch)
    row = bench.run_one({"problem": "toy", "noise": 0.0, "seed": 0, "engine": "hierarchical_extended"}, entries)
    assert row["status"] == "ok" and row["symbolic"] and row["accuracy_solution"], row


def test_the_summary_counts_every_outcome():
    rows = [{"engine": "l2", "family": "feynman", "noise": 0.0, "status": s, "seconds": 1.0, "symbolic": sym,
             "accuracy_solution": acc, "l2_outcome": out}
            for s, sym, acc, out in (("ok", True, True, "promoted_correct"), ("ok", False, True, "false_promotion"),
                                     ("timeout", None, False, None), ("ok", False, False, "no_promotion"))]
    cell = bench.summarise(rows)["l2|feynman|0.0"]
    assert (cell["n"], cell["symbolic"], cell["accuracy_solution"], cell["timeouts"]) == (4, 1, 2, 1)
    assert (cell["promoted_correct"], cell["false_promotion"], cell["no_promotion"]) == (1, 1, 1)
    assert cell["false_promotion_share_of_promotions"] == 0.5


def test_the_frozen_files_match_the_recorded_digest():
    recorded = json.loads(bench.DIGEST.read_text())["sha256"]
    assert recorded == bench.frozen_digests()
