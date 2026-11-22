"""Memory of confirmed laws: what is remembered of a law, and what is proposed from it."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import sympy as sp

from apedp_dyn.discovery_l2 import memory_stage
from apedp_dyn.proposers import structure_memory as sm
from apedp_dyn.proposers.portfolio import FixedFormulasProposer

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _memory(*laws: str) -> sm.Memory:
    memory = sm.Memory()
    for law in laws:
        memory.learn(law)
    return memory


# ---------------------------------------------------------------- what is remembered
def test_a_laws_motifs_are_its_one_variable_parts_without_names_or_numbers() -> None:
    assert sorted(sm.motifs_of("2.3*x0/(x0 + 1.7) - 0.7*exp(-0.45*x1)")) == ["exp(C*v)", "v/(C + v)"]
    assert sorted(sm.motifs_of("0.7*x0**2*sin(2.3*x1)")) == ["sin(C*v)", "v**2"]
    assert sorted(sm.motifs_of("1.3/(x1*(log(0.3*x0 + 1) + 1.7))")) == ["1/(C + log(C*v + 1))", "1/v", "log(C*v + 1)"]
    assert sorted(sm.motifs_of("0.45*x1**2/((x0**2 + 0.3)*(x1/(x1 + 2.9) + 0.3))")) == [
        "1/(C + v**2)", "v**2", "v**2/(C + v/(C + v))", "v/(C + v)"]          # a part, its factors, what it divides by
    assert sm.motifs_of("3.0*sqrt(x0 + 1.3)") == ["sqrt(C + v)"]              # one variable: the law itself, scale stripped
    assert sm.motifs_of("4.2") == []


def test_scale_and_offset_belong_to_the_join_and_not_to_the_motif() -> None:
    assert sm.shape_of(sp.sympify("5.0*log(x3) + 2.0")) == "log(v)"
    assert sm.shape_of(sp.sympify("-0.2*x1")) == "v"
    long = sp.sympify("sin(cos(exp(sqrt(log(x0 + 1.0) + 2.0) + 3.0) + 4.0) + 5.0)*tanh(x0)**3/(x0 + 1.5)")
    assert sm.shape_of(long) is None                                           # too large to be a building block


def test_a_whole_law_is_remembered_by_shape() -> None:
    shape, variables = sm.skeleton_of("2.3*b/(b + 1.7) - 0.7*exp(-0.45*a)")
    assert variables == 2 and "C" in shape and "u0" in shape and "u1" in shape and "a" not in shape.replace("u", "")
    again, _ = sm.skeleton_of("9.1*q/(q + 0.2) - 3.3*exp(-1.5*p)")
    assert again == shape                                                      # other names, other numbers: the same law


def test_memory_counts_motifs_and_survives_a_file(tmp_path) -> None:
    memory = _memory("2.0*x0/(x0 + 1.5) + 0.5*x1", "1.1*exp(-0.3*x0)*x1", "7.0")
    assert memory.laws == 2 and memory.motifs == {"v/(C + v)": 1, "v": 2, "exp(C*v)": 1}, memory.motifs
    assert len(memory.skeletons) == 2 and set(memory.skeletons.values()) == {2}
    memory.save(tmp_path / "memory.json")
    assert sm.Memory.load(tmp_path / "memory.json").as_dict() == memory.as_dict()
    assert json.loads((tmp_path / "memory.json").read_text())["laws"] == 2
    assert "0.3" not in (tmp_path / "memory.json").read_text()                 # no constant's value is kept


def test_every_constant_of_a_shape_becomes_a_parameter_of_its_own() -> None:
    expr, params = sm.with_parameters("C*v/(C + v)", {"v": sp.Symbol("x1")}, "_p")
    assert len(params) == 2 and expr.free_symbols == {sp.Symbol("x1"), *params}
    assert sp.simplify(expr.subs({params[0]: 2, params[1]: 3}) - 2 * sp.Symbol("x1") / (3 + sp.Symbol("x1"))) == 0


# ---------------------------------------------------------------- what is proposed
def _frame(law: str, rows: int = 150, noise: float = 0.0, seed: int = 0) -> tuple[pd.DataFrame, sp.Expr]:
    rng = np.random.default_rng(seed)
    expr = sp.sympify(law)
    names = sorted(str(s) for s in expr.free_symbols)
    X = rng.uniform(1.0, 5.0, (rows, len(names)))
    y = np.asarray(sp.lambdify([sp.Symbol(n) for n in names], expr, "numpy")(*X.T), dtype=float)
    y = y + noise * float(np.sqrt(np.mean(y ** 2))) * rng.standard_normal(rows)
    return pd.DataFrame({**{n: X[:, i] for i, n in enumerate(names)}, "target": y}), expr


def _same_function(text: str, law: sp.Expr, frame: pd.DataFrame, tolerance: float) -> bool:
    names = [c for c in frame.columns if c != "target"]
    values = np.asarray(sp.lambdify([sp.Symbol(n) for n in names], sp.sympify(text), "numpy")(*frame[names].to_numpy().T), dtype=float)
    truth = np.asarray(sp.lambdify([sp.Symbol(n) for n in names], law, "numpy")(*frame[names].to_numpy().T), dtype=float)
    return bool(np.sqrt(np.mean((values - truth) ** 2)) <= tolerance * np.std(truth))


def test_a_law_built_from_remembered_parts_is_proposed() -> None:
    memory = _memory("2.0*a/(a + 1.5) + 0.5*b", "1.1*exp(-0.3*a)*b", "0.9*sin(1.3*a) + b**2")
    frame, law = _frame("0.8*exp(-0.6*x0) + 1.9*x1/(x1 + 2.2)")          # a new law from two parts met in different laws
    proposer = sm.StructureMemoryProposer(memory, seconds=40.0)
    proposals = proposer.fit_and_propose(frame, target_col="target")
    assert 1 <= len(proposals) <= 3 and proposer.proposer_id == "structure_memory"
    assert _same_function(proposals[0].formula, law, frame, 1e-5)
    assert "exp" in proposals[0].formula and proposer.last_run["laws_remembered"] == 3
    assert all(sp.sympify(p.formula).free_symbols == {sp.Symbol("x0"), sp.Symbol("x1")} for p in proposals)
    values = proposals[0].compute_score(frame)
    assert values.shape == (len(frame),) and np.allclose(values, frame["target"], rtol=1e-4, atol=1e-6)


def test_a_frequency_is_found_by_scanning_and_not_by_luck() -> None:
    memory = _memory("0.9*sin(1.3*a) + b**2")
    frame, law = _frame("1.4*sin(3.4*x0)", rows=200, noise=0.01)           # far from every one of the plain starts
    proposals = sm.StructureMemoryProposer(memory, seconds=30.0).fit_and_propose(frame, target_col="target")
    assert proposals and _same_function(proposals[0].formula, law, frame, 0.02)
    sine = sm._compiled(["sin(C*v)"])["sin(C*v)"]
    scan = sine.scan(frame["x0"].to_numpy())
    assert sine.periodic and len(scan) == sm.HALF_PERIODS and min(abs(f - 3.4) for f in scan) < 0.25
    assert not sm._compiled(["exp(C*v)"])["exp(C*v)"].periodic and not sm._compiled(["v**2"])["v**2"].params


def test_the_same_law_in_other_variables_is_recalled_whole() -> None:
    memory = _memory("2.3*p/(p + 1.7) - 0.7*exp(-0.45*q)")
    frame, law = _frame("1.1*x1/(x1 + 0.6) - 2.0*exp(-0.9*x0)")
    names = ["x0", "x1"]
    recalled = sm.remembered_whole(memory, names, frame[names].to_numpy(), frame["target"].to_numpy())
    assert recalled and _same_function(str(recalled[0][1]), law, frame, 1e-6)
    assert sm.remembered_whole(memory, ["x0"], frame[["x0"]].to_numpy(), frame["target"].to_numpy()) == []   # one variable short


def test_what_was_never_met_cannot_be_proposed() -> None:
    memory = _memory("2.0*a + 0.5*b")                                         # only the plain variable is known
    frame, law = _frame("0.8*exp(-0.6*x0) + 1.9*x1/(x1 + 2.2)")
    proposals = sm.StructureMemoryProposer(memory, seconds=20.0).fit_and_propose(frame, target_col="target")
    assert not any(_same_function(p.formula, law, frame, 1e-3) for p in proposals)
    assert sm.StructureMemoryProposer(sm.Memory(), seconds=5.0).fit_and_propose(frame, target_col="target") == []


def test_growing_stops_on_time_and_reports_what_it_did() -> None:
    memory = _memory("2.0*a/(a + 1.5) + 0.5*b", "1.1*exp(-0.3*a)*b", "0.9*sin(1.3*a) + b**2", "log(a + 2.0)*sqrt(b + 0.3)")
    frame, _law = _frame("tanh(x0)*x1 + x2/x0", rows=120)
    names = ["x0", "x1", "x2"]
    best, motifs, report = sm.grow(memory, frame[names].to_numpy(), frame["target"].to_numpy(), seconds=1.5)
    assert report["stopped"] in ("out of time", "grown to the limit", "an exact fit") and report["fits"] > 0
    assert sm.CONSTANT in motifs and report["motifs"] == len(motifs) - 1
    assert best == sorted(best, key=lambda g: g.bic)
    for grown in best:
        assert grown.steps[0][0] == "start" and all(step[0] in sm.JOINS for step in grown.steps[1:])
        assert sum(step[1] == sm.CONSTANT for step in grown.steps) <= 1       # one intercept at most


@pytest.mark.parametrize("join,expected", [("add", "x0 + 2.0*x1"), ("multiply", "x0*x1"), ("times", "x0*(x1 + 2.0)"),
                                           ("divide", "x0/(x1 + 2.0)")])
def test_each_join_writes_what_it_computes(join: str, expected: str) -> None:
    motifs = sm._compiled(["v"])
    steps = (("start", "v", 0), (join, "v", 1))
    theta = (1.0,) if join == "multiply" else (1.0, 2.0)
    grown = sm.Grown(steps, theta, 0.0, 0.0)
    X = np.random.default_rng(0).uniform(1.0, 5.0, (20, 2))
    law = sm.written(grown, ["x0", "x1"], motifs)
    assert sp.simplify(law - sp.sympify(expected)) == 0
    assert np.allclose(sm._values(steps, np.array(theta), X, motifs), sp.lambdify(sp.symbols("x0 x1"), law, "numpy")(*X.T))


# ---------------------------------------------------------------- the stage, the memory file, the runner
def test_the_memory_takes_turns_with_the_searches_and_stands_alone_without_them() -> None:
    memory = _memory("2.0*a/(a + 1.5) + 0.5*b", "1.1*exp(-0.3*a)*b")
    frame, _law = _frame("0.8*exp(-0.6*x0) + 1.9*x1/(x1 + 2.2)")
    frame = frame.rename(columns={"target": "__target__"})
    both = memory_stage.with_memory(FixedFormulasProposer(["x0 + x1", "x0*x1"]), memory, seconds=20.0)
    formulas = [p.formula for p in both.fit_and_propose(frame, target_col="__target__")]
    assert formulas[0] == "x0 + x1" and "x0*x1" in formulas and len(formulas) > 2      # the searches' best stays first
    assert any("exp" in f for f in formulas[1:])
    assert both.last_run["generator"]["laws_remembered"] == 2
    alone = memory_stage.with_memory(None, memory, seconds=20.0)
    assert alone.proposer_id == "structure_memory" and alone.fit_and_propose(frame, target_col="__target__")


def test_the_recorded_memory_holds_shapes_only_and_what_its_builder_gives() -> None:
    builder = _load("build_structure_memory")
    record = json.loads(builder.OUT.read_text(encoding="utf-8"))
    assert record == builder.build()
    memory = memory_stage.recorded_memory()
    assert memory.laws == record["laws"] == sum(record["sources"].values()) and memory.laws > 200
    assert all(count >= builder.MIN_LAWS for count in memory.motifs.values()) and len(memory.motifs) < record["motifs_met"]
    for shape in [*memory.motifs, *memory.skeletons]:
        assert not sp.sympify(shape, locals={"C": sm.C}).atoms(sp.Float), shape        # no constant's value was kept
    assert {"v", "1/v", "v**2"} <= set(memory.motifs)


def test_the_blind_laws_are_new_to_the_benchmark_and_to_the_development_set() -> None:
    runner = _load("run_memory_laws")
    grammar = _load("build_invented_laws")
    development = _load("run_search_evolution")
    record = json.loads(runner.LAWS_FILE.read_text(encoding="utf-8"))
    assert len(record["laws"]) == runner.LAW_COUNT == 40 and record["seed"] == runner.LAW_SEED
    in_benchmark = set(json.loads(grammar.FINGERPRINTS.read_text(encoding="utf-8"))["fingerprints"])
    developed_on = {grammar.structure(sp.sympify(law["text"])) for law in development.draw_laws(80)}
    structures = [grammar.structure(sp.sympify(law["text"])) for law in record["laws"]]
    assert len(set(structures)) == 40 and not set(structures) & developed_on
    assert not {grammar.fingerprint(law["text"]) for law in record["laws"]} & in_benchmark
    assert len(runner.units()) == 80 and set(runner.entries()) == {f"memory_{law['id']}" for law in record["laws"]}
    assert runner.WRONG_LIMIT == 0.05


def _result_row(arm: str, noise: float, problem: int, *, symbolic: bool, accurate: bool = True, reported: bool = True) -> dict:
    return {"arm": arm, "noise": noise, "problem": f"memory_{problem}", "status": "ok", "symbolic": symbolic,
            "accuracy_solution": accurate, "model": "x0" if reported else None, "seconds": 1.0}


def test_the_memory_runs_summary_judges_the_expectations_as_written() -> None:
    runner = _load("run_memory_laws")
    rows = []
    for noise in runner.NOISE_LEVELS:
        for problem in range(10):
            rows.append(_result_row("default", noise, problem, symbolic=problem < 4))
            rows.append({**_result_row("memory", noise, problem, symbolic=problem < 6), "memory": {"the_law_among_them": problem < 5}})
    summary = runner.summarise(rows)
    e = summary["expectations"]
    assert e["1_holds"] and e["2_holds"] and e["2_recovered_memory"] == {"0.0": 6, "0.01": 6}
    assert summary["run_for_run"]["memory against default|0.0"] == {"gained": 2, "lost": 0}
    assert summary["memory|all"]["memory_handed_over_the_law"] == 10 and summary["default|all"]["recovered"] == 8
    tied = [dict(r, symbolic=(int(r["problem"].split("_")[1]) < 4)) if r["arm"] == "memory" else r for r in rows]
    assert not runner.summarise(tied)["expectations"]["2_holds"]                       # no more than the default: not a gain
    worse = [dict(r) for r in rows]
    for row in [r for r in worse if r["arm"] == "memory" and not r["symbolic"]][:2]:
        row["accuracy_solution"] = False                                               # two wrong of twenty: over the limit
    assert not runner.summarise(worse)["expectations"]["1_holds"]
