"""The invented laws: what the generator keeps, and that the committed file is what it gives."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]


def _builder():
    name = "build_invented_laws"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def test_the_committed_file_is_what_the_generator_gives():
    b = _builder()
    assert b.main(["--check"]) == 0
    record = json.loads(b.OUT.read_text(encoding="utf-8"))
    laws = record["laws"]
    assert len(laws) == b.COUNT and [law["id"] for law in laws] == list(range(9000, 9000 + b.COUNT))
    assert [law["split"] for law in laws].count("development") == b.DEVELOPMENT
    assert all(law["split"] == ("development" if i < b.DEVELOPMENT else "held_out") for i, law in enumerate(laws))
    assert len({b.structure(sp.sympify(law["text"])) for law in laws}) == b.COUNT


def test_every_law_is_finite_uses_its_inputs_and_is_no_low_polynomial():
    b = _builder()
    rng = np.random.default_rng(1)
    for law in json.loads(b.OUT.read_text(encoding="utf-8"))["laws"]:
        expr = sp.sympify(law["text"])
        symbols = [sp.Symbol(f"x{i}") for i in range(law["inputs"])]
        assert expr.free_symbols == set(symbols) and b.MIN_COST <= law["cost"] == b.cost(expr) <= b.MAX_COST
        assert not (expr.is_polynomial(*symbols) and sp.Poly(expr, *symbols).total_degree() <= 2)
        X = rng.uniform(b.LOW, b.HIGH, (200, law["inputs"]))
        y = np.asarray(sp.lambdify(symbols, expr, "numpy")(*X.T), dtype=float)
        assert np.all(np.isfinite(y)) and np.std(y) > 0


def test_a_law_with_a_part_no_data_could_reveal_is_not_kept():
    b = _builder()
    x0 = sp.Symbol("x0")
    rng = np.random.default_rng(2)
    flat = 0.45 * x0 ** 2 / (sp.tanh(3.7 * x0) + 1.7)        # the tanh is one on the whole domain
    assert not b.acceptable(flat, 1, rng)
    assert b.acceptable(0.45 * x0 ** 2 / (sp.tanh(0.3 * x0) + 1.7), 1, rng)
    assert not b.acceptable(2.0 * x0 ** 2 + 0.3 * x0, 1, rng)                 # a polynomial of degree two
    assert not b.acceptable(sp.Float(1.3) * sp.sin(0.7 * x0), 2, rng)         # does not use its second input


def test_structures_ignore_the_constants():
    b = _builder()
    x0 = sp.Symbol("x0")
    assert b.structure(1.3 * sp.sin(0.7 * x0)) == b.structure(2.9 * sp.sin(0.3 * x0))
    assert b.structure(1.3 * sp.sin(0.7 * x0)) != b.structure(1.3 * sp.cos(0.7 * x0))


# --- the development harness -----------------------------------------------------------------------------

def _harness():
    for name in ("run_srbench_baseline", "run_srbench_main_search", "run_invented_laws"):
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
    return sys.modules["run_invented_laws"]


def test_the_harness_reads_the_development_laws_and_no_held_back_one():
    h = _harness()
    development = h.entries()
    assert len(development) == 20 and all(name.startswith("invented_90") for name in development)
    assert max(int(name.split("_")[1]) for name in development) == 9019
    assert len(h.all_jobs()) == 20 * len(h.NOISE_LEVELS) * len(h.ARMS)
    assert not {job["problem"] for job in h.all_jobs()} & set(h.entries("held_out"))
    text = (ROOT / "scripts" / "run_invented_laws.py").read_text(encoding="utf-8")
    assert 'entries("held_out")' not in text and "split=\"held_out\"" not in text


def test_a_laws_rows_are_reproducible_and_follow_the_law():
    h = _harness()
    entry = h.entries()["invented_9002"]
    frame = h.frame_of(entry)
    assert frame.equals(h.frame_of(entry)) and len(frame) == 1500 and list(frame.columns) == ["x0", "target"]
    assert frame["x0"].between(1.0, 5.0).all()
    truth = sp.lambdify(sp.Symbol("x0"), sp.sympify(entry["formula"]), "numpy")
    assert np.allclose(frame["target"], truth(frame["x0"].to_numpy()))


def test_an_arm_with_a_scripted_model_is_run_scored_and_summarised(monkeypatch):
    h = _harness()

    class Scripted:
        calls = 0

        def __call__(self, prompt: str) -> str:
            Scripted.calls += 1
            if "No data is given" in prompt:
                return "unknown"
            if "exact fit" in prompt or Scripted.calls > 3:
                return "SUBMIT: c0*tanh(c1*x0)"
            return "THINK: it levels off\nFIT: c0*x0/(c1 + x0)\nFIT: c0*tanh(c1*x0)"

    monkeypatch.setattr(h, "_client", lambda: Scripted())
    row = h.run_one({"problem": "invented_9002", "noise": 0.01, "arm": "agent_alone"}, h.entries(), 5.0, 120.0)
    assert row["status"] == "ok" and row["symbolic"] and row["accuracy_solution"] and row["stage"] == "the agent alone"
    assert row["agent"]["sessions"][0]["stopped"] == "submitted" and row["truth"] == "2.9*tanh(0.3*x0)"
    cell = h.summarise([row])["agent_alone|0.01"]
    assert cell["symbolic"] == 1 and cell["reported_wrong"] == 0 and cell["sessions_submitted"] == 1
    assert cell["structures_by_agent"] == 2 and cell["median_seconds"] == row["seconds"]
    with np.errstate(all="ignore"):
        bad = h.run_one({"problem": "invented_9002", "noise": 0.01, "arm": "no_such_arm"}, h.entries(), 5.0, 60.0)
    assert bad["status"] == "error" and "unknown arm" in bad["error"]


def _harness():
    name = "run_invented_laws"
    if name not in sys.modules:
        if str(ROOT / "scripts") not in sys.path:
            sys.path.insert(0, str(ROOT / "scripts"))
        spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def test_a_study_record_keeps_the_summary_and_says_which_arm_recovered_what_the_other_did_not():
    dev = _harness()

    def row(arm, problem, noise, symbolic, model="x0"):
        return {"arm": arm, "problem": problem, "noise": noise, "status": "ok", "seconds": 1.0, "symbolic": symbolic,
                "accuracy_solution": True, "model": model, "formula": model, "truth": "x0", "candidates": ["bulk"],
                "agent": {"sessions": [{"fits": 3, "stopped": "submitted"}], "by_agent": 2, "structures_tried": 5,
                          "last_reply": "bulk"}}

    rows = [row("default", "a", 0.0, True), row("agentic", "a", 0.0, True), row("default", "b", 0.0, False),
            row("agentic", "b", 0.0, True), row("default", "c", 0.01, True), row("agentic", "c", 0.01, False, None)]
    record = dev.study_record(rows, {"where": "a test"})
    assert record["recovered_by_one_arm_and_not_the_other"] == {"default and not agentic": ["c at noise 0.01"],
                                                                "agentic and not default": ["b at noise 0"]}
    assert len(record["runs"]) == 6 and all("candidates" not in run and "agent" not in run for run in record["runs"])
    assert record["runs"][0]["sessions"] == [[3, "submitted"]] and record["provenance"] == {"where": "a test"}
    assert record["summary"]["agentic|all"]["symbolic"] == 2
    assert [run["reported"] for run in record["runs"] if run["problem"] == "c"] == [False, True]    # agentic, default


def test_the_recorded_development_run_says_what_the_documents_say():
    record = json.loads((ROOT / "benchmarks" / "agentic_variation_dev_study.json").read_text(encoding="utf-8"))
    summary = record["summary"]
    assert len(record["runs"]) == 120 and record["provenance"]["github_run"] == 44861670227
    assert [summary[f"{arm}|all"]["symbolic"] for arm in ("default", "agentic", "agent_alone")] == [13, 16, 1]
    assert [summary[f"{arm}|all"]["reported"] for arm in ("default", "agentic", "agent_alone")] == [26, 27, 12]
    assert all(summary[f"{arm}|all"]["reported_wrong"] == 0 for arm in ("default", "agentic", "agent_alone"))
    assert round(summary["default|all"]["median_seconds"]) == 277 and round(summary["agentic|all"]["median_seconds"]) == 504
    paired = record["recovered_by_one_arm_and_not_the_other"]
    assert len(paired["agentic and not default"]) == 4 and paired["default and not agentic"] == ["invented_9008 at noise 0.01"]
    assert "not a preregistered result" in record["what"]
