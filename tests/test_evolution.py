"""The loop in which a model rewrites a piece of the search: the check, the jail, the loop, the judge.

Everything here runs with a scripted stand-in for the model. What these show is that the machinery
does what it says: code outside the allowed subset never runs, code inside it runs where it can
reach nothing, the judge reads text and nothing else, every version is kept, and nothing is
installed. Whether a real coding model improves the search this way is not tested here.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import sympy as sp

from apedp_dyn.evolution import evolved_proposer as ep
from apedp_dyn.evolution import jail, restricted
from apedp_dyn.evolution import operator as loop
from apedp_dyn.evolution.seed_generator import SEED_GENERATOR
from apedp_dyn.grammar.proposer_interface import CandidateProposal, ProposerBase, StrategyArchetype

ROOT = Path(__file__).resolve().parents[1]
needs_linux = pytest.mark.skipif(sys.platform != "linux", reason="the jail's limits are Linux resource limits")


def _script(name: str) -> Any:
    if name not in sys.modules:
        if str(ROOT / "scripts") not in sys.path:
            sys.path.insert(0, str(ROOT / "scripts"))
        spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _wrap(body: str) -> str:
    return "def propose(X, y, names):\n" + "".join(f"    {line}\n" for line in body.splitlines())


X = np.random.default_rng(0).uniform(1.0, 5.0, (60, 2))
Y = 2.0 * X[:, 0] + 1.0
CALL = [X, Y, ["x0", "x1"]]


# ------------------------------------------------------------------ the check
def test_the_seed_and_the_stand_ins_improvement_are_inside_the_subset():
    restricted.check(SEED_GENERATOR, "propose", 3)
    restricted.check(_script("run_search_evolution").IMPROVED_GENERATOR, "propose", 3)


def test_what_a_coding_model_ordinarily_writes_is_accepted():
    code = '''
"""A docstring at the top."""
GRID = [0.5, 1.0, 2.0]


def helper(values: np.ndarray, power: int = 2) -> float:
    def inner(v):
        return float(np.sum(v ** power))
    with np.errstate(all="ignore"):
        total = inner(values)
    return total


def propose(X: np.ndarray, y: "np.ndarray", names: list[str]) -> list[str]:
    out = []
    for i, name in enumerate(names):
        try:
            coef = np.linalg.lstsq(np.column_stack([X[:, i], np.ones(len(y))]), y, rcond=None)[0]
        except Exception as problem:
            print("skipped", str(problem))
            continue
        if not np.all(np.isfinite(coef)):
            raise ValueError("not finite")
        assert len(coef) == 2
        out.append((helper(y), f"{coef[0]:.6g}*{name} + {coef[1]:.6g}"))
    ranked = sorted(out, key=lambda item: item[0])
    return [text for _score, text in ranked] if (n := len(ranked)) > 0 else []
'''.replace("_score", "score")
    restricted.check(code, "propose", 3)


@pytest.mark.parametrize("code, reason", [
    ("import os\n" + _wrap("return []"), "Import"),
    ("from os import path\n" + _wrap("return []"), "ImportFrom"),
    (_wrap("return [().__class__]"), "__class__"),
    (_wrap("return [propose.__globals__]"), "__globals__"),
    (_wrap("return [open('x')]"), "'open'"),
    (_wrap("return [eval('1')]"), "'eval'"),
    (_wrap("return [exec('1')]"), "'exec'"),
    (_wrap("return [__import__('os')]"), "underscore"),
    (_wrap("return [getattr(X, 'tofile')]"), "'getattr'"),
    (_wrap("return [globals()]"), "'globals'"),
    (_wrap("return [type(X)]"), "'type'"),
    (_wrap("np.save('f', X)\nreturn []"), "'save'"),
    (_wrap("return [np.load('f')]"), "'load'"),
    (_wrap("X.tofile('f')\nreturn []"), "'tofile'"),
    (_wrap("return [X.dumps()]"), "'dumps'"),
    (_wrap("return [X.ctypes]"), "'ctypes'"),
    (_wrap("return [np.lib]"), "'lib'"),
    (_wrap("return [np.ctypeslib]"), "'ctypeslib'"),
    (_wrap("return [np.random.default_rng(0).bit_generator]"), "'bit_generator'"),
    (_wrap("return ['{0.shape}'.format(X)]"), "'format'"),
    (_wrap("return [(x for x in names).gi_frame]"), "'gi_frame'"),
    (_wrap("np.sum = len\nreturn []"), "not assigned"),
    (_wrap("np = 3\nreturn []"), "may not be redefined"),
    ("def propose(X, y, math):\n    return []\n", "may not be redefined"),
    (_wrap("_hidden = 1\nreturn []"), "underscore"),
    ("class A:\n    pass\n" + _wrap("return []"), "ClassDef"),
    ("@staticmethod\n" + _wrap("return []"), "decorators"),
    (_wrap("global seen\nreturn []"), "Global"),
    (_wrap("del names\nreturn []"), "Delete"),
    (_wrap("yield 'x0'"), "Yield"),
    ("async def propose(X, y, names):\n    return []\n", "AsyncFunctionDef"),
    (_wrap("with open('f') as handle:\n    pass\nreturn []"), "np.errstate"),
    (_wrap("try:\n    pass\nfinally:\n    return []"), "finally"),
    (_wrap("try:\n    pass\nexcept:\n    pass\nreturn []"), "except has to name"),
    (_wrap("try:\n    pass\nexcept BaseException:\n    pass\nreturn []"), "except has to name"),
    (_wrap("try:\n    pass\nexcept np.linalg.LinAlgError:\n    pass\nreturn []"), "except has to name"),
    (_wrap("match names:\n    case _:\n        return []"), "Match"),
    (_wrap("return [b'bytes']"), "bytes"),
    ("def propose(X, y, *names):\n    return []\n", "*args"),
    ("def propose(X, y):\n    return []\n", "exactly 3"),
    ("def other(X, y, names):\n    return []\n", "exactly one function named 'propose'"),
    ("print('at the top')\n" + _wrap("return []"), "only function definitions"),
    (_wrap("return [undefined_name]"), "'undefined_name'"),
    ("def propose(X, y, names:\n", "not Python"),
    (_wrap("return []") + "# " + "x" * restricted.MAX_CHARS, "at most"),
])
def test_code_outside_the_subset_is_refused_before_anything_runs(code, reason):
    with pytest.raises(restricted.RejectedCodeError) as refusal:
        restricted.check(code, "propose", 3)
    assert reason in str(refusal.value), str(refusal.value)


def test_an_identifier_written_in_other_characters_is_read_as_what_python_reads():
    # Python folds full-width letters to ordinary ones; the check sees the folded name, as the interpreter would.
    for code, name in ((_wrap("return [\uff4f\uff50\uff45\uff4e('x')]"), "'open'"),
                       (_wrap("X.\uff54\uff4f\uff46\uff49\uff4c\uff45('f')\nreturn []"), "'tofile'")):
        with pytest.raises(restricted.RejectedCodeError) as refusal:
            restricted.check(code, "propose", 3)
        assert name in str(refusal.value)


def test_the_allowlists_hold_nothing_that_reads_writes_or_evaluates():
    never_an_attribute = {
        "load", "save", "savez", "savetxt", "loadtxt", "genfromtxt", "fromfile", "tofile", "memmap", "ctypeslib",
        "ctypes", "lib", "format", "format_map", "dump", "dumps", "tobytes", "view", "data", "base", "frompyfunc",
        "vectorize", "fromstring", "frombuffer", "testing", "f2py", "os", "sys", "bit_generator", "seed", "set_state",
        "get_state", "setflags", "getfield", "setfield", "apply_along_axis", "piecewise", "fromfunction", "r_", "c_",
        "bmat", "matrix", "DataSource", "seterr", "seterrcall", "encode", "decode", "translate", "mro", "send",
        "throw", "close", "gi_frame", "f_globals", "ndarray", "core", "char", "rec", "strings", "as_strided",
    }
    never_a_builtin = {"open", "eval", "exec", "compile", "getattr", "setattr", "delattr", "globals", "locals", "vars",
                       "type", "object", "super", "input", "breakpoint", "help", "memoryview", "classmethod",
                       "staticmethod", "property", "__import__", "dir", "format", "id"}
    assert not never_an_attribute & restricted.ALLOWED_ATTRIBUTES
    assert not never_a_builtin & restricted.ALLOWED_BUILTINS
    assert not any(name.startswith("_") for name in restricted.ALLOWED_ATTRIBUTES | restricted.ALLOWED_BUILTINS)
    assert "BaseException" not in restricted.ALLOWED_EXCEPTIONS and "KeyboardInterrupt" not in restricted.ALLOWED_EXCEPTIONS


def test_every_allowed_attribute_is_a_real_one():
    holders = [np, np.linalg, np.random, np.random.default_rng(0), np.ones(2), np.finfo(float), __import__("math"),
               [], {}, set(), "", 1.0]
    missing = sorted(name for name in restricted.ALLOWED_ATTRIBUTES if not any(hasattr(h, name) for h in holders))
    assert not missing


def test_the_rules_given_to_a_model_name_what_it_most_often_gets_wrong():
    text = restricted.describe()
    for phrase in ("No import", "underscore", "np.linalg.lstsq", "str.format", "except Exception", "print does nothing"):
        assert phrase in text


# ------------------------------------------------------------------ the jail
@needs_linux
def test_a_checked_candidate_runs_and_returns_its_text():
    code = _wrap("coef = np.linalg.lstsq(np.column_stack([X[:, 0], np.ones(len(y))]), y, rcond=None)[0]\n"
                 "print('nothing is printed')\n"
                 "return [f'{coef[0]}*{names[0]} + {coef[1]}', np.float64(1.5), np.arange(3)]")
    run = jail.run(code, "propose", [CALL, CALL], parameters=3)
    assert run.defined and run.error == "" and run.restarts == 0 and len(run.calls) == 2
    first = run.calls[0]
    assert first.ok and first.result[1] == 1.5 and first.result[2] == [0, 1, 2] and first.seconds >= 0
    expr = sp.sympify(first.result[0])
    assert abs(float(expr.coeff(sp.Symbol("x0"))) - 2.0) < 1e-9


@needs_linux
def test_ordinary_numpy_works_inside_the_jail():
    # numpy loads some helpers on first use; the jail cannot open a file, so each of these has to work regardless.
    expressions = [
        "str(X[:2])", "f'{X[0]}'", "f'{y[0]:.4g} {np.float64(2.5)} {float(y[0])!r}'", "X[:2].astype(str).tolist()",
        "X.astype(int).sum()", "np.unique(['b', 'a', 'b']).tolist()", "np.asarray([1, 2], dtype='f4').sum()",
        "np.random.default_rng(0).choice(names)", "np.linalg.lstsq(X, y, rcond=None)[0].tolist()",
        "np.einsum('ij,jk->ik', X.T, X, optimize=True).shape", "np.where(y > 3, 1.0, 0.0).sum()", "np.isfinite(X).all()",
        "np.full((2, 2), 3.0).sum()", "np.clip(y, 0, 3).max()", "str(X.dtype)", "np.sqrt(-1 + 0j).real",
        "np.quantile(y, [0.1, 0.9]).tolist()", "np.median(X, axis=0).tolist()", "np.percentile(y, 50)",
        "np.polyfit(X[:, 0], y, 2).tolist()", "np.std(y, ddof=1)", "np.corrcoef(X.T)[0, 1]",
        "np.linalg.svd(X, full_matrices=False)[1].tolist()", "np.linalg.pinv(X).shape", "np.sort(y)[0]",
        "np.argsort(y, kind='stable')[0]", "np.interp(2.0, [1, 3], [10, 30])", "np.cumsum(y)[-1]", "np.cov(X.T)[0, 0]",
        "np.linalg.cond(X)", "np.linalg.matrix_rank(X)", "np.gradient(y)[0]", "np.finfo(float).eps", "math.erf(0.3)",
        "np.polyfit([1, 1, 1], [1, 2, 3], 2)[0]", "(np.ones(2) / np.zeros(2))[0]", "np.nanmean(np.array([1.0, np.nan]))",
    ]
    body = "out = []\n" + "".join(f"out.append(str({e}))\n" for e in expressions) + "return out"
    run = jail.run(_wrap(body), "propose", [CALL], parameters=3)
    assert run.defined and run.calls[0].ok, run.calls[0].error if run.calls else run.error
    assert len(run.calls[0].result) == len(expressions)


@needs_linux
def test_the_helper_that_fetches_numpys_parts_hands_out_nothing_else():
    code = ("def f():\n    fetch = f.__builtins__['__import__']\n    out = []\n"
            "    for name in ('os', 'socket', 'subprocess', 'numpy.f2py', 'sys', 'builtins'):\n"
            "        try:\n            fetch(name)\n            out.append(name)\n"
            "        except Exception:\n            pass\n    return out\n")
    run = jail.run_unchecked(code, "f", [[]])
    assert run.defined and run.calls[0].ok and run.calls[0].result == []
    with pytest.raises(restricted.RejectedCodeError, match="underscore"):
        restricted.check(_wrap("return [__import__]"), "propose", 3)


@needs_linux
def test_type_annotations_are_never_evaluated():
    code = "def propose(X: [][0], y: undefined_thing, names: 1 / 0) -> open('nothing'):\n    return ['x0']\n"
    restricted.check(code, "propose", 3)
    run = jail.run(code, "propose", [CALL], parameters=3)
    assert run.defined and run.calls[0].ok and run.calls[0].result == ["x0"]


@needs_linux
def test_a_call_that_runs_over_its_time_fails_and_the_next_one_still_runs():
    code = _wrap("if names[0] == 'slow':\n    k = 0\n    while True:\n        k += 1\nreturn ['x0']")
    started = time.perf_counter()
    run = jail.run(code, "propose", [[X, Y, ["slow"]], CALL], parameters=3, seconds_per_call=0.5)
    assert [c.ok for c in run.calls] == [False, True] and "no answer within 0.5 s" in run.calls[0].error
    assert time.perf_counter() - started < 10 and run.restarts == 0


@needs_linux
def test_catching_exception_does_not_catch_the_time_limit():
    code = _wrap("try:\n    while True:\n        k = 1\nexcept Exception:\n    return ['caught']\nreturn []")
    run = jail.run(code, "propose", [CALL], parameters=3, seconds_per_call=0.5)
    assert not run.calls[0].ok and "no answer" in run.calls[0].error


@needs_linux
def test_a_call_stuck_inside_one_operation_ends_its_process_and_the_rest_run_in_another():
    # sum over a huge range never returns to Python, so no exception can reach it; only the CPU limit can.
    code = _wrap("if names[0] == 'stuck':\n    return [str(sum(range(10 ** 12)))]\nreturn ['x0']")
    run = jail.run(code, "propose", [CALL, [X, Y, ["stuck"]], CALL], parameters=3, seconds_per_call=0.5)
    assert [c.ok for c in run.calls] == [True, False, True] and run.restarts == 1
    assert "killed after" in run.calls[1].error


@needs_linux
def test_memory_is_capped_and_the_process_survives_it():
    code = _wrap("if names[0] == 'greedy':\n    return [str(np.ones((40000, 40000)).sum())]\nreturn ['x0']")
    run = jail.run(code, "propose", [[X, Y, ["greedy"]], CALL], parameters=3, memory_bytes=1024 ** 3)
    assert not run.calls[0].ok and "MemoryError" in run.calls[0].error and run.calls[1].ok


@needs_linux
def test_nothing_carries_over_from_one_call_to_the_next():
    code = ("seen = []\n\n\n" + _wrap("seen.append(1)\nreturn [len(seen), float(np.random.normal()), "
                                      "float(np.random.default_rng(1).uniform())]"))
    first, second = jail.run(code, "propose", [CALL] * 3, parameters=3), jail.run(code, "propose", [CALL], parameters=3)
    assert [c.result for c in first.calls] == [first.calls[0].result] * 3 and first.calls[0].result[0] == 1
    assert second.calls[0].result == first.calls[0].result


@needs_linux
def test_an_answer_too_large_to_be_a_law_is_not_read():
    run = jail.run(_wrap("return ['x' * 300000]"), "propose", [CALL], parameters=3)
    assert not run.calls[0].ok and "at most" in run.calls[0].error


@needs_linux
def test_a_candidate_that_cannot_be_defined_is_reported_and_nothing_is_called():
    run = jail.run("TABLE = [1, 2, 3][7]\n\n\n" + _wrap("return []"), "propose", [CALL], parameters=3)
    assert not run.defined and run.calls == [] and "IndexError" in run.error


@needs_linux
def test_the_jail_refuses_unchecked_code_through_run():
    with pytest.raises(restricted.RejectedCodeError):
        jail.run("import os\n" + _wrap("return []"), "propose", [CALL], parameters=3)


def test_the_jailed_process_imports_nothing_of_this_repository_and_compiles_as_the_check_assumes():
    import ast

    from apedp_dyn.evolution import _jail_child

    tree = ast.parse(jail.CHILD.read_text(encoding="utf-8"))
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    imported |= {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert not any(name.startswith(("apedp", "scripts")) for name in imported)
    assert imported <= {"__future__", "builtins", "json", "math", "os", "resource", "signal", "sys", "time", "typing",
                        "linecache", "traceback", "warnings", "numpy", "numpy.fft", "numpy.linalg", "numpy.ma",
                        "numpy.polynomial", "numpy.random"}
    assert jail.CHILD == Path(_jail_child.__file__) and "future.annotations.compiler_flag" in jail.CHILD.read_text()
    assert _jail_child.MAX_RESULT_CHARS == 200_000 and issubclass(_jail_child._OutOfTime, BaseException)
    assert not issubclass(_jail_child._OutOfTime, Exception)
    _jail_child._alarm(14, None)                        # with no candidate running, the alarm raises nothing
    _jail_child._RUNNING[0] = True
    with pytest.raises(_jail_child._OutOfTime):
        _jail_child._alarm(14, None)
    assert _jail_child._RUNNING == [False]              # and it raises once


def _unchecked(body: str) -> jail.CallResult:
    """The second barrier alone: code the check would refuse, given the whole of numpy's module to climb out of."""
    code = "def f():\n    get = np.__builtins__['__import__']\n" + "".join(f"    {line}\n" for line in body.splitlines())
    run = jail.run_unchecked(code, "f", [[]])
    assert run.defined, run.error
    return run.calls[0]


@needs_linux
def test_the_process_alone_stops_writing_reading_and_the_network():
    written = _unchecked("np.save('leak.npy', np.ones(3))\nreturn 'written'")
    assert not written.ok and "Too many open files" in written.error
    read = _unchecked("return get('builtins').open('/etc/hostname').read()")
    assert not read.ok and "Too many open files" in read.error
    by_numpy = _unchecked("return str(np.loadtxt('/etc/hostname', dtype=str))")
    assert not by_numpy.ok
    socket = _unchecked("get('socket').socket()\nreturn 'socket made'")
    assert not socket.ok


@needs_linux
def test_the_process_alone_cannot_start_another_process():
    forked = _unchecked("return get('os').fork()")
    assert not forked.ok, "the jailed process forked"


@needs_linux
def test_the_process_sees_none_of_the_callers_environment_and_works_in_a_directory_that_is_gone_afterwards(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "not-a-real-key")
    monkeypatch.setenv("CODING_MODEL_API_KEY", "not-a-real-key-either")
    monkeypatch.setenv("GITHUB_TOKEN", "nor-this")
    seen = _unchecked("os = get('os')\nreturn [sorted(os.environ), os.getcwd(), list(os.environ.values())]")
    assert seen.ok
    names, directory, values = seen.result
    assert set(names) <= {"LC_CTYPE", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "PYTHONHASHSEED",
                          "LD_LIBRARY_PATH"}
    assert not any("not-a-real-key" in value or "nor-this" in value for value in values)
    assert Path(directory).name.startswith("jail-") and not Path(directory).exists()
    assert str(ROOT) not in directory and directory.startswith(tempfile.gettempdir())
    assert "NVIDIA_API_KEY" not in jail.environment() and "CODING_MODEL_API_KEY" not in jail.environment()


# ------------------------------------------------------------------ the door from a candidate to the judge
@pytest.mark.parametrize("text, kept", [
    ("2.5*x0 + 1", True), ("exp(-1.3*x0)/(0.7 + x1)", True), ("(x0 + 1)**-2", True), ("x0**x1", True),
    ("x0**(2/3)", True), ("log(x0)*sin(x1)^2", True), ("1e-05*x0 + -0.5", True),
    ("9**9**9**9", False), ("x0**(9**9)", False), ("x0**(x1 + 9**9**9)", False), ("(9**9)**x0", False),
    ("x0**13", False), ("2**x0", False), ("c0*x0", False), ("3.5", False), ("x2", False),
    ("__import__('os').system('id')", False), ("x0.real", False), ("x0 if x1 else 2", False), ("x0 +", False),
    ("lambda: x0", False), ("x0; x1", False), ("x0*" + "+".join(["1.5"] * 14), False), ("x0 + " * 200 + "x1", False),
])
def test_only_expressions_the_whitelist_can_read_safely_pass(text, kept):
    started = time.perf_counter()
    expressions, refused = ep.read_candidates([text], 2)
    assert (len(expressions), refused) == ((1, 0) if kept else (0, 1))
    assert time.perf_counter() - started < 5.0


def test_what_is_not_a_list_of_text_is_refused_and_only_the_first_few_are_read():
    assert ep.read_candidates("x0", 2) == ([], 1)
    assert ep.read_candidates(None, 2) == ([], 1)
    expressions, refused = ep.read_candidates([1, None, "x0", "x1", "x0 + x1", "x0*x1"], 2)
    assert [str(e) for e in expressions] == ["x0", "x1"] and refused == 2 and ep.MAX_CANDIDATES == 4


# ------------------------------------------------------------------ the loop, on a judge that needs no jail
class Counts:
    """A judge for the loop's own logic: a version's score is the number it returns in ``SCORE = n``."""

    contract = loop.Contract("propose", 3, "propose(X, y, names)", "Return candidate laws.", "A note.")

    def __init__(self) -> None:
        self.seen: list[str] = []

    def evaluate(self, code: str) -> loop.Evaluation:
        self.seen.append(code)
        try:
            restricted.check(code, "propose", 3)
        except restricted.RejectedCodeError as exc:
            return loop.Evaluation(False, error=f"refused before running: {exc}")
        if "BROKEN" in code:
            return loop.Evaluation(False, error="failed on every problem; the first: IndexError")
        score = float(code.split("SCORE = ")[1].split()[0]) if "SCORE = " in code else 0.0
        return loop.Evaluation(True, (score,), f"score {score:g}", f"  missed: HELD_BACK_NEVER_SHOWN is not here; {score:g}")


def _version(score: float, extra: str = "") -> str:
    return f"SCORE = {score:g}\n{extra}\n\ndef propose(X, y, names):\n    return [names[0]]\n"


def _reply(code: str, thought: str = "a change") -> str:
    return f"THINK: {thought}\n```python\n{code}```"


class Scripted:
    def __init__(self, replies: list[str]) -> None:
        self.replies, self.prompts = list(replies), []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self.replies:
            raise RuntimeError("unreachable")
        return self.replies.pop(0)


def test_a_refused_version_is_repaired_and_every_attempt_is_kept():
    model = Scripted([_reply("import os\n" + _version(5)), _reply(_version(5, "BROKEN = 1")), _reply(_version(5), "works now")])
    task, archive = Counts(), loop.Archive()
    seed = archive.add(None, "seed", _version(1), Counts().evaluate(_version(1)))
    added = loop.vary(model, task, archive, seed, step=1, repairs=2)
    assert [v.evaluation.ran for v in added] == [False, False, True] and [v.repairs for v in added] == [0, 1, 2]
    assert "Import is not allowed" in model.prompts[1] and "IndexError" in model.prompts[2]
    assert archive.best() is added[-1] and added[-1].parent == seed.id and added[-1].thought == "a change"
    assert len(archive.versions) == 4 and loop.summary(archive)["refused_or_failed"] == 2


def test_the_prompt_holds_the_contract_the_rules_the_lineage_and_the_parent():
    model = Scripted([_reply(_version(2))])
    archive = loop.Archive()
    seed = archive.add(None, "seed", _version(1), Counts().evaluate(_version(1)))
    loop.vary(model, Counts(), archive, seed, steps_left=4)
    prompt = model.prompts[0]
    for part in ("propose(X, y, names)", "Return candidate laws.", "A note.", "No import", "version 0 (the seed): score 1",
                 "SCORE = 1", "missed:", "Variations left after this one: 4", "THINK:"):
        assert part in prompt, part


def test_a_repeated_version_is_not_scored_again_and_no_code_ends_the_variation():
    model = Scripted([_reply(_version(1)), "I have no idea.", "Still none."])
    task, archive = Counts(), loop.Archive()
    seed = archive.add(None, "seed", _version(1), task.evaluate(_version(1)))
    assert loop.vary(model, task, archive, seed, repairs=2) == []
    assert len(task.seen) == 1 and "exactly version 0" in model.prompts[1] and "no code" in model.prompts[2]


def test_a_model_that_cannot_be_reached_ends_a_variation_and_never_the_run():
    archive = loop.evolve(Scripted([]), Counts(), _version(1), steps=5)
    assert len(archive.versions) == 1 and archive.best().origin == "seed"


def test_the_run_starts_from_the_seed_rotates_its_parents_and_writes_every_version(tmp_path):
    model = Scripted([_reply(_version(3)), _reply(_version(2)), _reply(_version(9)), _reply(_version(4)), _reply(_version(5))])
    seen: list[int] = []
    archive = loop.evolve(model, Counts(), _version(1), steps=5, parents=3, on_version=lambda v: seen.append(v.id))
    assert seen == [0, 1, 2, 3, 4, 5]
    # step k starts from the (k mod 3)-th best so far: the seed while it is the only one, the second of two (the
    # seed), the third of three (the seed), then the best of four (version 3), then the second best (version 4)
    assert [v.parent for v in archive.versions] == [None, 0, 0, 0, 3, 4]
    assert archive.best().evaluation.score == (9.0,)
    archive.write(tmp_path / "versions.jsonl")
    rows = [json.loads(line) for line in (tmp_path / "versions.jsonl").read_text().splitlines()]
    assert [r["id"] for r in rows] == seen and all(len(r["sha256"]) == 64 and r["code"] for r in rows)
    assert loop.summary(archive) == {"versions": 6, "ran": 6, "refused_or_failed": 0, "by_model": 5, "repaired": 0,
                                     "best": {"id": 3, "origin": "model", "score": [9.0], "summary": "score 9",
                                              "sha256": archive.versions[3].sha256}, "seed_score": [1.0]}


def test_a_deadline_ends_the_run():
    model = Scripted([_reply(_version(k)) for k in range(2, 30)])
    archive = loop.evolve(model, Counts(), _version(1), steps=25, seconds=0.0)
    assert len(archive.versions) == 1


def test_the_lineage_names_the_leading_versions_their_changes_and_the_latest_refusals():
    model = Scripted([_reply(_version(3), "added ratios"), _reply("import os\n" + _version(4)), "none", "none"])
    archive = loop.evolve(model, Counts(), _version(1), steps=2, repairs=1)
    text = archive.lineage()
    assert text.splitlines()[0].startswith("  version 1 (from version 0): score 3") and 'change: "added ratios"' in text
    assert "1 versions were refused or did not run" in text and "Import is not allowed" in text


@pytest.mark.parametrize("reply, code", [
    ("THINK: x\n```python\ndef propose(X, y, names):\n    return []\n```", "def propose(X, y, names):\n    return []\n"),
    ("```\ndef propose(X, y, names):\n    return []\n```", "def propose(X, y, names):\n    return []\n"),
    ("first\n```python\ndef a():\n    pass\n```\nthen\n```py\ndef propose(X, y, names):\n    return []\n```", "def propose(X, y, names):\n    return []\n"),
    ("Here it is:\ndef propose(X, y, names):\n    return []", "def propose(X, y, names):\n    return []\n"),
    ("I have no idea.", None),
])
def test_the_code_of_a_reply_is_its_last_fenced_block(reply, code):
    assert loop.extract_code(reply) == code


def test_a_replys_code_is_only_ever_parsed_in_the_callers_process():
    # A reply whose code would mark the caller if anything of it ran here.
    marker = "MARK = [].append"
    task = Counts()
    archive = loop.evolve(Scripted([_reply(f"{marker}\nimport sys\nsys.modules['marked_by_candidate'] = 1\n" + _version(7))]),
                          task, _version(1), steps=1, repairs=0)
    assert "marked_by_candidate" not in sys.modules and not archive.versions[1].evaluation.ran


# ------------------------------------------------------------------ the judge of candidate generators
def _problems() -> list[ep.Problem]:
    rng = np.random.default_rng(3)
    x0, x1 = sp.symbols("x0 x1")
    out = []
    for name, truth, inputs in (("reachable", 2.0 * x0 ** 2 + 1.5 / x1, 2), ("inside", 1.7 * sp.exp(-0.7 * x0), 1),
                                ("ratio", 2.3 * x0 / (0.45 + x0) / (1.3 + x1), 2)):
        rows = rng.uniform(1.0, 5.0, (260, inputs))
        values = np.asarray(sp.lambdify([x0, x1][:inputs], truth, "numpy")(*rows.T), dtype=float)
        noisy = values[:160] + 0.01 * float(np.sqrt(np.mean(values ** 2))) * rng.standard_normal(160)
        out.append(ep.Problem(name, rows[:160], noisy, rows[160:], truth))
    return out


@needs_linux
def test_the_judge_counts_the_laws_a_version_proposes_and_names_the_ones_it_missed():
    script = _script("run_search_evolution")
    task = ep.CandidateGeneratorTask(_problems(), script.recovered, seconds_per_call=3.0)
    seed = task.evaluate(SEED_GENERATOR)
    assert seed.ran and seed.score[0] == 1.0 and seed.details["problems"]["reachable"]["solved"]
    assert "2 problems where the law was not among the candidates" in seed.feedback and "exp(-0.7*x0)" in seed.feedback
    better = task.evaluate(script.IMPROVED_GENERATOR)
    assert better.ran and better.score[0] >= 2.0 and better.details["problems"]["inside"]["solved"]
    assert better.score > seed.score and "law proposed in" in better.summary


@needs_linux
def test_the_judge_reports_a_refused_a_failing_and_a_useless_version_without_raising():
    script = _script("run_search_evolution")
    task = ep.CandidateGeneratorTask(_problems()[:2], script.recovered, seconds_per_call=1.0)
    refused = task.evaluate("import os\n" + _wrap("return []"))
    assert not refused.ran and refused.error.startswith("refused before running")
    failing = task.evaluate(_wrap("return [names[len(names)]]"))
    assert not failing.ran and "failed on every problem" in failing.error and "IndexError" in failing.error
    useless = task.evaluate(_wrap("return [\"__import__('os').system('id')\", '9**9**9**9', 'c0*' + names[0], 7]"))
    assert useless.ran and useless.score[:2] == (0.0, 0.0) and useless.details["refused_strings"] == 8
    assert "not readable as expressions" in useless.feedback


@needs_linux
def test_a_version_becomes_a_proposer_that_decides_nothing():
    frame = pd.DataFrame({"mass": X[:, 0], "speed": X[:, 1], "target": 2.0 * X[:, 0] ** 2 + 1.5 / X[:, 1]})
    proposer = ep.EvolvedProposer(SEED_GENERATOR)
    proposals = proposer.fit_and_propose(frame, target_col="target")
    assert 1 <= len(proposals) <= ep.MAX_CANDIDATES and all(isinstance(p, CandidateProposal) for p in proposals)
    best = proposals[0]
    assert "mass" in best.formula and "speed" in best.formula and "x0" not in best.formula
    assert np.allclose(best.compute_score(frame), frame["target"], rtol=1e-6)
    assert best.metadata["selection_pool_size"] == len(proposals) and best.candidate_id.startswith("evolved:")
    assert proposer.last_run["candidates"] == len(proposals) and proposer.last_run["defined"]
    with pytest.raises(restricted.RejectedCodeError):
        ep.EvolvedProposer("import os\n" + _wrap("return []"))
    broken = ep.EvolvedProposer(_wrap("return [names[len(names)]]"))
    assert list(broken.fit_and_propose(frame, target_col="target")) == [] and "IndexError" in broken.last_run["error"]


class _Searches(ProposerBase):
    """A stand-in for the default searches: three fixed candidates."""

    def __init__(self) -> None:
        self.last_run = {"members": 1}

    @property
    def proposer_id(self) -> str:
        return "searches"

    def fit_and_propose(self, train_data: Any, target_col: str | None = None, **kwargs: Any) -> list[CandidateProposal]:
        return [CandidateProposal(proposer_id="searches", candidate_id=f"s{i}", formula=f"{i + 2}*mass",
                                  strategy_archetype=StrategyArchetype.TACTICAL, declared_scale_invariant=False,
                                  compute_score=lambda df, i=i: (i + 2) * df["mass"].to_numpy(), parameters={}, metadata={})
                for i in range(3)]


@needs_linux
def test_the_stage_hands_on_the_searches_candidates_and_the_generators_in_turn(monkeypatch):
    from apedp_dyn.discovery_l2 import evolved_stage

    monkeypatch.setattr(evolved_stage, "default_search_stage", lambda budget, cores=None, seed=0: (_Searches(), "stub searches"))
    stage, described = evolved_stage.evolved_search_stage(SEED_GENERATOR, 5.0)
    frame = pd.DataFrame({"mass": X[:, 0], "speed": X[:, 1], "__target__": 2.0 * X[:, 0] ** 2 + 1.5 / X[:, 1]})
    proposals = stage.fit_and_propose(frame, target_col="__target__")
    assert [p.proposer_id for p in proposals[:4]] == ["searches", "evolved_proposer", "searches", "evolved_proposer"]
    assert len(proposals) <= evolved_stage.MAX_PROPOSALS and "jailed candidate generator" in described
    assert stage.last_run["generator"]["candidates"] >= 1 and stage.last_run["searches"] == {"members": 1}
    with pytest.raises(restricted.RejectedCodeError):
        evolved_stage.evolved_search_stage("import os\n" + _wrap("return []"), 5.0)


# ------------------------------------------------------------------ the harness
def test_the_training_laws_are_drawn_the_same_way_each_time_and_none_has_a_benchmark_structure():
    script, builder = _script("run_search_evolution"), _script("build_invented_laws")
    laws = script.draw_laws(24)
    assert laws == script.draw_laws(24) and len({law["name"] for law in laws}) == 24
    in_benchmark = set(json.loads(builder.FINGERPRINTS.read_text())["fingerprints"])
    assert len(in_benchmark) == builder.COUNT and not {builder.fingerprint(law["text"]) for law in laws} & in_benchmark
    assert script.PROBLEM_SEED != builder.SEED
    record = json.loads(builder.OUT.read_text())
    assert {builder.fingerprint(law["text"]) for law in record["laws"]} == in_benchmark
    assert not any(law["text"] in builder.FINGERPRINTS.read_text() for law in record["laws"])


def test_a_problem_gives_the_candidate_noisy_rows_and_the_judge_clean_ones():
    script = _script("run_search_evolution")
    law = script.draw_laws(3)[1]
    first, again = script.problem_of(law), script.problem_of(law)
    assert first.X.shape == (script.ROWS, law["inputs"]) and first.X_test.shape == (script.TEST_ROWS, law["inputs"])
    assert np.array_equal(first.y, again.y) and first.names == [f"x{i}" for i in range(law["inputs"])]
    clean = np.asarray(sp.lambdify([sp.Symbol(n) for n in first.names], first.truth, "numpy")(*first.X.T), dtype=float)
    spread = float(np.sqrt(np.mean((first.y - clean) ** 2)) / np.sqrt(np.mean(clean ** 2)))
    assert 0.005 < spread < 0.02


def test_no_run_with_a_real_model_starts_without_one_and_the_key_goes_nowhere_but_the_client(monkeypatch, capsys):
    script = _script("run_search_evolution")
    for name in ("CODING_MODEL", "CODING_MODEL_API_KEY", "CODING_MODEL_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(SystemExit, match="No coding model has been provided"):
        script.coding_model()
    monkeypatch.setenv("CODING_MODEL", "some/coder")
    monkeypatch.setenv("CODING_MODEL_API_KEY", "not-a-real-key")
    monkeypatch.setenv("CODING_MODEL_BASE_URL", "https://models.example/v1")
    from apedp_dyn.proposers import llm_https

    given: dict[str, Any] = {}
    monkeypatch.setattr(llm_https, "nvidia_client", lambda model, **options: given.update(model=model, **options) or (lambda p: ""))
    _client, name = script.coding_model()
    assert name == "some/coder" and given["api_key"] == "not-a-real-key" and given["base_url"] == "https://models.example/v1"
    assert given["thinking"] is None
    assert "not-a-real-key" not in capsys.readouterr().out


def test_a_plain_http_service_is_refused(monkeypatch):
    script = _script("run_search_evolution")
    monkeypatch.setenv("CODING_MODEL", "some/coder")
    monkeypatch.setenv("CODING_MODEL_API_KEY", "not-a-real-key")
    monkeypatch.setenv("CODING_MODEL_BASE_URL", "http://models.example/v1")
    with pytest.raises(ValueError, match="https"):
        script.coding_model()


@needs_linux
def test_the_loop_with_the_scripted_stand_in_ends_with_a_ledger_a_champion_and_nothing_installed(tmp_path):
    script = _script("run_search_evolution")
    model = script.ScriptedModel()
    report = script.run_loop(model, tmp_path, steps=3, train=8, validation=8, seconds_per_call=1.0, seconds=None,
                             model="scripted stand-in (no model)")
    rows = [json.loads(line) for line in (tmp_path / "versions.jsonl").read_text().splitlines()]
    assert [(r["origin"], r["ran"]) for r in rows] == [("seed", True), ("model", False), ("model", False), ("model", True),
                                                       ("model", False), ("model", True)]
    assert "Import is not allowed" in rows[1]["error"] and "IndexError" in rows[2]["error"]
    assert "no answer within" in rows[4]["error"] and rows[5]["details"]["refused_strings"] == 8 * 4
    assert report["versions"] == 6 and report["by_model"] == 5 and report["repaired"] == 1 and report["adopted"] is False
    assert report["model_calls"] == len(script.DEMO_REPLIES) == 7
    assert "was not accepted" in model.prompts[4] and "no answer within" in model.prompts[4]
    assert "exactly version 3" in model.prompts[5] and model.prompts[6].startswith("You are improving")
    # the champion is chosen on the validation laws, among versions that ran, and is only a text file
    champion = report["champion"]
    assert champion["id"] in (0, 3) and (tmp_path / "champion.txt").read_text() == rows[champion["id"]]["code"]
    assert {row["id"] for row in report["leading_versions_on_validation"]} == {0, 3, 5}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["champion.txt", "summary.json", "versions.jsonl"]
    # no validation law was ever in a prompt
    validation = [str(p.truth) for p in script.tasks(8, 8, 1.0)[1].problems]
    assert not any(law in prompt for law in validation for prompt in model.prompts)


def test_a_version_through_the_path_is_an_arm_of_its_own_and_its_rows_carry_the_codes_hash(monkeypatch):
    script, dev = _script("run_search_evolution"), _script("run_invented_laws")
    jobs = script.path_jobs("champion")
    assert len(jobs) == 20 * len(dev.NOISE_LEVELS) and {job["arm"] for job in jobs} == {"champion"}
    asked: list[tuple[dict[str, Any], float]] = []
    monkeypatch.setattr(dev, "run_one", lambda job, laws, budget: asked.append((job, budget)) or {**job, "status": "ok"})
    monkeypatch.setattr(dev, "EXTRA_STAGES", {})
    rows = script.run_path(SEED_GENERATOR, jobs[:2], 40.0)
    assert [job for job, _budget in asked] == jobs[:2] and {budget for _job, budget in asked} == {40.0}
    assert list(dev.EXTRA_STAGES) == ["champion"] and callable(dev.EXTRA_STAGES["champion"])
    assert {row["code_sha256"] for row in rows} == {hashlib.sha256(SEED_GENERATOR.encode()).hexdigest()}
    with pytest.raises(ValueError, match="own arms"):
        script.run_path(SEED_GENERATOR, [{"problem": "invented_9000", "noise": 0.0, "arm": "default"}], 40.0)
    assert script.read_code("seed") == SEED_GENERATOR


def test_the_development_harness_takes_a_further_arm_and_refuses_an_unknown_one():
    dev = _script("run_invented_laws")
    assert "evolved" not in dev.ARMS and isinstance(dev.EXTRA_STAGES, dict)
    with pytest.raises(ValueError, match="unknown arm"):
        dev.run_arm("no_such_arm", {"names": ["x0"], "X_train": np.ones((10, 1)), "y_train": np.ones(10)},
                    {"name": "invented_9000"})
