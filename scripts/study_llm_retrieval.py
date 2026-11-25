#!/usr/bin/env python3
"""Development: does a language model propose the law earlier when shown laws the system solved on tables like this one?

Status: audit-only

**Development, not a result.** Nothing was fixed beforehand; one model, one run, a ceiling on calls.
The measurement is ``apedp_dyn/recursion/order_score.py``: on fresh problems of the curriculum, the
place of the first candidate the judge would count solved in the list a proposer hands over.

Three lists for each problem, of at most ``WANTED`` laws:

* ``model alone``: a hosted language model is shown rows of the table (numbers only, inputs named
  ``x0, x1, ...``) and names laws with free constants, most likely first; the constants are fitted;
* ``model with cases``: the same prompt, with the laws of the ``CASES`` solved tables that looked
  most like this one (``apedp_dyn/proposers/experience.py``), constants removed;
* ``cases alone``: those cases' laws refitted on this table, nearest first, with no model at all.

The cases come from a teaching set drawn apart from the test problems: no test problem's structure
is among them, so a list can hold the law only by composing parts, not by reading it back. The
teaching laws are the generator's own: an upper bound on what the same number of laws confirmed by
the referee would give.

Every list is kept with its fitted laws and what the judge made of each, so that lists can be
joined or ordered otherwise afterwards without another call (the first run, of 25 November, kept
only the places).

Two calls a problem; ``--max-calls`` is a hard ceiling. The key is read from the environment
(``NVIDIA_API_KEY`` or ``CODING_MODEL_API_KEY``) and is written nowhere.

    NVIDIA_API_KEY=... python scripts/study_llm_retrieval.py --out DIR [--problems 60] [--max-calls 200]
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import signal
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from apedp_dyn.proposers import experience, llm_proposer  # noqa: E402
from apedp_dyn.recursion import order_score  # noqa: E402
from apedp_dyn.recursion.curriculum import Problem  # noqa: E402

TAG = "DEV-LLM-RETRIEVAL-2026-11"
PROBLEMS, TEACHING = 60, 320
WANTED, CASES, ROWS_SHOWN = 10, 8, 40
MAX_CALLS = 200
MAX_CONSTANTS = 10
SECONDS_A_LAW = 60                 # reading and fitting one proposed law; a law that takes longer is counted unread
ARMS = ("model alone", "model with cases", "cases alone")
#: strongest first; the first that answers is used (apedp_dyn.proposers.llm_clients.pick_model)
MODELS = ("nvidia/nemotron-3-ultra-550b-a55b", "nvidia/nemotron-3.5-lightning-30b-a3b", "minimaxai/minimax-m3.1-flash")


def _measure_script() -> Any:
    spec = importlib.util.spec_from_file_location("measure_order_score", ROOT / "scripts" / "measure_order_score.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def problems_and_store(count: int = PROBLEMS, teaching: int = TEACHING) -> tuple[list[Problem], experience.Experience]:
    """The first ``count`` test problems of the order-score set, and the teaching set as solved cases."""
    script = _measure_script()
    tests = script.test_problems(script.PROBLEMS)
    store = experience.Experience()
    for problem in script.teaching_problems(teaching, tests):
        rows = problem.rows()
        store.add(problem.key, problem.text, rows["X"], rows["y"])
    return tests[:count], store


def prompt(X: np.ndarray, y: np.ndarray, cases: list[experience.Case] | None = None) -> str:
    """Rows of the table spread over the target's range, the form the reply must take, and (if given) solved cases."""
    names = [f"x{i}" for i in range(X.shape[1])]
    picked = np.argsort(y, kind="stable")[np.linspace(0, len(y) - 1, min(ROWS_SHOWN, len(y))).round().astype(int)]
    lines = [
        f"Here are {len(picked)} measurements of y for the inputs {', '.join(names)}, taken from a table of {len(y)} rows.",
        f"Find the law y = f({', '.join(names)}).",
        f"Write up to {WANTED} candidate laws, the most likely first, one per line.",
        f"Use only {', '.join(names)}, free constants c0, c1, ... (each is fitted to the table for you), the operators "
        "+ - * / **, the functions sqrt exp log sin cos tanh, and small whole numbers as exponents.",
        "Reply with the expressions only, no explanation.",
    ]
    if cases:
        lines += ["", "Laws that held on earlier tables which looked like this one, constants removed:",
                  *[experience.skeleton(case.law) for case in cases],
                  "The law of this table need not be one of them; it is often built from the same kinds of parts."]
    lines += ["", f"Rows ({', '.join([*names, 'y'])}):", *[", ".join(f"{v:.4g}" for v in (*X[i], y[i])) for i in picked]]
    return "\n".join(lines)


@contextlib.contextmanager
def _limit(seconds: int) -> Iterator[None]:
    """Give up on the block after ``seconds`` (where the platform has an alarm; elsewhere no limit)."""
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def _late(_signum: int, _frame: object) -> None:
        raise TimeoutError(f"no answer in {seconds} s")

    before = signal.signal(signal.SIGALRM, _late)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, before)


def fitted(texts: list[str], X: np.ndarray, y: np.ndarray, wanted: int = WANTED) -> tuple[list[str], int]:
    """Skeletons with their constants fitted, in the order given, each structure once; and how many could not be read."""
    out: list[str] = []
    seen: set[str] = set()
    unread = 0
    for text in texts:
        if len(out) == wanted:
            break
        try:
            with _limit(SECONDS_A_LAW):
                expr = llm_proposer.parse_skeleton(text, X.shape[1], max_constants=MAX_CONSTANTS)
                if not expr.free_symbols & {sp.Symbol(f"x{i}") for i in range(X.shape[1])}:
                    raise llm_proposer.UnsafeExpressionError("no input in it")
                key = llm_proposer.structure_key(expr)
                if key in seen:
                    continue
                constants, _error = llm_proposer.fit_constants(expr, X, y)
                law = expr.xreplace({sp.Symbol(name): sp.Float(float(f"{value:.8g}")) for name, value in constants.items()})
        except Exception:  # noqa: BLE001 - whatever cannot be read or fitted in time is counted, not raised
            unread += 1
            continue
        seen.add(key)
        out.append(str(law))
    return out, unread


def one(problem: Problem, store: experience.Experience, client: Callable[[str], str] | None) -> dict[str, Any]:
    rows = problem.rows()
    X, y = rows["X"], rows["y"]
    cases = store.nearest(X, y, k=CASES)
    out: dict[str, Any] = {"key": problem.key, "level": problem.level, "inputs": problem.inputs, "noise": problem.noise,
                           "text": problem.text, "cases": [case.key for case in cases], "arms": {}}
    lists: dict[str, tuple[list[str], int, float | None, str | None]] = {}
    for arm, shown in (("model alone", None), ("model with cases", cases)):
        if client is None:
            continue
        started = time.perf_counter()
        try:
            reply = client(prompt(X, y, shown))
        except Exception as exc:  # noqa: BLE001 - a call that fails is recorded; the ceiling on calls is raised by the caller's client
            if type(exc).__name__ == "CallLimitReachedError":
                raise
            lists[arm] = ([], 0, round(time.perf_counter() - started, 2), f"{type(exc).__name__}: {exc}"[:160])
            continue
        laws, unread = fitted(llm_proposer._replies(reply), X, y)
        lists[arm] = (laws, unread, round(time.perf_counter() - started, 2), None)
    laws, unread = fitted([experience.skeleton(case.law) for case in cases], X, y)
    lists["cases alone"] = (laws, unread, None, None)
    for arm, (laws, unread, seconds, error) in lists.items():
        row = order_score.measure(problem, laws)
        out["arms"][arm] = {"proposals": row["proposals"], "unread": unread, "seconds": seconds, "error": error,
                            **{what: row[what] for what in order_score.PLACES},
                            "candidates": [{key: item[key] for key in ("text", "r2", "bits", "accurate", "solved", "same_form")}
                                           for item in row["candidates"]]}
    return out


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def as_rows(arm: str) -> list[dict[str, Any]]:
        return [{"key": row["key"], "level": row["level"], **{what: row["arms"][arm][what] for what in order_score.PLACES}}
                for row in rows if arm in row["arms"] and row["arms"][arm]["error"] is None]
    arms = {arm: as_rows(arm) for arm in ARMS}
    read = {arm: {"problems": len(found), **{what: order_score.summarise(found, what) for what in order_score.PLACES}}
            for arm, found in arms.items() if found}
    against: dict[str, Any] = {}
    if arms["model alone"] and arms["model with cases"]:
        against["model with cases against model alone"] = {
            what: order_score.compare(arms["model alone"], arms["model with cases"], what) for what in order_score.PLACES}
    if arms["cases alone"] and arms["model with cases"]:
        against["model with cases against cases alone"] = {
            what: order_score.compare(arms["cases alone"], arms["model with cases"], what) for what in order_score.PLACES}
    return {"arms": read, "against": against}


def main(argv: list[str] | None = None, client: Callable[[str], str] | None = None, model: str | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--problems", type=int, default=PROBLEMS)
    parser.add_argument("--max-calls", type=int, default=MAX_CALLS)
    parser.add_argument("--no-model", action="store_true", help="only the list that needs no model")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    tests, store = problems_and_store(args.problems)
    timed: Any = None
    if client is None and not args.no_model:
        import run_fast_generator_loop as fast

        from apedp_dyn.proposers import llm_clients

        key, address = fast.key_and_address()
        model = llm_clients.pick_model(MODELS, api_key=key, base_url=address)
        timed = fast.TimedClient(fast.hosted_client(model, key, address), args.max_calls)
        client = timed
    rows: list[dict[str, Any]] = []
    stopped = None
    with (args.out / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for problem in tests:
            try:
                row = one(problem, store, client)
            except Exception as exc:  # noqa: BLE001 - the ceiling on calls ends the run with what was measured
                if type(exc).__name__ != "CallLimitReachedError":
                    raise
                stopped = str(exc)
                break
            rows.append(row)
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
    record = {"tag": TAG, "development": True,
              "what": "development: a language model's proposals with and without the laws of solved tables that looked like "
                      "this one; one model, one run; not a result",
              "model": model, "calls": getattr(timed, "calls", None), "max_calls": args.max_calls, "stopped": stopped,
              "problems": len(rows), "cases_in_the_store": len(store.cases), "cases_shown": CASES, "laws_asked_for": WANTED,
              **summary(rows), "rows": rows}
    (args.out / "llm_retrieval.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    for arm, block in record["arms"].items():
        solved = block["solved_place"]
        print(f"{arm:17s} solved on the list {solved['found']:3d}, first {solved['first']:3d}, within 3 {solved['within_3']:3d} "
              f"of {solved['problems']}; true form on the list {block['form_place']['found']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
