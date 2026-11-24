#!/usr/bin/env python3
"""Development: does the learning loop improve the route over a stream of new laws?

**Development, not a result.** ``apedp_dyn/discovery_l2/learning_loop.py`` lets
the route learn from the laws its referee confirms. This script runs it over a
stream of laws it has never seen, one after the other, in two arms from the
same starting state:

* ``learning``: after each law, what the route reported and the last checks
  vouched for is learned (``learning_loop.learnable``);
* ``frozen``: the same route and state, never learning.

Laws are drawn from the invented-law grammar with a seed used nowhere else,
leaving out every structure used before (``run_vouching_v2_blind.used`` and
that run's own laws). Each law gets 1,000 rows (700 to fit on, 300 in the pool)
at noise 0 or 1 % in turn, and 400 clean rows to score on. No search engine
runs: the candidates come from the first pass, ordered by the law model, the
polynomial library, and the memory. Self-improvement shows as more laws
recovered later in the stream by the learning arm than by the frozen one, with
no more wrong laws.

The laws come from one grammar, so what is learned is that grammar's parts.
Real fields repeat their parts too (physics laws share motifs), but a gain here
says nothing yet about the physics benchmark.

    python scripts/run_learning_stream.py run --laws 40 --out DIR --memory-seconds 20 [--empty-memory]

With ``--empty-memory`` both arms start remembering nothing: the stream is a field
the system has not met, and only the learning arm can build a memory of it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import build_invented_laws as grammar  # noqa: E402
import run_srbench_baseline as base  # noqa: E402
import run_vouching_v2_blind as blind  # noqa: E402

from apedp_dyn.discovery_l2 import learning_loop, memory_stage  # noqa: E402
from apedp_dyn.proposers.structure_memory import Memory  # noqa: E402
from apedp_mmlm.ledger_learning import parse_law  # noqa: E402

STREAM_SEED = 20261124
ROWS, POOL, TEST_ROWS = 700, 300, 400
NOISES = (0.0, 0.01)
ARMS = ("learning", "frozen")


def taken() -> set[str]:
    """Every structure used in an earlier run, the blind test of the second version's laws included."""
    out: set[str] = set(blind.used())
    made = json.loads(blind.LAWS_FILE.read_text(encoding="utf-8"))
    for law in made["handed"]["laws"]:
        out.add(grammar.structure(sp.sympify(law["law"])))
    for law in made["searched"]:
        out.add(grammar.structure(sp.sympify(law["text"])))
    return out


def draw_stream(count: int, seed: int = STREAM_SEED) -> list[dict[str, Any]]:
    used = taken()
    in_benchmark = set(json.loads(grammar.FINGERPRINTS.read_text(encoding="utf-8"))["fingerprints"])
    rng = np.random.default_rng(seed)
    laws: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _ in range(grammar.MAX_DRAWS):
        if len(laws) == count:
            break
        inputs = int(rng.integers(1, 4))
        expr = grammar.draw(rng, inputs)
        key = grammar.structure(expr)
        if key in seen or key in used or not grammar.acceptable(expr, inputs, rng):
            continue
        seen.add(key)
        if grammar.fingerprint(str(expr)) in in_benchmark:
            continue
        laws.append({"index": len(laws), "inputs": inputs, "text": str(expr), "noise": NOISES[len(laws) % len(NOISES)]})
    if len(laws) != count:
        raise SystemExit(f"only {len(laws)} laws")
    return laws


def seed_state(empty_memory: bool = False) -> learning_loop.LearningState:
    """The starting state of both arms: the memory in force (or none), and the law model fitted on SRBench's formulas."""
    laws = [str(base.truth_expr(entry)) for entry in base.load_manifest()]
    return learning_loop.LearningState(Memory() if empty_memory else memory_stage.recorded_memory(), laws)


def one_law(state: learning_loop.LearningState, law: dict[str, Any], learn: bool, memory_seconds: float,
            searches: float = 0.0) -> dict[str, Any]:
    names = [f"x{i}" for i in range(law["inputs"])]
    symbols = [sp.Symbol(n) for n in names]
    truth = sp.sympify(law["text"], locals=dict(zip(names, symbols, strict=True)))
    key = hashlib.sha256(f"learning-stream|{law['index']}".encode()).digest()[:8]
    rng = np.random.default_rng(int.from_bytes(key, "big"))
    X = rng.uniform(grammar.LOW, grammar.HIGH, (ROWS + POOL + TEST_ROWS, law["inputs"]))
    clean = np.asarray(sp.lambdify(symbols, truth, "numpy")(*X.T), dtype=float) * np.ones(len(X))
    y = clean + law["noise"] * float(np.sqrt(np.mean(clean[:ROWS + POOL] ** 2))) * rng.standard_normal(len(X))
    dev, pool, test = slice(0, ROWS), slice(ROWS, ROWS + POOL), slice(ROWS + POOL, None)
    pool_y = y[pool]
    tree = parse_law(law["text"])
    bits = None if tree is None else round(float(state.model.bits(tree)), 3)    # what the law cost the model before
    proposer = None
    if searches > 0:                       # the default searches, without the memory in force: the state's memory replaces it
        from apedp_dyn.discovery_l2.search_stage import default_search_stage
        proposer = default_search_stage(searches, seed=law["index"])[0]
    out = learning_loop.discover_law({n: X[dev, i] for i, n in enumerate(names)}, y[dev], X[pool],
                                     lambda rows: pool_y[rows], state=state, proposer=proposer, learn=learn,
                                     source=f"stream {law['index']}", memory_seconds=memory_seconds, seed=law["index"])
    reported = out.get("law")
    recovered = accurate = False
    if reported:
        model = sp.sympify(reported, locals=dict(zip(names, symbols, strict=True)))
        recovered = bool(base.symbolic_recovery(model, truth, names, X[test])["recovered"])
        r2 = base._r2(clean[test], np.asarray(sp.lambdify(symbols, model, "numpy")(*X[test].T), dtype=float) * np.ones(TEST_ROWS))
        accurate = bool(r2 is not None and r2 >= base.R2_ACCURACY)
    return {"index": law["index"], "noise": law["noise"], "inputs": law["inputs"], "truth": law["text"],
            "reported": bool(reported), "law": reported, "recovered": recovered,
            "wrong": bool(reported) and not recovered and not accurate, "accurate": accurate,
            "learned": out["learning"]["learned"], "why_not": out["learning"]["why_not"],
            "bits_before": bits, "seconds": out["learning"]["seconds"], "first_pass": out["learning"]["first_pass"],
            "motifs": len(state.memory.motifs)}


def run_arm(job: tuple[str, list[dict[str, Any]], float, str, bool, float]) -> list[dict[str, Any]]:
    arm, laws, memory_seconds, out_dir, empty_memory, searches = job
    state = seed_state(empty_memory)
    rows = []
    tag = f"{arm}{'-empty' if empty_memory else ''}"
    path = Path(out_dir) / f"{tag}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for law in laws:
            try:
                row = one_law(state, law, arm == "learning", memory_seconds, searches)
            except Exception as exc:  # noqa: BLE001 - a law that cannot be run is counted with its error
                row = {"index": law["index"], "noise": law["noise"], "error": f"{type(exc).__name__}: {exc}"[:200]}
            row["arm"] = arm
            rows.append(row)
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
    if arm == "learning":
        state.save(Path(out_dir) / f"{tag}_journal.json")
    return rows


def summarise(rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm, sel in rows.items():
        ok = [r for r in sel if "error" not in r]
        half = len(sel) // 2
        first, second = [r for r in ok if r["index"] < half], [r for r in ok if r["index"] >= half]
        out[arm] = {part: {"laws": len(s), "recovered": sum(r["recovered"] for r in s), "wrong": sum(r["wrong"] for r in s),
                           "reported": sum(r["reported"] for r in s), "learned": sum(r["learned"] for r in s),
                           "median_bits_before": float(np.median([r["bits_before"] for r in s if r["bits_before"] is not None]))
                           if s else None,
                           "median_seconds": float(np.median([r["seconds"] for r in s])) if s else None}
                    for part, s in (("first_half", first), ("second_half", second), ("all", ok))}
        out[arm]["errors"] = len(sel) - len(ok)
    learning = {r["index"]: r for r in rows.get("learning", []) if "error" not in r}
    frozen = {r["index"]: r for r in rows.get("frozen", []) if "error" not in r}
    both = sorted(set(learning) & set(frozen))
    out["paired"] = {"gained": [i for i in both if learning[i]["recovered"] and not frozen[i]["recovered"]],
                     "lost": [i for i in both if frozen[i]["recovered"] and not learning[i]["recovered"]]}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("run", "laws", "collect"))
    parser.add_argument("--arm", choices=(*ARMS, "both"), default="both")
    parser.add_argument("--searches", type=float, default=0.0,
                        help="seconds for the default searches beside the memory (0: no search engine)")
    parser.add_argument("--laws", type=int, default=40)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--memory-seconds", type=float, default=learning_loop.SECONDS)
    parser.add_argument("--empty-memory", action="store_true",
                        help="both arms start with no remembered law: a field the system has not met")
    args = parser.parse_args()
    laws = draw_stream(args.laws)
    if args.command == "laws":
        print(json.dumps(laws, indent=1))
        return 0
    if args.out is None:
        raise SystemExit("--out is needed")
    if args.command == "collect":
        settings: dict[str, Any] = {}
        record = {"what": "development stream of the learning loop; not a result", "laws": len(laws), "settings": settings}
        for empty in (False, True):
            suffix = "-empty" if empty else ""
            found = {arm: [json.loads(line) for line in (args.out / f"{arm}{suffix}.jsonl").read_text().splitlines()]
                     for arm in ARMS if (args.out / f"{arm}{suffix}.jsonl").exists()}
            if found:
                settings["empty memory" if empty else "memory in force"] = summarise(found)
        (args.out / "summary.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(record, indent=1))
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "laws.json").write_text(json.dumps(laws, indent=1) + "\n", encoding="utf-8")
    started = time.perf_counter()
    arms: tuple[str, ...] = ARMS if args.arm == "both" else (args.arm,)
    jobs = [(arm, laws, args.memory_seconds, str(args.out), args.empty_memory, args.searches) for arm in arms]
    if len(jobs) == 1:
        results = [run_arm(jobs[0])]
    else:
        with mp.get_context("fork").Pool(len(jobs)) as pool:
            results = pool.map(run_arm, jobs)
    summary = summarise(dict(zip(arms, results, strict=True)))
    record = {"what": "development stream of the learning loop; not a result", "laws": len(laws),
              "empty_memory": args.empty_memory,
              "memory_seconds": args.memory_seconds, "seconds": round(time.perf_counter() - started, 1), "summary": summary}
    (args.out / "summary.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
