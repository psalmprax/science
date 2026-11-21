#!/usr/bin/env python3
"""A model rewrites the search's candidate generator; a fixed judge scores every version.

Development harness for ``apedp_dyn.evolution``: **not a preregistered run, and no real model has
driven it.** It exists so that, when a coding model is provided, one command runs the loop and the
checks around it.

    python scripts/run_search_evolution.py baseline --out DIR     # the seed on the training and validation laws
    python scripts/run_search_evolution.py demo     --out DIR     # the loop driven by a scripted stand-in: no model, no key
    python scripts/run_search_evolution.py evolve   --out DIR     # the loop driven by a real coding model (see below)
    python scripts/run_search_evolution.py score    --out DIR --code FILE --arm NAME --shard k/N
    python scripts/run_search_evolution.py collect  --out DIR

**Problems.** ``TRAIN + VALIDATION`` laws drawn by the invented-law generator
(``build_invented_laws``) with another seed. A structure that occurs in the benchmark is left out,
by its fingerprint: no law of the benchmark is read, held back or not. The model is shown what a
version missed on the training laws; the validation laws are never shown to it and only choose the
champion among the leading versions.

**The judge** (``evolved_proposer.CandidateGeneratorTask``): on each law, 300 rows with 1 % noise
go to the candidate in its jail; it returns up to four candidate laws as text; the law counts as
proposed if one of them is the law by the benchmark's own criterion
(``run_srbench_baseline.symbolic_recovery``) on 200 clean rows.

**A real model** is whatever serves chat completions in OpenAI's format: ``CODING_MODEL`` (its
name), ``CODING_MODEL_API_KEY``, and ``CODING_MODEL_BASE_URL`` unless it is NVIDIA's service. The
key is read from the environment, handed to the client and written nowhere.

**What a run produces, and what it does not.** A ledger of every version scored, with code, score
and parent; the champion's code as a text file; nothing else. No version is installed: ``score``
runs a version through the refined path beside the default searches on the twenty development
laws, and a version enters the route only by a reviewed commit, after one preregistered test on
laws it was never scored on.

**The risk this cannot remove by itself.** Training and validation laws come from one grammar.
A version can get better at that grammar without getting better at laws; a test that settles the
question has to use laws from elsewhere.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import build_invented_laws as grammar  # noqa: E402
import run_invented_laws as dev  # noqa: E402
import run_srbench_baseline as base  # noqa: E402

from apedp_dyn.evolution import operator as loop  # noqa: E402
from apedp_dyn.evolution.evolved_proposer import CandidateGeneratorTask, Problem  # noqa: E402
from apedp_dyn.evolution.seed_generator import SEED_GENERATOR  # noqa: E402

TAG = "DEV-SEARCH-EVOLUTION-2026-11"
PROBLEM_SEED = 20261122         # not the benchmark's
TRAIN, VALIDATION = 40, 40
ROWS, TEST_ROWS = 300, 200
NOISE = 0.01
SECONDS_PER_CALL = 3.0
STEPS = 30
REPAIRS = 2
CHAMPION_AMONG = 5              # leading versions by training score that the validation laws choose between


def draw_laws(count: int, seed: int = PROBLEM_SEED) -> list[dict[str, Any]]:
    """``count`` laws from the invented-law grammar whose structures are not in the benchmark."""
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
        if key in seen or not grammar.acceptable(expr, inputs, rng):
            continue
        seen.add(key)
        if grammar.fingerprint(str(expr)) in in_benchmark:
            continue
        laws.append({"name": f"training_{len(laws):03d}", "inputs": inputs, "text": str(expr)})
    if len(laws) != count:
        raise SystemExit(f"only {len(laws)} laws after {grammar.MAX_DRAWS} draws")
    return laws


def problem_of(law: dict[str, Any]) -> Problem:
    """Rows on the laws' domain: noisy targets for the candidate, clean rows for the judge."""
    symbols = [sp.Symbol(f"x{i}") for i in range(law["inputs"])]
    truth = sp.sympify(law["text"], locals={str(s): s for s in symbols})
    rng = np.random.default_rng(base._seed(TAG, law["name"], "rows"))
    X = rng.uniform(grammar.LOW, grammar.HIGH, (ROWS + TEST_ROWS, law["inputs"]))
    y = np.asarray(sp.lambdify(symbols, truth, "numpy")(*X.T), dtype=float) * np.ones(len(X))
    noisy = y[:ROWS] + rng.standard_normal(ROWS) * NOISE * float(np.sqrt(np.mean(y[:ROWS] ** 2)))
    return Problem(law["name"], X[:ROWS], noisy, X[ROWS:], truth)


def recovered(expr: sp.Expr, problem: Problem) -> bool:
    """The benchmark's own criterion for 'this is the law'."""
    return bool(base.symbolic_recovery(expr, problem.truth, problem.names, problem.X_test)["recovered"])


def tasks(train: int = TRAIN, validation: int = VALIDATION, seconds_per_call: float = SECONDS_PER_CALL
          ) -> tuple[CandidateGeneratorTask, CandidateGeneratorTask]:
    """The judge on the training laws (whose misses the model is shown) and on the validation laws (never shown)."""
    problems = [problem_of(law) for law in draw_laws(train + validation)]
    return (CandidateGeneratorTask(problems[:train], recovered, seconds_per_call=seconds_per_call),
            CandidateGeneratorTask(problems[train:], recovered, seconds_per_call=seconds_per_call))


def choose_champion(archive: loop.Archive, validation: CandidateGeneratorTask, among: int = CHAMPION_AMONG
                    ) -> tuple[loop.Version | None, list[dict[str, Any]]]:
    """Score the leading versions (and the seed) on the validation laws; the best there is the champion."""
    leading = archive.ranked()[:among]
    seed = archive.versions[0] if archive.versions else None
    if seed is not None and seed.evaluation.ran and seed not in leading:
        leading.append(seed)
    scored: list[dict[str, Any]] = []
    for rank, version in enumerate(leading):
        evaluation = validation.evaluate(version.code)
        scored.append({"id": version.id, "origin": version.origin, "sha256": version.sha256, "training_rank": rank,
                       "training_score": list(version.evaluation.score), "validation_ran": evaluation.ran,
                       "validation_score": list(evaluation.score), "validation_summary": evaluation.summary})
    usable = [row for row in scored if row["validation_ran"]]
    if not usable:
        return None, scored
    best = min(usable, key=lambda row: (tuple(-s for s in row["validation_score"]), row["training_rank"]))
    return archive.versions[best["id"]], scored


def run_loop(client: Callable[[str], str], out: Path, *, steps: int, train: int, validation: int,
             seconds_per_call: float, seconds: float | None, model: str) -> dict[str, Any]:
    """The loop from the seed, its ledger written as it goes, the champion chosen on the validation laws."""
    out.mkdir(parents=True, exist_ok=True)
    training_task, validation_task = tasks(train, validation, seconds_per_call)
    archive = loop.Archive()
    started = time.perf_counter()

    def record(version: loop.Version) -> None:
        archive.write(out / "versions.jsonl")
        e = version.evaluation
        print(f"version {version.id} ({version.origin}, step {version.step}, repair {version.repairs}): "
              f"{e.summary if e.ran else 'did not run: ' + e.error[:160]}", flush=True)

    loop.evolve(client, training_task, SEED_GENERATOR, steps=steps, repairs=REPAIRS, seconds=seconds, archive=archive,
                on_version=record)
    champion, scored = choose_champion(archive, validation_task)
    report = {"tag": TAG, "model": model, "steps": steps, "training_laws": train, "validation_laws": validation,
              "seconds_per_call": seconds_per_call, "seconds": round(time.perf_counter() - started, 1),
              "model_calls": int(getattr(client, "calls", 0) or 0), **loop.summary(archive),
              "leading_versions_on_validation": scored,
              "champion": None if champion is None else {"id": champion.id, "origin": champion.origin,
                                                         "sha256": champion.sha256},
              "adopted": False,
              "note": "Nothing here is installed. The champion's training and validation scores were used to choose "
                      "it and are not a measurement of it."}
    if champion is not None:
        (out / "champion.txt").write_text(champion.code, encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return report


# ------------------------------------------------------------------ a scripted stand-in for a coding model
IMPROVED_GENERATOR = '''\
def fit(columns, y):
    A = np.column_stack(columns + [np.ones(len(y))])
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    rest = y - A.dot(coef)
    return coef, float(np.sqrt(np.mean(rest ** 2)))


def written(coef, terms):
    parts = [f"{coef[i]}*{terms[i]}" for i in range(len(terms))]
    return " + ".join(parts + [f"{coef[len(terms)]}"])


def best_inside(make, x, y):
    """The constant inside a shape that lets the shape alone fit y best, from a grid."""
    best = None
    for a in np.geomspace(0.2, 5.0, 14):
        values = make(a, x)
        if not np.all(np.isfinite(values)):
            continue
        coef, err = fit([values], y)
        if best is None or err < best[0]:
            best = (err, float(a))
    return best[1] if best is not None else 1.0


def shapes(name, x, y):
    """Fixed shapes of one variable, and shapes with a constant inside chosen from a grid."""
    out = [(name, x), (f"{name}**2", x ** 2), (f"{name}**3", x ** 3), (f"1/{name}", 1.0 / x)]
    families = [("exp(-{a}*{v})", lambda a, x: np.exp(-a * x)), ("log(1 + {a}*{v})", lambda a, x: np.log(1.0 + a * x)),
                ("sin({a}*{v})", lambda a, x: np.sin(a * x)), ("{v}/({a} + {v})", lambda a, x: x / (a + x)),
                ("sqrt({a} + {v})", lambda a, x: np.sqrt(a + x)), ("tanh({a}*{v})", lambda a, x: np.tanh(a * x))]
    for text, make in families:
        a = best_inside(make, x, y)
        out.append((text.replace("{a}", f"{a:.4g}").replace("{v}", name), make(a, x)))
    return out


def propose(X, y, names):
    found = []
    per_variable = [shapes(names[i], X[:, i], y) for i in range(X.shape[1])]
    sums = [[]]
    for options in per_variable:
        sums = [combo + [shape] for combo in sums for shape in options]
    for combo in sums:
        coef, err = fit([values for text, values in combo], y)
        if np.all(np.isfinite(coef)):
            found.append((err, written(coef, [text for text, values in combo])))
        if len(combo) > 1:                      # the same shapes multiplied instead of added
            product = np.ones(len(y))
            for text, values in combo:
                product = product * values
            coef, err = fit([product], y)
            if np.all(np.isfinite(coef)):
                found.append((err, written(coef, ["*".join(f"({text})" for text, values in combo)])))
    found.sort()
    return [text for err, text in found[:4]]
'''
_BLOCK = "```python\n{code}```"
DEMO_REPLIES = (
    # step 1: refused by the check, then fails when run, then an improvement
    "THINK: enumerate products with itertools.\n" + _BLOCK.format(
        code="import itertools\n\ndef propose(X, y, names):\n    return [names[0]]\n"),
    "THINK: no import this time.\n" + _BLOCK.format(
        code="def propose(X, y, names):\n    return [names[len(names)]]\n"),
    "THINK: fit a constant inside each shape from a grid, and try products as well as sums.\n"
    + _BLOCK.format(code=IMPROVED_GENERATOR),
    # step 2: never returns; the repair repeats a version already scored, then gives up with no code
    "THINK: search longer.\n" + _BLOCK.format(
        code="def propose(X, y, names):\n    k = 0\n    while True:\n        k += 1\n    return []\n"),
    "THINK: back to the last one.\n" + _BLOCK.format(code=IMPROVED_GENERATOR),
    "I have no further idea.",
    # step 3: runs, and returns text the judge refuses to read
    "THINK: return whatever comes to mind.\n" + _BLOCK.format(
        code="def propose(X, y, names):\n    return [\"__import__('os').system('id')\", '9**9**9**9', names[0] + ' +', 'c0*' + names[0]]\n"),
)


class ScriptedModel:
    """A stand-in for a coding model: its replies in a fixed order, whatever it is asked. Then it is unreachable."""

    def __init__(self, replies: tuple[str, ...] = DEMO_REPLIES) -> None:
        self._replies = list(replies)
        self.calls = 0
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self.calls >= len(self._replies):
            raise RuntimeError("the scripted model has no further reply")
        self.calls += 1
        return self._replies[self.calls - 1]


def coding_model() -> tuple[Callable[[str], str], str]:
    """The model named by the environment, or a clear refusal to start without one."""
    name, key = os.environ.get("CODING_MODEL", "").strip(), os.environ.get("CODING_MODEL_API_KEY", "").strip()
    if not name or not key:
        raise SystemExit("No coding model has been provided. Set CODING_MODEL (its name) and CODING_MODEL_API_KEY, and "
                         "CODING_MODEL_BASE_URL unless it is NVIDIA's service. Nothing was run.")
    from apedp_dyn.proposers import llm_clients, llm_https

    client = llm_https.nvidia_client(
        name, api_key=key, base_url=os.environ.get("CODING_MODEL_BASE_URL", "").strip() or llm_clients.NVIDIA_BASE_URL,
        thinking=None, max_tokens=int(os.environ.get("CODING_MODEL_MAX_TOKENS", "8192")),
        min_interval=float(os.environ.get("LLM_INTERVAL_SECONDS", "2.0")), retries=3, timeout=600.0)
    return client, name


# ------------------------------------------------------------------ a version through the refined path
def read_code(source: str) -> str:
    return SEED_GENERATOR if source == "seed" else Path(source).read_text(encoding="utf-8")


def path_jobs(arm: str) -> list[dict[str, Any]]:
    """Every development law at both noise levels, for the arm that carries one version of ``propose``."""
    return [{"problem": name, "noise": noise, "arm": arm} for name in dev.entries() for noise in dev.NOISE_LEVELS]


def run_path(code: str, jobs: list[dict[str, Any]], budget: float, handle: Any = None) -> list[dict[str, Any]]:
    """Each job through ``discover_refined`` with the generator beside the default searches."""
    from apedp_dyn.discovery_l2 import evolved_stage

    digest = hashlib.sha256(code.encode()).hexdigest()
    for arm in {job["arm"] for job in jobs}:
        if arm in dev.ARMS:
            raise ValueError(f"{arm!r} is one of the harness's own arms")
        dev.EXTRA_STAGES[arm] = lambda seconds: evolved_stage.evolved_search_stage(code, seconds, seed=base.SEED)
    laws = dev.entries()
    rows = []
    for job in jobs:
        row = {**dev.run_one(job, laws, budget), "code_sha256": digest}
        rows.append(row)
        if handle is not None:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
            handle.flush()
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("command", choices=("baseline", "demo", "evolve", "score", "collect"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--train", type=int, default=TRAIN)
    parser.add_argument("--validation", type=int, default=VALIDATION)
    parser.add_argument("--seconds-per-call", type=float, default=SECONDS_PER_CALL)
    parser.add_argument("--seconds", type=float, default=None, help="wall-clock limit of the whole loop")
    parser.add_argument("--code", default="seed", help="for score: 'seed' or a file holding a version of propose")
    parser.add_argument("--arm", default="seed_generator", help="for score: what the rows call this version")
    parser.add_argument("--shard", default="0/1", help="for score: k/N")
    parser.add_argument("--budget", type=float, default=dev.SEARCH_BUDGET_S, help="for score: seconds of search")
    parser.add_argument("--limit", type=int, default=0, help="for score: only the first jobs (a smoke)")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == "baseline":
        training_task, validation_task = tasks(args.train, args.validation, args.seconds_per_call)
        of_seed: dict[str, dict[str, Any]] = {}
        for label, task in (("training", training_task), ("validation", validation_task)):
            evaluation = task.evaluate(SEED_GENERATOR)
            of_seed[label] = {"ran": evaluation.ran, "score": list(evaluation.score), "summary": evaluation.summary,
                              "error": evaluation.error}
            print(f"seed on the {label} laws: {evaluation.summary or evaluation.error}", flush=True)
        (args.out / "baseline.json").write_text(json.dumps(of_seed, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        return 0 if all(r["ran"] for r in of_seed.values()) else 1
    if args.command in ("demo", "evolve"):
        if args.command == "demo":
            client: Callable[[str], str] = ScriptedModel()
            model, steps = "scripted stand-in (no model)", 3
        else:
            client, model = coding_model()
            steps = args.steps
        report = run_loop(client, args.out, steps=steps, train=args.train, validation=args.validation,
                          seconds_per_call=args.seconds_per_call, seconds=args.seconds, model=model)
        print(json.dumps({k: v for k, v in report.items() if k != "leading_versions_on_validation"}, indent=1))
        for scored in report["leading_versions_on_validation"]:
            print(f"  on the validation laws, version {scored['id']}: {scored['validation_summary']}")
        return 0
    if args.command == "score":
        code = read_code(args.code)
        k, n = (int(v) for v in args.shard.split("/"))
        jobs = [job for position, job in enumerate(path_jobs(args.arm)) if position % n == k]
        jobs = jobs[: args.limit] if args.limit else jobs
        path = args.out / f"path-shard-{args.arm}-{k:03d}-of-{n:03d}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            rows = run_path(code, jobs, args.budget, handle)
        for done in rows:
            print(dev._printable(done), flush=True)
        return 1 if args.limit and any(done["status"] != "ok" for done in rows) else 0
    collected = [json.loads(line) for shard in sorted(args.out.glob("path-shard-*.jsonl"))
                 for line in shard.read_text().splitlines()]
    (args.out / "path_runs.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in collected),
                                              encoding="utf-8")
    per_arm = dev.summarise(collected)
    (args.out / "path_results.json").write_text(json.dumps(
        {"tag": TAG, "rows": len(collected), "code_sha256": sorted({r.get("code_sha256", "") for r in collected}),
         "summary": per_arm}, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"collected {len(collected)} runs through the path into {args.out}")
    print(json.dumps(per_arm, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
