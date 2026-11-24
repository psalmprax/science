#!/usr/bin/env python3
"""A fast hosted model drives the loop that rewrites the search's candidate generator.

Status: audit-only

**Development, not a result.** ``scripts/run_search_evolution.py`` is the loop: a model rewrites
``propose``, a fixed judge scores every version on laws it is never shown, and nothing is
installed. Until 2026-11-24 only a scripted stand-in had driven it. This script puts a hosted
model in that place, chosen for speed, and measures three things:

* ``speed``: which models the key reaches, and for a handful of them, how long each takes to answer
  the loop's own first prompt and whether what it wrote is accepted and runs. The one picked is
  the model that produced a running version soonest;
* ``evolve``: the loop itself with that model, with a ceiling on calls, and **where the time went**:
  seconds waiting for the model against seconds everything else took (mostly the judge scoring a
  version). A faster model shortens a round only by the first of these;
* ``fresh``: the champion and the seed on laws drawn after the run with another seed, none with
  a structure the loop was scored or taught on. The loop chose its champion on its own validation
  laws, so those scores are not a measurement of it; these are, within the same grammar.

    python scripts/run_fast_generator_loop.py speed  --out DIR [--candidates 6]
    python scripts/run_fast_generator_loop.py evolve --out DIR [--model NAME] [--steps 100] [--max-calls 300]
    python scripts/run_fast_generator_loop.py fresh  --out DIR [--laws 60]

**The key** is read from ``CODING_MODEL_API_KEY``, or ``NVIDIA_API_KEY`` if that is not set, handed
to the client and written nowhere; ``CODING_MODEL_BASE_URL`` names another service than NVIDIA's.
Code a model wrote runs in the jail (``apedp_dyn/evolution/jail.py``), which is given none of the
environment. No law of the benchmark, held back or not, is read: structures of the benchmark are
left out by fingerprint, as in the loop.

**What this cannot show.** Training, validation and fresh laws come from one grammar. A champion
that proposes more of them has got better at that grammar; whether it helps the route on other laws
needs the route's own test, beside the searches, on laws from elsewhere.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import build_invented_laws as grammar  # noqa: E402
import run_search_evolution as rse  # noqa: E402

from apedp_dyn.evolution import operator as loop  # noqa: E402
from apedp_dyn.evolution.evolved_proposer import CandidateGeneratorTask  # noqa: E402
from apedp_dyn.evolution.seed_generator import SEED_GENERATOR  # noqa: E402

TAG = "DEV-FAST-GENERATOR-LOOP-2026-11"
# Names that suggest a model built to answer fast, and three the key was seen to reach on 2026-11-24.
FAST_HINTS = ("lightning", "flash", "fast", "instant", "turbo", "mini", "small", "nano", "coder", "codestral")
KNOWN = ("nvidia/nemotron-3.5-lightning-30b-a3b", "minimaxai/minimax-m3.1-flash", "nvidia/nemotron-3-ultra-550b-a55b")
CANDIDATES = 6
PROBE_TRAIN, PROBE_SECONDS_PER_CALL = 16, 1.5      # the judge for the speed check: the demo's small one
MAX_CALLS = 300                                    # the first run's ceiling on calls to the model
INTERVAL = 1.5                                     # seconds between calls: the hosted service is rate limited
FRESH_SEED = 2026112491
FRESH_LAWS = 60


class CallLimitReachedError(RuntimeError):
    """The run's ceiling on calls to the model was reached."""


class TimedClient:
    """A model client that counts its calls, stops at a ceiling and records how long each answer took.

    The wait between calls (the service's rate limit) is taken before the clock starts, so
    ``seconds`` is the model's own time for an answer, with its retries if any.
    """

    def __init__(self, client: Callable[[str], str], max_calls: int, *, interval: float = INTERVAL,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic) -> None:
        self._client, self.max_calls, self._interval = client, int(max_calls), float(interval)
        self._sleep, self._clock = sleep, clock
        self._last: float | None = None
        self.calls = 0
        self.records: list[dict[str, Any]] = []

    def __call__(self, prompt: str) -> str:
        if self.calls >= self.max_calls:
            raise CallLimitReachedError(f"the ceiling of {self.max_calls} calls to the model was reached")
        if self._last is not None:
            pause = self._last + self._interval - self._clock()
            if pause > 0:
                self._sleep(pause)
        self.calls += 1
        started = self._clock()
        try:
            reply = self._client(prompt)
        except Exception as exc:
            self._last = self._clock()
            self.records.append({"ok": False, "seconds": round(self._last - started, 3), "error": type(exc).__name__})
            raise
        self._last = self._clock()
        usage = dict((getattr(self._client, "last", None) or {}).get("usage") or {})
        self.records.append({"ok": True, "seconds": round(self._last - started, 3), "prompt_characters": len(prompt),
                             "reply_characters": len(reply), "prompt_tokens": usage.get("prompt_tokens"),
                             "completion_tokens": usage.get("completion_tokens")})
        return reply

    def time(self) -> dict[str, Any]:
        """What the calls cost: how many, how long, how many tokens came back and how fast."""
        answered = [r for r in self.records if r["ok"]]
        seconds = float(sum(r["seconds"] for r in self.records))
        tokens = sum(int(r["completion_tokens"] or 0) for r in answered)
        return {"calls": self.calls, "answered": len(answered), "model_seconds": round(seconds, 1),
                "median_seconds_a_call": round(statistics.median(r["seconds"] for r in answered), 2) if answered else None,
                "slowest_call_seconds": round(max(r["seconds"] for r in answered), 2) if answered else None,
                "completion_tokens": tokens,
                "tokens_a_second": round(tokens / seconds, 1) if seconds > 0 and tokens else None}


def key_and_address() -> tuple[str, str]:
    """The key and the service's address from the environment, or a clear refusal to start without a key."""
    from apedp_dyn.proposers import llm_clients

    key = os.environ.get("CODING_MODEL_API_KEY", "").strip() or os.environ.get(llm_clients.KEY_VARIABLE, "").strip()
    if not key:
        raise SystemExit(f"No key: set CODING_MODEL_API_KEY or {llm_clients.KEY_VARIABLE}. Nothing was run.")
    return key, os.environ.get("CODING_MODEL_BASE_URL", "").strip() or llm_clients.NVIDIA_BASE_URL


def hosted_client(model: str, key: str, address: str, *, thinking: bool | None = False, timeout: float = 300.0) -> Any:
    """One model on the hosted service. Its own pacing is off: ``TimedClient`` paces, outside the clock."""
    from apedp_dyn.proposers import llm_https

    return llm_https.nvidia_client(model, api_key=key, base_url=address, thinking=thinking,
                                   max_tokens=int(os.environ.get("CODING_MODEL_MAX_TOKENS", "8192")),
                                   min_interval=0.0, retries=2, timeout=timeout)


def candidates(listed: Sequence[str], limit: int = CANDIDATES) -> list[str]:
    """Which of the models the key reaches are tried for speed: the known ones first, then names that suggest speed."""
    reachable = set(listed)
    chosen = [name for name in KNOWN if name in reachable]
    for name in sorted(listed):
        if name not in chosen and any(hint in name.lower() for hint in FAST_HINTS):
            chosen.append(name)
    return chosen[: max(int(limit), 1)]


def first_prompt(task: CandidateGeneratorTask) -> tuple[str, loop.Archive]:
    """The loop's own first prompt on ``task``: the seed, scored, and the request to improve it."""
    archive = loop.Archive()
    seed = archive.add(None, "seed", SEED_GENERATOR, task.evaluate(SEED_GENERATOR))
    return loop._prompt(task, archive, seed, steps_left=0), archive


def probe(model: str, make: Callable[[str], Any], prompt: str, task: CandidateGeneratorTask,
          clock: Callable[[], float] = time.monotonic) -> dict[str, Any]:
    """One model, one answer to the loop's first prompt: how long, how much, and whether its code is accepted and runs."""
    timed = TimedClient(make(model), max_calls=1, clock=clock)
    row: dict[str, Any] = {"model": model}
    try:
        reply = timed(prompt)
    except Exception as exc:  # noqa: BLE001 - a model that cannot be reached is a row of the table, not the end of it
        return {**row, "answered": False, "seconds": timed.records[-1]["seconds"], "error": f"{type(exc).__name__}: {exc}"[:160]}
    call = timed.records[-1]
    tokens = call.get("completion_tokens")
    row.update({"answered": True, "seconds": call["seconds"], "completion_tokens": tokens,
                "tokens_a_second": round(tokens / call["seconds"], 1) if tokens and call["seconds"] > 0 else None})
    code = loop.extract_code(reply)
    if code is None:
        return {**row, "code": False, "ran": False}
    evaluation = task.evaluate(code)
    return {**row, "code": True, "ran": bool(evaluation.ran), "score": list(evaluation.score),
            "summary": evaluation.summary, "error": evaluation.error[:160]}


def pick(rows: Sequence[dict[str, Any]]) -> str | None:
    """The model that gave a running version soonest; None if none did."""
    ran = [row for row in rows if row.get("ran")]
    return min(ran, key=lambda row: (row["seconds"], row["model"]))["model"] if ran else None


def speed(out: Path, limit: int, make: Callable[[str], Any], listed: Sequence[str],
          clock: Callable[[], float] = time.monotonic) -> dict[str, Any]:
    training, _validation = rse.tasks(PROBE_TRAIN, PROBE_TRAIN, PROBE_SECONDS_PER_CALL)
    prompt, archive = first_prompt(training)
    rows = []
    for model in candidates(listed, limit):
        rows.append(probe(model, make, prompt, training, clock))
        print(json.dumps(rows[-1]), flush=True)
    record = {"tag": TAG, "what": "development: how long each model takes to answer the loop's first prompt, and whether "
                                  "what it wrote is accepted and runs", "models_the_key_reaches": len(listed),
              "prompt_characters": len(prompt), "judge": f"{PROBE_TRAIN} training laws, {PROBE_SECONDS_PER_CALL:g} s a call",
              "seed": archive.versions[0].evaluation.summary, "rows": rows, "picked": pick(rows),
              "picked_by": "the shortest time to a version that ran"}
    (out / "speed.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    return record


def with_time(report: dict[str, Any], timed: TimedClient) -> dict[str, Any]:
    """The loop's report with where its time went: the model, and everything else."""
    cost = timed.time()
    whole = float(report["seconds"])
    other = max(whole - cost["model_seconds"], 0.0)
    versions = int(report.get("versions") or 0)
    report["time"] = {**cost, "loop_seconds": round(whole, 1), "everything_else_seconds": round(other, 1),
                      "share_of_the_time_spent_waiting_for_the_model": round(cost["model_seconds"] / whole, 3) if whole > 0 else None,
                      "seconds_a_version_outside_the_model": round(other / versions, 1) if versions else None,
                      "what_an_instant_model_would_save": "at most the model's seconds: the rest is the judge scoring "
                                                          "each version, and the wait between calls"}
    report["ceiling_on_calls"] = timed.max_calls
    return report


def fresh_laws(count: int, seed: int = FRESH_SEED) -> list[dict[str, Any]]:
    """Laws of the loop's grammar drawn with another seed, none with a structure the loop was scored or taught on."""
    in_benchmark = set(json.loads(grammar.FINGERPRINTS.read_text(encoding="utf-8"))["fingerprints"])
    used = {grammar.structure(sp.sympify(law["text"])) for law in rse.draw_laws(rse.TRAIN + rse.VALIDATION + rse.TEACH)}
    rng = np.random.default_rng(seed)
    laws: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _ in range(grammar.MAX_DRAWS):
        if len(laws) == count:
            break
        inputs = int(rng.integers(1, 4))
        expr = grammar.draw(rng, inputs)
        key = grammar.structure(expr)
        # compared as the loop's laws are kept, as text read back, so that the two structures are written alike
        if key in seen or grammar.structure(sp.sympify(str(expr))) in used or not grammar.acceptable(expr, inputs, rng):
            continue
        seen.add(key)
        if grammar.fingerprint(str(expr)) in in_benchmark:
            continue
        laws.append({"name": f"fresh_{len(laws):03d}", "inputs": inputs, "text": str(expr)})
    if len(laws) != count:
        raise SystemExit(f"only {len(laws)} fresh laws after {grammar.MAX_DRAWS} draws")
    return laws


def sign_test(gained: int, lost: int) -> float:
    """Two-sided exact p-value for ``gained`` against ``lost`` under an even chance."""
    n = gained + lost
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(gained, lost) + 1)) / 2 ** n
    return float(min(1.0, 2 * tail))


def compare_on_fresh(champion: str, count: int, seconds_per_call: float = rse.SECONDS_PER_CALL) -> dict[str, Any]:
    """The champion and the seed on the fresh laws: how many each proposes, and law by law which gained and lost."""
    laws = fresh_laws(count)
    task = CandidateGeneratorTask([rse.problem_of(law) for law in laws], rse.recovered, seconds_per_call=seconds_per_call)
    of: dict[str, Any] = {}
    solved: dict[str, set[str]] = {}
    for label, code in (("seed", SEED_GENERATOR), ("champion", champion)):
        evaluation = task.evaluate(code)
        per_problem = (evaluation.details or {}).get("problems") or {}
        solved[label] = {name for name, row in per_problem.items() if row.get("solved")}
        of[label] = {"ran": bool(evaluation.ran), "laws_proposed": len(solved[label]), "summary": evaluation.summary,
                     "error": evaluation.error[:160]}
    gained, lost = sorted(solved["champion"] - solved["seed"]), sorted(solved["seed"] - solved["champion"])
    return {"tag": TAG, "what": "development: the champion and the seed on laws drawn after the run with another seed, none "
                                "with a structure the loop was scored or taught on; the same grammar",
            "laws": count, "seed": of["seed"], "champion": of["champion"],
            "the_champion_is_the_seed": champion.strip() == SEED_GENERATOR.strip(),
            "gained_by_the_champion": len(gained), "lost_by_the_champion": len(lost),
            "could_be_chance_p": round(sign_test(len(gained), len(lost)), 4)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("command", choices=("speed", "evolve", "fresh"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--candidates", type=int, default=CANDIDATES, help="for speed: how many models are tried")
    parser.add_argument("--model", default="", help="for evolve: the model; default: the one speed picked")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--max-calls", type=int, default=MAX_CALLS)
    parser.add_argument("--seconds", type=float, default=None, help="for evolve: wall-clock limit of the loop")
    parser.add_argument("--laws", type=int, default=FRESH_LAWS, help="for fresh: how many fresh laws")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == "fresh":
        champion = args.out / "champion.txt"
        if not champion.exists():
            raise SystemExit(f"{champion} is not there: the loop gave no champion. Nothing was run.")
        record = compare_on_fresh(champion.read_text(encoding="utf-8"), args.laws)
        (args.out / "fresh.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
        print(json.dumps(record, indent=1))
        return 0
    key, address = key_and_address()

    def make(model: str) -> Any:
        return hosted_client(model, key, address)

    if args.command == "speed":
        from apedp_dyn.proposers import llm_https

        listed = llm_https.available_models(key, address)
        record = speed(args.out, args.candidates, make, listed)
        print(json.dumps({k: v for k, v in record.items() if k != "rows"}, indent=1))
        return 0 if record["picked"] else 1
    model = args.model.strip()
    if not model:
        picked = json.loads((args.out / "speed.json").read_text(encoding="utf-8")).get("picked") if (args.out / "speed.json").exists() else None
        if not picked:
            raise SystemExit("No model: give --model, or run speed first so that one is picked. Nothing was run.")
        model = str(picked)
    steps = min(int(args.steps), int(args.max_calls) // (rse.REPAIRS + 1))     # a step is one call and up to REPAIRS more
    timed = TimedClient(make(model), max_calls=args.max_calls)
    report = rse.run_loop(timed, args.out, steps=steps, train=rse.TRAIN, validation=rse.VALIDATION,
                          seconds_per_call=rse.SECONDS_PER_CALL, seconds=args.seconds, model=model)
    report = with_time(report, timed)
    report["tag"] = TAG
    (args.out / "fast_loop.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "leading_versions_on_validation"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
