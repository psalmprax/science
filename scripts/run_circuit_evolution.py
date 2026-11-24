#!/usr/bin/env python3
"""A model rewrites a quantum circuit; the competition's verifier judges every version.

Development harness for ``apedp_dyn.evolution.circuit_task``: **no model has driven it.** It
exists so that, when a coding model is provided on a machine that holds the competition's
simulator, one command runs the loop.

    python scripts/run_circuit_evolution.py profile --start FILE.kmx      # counts only; needs no simulator
    python scripts/run_circuit_evolution.py text    --start FILE.kmb --out FILE.kmx
    python scripts/run_circuit_evolution.py demo    --out DIR [--start FILE] [--whole]
    python scripts/run_circuit_evolution.py evolve  --out DIR [--start FILE] [--steps N] [--whole]

**The start** is a circuit that passes: a ``.kmx`` text or a ``.kmb`` file. Without ``--start``
it is the reference circuit the competition ships (``sample_submission/invsqrt.kmb``, 260 qubits).
The repository's own 89-qubit circuit is rebuilt by ``scripts/test_shared_pddd_circuit.py``.

**The judge** is the competition's verifier for Benchmark I, run by
``apedp_dyn.quantum_compilation.referee`` against its pinned contract, first on 4,096 inputs, then
on a million, and with ``--whole`` on every input of the benchmark for a circuit that would lead.
It is never given a salt. ``demo`` drives the loop with scripted replies instead of a model: the
seed, a renaming of qubits that makes the reference circuit narrower, a pair of flips that changes
nothing, a missing gate, a missing rotation and a line that is not an instruction.

**A real model** is whatever serves chat completions in OpenAI's format: ``CODING_MODEL`` (its
name), ``CODING_MODEL_API_KEY``, and ``CODING_MODEL_BASE_URL`` unless it is NVIDIA's service. The
key is read from the environment, handed to the client and written nowhere.

**What a run produces, and what it does not.** A ledger of every version (code, score, parent,
what the checker said), the leading circuit as text, and the receipts of every verifier run in a
``QuantumCandidateLedger``. A circuit that leads here passed the verifier without a salt; a claim
needs the one salted run on every input, made once, outside this loop.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from apedp_dyn.evolution import circuit_task, circuit_text  # noqa: E402
from apedp_dyn.evolution import operator as loop  # noqa: E402

TAG = "circuit-evolution-dev"
STEPS = 20
REPAIRS = 2
SAMPLE = ("sample_submission", "invsqrt.kmb")

# ------------------------------------------------------------------ scripted versions: what a model might send
PACK = '''def rewrite(lines, facts):
    # A working qubit is free again once it has been measured. Give every stretch of use the lowest free number.
    inputs = facts["inputs"]
    split = [line.split(" ") for line in lines]

    def working(words):
        places = []
        if words[0] == "INFO_TAG":
            return places
        for k in range(1, len(words)):
            word = words[k]
            if word == "if":
                break
            if word[0] == "q" and int(word[1:]) >= inputs:
                places.append(k)
        return places

    opened = {}
    last = {}
    begins = {}
    for i in range(len(split)):
        words = split[i]
        for k in working(words):
            q = int(words[k][1:])
            if q not in opened:
                opened[q] = i
            last[q] = i
            if words[0] == "HMR" and "if" not in words:
                begins.setdefault(opened.pop(q), []).append((q, i))
    for q in opened:
        begins.setdefault(opened[q], []).append((q, last[q]))

    free = []
    following = inputs
    now = {}
    ends = {}
    out = []
    for i in range(len(split)):
        for q, end in begins.get(i, []):
            if free:
                new = min(free)
                free.remove(new)
            else:
                new = following
                following += 1
            now[q] = new
            ends.setdefault(end, []).append(q)
        words = split[i]
        for k in working(words):
            words[k] = "q" + str(now[int(words[k][1:])])
        out.append(" ".join(words))
        for q in ends.get(i, []):
            free.append(now.pop(q))
    return out
'''

HARMLESS = '''def rewrite(lines, facts):
    # Two flips of the same qubit do nothing: another circuit, the same behaviour.
    return ["X q0", "X q0"] + lines
'''

DROP_A_GATE = '''def rewrite(lines, facts):
    # One costly gate less.
    out = []
    dropped = False
    for line in lines:
        if not dropped and line.startswith("CCX "):
            dropped = True
            continue
        out.append(line)
    return out
'''

DROP_A_ROTATION = '''def rewrite(lines, facts):
    # The finest rotation turns the phase least; perhaps the tolerance has room for leaving it out.
    finest = None
    smallest = 1.0
    for i in range(len(lines)):
        words = lines[i].split(" ")
        if words[0] == "Z_POW" and 0 < abs(float(words[2])) < smallest:
            smallest = abs(float(words[2]))
            finest = i
    return [lines[i] for i in range(len(lines)) if i != finest]
'''

OUTSIDE_THE_FORMAT = '''def rewrite(lines, facts):
    # A second register would let the circuit keep a result.
    return ['REGISTER r1 "out"'] + lines
'''


def _reply(thought: str, code: str) -> str:
    return f"THINK: {thought}\n```python\n{code}```"


DEMO_REPLIES: tuple[str, ...] = (
    _reply("a measured working qubit is free again, so give every stretch of use the lowest free number", PACK),
    _reply("two flips of the same qubit, to see what the judge does with a circuit that behaves alike", HARMLESS),
    _reply("leave out one costly gate", DROP_A_GATE),
    _reply("leave out the finest rotation", DROP_A_ROTATION),
    _reply("declare a second register", OUTSIDE_THE_FORMAT),
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


# ------------------------------------------------------------------ the loop
def read_start(path: Path | None) -> circuit_task.Start:
    """The starting circuit: a text file as it is, anything else through the pinned simulator."""
    if path is None:
        from apedp_dyn.quantum_compilation import contracts

        path = contracts.default_competition_root().joinpath(*SAMPLE)
    if path.suffix == ".kmx":
        return circuit_task.Start.from_text(path.name, path.read_text(encoding="utf-8"))
    return circuit_task.Start.from_text(path.name, circuit_task.circuit_file_as_text(path))


def run_loop(client: Callable[[str], str], out: Path, task: circuit_task.RewriteTask, *, steps: int,
             seconds: float | None, model: str) -> dict[str, Any]:
    """The loop from the seed, its ledger written as it goes."""
    out.mkdir(parents=True, exist_ok=True)
    archive = loop.Archive()
    started = time.perf_counter()

    def record(version: loop.Version) -> None:
        archive.write(out / "versions.jsonl")
        e = version.evaluation
        print(f"version {version.id} ({version.origin}, step {version.step}, repair {version.repairs}): "
              f"{e.summary if e.ran else 'did not run: ' + e.error[:200]}", flush=True)

    loop.evolve(client, task, circuit_task.SEED_CODE, steps=steps, repairs=REPAIRS, seconds=seconds, archive=archive,
                on_version=record)
    seed, best = archive.versions[0], archive.best()
    seed_cost = seed.evaluation.details.get("cost") if seed.evaluation.ran else None
    best_cost = best.evaluation.details.get("cost") if best is not None else None
    complete = bool(best is not None and best.evaluation.details.get("complete"))
    report = {"tag": TAG, "model": model, "steps": steps, "start": task.start.name, "start_sha256": task.start_sha256,
              "start_numbers": {k: task.profile[k] for k in ("lines", "width", "held", "toffoli_like", "rotations")},
              "scopes": [scope.name for scope in task.scopes], "seconds": round(time.perf_counter() - started, 1),
              "model_calls": int(getattr(client, "calls", 0) or 0), **loop.summary(archive),
              "checker_runs": dict(task.checked), "start_cost": seed_cost, "leading_cost": best_cost,
              "leading_passed_every_scope": complete,
              "saved": None if not (seed_cost and best_cost and complete) else round(1.0 - best_cost / seed_cost, 6),
              "leading_circuit_sha256": None if best is None else best.evaluation.details.get("sha256"),
              "note": "A circuit that leads here passed the verifier without a salt, on the scopes named. A claim needs "
                      "the one salted run on every input, made once, outside this loop."}
    if best is not None:
        (out / "leading_code.txt").write_text(best.code, encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(report, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return report


def coding_model() -> tuple[Callable[[str], str], str]:
    """The model named by the environment, reached the way the other loop reaches it; a refusal without one."""
    import run_search_evolution

    client, name = run_search_evolution.coding_model()
    return client, str(name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("command", choices=("profile", "text", "demo", "evolve"))
    parser.add_argument("--start", type=Path, default=None, help="a circuit that passes (.kmx or .kmb); default: the reference")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--seconds", type=float, default=None, help="wall-clock limit of the whole loop")
    parser.add_argument("--whole", action="store_true", help="also check a circuit that would lead on every input")
    args = parser.parse_args(argv)

    if args.command == "profile":
        start = read_start(args.start)
        profile = circuit_text.profile(circuit_text.read(list(start.body)), start.inputs)
        print(json.dumps({"start": start.name, **profile}, indent=1, default=str))
        return 0
    if args.out is None:
        parser.error(f"{args.command} needs --out")
    if args.command == "text":
        if args.start is None:
            parser.error("text needs --start")
        text = circuit_task.circuit_file_as_text(args.start)
        circuit_task.Start.from_text(args.start.name, text)             # readable by the strict reader, or it is refused here
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out}: {text.count(chr(10)):,} lines")
        return 0

    from apedp_dyn.quantum_compilation.ledger import QuantumCandidateLedger

    client: Callable[[str], str]
    if args.command == "evolve":
        client, name = coding_model()                                   # before anything is built: no model, no run
        steps = args.steps
    else:
        client, name, steps = ScriptedModel(), "scripted stand-in (no model)", len(DEMO_REPLIES)
    args.out.mkdir(parents=True, exist_ok=True)
    start = read_start(args.start)
    check = circuit_task.official_checker(ledger=QuantumCandidateLedger(args.out / "ledger.jsonl"))
    task = circuit_task.RewriteTask(start, check, scopes=circuit_task.COMPLETE if args.whole else circuit_task.DEVELOPMENT,
                                    keep=args.out / "circuits")
    report = run_loop(client, args.out, task, steps=steps, seconds=args.seconds, model=name)
    print(json.dumps({k: report[k] for k in ("model", "versions", "ran", "start_cost", "leading_cost", "saved",
                                             "checker_runs")}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
