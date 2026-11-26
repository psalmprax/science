#!/usr/bin/env python3
"""A language model rewrites a program that builds GF(2^n) multiplication decompositions; a fixed judge scores it.

Status: deprecated (retired 27 Nov 2026 on the owner's word; kept to replicate its record)

**Development, not a result.** The fourth of the four approaches the owner asked on 26 November to build
and test toward the GF(2^12) target (41 products or fewer): an AlphaEvolve-style loop (Novikov et al.,
2025), with this repository's evolution machinery (``apedp_dyn.evolution``: the restricted language, the
jail, the operator). The program a model writes is ``construct(n, modulus, seconds)``: it returns a
bilinear decomposition of multiplication in GF(2^n) = GF(2)[u]/(modulus), as products [a, b, c] meaning
(a.x)(b.y) c on the coefficient bits. The seed builds the schoolbook and runs a small flip walk; the model
may write anything in the restricted language (towers, interpolation, Karatsuba, better walks).

**The judge** calls it in the jail for GF(2^5), ..., GF(2^10) and GF(2^12) with ``SECONDS`` each, checks
every entry of every tensor, and scores (fields where the best known count is met or beaten, minus the
total excess over the best known counts, minus the count for GF(2^12)). A count below the best known is
flagged and kept whole; for GF(2^12) it is checked again by ``field_tensor_search.verify_tensor_decomposition``.
The model is told its counts and errors, never anything it could not compute itself.

**Model calls are capped** by ``--max-calls`` (a client that refuses beyond it); the first runs stay at a
few hundred calls at most, as agreed. The key is read from the environment (``CODING_MODEL_API_KEY`` with
``CODING_MODEL``, or ``NVIDIA_API_KEY``), handed to the client and written nowhere; code a model wrote runs
in a process given none of the job's environment.

    python scripts/run_gf2_evolution.py baseline --out DIR            # the seed scored, no model
    python scripts/run_gf2_evolution.py demo --out DIR                # the loop with a scripted stand-in, no model
    python scripts/run_gf2_evolution.py evolve --out DIR --steps 40 --max-calls 120
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import gf2_towers as towers  # noqa: E402

from apedp_dyn.evolution import jail, restricted  # noqa: E402
from apedp_dyn.evolution import operator as loop  # noqa: E402

TAG = "DEV-GF2-12-EVOLUTION-2026-11"
FIELDS = (5, 6, 7, 8, 9, 10, 12)
KNOWN = {5: 13, 6: 15, 7: 22, 8: 24, 9: 30, 10: 33, 12: 42}
SECONDS = 15.0

CONTRACT = loop.Contract(
    function="construct",
    parameters=3,
    signature="construct(n, modulus, seconds)",
    description=(
        "Return a bilinear decomposition of multiplication in GF(2^n) = GF(2)[u]/(modulus), with as few products as "
        "you can. An element is an n-bit integer, bit i the coefficient of u^i; modulus is the integer of the "
        "irreducible polynomial (bit n set). Return a list of products [a, b, c], three integers below 2^n: the "
        "product is (parity of a & x) times (parity of b & y), added (XOR) into the output bits of c. The products "
        "must give x*y mod modulus for every x and y; the judge checks all n^3 entries. You are called for "
        "n = 5, 6, 7, 8, 9, 10 and 12 with `seconds` of time each; a call that runs over is killed and counts as "
        "nothing, so bound your loops (there is no clock: count work). Score: first the number of fields where your "
        "count is at most the best known (13, 15, 22, 24, 30, 33, 42 for n = 5..10, 12), then the total excess over "
        "those counts (smaller is better), then the count for n = 12. Fewer than 42 products for n = 12 would be a "
        "new result."),
    notes=(
        "Known constructions: Karatsuba-style splitting; towers (GF(2^km) as GF(2^k)^m with interpolation at 2m-1 "
        "points of the projective line over GF(2^k), each GF(2^k) product written with the best decomposition of "
        "GF(2^k)); Chinese remaindering with small polynomials; local search on the products (flip graphs: two "
        "products sharing a factor, a(x)b(x)c + a(x)b'(x)c' = a(x)(b+b')(x)c + a(x)b'(x)(c+c'), and merges when two "
        "share two factors). Symmetric decompositions (a = b) suffice for the best known counts. A product whose "
        "a, b or c is zero is wasted."),
    subject="a program that builds decompositions of multiplication in binary fields",
)

SEED = '''
def construct(n, modulus, seconds):
    def mul(a, b):
        r = 0
        while b:
            if b & 1:
                r ^= a
            b >>= 1
            a <<= 1
            if (a >> n) & 1:
                a ^= modulus
        return r

    terms = []
    for i in range(n):
        for j in range(n):
            c = mul(1 << i, 1 << j)
            if c:
                terms.append([1 << i, 1 << j, c])
    state = [88172645463325252 ^ (n * 2654435761)]
    mask = (1 << 64) - 1

    def rnd():
        x = state[0]
        x ^= (x << 13) & mask
        x ^= x >> 7
        x ^= (x << 17) & mask
        state[0] = x
        return x

    def tidy(idx):
        changed = True
        while changed:
            changed = False
            for t in list(idx):
                if t >= len(terms):
                    continue
                if terms[t][0] == 0 or terms[t][1] == 0 or terms[t][2] == 0:
                    terms[t] = terms[-1]
                    terms.pop()
                    changed = True
                    break
                for s in range(len(terms)):
                    if s == t:
                        continue
                    same = [m for m in range(3) if terms[s][m] == terms[t][m]]
                    if len(same) >= 2:
                        q = 3 - same[0] - same[1]
                        terms[t][q] ^= terms[s][q]
                        terms[s] = terms[-1]
                        terms.pop()
                        changed = True
                        break
                if changed:
                    break
            idx = list(range(len(terms))) if changed else idx

    best = [list(t) for t in terms]
    steps = int(20000 * seconds)
    for step in range(steps):
        m = len(terms)
        if m < 2:
            break
        i = rnd() % m
        k = rnd() % 3
        partners = [j for j in range(m) if j != i and terms[j][k] == terms[i][k]]
        if not partners:
            continue
        j = partners[rnd() % len(partners)]
        p = (k + 1) % 3
        q = (k + 2) % 3
        if rnd() & 1:
            p, q = q, p
        terms[i][p] ^= terms[j][p]
        terms[j][q] ^= terms[i][q]
        tidy([i, j])
        if len(terms) < len(best):
            best = [list(t) for t in terms]
    return best
'''


def read_decomposition(value: Any, n: int) -> list[tuple[int, int, int]] | str:
    """The products, or why they are not a valid answer."""
    if not isinstance(value, list) or not value:
        return "did not return a non-empty list"
    if len(value) > 4 * n * n:
        return f"returned {len(value)} products, more than 4 n^2"
    out = []
    for item in value:
        if not isinstance(item, list | tuple) or len(item) != 3 or not all(isinstance(v, int) and not isinstance(v, bool) for v in item):
            return "a product is not three integers"
        if not all(0 <= v < (1 << n) for v in item):
            return "a product has an integer outside 0 .. 2^n - 1"
        out.append((int(item[0]), int(item[1]), int(item[2])))
    return out


class ConstructionTask:
    """Scores one version of ``construct`` on every field; never raises for a bad candidate."""

    contract = CONTRACT

    def __init__(self, fields: tuple[int, ...] = FIELDS, seconds: float = SECONDS) -> None:
        self.fields, self.seconds = fields, seconds

    def evaluate(self, code: str) -> loop.Evaluation:
        calls = [(n, towers.MODULI[n], self.seconds) for n in self.fields]
        try:
            run = jail.run(code, "construct", calls, parameters=3, seconds_per_call=self.seconds + 10.0,
                           result_chars=20_000_000)
        except restricted.RejectedCodeError as exc:
            return loop.Evaluation(ran=False, error=f"refused by the check: {exc}")
        if not run.defined:
            return loop.Evaluation(ran=False, error=f"could not be defined: {run.error}")
        counts: dict[str, Any] = {}
        lines, below = [], []
        met, excess = 0, 0
        kept: dict[str, Any] = {}
        for n, call in zip(self.fields, run.calls, strict=True):
            known = KNOWN[n]
            if not call.ok:
                counts[str(n)] = None
                excess += n * n
                lines.append(f"GF(2^{n}): failed ({call.error[:120]})")
                continue
            got = read_decomposition(call.result, n)
            if isinstance(got, str):
                counts[str(n)] = None
                excess += n * n
                lines.append(f"GF(2^{n}): {got}")
                continue
            if not towers.exact(got, n):
                counts[str(n)] = None
                excess += n * n
                lines.append(f"GF(2^{n}): {len(got)} products, but they do not give the multiplication")
                continue
            r = len(got)
            counts[str(n)] = r
            met += int(r <= known)
            excess += max(0, r - known)
            lines.append(f"GF(2^{n}): {r} products (best known {known})")
            if r < known:
                exact_here = True
                if n == 12:
                    from apedp_dyn.quantum import flip_search
                    exact_here = flip_search.verified(got)
                if exact_here:
                    below.append(n)
                    kept[str(n)] = [list(t) for t in got]
        twelve = counts.get("12") or 10**6
        ran = any(v is not None for v in counts.values())
        return loop.Evaluation(
            ran=ran, score=(float(met), float(-excess), float(-twelve)),
            summary=", ".join(f"{n}:{counts[str(n)]}" for n in self.fields) + f" | met {met}, excess {excess}",
            feedback="\n".join(lines), error="" if ran else "it returned no valid decomposition for any field",
            details={"counts": counts, "below_known": below, "decompositions_below_known": kept})


class ScriptedStandIn:
    """Replies a model might give, in order: the seed with a longer walk, then a version that does not run."""

    def __init__(self) -> None:
        self.calls = 0
        self._replies = [
            "THINK: walk twice as long\n```python\n" + SEED.replace("int(20000 * seconds)", "int(40000 * seconds)") + "```",
            "THINK: this one breaks the rules\n```python\nimport os\ndef construct(n, modulus, seconds):\n    return []\n```",
        ]

    def __call__(self, prompt: str) -> str:
        if self.calls >= len(self._replies):
            raise RuntimeError("the stand-in has no more replies")
        self.calls += 1
        return self._replies[self.calls - 1]


class CappedClient:
    """A client that refuses once ``max_calls`` calls have been made: the run's hard limit on model calls."""

    def __init__(self, client: Callable[[str], str], max_calls: int) -> None:
        self.client, self.max_calls, self.calls = client, max_calls, 0

    def __call__(self, prompt: str) -> str:
        if self.calls >= self.max_calls:
            raise RuntimeError(f"the cap of {self.max_calls} model calls is reached")
        self.calls += 1
        return self.client(prompt)


def coding_model() -> tuple[Callable[[str], str], str]:
    """``CODING_MODEL`` with ``CODING_MODEL_API_KEY`` if both are set, else NVIDIA's service with ``NVIDIA_API_KEY``."""
    from apedp_dyn.proposers import llm_clients, llm_https
    name, key = os.environ.get("CODING_MODEL", "").strip(), os.environ.get("CODING_MODEL_API_KEY", "").strip()
    base = os.environ.get("CODING_MODEL_BASE_URL", "").strip() or llm_clients.NVIDIA_BASE_URL
    if not (name and key):
        key = os.environ.get(llm_clients.KEY_VARIABLE, "").strip()
        if not key:
            raise SystemExit("No model key: set CODING_MODEL and CODING_MODEL_API_KEY, or NVIDIA_API_KEY. Nothing was run.")
        name = llm_clients.pick_model(api_key=key)
    client = llm_https.nvidia_client(name, api_key=key, base_url=base, thinking=None, max_tokens=8192,
                                     min_interval=float(os.environ.get("LLM_INTERVAL_SECONDS", "2.0")), retries=3,
                                     timeout=600.0)
    return client, name


def write_record(out: Path, archive: loop.Archive, *, mode: str, model: str, calls: int, steps: int, seconds: float) -> dict[str, Any]:
    archive.write(out / "ledger.jsonl")
    best = archive.best()
    below = sorted({n for v in archive.versions for n in v.evaluation.details.get("below_known", [])})
    record = {"tag": TAG, "development_not_a_result": True, "mode": mode, "model": model, "model_calls": calls,
              "steps": steps, "seconds_per_field": seconds, "versions": len(archive.versions),
              "versions_that_ran": sum(v.evaluation.ran for v in archive.versions),
              "seed": next((v.evaluation.details.get("counts") for v in archive.versions if v.origin == "seed"), None),
              "best": None if best is None else {"id": best.id, "counts": best.evaluation.details.get("counts"),
                                                 "score": list(best.evaluation.score), "thought": best.thought},
              "fields_below_the_best_known": below, "target_met": 12 in below,
              "loop": loop.summary(archive)}
    (out / "gf2_evolution.json").write_text(json.dumps(record, indent=1) + "\n")
    if best is not None:
        (out / "champion.py").write_text(best.code)
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("baseline", "demo", "evolve"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--repairs", type=int, default=2)
    parser.add_argument("--max-calls", type=int, default=120)
    parser.add_argument("--seconds", type=float, default=14400.0)
    parser.add_argument("--field-seconds", type=float, default=SECONDS)
    parser.add_argument("--fields", type=int, nargs="+", default=list(FIELDS))
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    task = ConstructionTask(tuple(args.fields), args.field_seconds)
    began = time.time()
    if args.command == "baseline":
        archive = loop.Archive()
        archive.add(None, "seed", SEED, task.evaluate(SEED))
        record = write_record(args.out, archive, mode="baseline", model="none", calls=0, steps=0, seconds=args.field_seconds)
    else:
        if args.command == "demo":
            client: Callable[[str], str] = ScriptedStandIn()
            model = "scripted stand-in"
            steps = 2
        else:
            raw, model = coding_model()
            client = CappedClient(raw, args.max_calls)
            steps = args.steps
        if args.max_calls < steps:
            raise SystemExit("--max-calls must allow at least one call per step")
        archive = loop.evolve(client, task, SEED, steps=steps, repairs=args.repairs, seconds=args.seconds,
                              on_version=lambda _v: None)
        record = write_record(args.out, archive, mode=args.command, model=model,
                              calls=int(getattr(client, "calls", 0)), steps=steps, seconds=args.field_seconds)
    record["wall_seconds"] = round(time.time() - began, 1)
    (args.out / "gf2_evolution.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k not in ("top",)}, indent=1)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
