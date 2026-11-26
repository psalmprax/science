#!/usr/bin/env python3
"""The flip walk with look-ahead and a scorer it learns from its own descents (development).

Status: deprecated (retired 27 Nov 2026 on the owner's word; kept to replicate its record)

**Development, not a result.** On 26 November the owner asked whether the loops of the non-neural
program search (psalmprax/non-neural-program-search) can help the GF(2^n) searcher, which had not
found 41 products for GF(2^12) and had shown nothing that gets better on its own. Two loops from there
are tried on the flip walk (``scripts/gf2_flip_walk.c``):

* **look-ahead** (that repository's study 3, its largest gain per candidate): at every step the walk
  draws K candidate flips and scores each by what it would do one step ahead, without applying it
  (``scripts/gf2_guided_walk.c``: would a product go, would a changed product share two factors and
  merge, how many products already hold the new values, how much sharing is lost, the change in bits,
  the size of the shared bucket);
* **self-improvement** (studies 1, 2 and 5: learn only from the system's own solutions, then search
  again with what was learned, round after round): the steps that led to each new best are written out
  with every candidate the walk passed over in the same state, and a conditional logit is fitted to
  them (the move taken against the moves not taken, offset by the walk's own probability of each, so
  that later rounds, which no longer choose uniformly, are fitted without bias). The next round walks
  with the learned weights. Anything learned is checked on walks it was not fitted to (held out), by
  position before the new best, because the last move before a new best is a merge by construction and
  its "signal" says nothing.

Both are compared with the plain walk at equal time (the look-ahead costs time per move). Nothing
counts until it is checked against every entry of its tensor.

    python scripts/search_gf2_guided_walk.py study --train-field 8 --test-fields 8 9 10 --rounds 2 --out DIR
    python scripts/search_gf2_guided_walk.py settings --train-fields 7 8 --test-fields 9 10 --rounds 3 --out DIR

A third loop sits one level up (``settings``): the search learns from its own runs which of its
settings (plateau length, slack above the best) to spend time on, and the setting it settles on is
tested against the default on larger fields it never saw.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import search_gf2_mul_flip_c as plain  # noqa: E402

TAG = "DEV-GF2-GUIDED-WALK-2026-11"
SOURCE = ROOT / "scripts" / "gf2_guided_walk.c"
KNOWN = {4: 9, 5: 13, 6: 15, **plain.KNOWN}   #: least counts known (4 to 7 proven, Barbulescu et al.)
FEATURES = ("zero", "merge", "already_held", "sharing_lost", "bits_change", "log2_bucket")
BANDS = (("last move", 0, 1), ("moves 2-8 before", 1, 8), ("moves 9-64 before", 8, 64))
GREEDY = [0.0, 10.0, 0.0, 0.0, 0.0, 0.0]             #: take a merge when one of the K candidates gives one


def build(binary: Path) -> Path:
    subprocess.run(["gcc", "-O3", "-march=native", "-o", str(binary), str(SOURCE), "-lm"], check=True)  # noqa: S603,S607 - the repository's own source
    return binary


def walk(binary: Path, field: int, seed: int, seconds: float, k: int = 1, temperature: float = 1.0,
         weights: Sequence[float] = (), log: Path | None = None, spec: str = "schoolbook",
         plateau: int = 100, slack: int = 2, reset: int = 300_000) -> dict[str, Any]:
    triples = plain.start(field, spec)
    text = f"{field} {len(triples)}\n" + "".join(f"{a} {b} {c}\n" for a, b, c in triples)
    argv = [str(binary), str(seconds), str(seed), str(plateau), str(slack), str(reset), "0", str(k), str(temperature),
            *(str(float(w)) for w in weights)]
    env = {"LOG": str(log)} if log else {}
    done = subprocess.run(argv, input=text, capture_output=True, text=True, timeout=seconds + 600,  # noqa: S603 - built here
                          env={**_base_env(), **env})
    lines = done.stdout.strip().split("\n")
    best, flips, pluses, resets, took = lines[0].split()
    found = [tuple(int(v) for v in line.split()) for line in lines[1:] if line.strip()]
    ok = len(found) == int(best) and plain.verified(found, field)
    row = {"field": field, "start": spec, "seed": seed, "k": k, "temperature": temperature,
           "weights": [float(w) for w in weights], "seconds": float(took), "best": int(best), "flips": int(flips),
           "plus_transitions": int(pluses), "resets": int(resets), "exact": ok}
    if ok and int(best) < KNOWN.get(field, 0):
        row["triples"] = [list(t) for t in found]          # anything below what is known is kept whole
    return row


def _base_env() -> dict[str, str]:
    import os
    return {key: value for key, value in os.environ.items() if key != "LOG"}


Step = tuple[int, np.ndarray, np.ndarray, int]            #: position before the new best, log pi, phi, taken


def read_log(path: Path) -> list[Step]:
    """Blocks ``B n`` of steps ``S nc pick`` (latest first), each followed by nc rows ``logpi phi...``."""
    steps: list[Step] = []
    lines = path.read_text().split("\n")
    i, pos = 0, 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("B "):
            pos, i = 0, i + 1
        elif line.startswith("S "):
            _, nc_text, pick_text = line.split()
            nc = int(nc_text)
            rows = np.array([[float(v) for v in lines[i + 1 + j].split()] for j in range(nc)])
            steps.append((pos, rows[:, 0], rows[:, 1:], int(pick_text)))
            i, pos = i + nc + 1, pos + 1
        else:
            i += 1
    return steps


def pairs(x: np.ndarray) -> np.ndarray:
    """The six features and their pairwise products (a small synthesised set, after the featsynth study)."""
    n = x.shape[-1]
    extra = [x[..., i:i + 1] * x[..., j:j + 1] for i in range(1, n) for j in range(i, n)]
    out: np.ndarray = np.concatenate([x, *extra], axis=-1)
    return out


def _stack(steps: Sequence[Step], expand: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nc = {len(s[1]) for s in steps}
    if len(nc) != 1:
        raise ValueError("steps with different numbers of candidates")
    offsets = np.stack([s[1] for s in steps])
    phi = np.stack([s[2] for s in steps])
    picks = np.array([s[3] for s in steps])
    return offsets, (pairs(phi) if expand else phi), picks


def fit(steps: Sequence[Step], l2: float = 10.0, iters: int = 30, expand: bool = False) -> np.ndarray:
    """Conditional logit, the move taken against the candidates passed over, offset by log pi (Newton)."""
    offsets, phi, picks = _stack(steps, expand)
    n, _, d = phi.shape
    v = np.zeros(d)
    rows = np.arange(n)
    for _ in range(iters):
        u = offsets + phi @ v
        u -= u.max(axis=1, keepdims=True)
        p = np.exp(u)
        p /= p.sum(axis=1, keepdims=True)
        mean = np.einsum("nk,nkd->nd", p, phi)
        grad = l2 * v - (phi[rows, picks] - mean).sum(axis=0)
        hess = l2 * np.eye(d) + np.einsum("nk,nkd,nke->de", p, phi, phi) - mean.T @ mean
        v = v - np.linalg.solve(hess, grad)
    result: np.ndarray = np.asarray(v, dtype=float)
    return result


def gain_per_step(v: np.ndarray, steps: Sequence[Step], expand: bool = False) -> float:
    """Log-likelihood per step of the moves taken, under the learned weights, above the walk's own choice."""
    offsets, phi, picks = _stack(steps, expand)
    rows = np.arange(len(picks))

    def loglik(u: np.ndarray) -> float:
        u = u - u.max(axis=1, keepdims=True)
        return float((u[rows, picks] - np.log(np.exp(u).sum(axis=1))).sum())
    return (loglik(offsets + phi @ v) - loglik(offsets)) / len(picks)


def _band(steps: Sequence[Step], lo: int, hi: int) -> list[Step]:
    return [s for s in steps if lo <= s[0] < hi]


def study(args: argparse.Namespace) -> int:
    args.out.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {"tag": TAG, "development_not_a_result": True, "features": list(FEATURES),
                              "train_field": args.train_field, "seconds": args.seconds, "candidates": args.k,
                              "rounds": []}
    with tempfile.TemporaryDirectory() as tmp:
        binary = build(Path(tmp) / "guided")
        weights = np.zeros(len(FEATURES))
        pooled: list[Step] = []
        with ThreadPoolExecutor(max_workers=args.processes) as pool:
            for r in range(args.rounds):
                seeds = [1000 * (r + 1) + i for i in range(2 * args.walks)]
                logs = [Path(tmp) / f"log_r{r}_{s}.txt" for s in seeds]
                now = weights.tolist()
                rows = list(pool.map(lambda sl, w=now: walk(binary, args.train_field, sl[0], args.seconds, args.k, 1.0, w, sl[1]),
                                     zip(seeds, logs, strict=True)))
                train = [s for p in logs[:args.walks] for s in read_log(p)]
                held = [s for p in logs[args.walks:] for s in read_log(p)]
                bands = {}
                for name, lo, hi in BANDS:
                    for expand in (False, True):
                        tr, te = _band(train, lo, hi), _band(held, lo, hi)
                        v = fit(tr, expand=expand)
                        bands[f"{name}, {'six and their products' if expand else 'six'}"] = {
                            "steps_fitted": len(tr), "steps_held_out": len(te),
                            "held_out_gain_nats_per_step": round(gain_per_step(v, te, expand), 4),
                            "weights": np.round(v, 3).tolist() if not expand else None}
                pooled += train + held
                learned = fit(_band(pooled, 0, 64))
                record["rounds"].append({"round": r, "walked_with_weights": np.round(weights, 3).tolist(),
                                         "walks": [{k: v for k, v in row.items() if k != "triples"} for row in rows],
                                         "steps_logged": len(train) + len(held), "held_out": bands,
                                         "learned_for_next_round": np.round(learned, 3).tolist()})
                print(json.dumps({"round": r, "best": [row["best"] for row in rows], "held_out": bands}), flush=True)
                weights = learned
            arms = {"plain": (1, 1.0, []), "merge-greedy K=2": (2, 0.1, GREEDY), "merge-greedy K=4": (4, 0.1, GREEDY),
                    f"learned K={args.k}": (args.k, 1.0, weights.tolist())}
            jobs = [(name, field, seed) for field in args.test_fields for seed in range(1, args.test_seeds + 1) for name in arms]
            results = list(pool.map(lambda j: (j[0], walk(binary, j[1], j[2], args.test_seconds, *arms[j[0]])), jobs))
    table: dict[str, dict[str, Any]] = {}
    for name, row in results:
        cell = table.setdefault(str(row["field"]), {}).setdefault(name, {"best": [], "flips": [], "exact": True})
        cell["best"].append(row["best"])
        cell["flips"].append(row["flips"])
        cell["exact"] = cell["exact"] and row["exact"]
        if "triples" in row:
            cell.setdefault("below_known", []).append(row)
    for field, cells in table.items():
        for cell in cells.values():
            cell["mean_best"] = round(float(np.mean(cell["best"])), 2)
        cells["known"] = KNOWN.get(int(field))
    record["equal_time"] = {"seconds_each": args.test_seconds, "seeds": args.test_seeds, "by_field": table}
    record["below_known_anywhere"] = any("below_known" in c for cells in table.values() for c in cells.values() if isinstance(c, dict))
    (args.out / "guided_walk.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({field: {name: (c["best"] if isinstance(c, dict) else c) for name, c in cells.items()}
                      for field, cells in table.items()}, indent=1))
    return 0


SETTINGS = [(plateau, slack) for plateau in (30, 100, 1000) for slack in (1, 2, 3)]   #: the walk's own knobs
DEFAULT = (100, 2)


def settings(args: argparse.Namespace) -> int:
    """A loop one level up: the search learns from its own runs which settings to spend its time on.

    Round 0 runs every setting on the training fields; each later round spends the same time on the
    settings that have done best so far (lowest mean excess over the least known count, ties to fewer
    runs); at the end the setting it settled on is run against the default on fields it never saw, at
    equal time and on seeds it never used."""
    args.out.mkdir(parents=True, exist_ok=True)
    history: dict[str, list[int]] = {f"{p}/{s}": [] for p, s in SETTINGS}
    rounds = []
    with tempfile.TemporaryDirectory() as tmp:
        binary = build(Path(tmp) / "guided")
        with ThreadPoolExecutor(max_workers=args.processes) as pool:
            def go(job: tuple[tuple[int, int], int, int, float]) -> tuple[str, dict[str, Any]]:
                (plateau, slack), field, seed, seconds = job
                return f"{plateau}/{slack}", walk(binary, field, seed, seconds, plateau=plateau, slack=slack)
            for r in range(args.rounds):
                if r == 0:
                    chosen = list(SETTINGS)
                else:
                    ranked = sorted(SETTINGS, key=lambda ps: (np.mean(history[f"{ps[0]}/{ps[1]}"]), len(history[f"{ps[0]}/{ps[1]}"])))
                    chosen = (ranked[:3] * len(SETTINGS))[:len(SETTINGS)]   # the same time as round 0, on the best three
                jobs = [(ps, field, 100 * r + i, args.seconds) for i, ps in enumerate(chosen) for field in args.train_fields]
                done = list(pool.map(go, jobs))
                for name, row in done:
                    history[name].append(row["best"] - KNOWN[row["field"]] if row["exact"] else 10**3)
                excess = [row["best"] - KNOWN[row["field"]] for _, row in done]
                rounds.append({"round": r, "settings_run": sorted({f"{p}/{s}" for p, s in chosen}),
                               "mean_excess_this_round": round(float(np.mean(excess)), 2),
                               "all_exact": all(row["exact"] for _, row in done)})
                print(json.dumps(rounds[-1]), flush=True)
            best = min(SETTINGS, key=lambda ps: (np.mean(history[f"{ps[0]}/{ps[1]}"]), -len(history[f"{ps[0]}/{ps[1]}"])))
            jobs = [(ps, field, 10_000 + seed, args.test_seconds) for field in args.test_fields
                    for seed in range(args.test_seeds) for ps in dict.fromkeys((best, DEFAULT))]
            tested = list(pool.map(go, jobs))
    transfer: dict[str, dict[str, list[int]]] = {}
    for name, row in tested:
        transfer.setdefault(str(row["field"]), {}).setdefault(name, []).append(row["best"])
    record = {"tag": TAG + "-SETTINGS", "development_not_a_result": True, "train_fields": args.train_fields,
              "seconds": args.seconds, "rounds": rounds,
              "history_mean_excess": {k: (round(float(np.mean(v)), 2) if v else None) for k, v in history.items()},
              "settled_on": f"{best[0]}/{best[1]}", "default": f"{DEFAULT[0]}/{DEFAULT[1]}",
              "settled_on_the_default": best == DEFAULT,
              "transfer": {"seconds_each": args.test_seconds, "by_field": transfer,
                           "known": {str(f): KNOWN.get(f) for f in args.test_fields}},
              "all_exact": all(row["exact"] for _, row in tested)}
    (args.out / "guided_walk_settings.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: record[k] for k in ("settled_on", "transfer")}, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("study", "settings"))
    parser.add_argument("--train-field", type=int, default=8)
    parser.add_argument("--train-fields", type=int, nargs="+", default=[7, 8])
    parser.add_argument("--test-fields", type=int, nargs="+", default=[8, 9, 10])
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--walks", type=int, default=2, help="walks fitted per round (as many again are held out)")
    parser.add_argument("--k", type=int, default=8, help="candidates per step while learning")
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--test-seconds", type=float, default=30.0)
    parser.add_argument("--test-seeds", type=int, default=4)
    parser.add_argument("--processes", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    return study(args) if args.command == "study" else settings(args)


if __name__ == "__main__":
    raise SystemExit(main())
