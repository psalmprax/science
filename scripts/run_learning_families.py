#!/usr/bin/env python3
"""Development: the learning loop over streams of laws from several families, with arms that take it apart.

**Development, not a result.** ``run_learning_stream.py`` ran the learning loop
over invented laws only. This runner adds the other families the repository
can read, and arms that say which learner does what:

Families (``--family``):

* ``invented``: the invented-law grammar, as ``run_learning_stream.py`` draws it;
* ``srbench``: SRBench's 133 Feynman and Strogatz laws in a fixed random order,
  on the benchmark's own rows (1,000 to train on: 700 to fit, 300 in the pool;
  500 to score). **The memory starts empty**: the memory in force was built
  partly from SRBench runs. **The law model is seeded on invented laws**,
  since it was otherwise fitted on SRBench's own formulas;
* ``ode``: the 46 systems of PREREG-ODE-ADDITIONS (138 components), through the
  ODE route in force, where only the memory learns;
* ``llmsrbench``: LLM-SRBench, gated (``llm_srbench.py``). Rows carry no
  equation, and the journal is not written.

Arms (``--arms``):

* ``frozen``: never learns;
* ``learning``: both learners learn what ``learning_loop.learnable`` allows;
* ``memory-only`` and ``model-only``: one learner learns, the other stays at its seed;
* ``all-inputs``: both learn, without the measured-range condition, so laws with
  more than three inputs are learned too. The last checks were never tested
  there, which is why the loop does not do this by itself;
* ``transfer``: frozen, from a state that has learned the laws of another
  family's journal (``--transfer-journal``).

A stream can be cut into chunks (``--start``, ``--stop``), and an arm that
learns carries its journal from one chunk to the next (``--journal-in``). Every
chunk writes ``<family>-<arm>-<start>.jsonl`` and, for an arm that learns, the
journal after it.

    python scripts/run_learning_families.py run --family srbench --arms frozen,learning --start 0 --stop 45 --searches 120 --out DIR
    python scripts/run_learning_families.py collect --out DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_learning_stream as invented_stream  # noqa: E402
import run_srbench_baseline as base  # noqa: E402

from apedp_dyn.discovery_l2 import learning_loop, memory_stage  # noqa: E402
from apedp_dyn.proposers.structure_memory import Memory  # noqa: E402

FAMILIES = ("invented", "srbench", "ode", "llmsrbench")
ARMS = {
    "frozen": {"learn": False, "learners": learning_loop.LEARNERS},
    "learning": {"learn": True, "learners": learning_loop.LEARNERS},
    "memory-only": {"learn": True, "learners": ("memory",)},
    "model-only": {"learn": True, "learners": ("model",)},
    "all-inputs": {"learn": True, "learners": learning_loop.LEARNERS, "beyond_range": True},
    "transfer": {"learn": False, "learners": learning_loop.LEARNERS, "transfer": True},
}
ORDER_SEED = 20261124
CORPUS_SEED = 20261125
NOISES = (0.0, 0.01)
DEV, POOL = 700, 300
ODE_SEED = 20261119
PRIVATE = ("llmsrbench",)


def corpus(kind: str) -> list[str]:
    """The law model's seed: SRBench's formulas, or invented laws drawn with a seed of their own."""
    if kind == "srbench":
        return [str(base.truth_expr(entry)) for entry in base.load_manifest()]
    return [law["text"] for law in invented_stream.draw_stream(133, seed=CORPUS_SEED)]


def units(family: str, count: int | None) -> list[dict[str, Any]]:
    """The stream of one family, in its fixed order; each unit says what to run and at what noise."""
    if family == "invented":
        made = invented_stream.draw_stream(count or 40)
        return [{**law, "unit": law["index"]} for law in made]
    if family == "srbench":
        entries = base.load_manifest()
        order = np.random.default_rng(ORDER_SEED).permutation(len(entries))
        return [{"unit": k, "name": entries[i]["name"], "noise": NOISES[k % 2]} for k, i in enumerate(order)][:count]
    if family == "ode":
        import run_ode_additions as ode_additions
        systems = ode_additions.systems()
        order = np.random.default_rng(ORDER_SEED).permutation(len(systems))
        out: list[dict[str, Any]] = []
        for i in order:
            for component in range(systems[i]["dim"]):
                out.append({"unit": len(out), "system": int(i), "component": component, "noise": NOISES[len(out) % 2]})
        return out[:count]
    if family == "llmsrbench":
        import llm_srbench
        problems, _ = llm_srbench.load(llm_srbench.download())
        order = np.random.default_rng(ORDER_SEED).permutation(len(problems))
        return [{"unit": k, "key": problems[i].key, "noise": NOISES[k % 2]} for k, i in enumerate(order)][:count]
    raise ValueError(family)


def state_for(family: str, arm: str, empty_memory: bool, law_model: str, journal_in: Path | None,
              transfer_journal: Path | None) -> learning_loop.LearningState:
    spec = ARMS[arm]
    memory = Memory() if empty_memory else memory_stage.recorded_memory()
    seeds = corpus(law_model)
    if journal_in is not None and spec["learn"]:
        return learning_loop.LearningState.replay(journal_in, memory, seeds, learners=spec["learners"])
    state = learning_loop.LearningState(memory, seeds, learners=spec["learners"])
    if spec.get("transfer"):
        if transfer_journal is None:
            raise SystemExit("the transfer arm needs --transfer-journal")
        for item in json.loads(transfer_journal.read_text(encoding="utf-8"))["journal"]:
            if not item.get("retracted"):
                state.learn(item["law"], domain=item.get("domain"), source=f"transfer: {item.get('source', '')}")
    return state


def _score_table(law: str | None, truth: sp.Expr, names: list[str], X_test: np.ndarray, y_test: np.ndarray) -> dict[str, Any]:
    if not law:
        return {"reported": False, "recovered": False, "accurate": False, "wrong": False}
    model = sp.sympify(law, locals={n: sp.Symbol(n) for n in names})
    recovered = bool(base.symbolic_recovery(model, truth, names, X_test)["recovered"])
    with np.errstate(all="ignore"):
        pred = np.asarray(sp.lambdify([sp.Symbol(n) for n in names], model, "numpy")(*X_test.T), dtype=float) * np.ones(len(X_test))
    r2 = base._r2(y_test, pred) if np.all(np.isfinite(pred)) else None
    accurate = bool(r2 is not None and r2 >= base.R2_ACCURACY)
    return {"reported": True, "recovered": recovered, "accurate": accurate, "wrong": not recovered and not accurate}


def _table_rows(names: list[str], X: np.ndarray, y: np.ndarray, noise: float, key: str) -> tuple[Any, ...]:
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big"))
    order = rng.permutation(len(y))[:DEV + POOL]
    noisy = y + noise * float(np.sqrt(np.mean(y ** 2))) * rng.standard_normal(len(y))
    dev, pool = order[:DEV], order[DEV:]
    return X[dev], noisy[dev], X[pool], noisy[pool]


def run_unit(family: str, unit: dict[str, Any], state: learning_loop.LearningState, spec: dict[str, Any],
             searches: float, memory_seconds: float, cache: dict[str, Any]) -> dict[str, Any]:
    head = {"unit": unit["unit"], "noise": unit["noise"]}
    proposer = None
    if family != "ode" and searches > 0:
        from apedp_dyn.discovery_l2.search_stage import default_search_stage
        proposer = default_search_stage(searches, seed=int(unit["unit"]))[0]
    if family == "invented":
        row = invented_stream.one_law(state, {**unit, "index": unit["unit"]}, spec["learn"], memory_seconds, searches)
        return {**head, **{k: row[k] for k in ("reported", "recovered", "wrong", "accurate", "learned", "why_not", "seconds")},
                "inputs": unit["inputs"], "name": f"invented {unit['unit']}"}
    if family in ("srbench", "llmsrbench"):
        if family == "srbench":
            entry = next(e for e in cache.setdefault("manifest", base.load_manifest()) if e["name"] == unit["name"])
            import run_srbench_main_search as path_run
            split = base.make_split(entry, path_run.srb.load_dataset(entry), 0.0)
            names, X_all, y_all = split["names"], split["X_train"], split["y_train"]
            X_test, y_test, truth = split["X_test"], split["y_test"], base.truth_expr(entry)
            label = unit["name"]
        else:
            problem = next(p for p in cache["problems"] if p.key == unit["key"])
            names = [f"x{i}" for i in range(problem.inputs)]
            X_all, y_all = problem.train[:, :-1], problem.train[:, -1]
            X_test, y_test, truth = problem.test[:, :-1], problem.test[:, -1], problem.truth
            label = problem.key
        X, y, pool_X, pool_y = _table_rows(names, X_all, y_all, unit["noise"], f"{family}|{label}|{unit['noise']}")
        started = time.perf_counter()
        out = learning_loop.discover_law({n: X[:, i] for i, n in enumerate(names)}, y, pool_X, lambda rows: pool_y[rows],
                                         state=state, proposer=proposer, learn=spec["learn"], source=f"{family} {label}",
                                         memory_seconds=memory_seconds, seed=int(unit["unit"]),
                                         beyond_range=bool(spec.get("beyond_range")))
        scored = _score_table(out.get("law"), truth, names, X_test, y_test)
        row = {**head, **scored, "inputs": len(names), "learned": out["learning"]["learned"],
               "why_not": out["learning"]["why_not"], "range": (out.get("measured_range") or {}).get("state"),
               "seconds": round(time.perf_counter() - started, 1)}
        if family not in PRIVATE:
            row.update({"name": label, "law": out.get("law")})
        return row
    if family == "ode":
        import odebench_worlds as worlds
        import run_ode_additions as ode_additions
        import run_odebench_main_search as ode_run
        system = cache.setdefault("systems", ode_additions.systems())[unit["system"]]
        world = worlds.world(system, unit["noise"], tag="learning-families")
        if "excluded" in world:
            return {**head, "excluded": world["excluded"]}
        component = unit["component"]
        started = time.perf_counter()
        found, done = learning_loop.discover_component(world["dev"], world["pool"], component, state=state,
                                                       budget_seconds=searches, seed=ODE_SEED, learn=spec["learn"],
                                                       source=f"ode {system['id']}.{component}", memory_seconds=memory_seconds)
        names = [f"x{i}" for i in range(system["dim"])]
        model = sp.sympify(found.law, locals={n: sp.Symbol(n) for n in names}) if found.law else None
        scored = ode_run.score({**world, "test_rates_component": world["test_rates"][:, component]},
                               system["components"][component], {"model": model, "formula": found.law})
        recovered, accurate = bool(scored["symbolic"]), bool(scored["accuracy_solution"])
        return {**head, "name": f"{system['name']} x{component}", "inputs": system["dim"], "law": found.law,
                "reported": found.law is not None, "recovered": recovered, "accurate": accurate,
                "wrong": found.law is not None and not recovered and not accurate,
                "learned": done["learning"]["learned"], "why_not": done["learning"]["why_not"],
                "seconds": round(time.perf_counter() - started, 1)}
    raise ValueError(family)


def run(args: argparse.Namespace) -> int:
    family = args.family
    if family == "srbench" and not args.empty_memory:
        raise SystemExit("the memory in force was built partly from SRBench runs: an SRBench stream starts from --empty-memory")
    law_model = args.law_model or ("srbench" if family == "invented" else "invented")
    stream = units(family, args.count)
    chunk = [u for u in stream if args.start <= u["unit"] < (args.stop if args.stop is not None else len(stream))]
    cache: dict[str, Any] = {}
    if family == "llmsrbench":
        import llm_srbench
        cache["problems"], _ = llm_srbench.load(llm_srbench.download())
    args.out.mkdir(parents=True, exist_ok=True)
    suffix = "-empty" if args.empty_memory else ""
    for arm in args.arms.split(","):
        spec = ARMS[arm]
        state = state_for(family, arm, args.empty_memory, law_model, args.journal_in if spec["learn"] else None,
                          args.transfer_journal)
        tag = f"{family}-{arm}{suffix}"
        with (args.out / f"{tag}-{args.start:03d}.jsonl").open("w", encoding="utf-8") as handle:
            for unit in chunk:
                try:
                    row = run_unit(family, unit, state, spec, args.searches, args.memory_seconds, cache)
                except Exception as exc:  # noqa: BLE001 - a unit that cannot be run is counted, with the class of its error
                    row = {"unit": unit["unit"], "noise": unit["noise"], "error": type(exc).__name__}
                row.update({"family": family, "arm": arm, "memory": "empty" if args.empty_memory else "in force",
                            "learned_so_far": len(state.learned()), "motifs": len(state.memory.motifs)})
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
        if spec["learn"] and family not in PRIVATE:
            state.save(args.out / f"{tag}-journal.json")
    return 0


def collect(out_dir: Path) -> dict[str, Any]:
    rows: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(out_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[(row["family"], row["memory"], row["arm"])].append(row)
    summary: dict[str, Any] = {}
    for (family, memory, arm), sel in sorted(rows.items()):
        sel = sorted(sel, key=lambda r: r["unit"])
        ok = [r for r in sel if "error" not in r and "excluded" not in r]
        n = max((r["unit"] for r in sel), default=-1) + 1
        halves = {"first half": [r for r in ok if r["unit"] < n / 2], "second half": [r for r in ok if r["unit"] >= n / 2], "all": ok}
        cell: dict[str, Any] = {part: {"units": len(s), "recovered": sum(r["recovered"] for r in s), "wrong": sum(r["wrong"] for r in s),
                       "reported": sum(r["reported"] for r in s), "learned": sum(r.get("learned", False) for r in s)}
                for part, s in halves.items()}
        cell["errors"] = sum("error" in r for r in sel)
        cell["excluded"] = sum("excluded" in r for r in sel)
        frozen = {r["unit"]: r for r in rows.get((family, memory, "frozen"), []) if "error" not in r and "excluded" not in r}
        if arm != "frozen" and frozen:
            mine = {r["unit"]: r for r in ok}
            both = sorted(set(mine) & set(frozen))
            cell["against frozen"] = {"gained": sum(mine[u]["recovered"] and not frozen[u]["recovered"] for u in both),
                                      "lost": sum(frozen[u]["recovered"] and not mine[u]["recovered"] for u in both),
                                      "gained in second half": sum(mine[u]["recovered"] and not frozen[u]["recovered"]
                                                                   for u in both if u >= n / 2)}
        summary.setdefault(family, {}).setdefault(memory, {})[arm] = cell
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("run", "collect", "units"))
    parser.add_argument("--family", choices=FAMILIES, default="srbench")
    parser.add_argument("--arms", default="frozen,learning")
    parser.add_argument("--count", type=int)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--searches", type=float, default=0.0)
    parser.add_argument("--memory-seconds", type=float, default=learning_loop.SECONDS)
    parser.add_argument("--empty-memory", action="store_true")
    parser.add_argument("--law-model", choices=("srbench", "invented"))
    parser.add_argument("--journal-in", type=Path)
    parser.add_argument("--transfer-journal", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "units":
        made = units(args.family, args.count)
        print(json.dumps({"units": len(made)} if args.family in PRIVATE else made[:5], indent=1))
        return 0
    if args.command == "collect":
        summary = collect(args.out)
        (args.out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=1))
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
