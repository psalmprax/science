"""Outside engines on the same SRBench protocol as the baseline (PREREG-SRBENCH-EXTERNAL-2026-11).

    python scripts/run_srbench_external.py smoke                         # adapters work on a toy law
    python scripts/run_srbench_external.py run --shard 3/32 --out <dir>
    python scripts/run_srbench_external.py collect --out <dir>

Rows, renamed variables, noise, isolation and scoring are imported from
``run_srbench_baseline`` unchanged, so a number here is comparable with the
baseline's cell for the same problem and noise level. Two things differ and are
fixed by this run's own preregistration: the engines, and the time limit, since
these engines are built to search for minutes.

Engines:

- **PySR** (multi-population genetic programming; Julia backend). Serial and
  deterministic, stopped by its own ``timeout_in_seconds``.
- **PhySO** (unit-constrained reinforcement search). Receives the units, as L2
  does. It has no time stop of its own, so its logger ends the search at the
  budget and the best program so far is returned.
- **exhaustive_mdl**: the repository's own ``discovery_l2/exhaustive_search.py``,
  added after the baseline was frozen and so measured here.

PySR and PhySO are not dependencies of the package: they are installed only by
the workflow that runs this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_srbench_baseline as base  # noqa: E402

PREREGISTRATION_ID = "PREREG-SRBENCH-EXTERNAL-2026-11"
DIGEST = ROOT / "preregistration" / "DIGEST-SRBENCH-EXTERNAL.json"
FROZEN = ("preregistration/PREREG-SRBENCH-EXTERNAL-2026-11.md", "scripts/run_srbench_external.py",
          "scripts/run_srbench_baseline.py", "scripts/fetch_srbench_ground_truth.py",
          "benchmarks/srbench/manifest.json")

ENGINES = ("pysr", "physo", "exhaustive_mdl")
NOISE_LEVELS = (0.0, 0.01)
SEARCH_BUDGET_S = 240.0   # what each engine is told it may search for
TIME_LIMIT_S = 330.0      # when its process is killed
UNIT_ORDER = ("m", "l", "t", "th", "i")
BINARY = ("+", "-", "*", "/")
UNARY = ("sqrt", "exp", "log", "sin", "cos", "square")


def _predict(model: sp.Expr, names: list[str], X: np.ndarray) -> np.ndarray:
    fn = sp.lambdify([sp.Symbol(n) for n in names], model, "numpy", dummify=True)
    with np.errstate(all="ignore"):
        out: np.ndarray = np.asarray(fn(*X.T), dtype=float) * np.ones(len(X))
    return out


def run_pysr(split: dict[str, Any], entry: dict[str, Any], budget: float = SEARCH_BUDGET_S) -> dict[str, Any]:
    from pysr import PySRRegressor

    model = PySRRegressor(
        niterations=10**7, timeout_in_seconds=budget, binary_operators=list(BINARY), unary_operators=list(UNARY),
        model_selection="best", parallelism="serial", deterministic=True, random_state=base.SEED,
        progress=False, verbosity=0, temp_equation_file=True,
    )
    model.fit(split["X_train"], split["y_train"], variable_names=split["names"])
    expr = sp.sympify(model.sympy())
    return {"model": expr, "pred": _predict(expr, split["names"], split["X_test"]), "formula": str(expr)}


class _BudgetReached(Exception):
    pass


def run_physo(split: dict[str, Any], entry: dict[str, Any], budget: float = SEARCH_BUDGET_S) -> dict[str, Any]:
    import physo
    import physo.learn.monitoring as monitoring
    import torch

    deadline = time.perf_counter() + budget

    class BudgetLogger(monitoring.RunLogger):  # type: ignore[misc]
        def log(self, *args: Any, **kwargs: Any) -> None:
            super().log(*args, **kwargs)
            if time.perf_counter() > deadline:
                raise _BudgetReached

    logger = BudgetLogger(save_path=None, do_save=False)

    def units(u: dict[str, str]) -> list[float]:
        return [float(Fraction(u.get(k, "0"))) for k in UNIT_ORDER]

    torch.manual_seed(base.SEED)
    np.random.seed(base.SEED)
    try:
        physo.SR(
            split["X_train"].T, split["y_train"], X_names=split["names"],
            X_units=[units(entry["units"][f]) for f in entry["features"]],
            y_name="y", y_units=units(entry["target_units"]),
            fixed_consts=[1.0], fixed_consts_units=[[0.0] * len(UNIT_ORDER)],
            free_consts_names=["c1", "c2"], free_consts_units=[[0.0] * len(UNIT_ORDER)] * 2,
            op_names=["mul", "add", "sub", "div", "inv", "n2", "sqrt", "neg", "exp", "log", "sin", "cos"],
            get_run_logger=lambda: logger,
            get_run_visualiser=lambda: monitoring.RunVisualiser(
                epoch_refresh_rate=10**9, save_path=None, do_show=False, do_prints=False, do_save=False),
            run_config=physo.config.config0.config0, parallel_mode=False, epochs=10**9,
        )
    except _BudgetReached:
        pass
    best = logger.hall_of_fame[-1]
    expr = sp.sympify(np.ravel(best.get_infix_sympy(evaluate_consts=True))[0])
    return {"model": expr, "pred": _predict(expr, split["names"], split["X_test"]), "formula": str(expr)}


def run_exhaustive_mdl(split: dict[str, Any], entry: dict[str, Any], budget: float = SEARCH_BUDGET_S) -> dict[str, Any]:
    """The repository's exhaustive search: 6 nodes up to five variables, 5 beyond (it needs no time budget)."""
    from apedp_dyn.discovery_l2 import exhaustive_search

    names = split["names"]
    best = exhaustive_search.search(split["X_train"], split["y_train"], names, 6 if len(names) <= 5 else 5)[0]
    expr = sp.sympify(exhaustive_search.model_string(best))
    return {"model": expr, "pred": _predict(expr, names, split["X_test"]), "formula": best.expression}


RUNNERS = {"pysr": run_pysr, "physo": run_physo, "exhaustive_mdl": run_exhaustive_mdl}
base.RUNNERS.update(RUNNERS)  # base.run_one isolates, times and scores any registered engine


def all_jobs(problems: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    problems = problems if problems is not None else base.load_manifest()
    return [{"problem": p["name"], "noise": noise, "seed": base.SEED, "engine": engine}
            for p in problems for noise in NOISE_LEVELS for engine in ENGINES]


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for engine in ENGINES:
        for family in ("feynman", "strogatz", "all"):
            for noise in (*NOISE_LEVELS, "all"):
                sel = [r for r in rows if r["engine"] == engine and (family == "all" or r["family"] == family)
                       and (noise == "all" or r["noise"] == noise)]
                if not sel:
                    continue
                n = len(sel)
                sym = sum(bool(r.get("symbolic")) for r in sel)
                acc = sum(bool(r.get("accuracy_solution")) for r in sel)
                out[f"{engine}|{family}|{noise}"] = {
                    "n": n, "symbolic": sym, "symbolic_rate": sym / n,
                    "symbolic_interval_95": base.clopper_pearson(sym, n),
                    "accuracy_solution": acc, "accuracy_rate": acc / n,
                    "timeouts": sum(r["status"] == "timeout" for r in sel),
                    "errors": sum(r["status"] == "error" for r in sel),
                    "median_seconds": float(np.median([r["seconds"] for r in sel])),
                }
    return out


def frozen_digests(root: Path = ROOT) -> dict[str, str]:
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in FROZEN}


def check_frozen() -> None:
    if json.loads(DIGEST.read_text(encoding="utf-8"))["sha256"] != frozen_digests():
        raise SystemExit(f"{PREREGISTRATION_ID} changed since it was frozen")


def smoke(budget: float = 20.0, engines: tuple[str, ...] = ENGINES) -> list[dict[str, Any]]:
    """Each adapter on a toy law that is in no benchmark: y = x0 * x1 + x0."""
    rng = np.random.default_rng(0)
    X = rng.uniform(1, 5, (300, 2))
    split = {"names": ["x0", "x1"], "X_train": X[:200], "y_train": X[:200, 0] * X[:200, 1] + X[:200, 0],
             "X_test": X[200:], "y_test": X[200:, 0] * X[200:, 1] + X[200:, 0]}
    entry = {"name": "smoke", "family": "feynman", "features": ["a", "b"], "formula": "a*b + a",
             "units": {"a": {}, "b": {}}, "target_units": {}}
    rows = []
    for engine in engines:
        raw = RUNNERS[engine](split, entry, budget)
        rows.append({"engine": engine, **base.score_run(engine, entry, split, raw)})
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("command", choices=("smoke", "run", "collect"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--shard", default="0/1")
    args = parser.parse_args(argv)
    if args.command == "smoke":
        rows = smoke()
        for row in rows:
            print(row["engine"], "model:", row["model"], "| symbolic:", row["symbolic"], "| R2:", row["test_r2"])
        return 0 if all(r["test_r2"] is not None for r in rows) else 1
    check_frozen()
    entries = {p["name"]: p for p in base.load_manifest()}
    jobs = all_jobs(list(entries.values()))
    if args.command == "run":
        k, n = (int(v) for v in args.shard.split("/"))
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / f"runs-shard-{k:03d}-of-{n:03d}.jsonl"
        if path.exists():
            raise SystemExit(f"{path} exists; each shard runs once")
        with path.open("w", encoding="utf-8") as fh:
            for job in base.shard(jobs, k, n):
                row = base.run_one(job, entries, time_limit=TIME_LIMIT_S)
                fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")
                fh.flush()
                print(f"{row['problem']:24s} {row['engine']:6s} noise={row['noise']:<5} {row['status']:8s} "
                      f"sym={row.get('symbolic')} r2={row.get('test_r2')} {row['seconds']}s", flush=True)
        return 0
    rows = [json.loads(line) for path in sorted(args.out.glob("runs-shard-*.jsonl"))
            for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected = {(j["problem"], j["noise"], j["engine"]) for j in jobs}
    got = [(r["problem"], r["noise"], r["engine"]) for r in rows]
    if set(got) != expected or len(got) != len(expected):
        raise SystemExit(f"shards are incomplete or duplicated: {len(set(got))} distinct of {len(expected)}, {len(got)} rows")
    rows.sort(key=lambda r: (r["problem"], r["noise"], r["engine"]))
    (args.out / "runs.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    results = {"preregistration": PREREGISTRATION_ID, "frozen": frozen_digests(), "summary": summarise(rows)}
    (args.out / "results.json").write_text(json.dumps(results, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"collected {len(rows)} runs into {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
