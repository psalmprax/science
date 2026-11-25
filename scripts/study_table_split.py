#!/usr/bin/env python3
"""Development: is the shape of a law read rightly from its table? Held against the law's own exact shape.

**Development, not a result.** ``apedp_dyn/proposers/table_shape.py`` reads from a table how its law is
put together from sums and products of smaller parts, so that a table of five inputs need not be searched
as one. This study measures that reading, and only that: no search is run and no law is proposed.

For each table it compares two things. **The exact shape**: the law's own values on a full grid, with no
noise and no model, split by the same forms for as long as a form holds to rounding (``exact_shape``).
**The shape read** from 1,000 noisy rows. A reading is scored by how far the law is from the shape read
(``departure``: the law's values on the grid, projected on that shape, as a share of the law's spread):

- *exact*: the law has that shape (departure within ``EXACT``);
- *close*: it does not, but is within ``CLOSE`` of its spread of a law that has;
- *wrong*: further off than that. A search of the parts of a wrong shape cannot find the law.

Beside the verdict it counts the shapes *within the accuracy line*: a law of that shape could still
have R² of ``R2_ACCURACY`` against the law itself, which is the line the benchmark runs draw between
an accurate law and a wrong one. Beyond it, whatever the parts' searches find is a wrong law.

Three sets of laws:

- ``first``: the 30 laws of ``scripts/run_wider_conditions.py`` and the tables that run read. The
  reading's forms and its rule were worked out looking at three of these.
- ``second``: 30 laws drawn the same way with another seed, after the rule was fixed.
- ``traps``: laws written by hand in which inputs meet inside a function (a sine of a product, a root of
  a sum of squares), so that no form of the reading holds between them. The development grammar cannot
  produce these; every law it produces splits down to tables of one input.

**What the system already had.** The older package has a test of its own (``apedp/separability.py``): each
pair of inputs judged by the mixed curvature of local quadratic fits, groups that add or groups that multiply,
used to seed that package's search. The table route does not use it. With ``--earlier`` the same tables are
read with it as well and scored the same way, so that the two can be set side by side.

    python scripts/study_table_split.py --set first --out benchmarks/table_split/dev_first.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_srbench_baseline as base  # noqa: E402
import run_wider_conditions as wider  # noqa: E402

from apedp_dyn.proposers import table_shape as sep  # noqa: E402
from apedp_dyn.recursion import curriculum  # noqa: E402

TAG = "DEV-TABLE-SPLIT-2026-11"
SECOND_SEED = 2026112601
GRID = 7                  #: points an input on the full grid the exact shape is read from
HOLDS = 1.0e-7            #: a form holds on exact values when what it leaves is below this share of the leading part
EXACT, CLOSE = 1.0e-6, 1.0e-2
ACCURACY_LINE = float(np.sqrt(1.0 - base.R2_ACCURACY))   #: the departure at which a law of the shape read stops being accurate
TRAPS = (
    "sin(0.7*x0*x1) + 0.5*x2",
    "x2*exp(-0.3*x0*x1)",
    "x2*sqrt(x0**2 + x1**2)",
    "log(x0 + x1*x2)",
    "x0**(0.5*x1) + x2",
    "sqrt(x0**2 + x1**2 - 1.2*x0*x1*cos(0.5*x2))",
    "x0*x1/(x0 + x1) + 0.3*x2",
    "x2*exp(-((x0 - x1)/2)**2)",
    "tanh(x0 - x1) + 0.2*x2*x3",
    "x2*sin(x0 + x1)",
    "x2*(x0 - x1)**2 + x3",
    "x3/(1 + x0*x1*x2)",
    "sqrt(x0*x1 + x2*x3)",
    "x4*exp(-x0/x1) + sin(0.5*x2*x3)",
    "(x0 + x1)**1.5/(x2 + x3) + x4",
)


# --- exact shapes: the law's own values on a full grid -------------------------------------------------------


def exact_tensor(text: str, inputs: int, seed: int = 0) -> np.ndarray:
    """The law's values on a full grid of ``GRID`` points an input, drawn in the range the tables are drawn in."""
    names = [f"x{i}" for i in range(inputs)]
    symbols = [sp.Symbol(n) for n in names]
    law = sp.sympify(text, locals=dict(zip(names, symbols, strict=True)))
    rng = np.random.default_rng(seed)
    axes = [np.sort(rng.uniform(curriculum.LOW, curriculum.HIGH, GRID)) for _ in range(inputs)]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.asarray(sp.lambdify(symbols, law, "numpy")(*mesh), dtype=float) * np.ones((GRID,) * inputs)


def _triple(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    return u, s, vt


def _constant(matrix: np.ndarray) -> float | None:
    rng = np.random.default_rng(0)
    rows, cols = matrix.shape
    i, k, j, m = (rng.integers(0, rows, 4000), rng.integers(0, rows, 4000), rng.integers(0, cols, 4000),
                  rng.integers(0, cols, 4000))
    mixed = matrix[i, j] + matrix[k, m] - matrix[i, m] - matrix[k, j]
    det = matrix[i, j] * matrix[k, m] - matrix[i, m] * matrix[k, j]
    weight = float(np.sum(mixed * mixed))
    spread = float(np.linalg.norm(matrix - matrix.mean()))
    if weight <= (HOLDS * spread) ** 2:
        return None
    constant = float(np.sum(det * mixed) / weight)
    return constant if abs(constant) <= sep.MAX_CONSTANT * float(np.max(np.abs(matrix))) else None


def project(matrix: np.ndarray, form: str) -> tuple[float, list[tuple[int, np.ndarray]], float] | None:
    """A matrix's nearest shape of a form: its constant, its parts (0 = of the rows, 1 = of the columns), what is left.

    What is left is given as a share of the part kept, so that zero to rounding means the form holds.
    """
    spread = float(np.linalg.norm(matrix - matrix.mean()))
    if not spread > 1e-12 * (float(np.abs(matrix).max()) + 1e-300):
        return None
    if form == sep.SUM:
        rows, cols = matrix.mean(axis=1), matrix.mean(axis=0) - matrix.mean()
        left = float(np.linalg.norm(matrix - rows[:, None] - cols[None, :]))
        return 0.0, [(0, rows), (1, cols)], left / spread
    if form in (sep.PRODUCT, sep.PRODUCT_AND_CONSTANT):
        constant = 0.0
        if form == sep.PRODUCT_AND_CONSTANT:
            found = _constant(matrix)
            if found is None:
                return None
            constant = found
        u, s, vt = _triple(matrix - constant)
        if not s[0] > 0:
            return None
        return constant, [(0, s[0] * u[:, 0]), (1, vt[0])], float(np.sqrt(np.sum(s[1:] ** 2))) / float(s[0])
    if form == sep.WITH_SECOND:
        means = matrix.mean(axis=0)
        u, s, vt = _triple(matrix - means[None, :])
        if not s[0] > HOLDS * spread:
            return None
        return 0.0, [(0, s[0] * u[:, 0]), (1, vt[0]), (1, means)], float(np.sqrt(np.sum(s[1:] ** 2))) / float(s[0])
    if form == sep.WITH_FIRST:
        means = matrix.mean(axis=1)
        u, s, vt = _triple(matrix - means[:, None])
        if not s[0] > HOLDS * spread:
            return None
        return 0.0, [(0, s[0] * u[:, 0]), (1, vt[0]), (0, means)], float(np.sqrt(np.sum(s[1:] ** 2))) / float(s[0])
    raise ValueError(form)


def _matrix(tensor: np.ndarray, group: tuple[int, ...], first: tuple[int, ...], second: tuple[int, ...]) -> np.ndarray:
    where = [group.index(a) for a in first] + [group.index(b) for b in second]
    return np.transpose(tensor, where).reshape(GRID ** len(first), GRID ** len(second))


def exact_shape(tensor: np.ndarray, group: tuple[int, ...], forms: tuple[str, ...] = sep.FORMS) -> sep.Shape:
    """The finest shape the law's exact values allow: at each step the split with the fewest parts that holds."""
    if len(group) < 2:
        return sep.Shape(sep.LEAF, group)
    best: tuple[tuple[float, int], str, tuple[int, ...], tuple[int, ...], list[tuple[int, np.ndarray]]] | None = None
    for first, second in sep.bipartitions(group):
        matrix = _matrix(tensor, group, first, second)
        for form in forms:
            got = project(matrix, form)
            if got is None or got[2] > HOLDS:
                continue
            cost = (sep._COST[form], max(len(first), len(second)))
            if best is None or cost < best[0]:
                best = (cost, form, first, second, got[1])
            break
    if best is None:
        return sep.Shape(sep.LEAF, group)
    _, form, first, second, parts = best
    inner = tuple(exact_shape(vector.reshape((GRID,) * len(side_group)), side_group, forms)
                  for side_group, vector in ((first if side == 0 else second, vector) for side, vector in parts))
    return sep.Shape(form, group, inner)


def shaped(tensor: np.ndarray, shape: sep.Shape) -> np.ndarray | None:
    """The law's exact values projected on a shape: the nearest thing of that shape, built part by part as a reading is."""
    if shape.form == sep.LEAF:
        return tensor
    group, first, second = shape.group, shape.parts[0].group, shape.parts[1].group
    got = project(_matrix(tensor, group, first, second), shape.form)
    if got is None:
        return None
    constant, parts, _ = got
    values = []
    for (side, vector), inner in zip(parts, shape.parts, strict=True):
        side_group = first if side == 0 else second
        if inner.group != side_group:
            return None
        value = shaped(vector.reshape((GRID,) * len(side_group)), inner)
        if value is None:
            return None
        flat = value.reshape(-1)
        values.append(flat[:, None] if side == 0 else flat[None, :])
    total: Any = constant
    for term in sep._TERMS[shape.form]:
        product = values[term[0]]
        for i in term[1:]:
            product = product * values[i]
        total = total + product
    matrix = np.broadcast_to(total, (GRID ** len(first), GRID ** len(second)))
    back = np.argsort([group.index(a) for a in first] + [group.index(b) for b in second])
    return np.transpose(matrix.reshape((GRID,) * len(group)), back)


def departure(tensor: np.ndarray, shape: sep.Shape) -> float:
    """How far the law is from a shape, as a share of the law's spread: zero to rounding when the law has that shape."""
    near = shaped(tensor, shape)
    spread = float(np.linalg.norm(tensor - tensor.mean())) or 1.0
    return 1.0 if near is None else float(np.linalg.norm(tensor - near)) / spread


def verdict(far: float) -> str:
    return "exact" if far <= EXACT else "close" if far <= CLOSE else "wrong"


# --- the tables ----------------------------------------------------------------------------------------------


def jobs(which: str) -> list[dict[str, Any]]:
    if which == "first":
        return wider.draw_laws()
    if which == "second":
        return wider.draw_laws(SECOND_SEED)
    if which != "traps":
        raise ValueError(which)
    out: list[dict[str, Any]] = []
    for law, text in enumerate(TRAPS):
        inputs = 1 + max(int(str(s)[1:]) for s in sp.sympify(text).free_symbols)
        for noise in wider.NOISES:
            out.append({"index": len(out), "law": law, "text": text, "inputs": inputs, "noise": noise})
    return out


def table(job: dict[str, Any], which: str) -> tuple[np.ndarray, np.ndarray]:
    """The rows the wider-conditions run read for this job; for the other sets, rows drawn the same way under another name."""
    names = [f"x{i}" for i in range(job["inputs"])]
    symbols = [sp.Symbol(n) for n in names]
    law = sp.sympify(job["text"], locals=dict(zip(names, symbols, strict=True)))
    name = "wider-conditions" if which == "first" else f"table-split-{which}"
    key = hashlib.sha256(f"{name}|{job['index']}".encode()).digest()[:8]
    rng = np.random.default_rng(int.from_bytes(key, "big"))
    X = rng.uniform(curriculum.LOW, curriculum.HIGH, (wider.ROWS + wider.TEST_ROWS, job["inputs"]))
    clean = np.asarray(sp.lambdify(symbols, law, "numpy")(*X.T), dtype=float) * np.ones(len(X))
    y = clean[:wider.ROWS] + job["noise"] * float(np.sqrt(np.mean(clean[:wider.ROWS] ** 2))) * rng.standard_normal(wider.ROWS)
    return X[:wider.ROWS], y


def one(job: dict[str, Any], which: str, exact: dict[int, tuple[np.ndarray, sep.Shape]], **settings: Any) -> dict[str, Any]:
    if job["law"] not in exact:
        tensor = exact_tensor(job["text"], job["inputs"])
        exact[job["law"]] = (tensor, exact_shape(tensor, tuple(range(job["inputs"]))))
    tensor, finest = exact[job["law"]]
    X, y = table(job, which)
    started = time.perf_counter()
    reading = sep.read(X, y, **settings)
    seconds = time.perf_counter() - started
    far = departure(tensor, reading.shape)
    report = reading.report
    return {**{k: job[k] for k in ("index", "law", "inputs", "noise")}, "text": job["text"],
            "exact_shape": finest.text(), "exact_widest": finest.widest(), "exact_parts": len(finest.leaves()),
            "read": bool(report["read"]), "shape": reading.shape.text(), "widest": reading.shape.widest(),
            "parts": len(reading.shape.leaves()), "departure": far, "verdict": verdict(far),
            "unsplit_error_of_spread": report.get("unsplit_error_of_spread"),
            "shaped_error_of_spread": report.get("shaped_error_of_spread"),
            "steps": report["steps"], "seconds": round(seconds, 2)}


def earlier_shape(X: np.ndarray, y: np.ndarray) -> sep.Shape:
    """What the analyzer the older package already had makes of the table (``apedp.separability``), as a shape.

    It judges every pair of inputs by the mixed curvature of local quadratic fits and returns groups of inputs that
    add, or failing that groups that multiply (only where the target keeps one sign). It is called as the older
    search calls it, with its own settings. Its groups are joined here left to right into a shape, so that the same
    score applies. It has no shape inside a group, and none of the forms with a constant or a term of one group.
    """
    from apedp.separability import SeparabilityAnalyzer

    inputs = X.shape[1]
    names = [f"x{i:02d}" for i in range(inputs)]
    got = SeparabilityAnalyzer().decompose({name: X[:, i] for i, name in enumerate(names)}, y)
    groups, form = [list(range(inputs))], sep.SUM
    if len(got.additive_groups) > 1:
        groups, form = got.additive_groups, sep.SUM
    elif len(got.multiplicative_groups) > 1:
        groups, form = got.multiplicative_groups, sep.PRODUCT
    members = sorted((tuple(sorted(names.index(name) if isinstance(name, str) else int(name) for name in group)) for group in groups),
                     key=lambda group: group[0])

    def joined(rest: list[tuple[int, ...]]) -> sep.Shape:
        if len(rest) == 1:
            return sep.Shape(sep.LEAF, rest[0])
        return sep.Shape(form, tuple(sorted(i for group in rest for i in group)), (sep.Shape(sep.LEAF, rest[0]), joined(rest[1:])))

    return joined(members)


def with_the_earlier_analyzer(rows: list[dict[str, Any]], which: str) -> list[dict[str, Any]]:
    """The same tables read by the analyzer the older package already had, scored the same way."""
    by_index = {job["index"]: job for job in jobs(which)}
    tensors: dict[int, np.ndarray] = {}
    out = []
    for row in rows:
        job = by_index[row["index"]]
        if job["law"] not in tensors:
            tensors[job["law"]] = exact_tensor(job["text"], job["inputs"])
        X, y = table(job, which)
        started = time.perf_counter()
        shape = earlier_shape(X, y)
        far = departure(tensors[job["law"]], shape)
        out.append({**{k: row[k] for k in ("index", "law", "inputs", "noise", "exact_widest")}, "read": True,
                    "shape": shape.text(), "widest": shape.widest(), "parts": len(shape.leaves()), "departure": far,
                    "verdict": verdict(far), "seconds": round(time.perf_counter() - started, 2)})
    return out


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def block(some: list[dict[str, Any]]) -> dict[str, Any]:
        split = [r for r in some if r["parts"] > 1]
        return {"tables": len(some),
                "not_read": sum(not r["read"] for r in some),
                "exact": sum(r["verdict"] == "exact" for r in some),
                "close": sum(r["verdict"] == "close" for r in some),
                "wrong": sum(r["verdict"] == "wrong" for r in some),
                "split_at_all": len(split),
                "split_and_within_the_accuracy_line": sum(r["departure"] <= ACCURACY_LINE for r in split),
                "split_and_beyond_the_accuracy_line": sum(r["departure"] > ACCURACY_LINE for r in split),
                "down_to_the_exact_widest": sum(r["widest"] == r["exact_widest"] and r["verdict"] != "wrong" for r in some),
                "finer_than_the_law_allows": sum(r["widest"] < r["exact_widest"] for r in some),
                "exact_and_as_fine": sum(r["verdict"] == "exact" and r["widest"] == r["exact_widest"] for r in some),
                "widest_read": {str(w): sum(r["widest"] == w for r in some) for w in sorted({r["widest"] for r in some})},
                "widest_exact": {str(w): sum(r["exact_widest"] == w for r in some) for w in sorted({r["exact_widest"] for r in some})},
                "median_seconds": float(np.median([r["seconds"] for r in some])) if some else None}

    out: dict[str, Any] = {"all": block(rows), "by_noise": {}, "by_inputs": {}}
    for noise in sorted({r["noise"] for r in rows}):
        out["by_noise"][str(noise)] = block([r for r in rows if r["noise"] == noise])
    for inputs in sorted({r["inputs"] for r in rows}):
        out["by_inputs"][str(inputs)] = block([r for r in rows if r["inputs"] == inputs])
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--set", dest="which", choices=("first", "second", "traps"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--z-refused", type=float, default=sep.Z_REFUSED)
    parser.add_argument("--collect", nargs="*", type=Path, default=None, help="shard records to put together")
    parser.add_argument("--earlier", action="store_true",
                        help="with --collect: read the same tables with the older package's analyzer as well")
    args = parser.parse_args(argv)

    earlier: list[dict[str, Any]] | None = None
    if args.collect is not None:
        rows = sorted((r for path in args.collect for r in json.loads(path.read_text())["tables"]), key=lambda r: r["index"])
        settings = json.loads(args.collect[0].read_text())["settings"]
        if args.earlier:
            earlier = with_the_earlier_analyzer(rows, args.which)
    else:
        todo = [j for j in jobs(args.which) if j["index"] % args.shards == args.shard]
        if args.limit:
            todo = todo[:args.limit]
        exact: dict[int, tuple[np.ndarray, sep.Shape]] = {}
        rows = []
        for job in todo:
            rows.append(one(job, args.which, exact, z_refused=args.z_refused))
            r = rows[-1]
            print(f"{r['index']:4d} law {r['law']:2d} inputs {r['inputs']} noise {r['noise']:.2f}  {r['verdict']:5s} "
                  f"departure {r['departure']:.2e}  widest {r['widest']} (exact {r['exact_widest']})  {r['shape']}", flush=True)
        settings = {"z_refused": args.z_refused, "folds": sep.FOLDS, "max_rows": sep.MAX_ROWS, "grid": sep.GRID,
                    "forms": list(sep.FORMS), "max_unsplit_error": sep.MAX_UNSPLIT_ERROR}
    record = {"tag": TAG, "development_not_a_result": True, "set": args.which, "settings": settings,
              "exact_within": EXACT, "close_within": CLOSE, "accuracy_line": ACCURACY_LINE,
              "summary": summarise(rows), "tables": rows}
    if earlier is not None:
        record["the_earlier_analyzer"] = {"what": "apedp.separability.SeparabilityAnalyzer().decompose on the same tables, "
                                                  "its groups joined into a shape and scored the same way",
                                          "summary": summarise(earlier), "tables": earlier}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(record["summary"]["all"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
