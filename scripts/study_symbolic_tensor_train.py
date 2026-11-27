#!/usr/bin/env python3
"""Development: can a tensor train read a wide table? A feasibility check for a proposal, not a build of it.

Status: audit-only

**Development, not a result.** Nothing was fixed beforehand, the tables are the development set of
wide laws (``scripts/run_wider_conditions.py``: the invented-law grammar at four and five inputs),
and the one-input search used at the end has every block of that grammar in its vocabulary. It
answers one question asked while a proposal was written
(``docs/SOLUTIONS_FOR_NO_GAIN_ITEMS_2026-11-27.md``, item 4): is the idea worth a test?

**The idea.** The repository's tensor-train engine models the *text* of laws and is the worst
engine there (``apedp_mmlm``). A tensor train is a model of multilinear structure, and a law of many
inputs that is made of sums and products of one-input functions has exactly that structure:

    y = G1(x1) G2(x2) ... Gd(xd)

with ``Gi(xi)`` a small matrix of one-input functions (a row at the left end, a column at the right).
One product of one-input functions is rank one; a sum of ``r`` products has rank at most ``r`` at every
cut; the five forms of ``apedp_dyn/proposers/table_shape.py`` are the cases of rank one and two for
two groups of inputs. The cost of a fit grows in proportion to the number of inputs.

**What is measured here**, on the 700 rows a proposer has, with the noise the table has:

1. *The fit.* Every entry of every core is a cubic spline of its input. With the other cores held,
   the model is linear in one core, so each step is a least-squares problem (alternating least
   squares) with a smoothness penalty. Ranks 1, 2 and 3 at every cut are grown one product at a
   time (a product fitted to what is left, then all cores refined together), and the rank is the
   smallest one that no higher rank beats on rows left out of the fit (five folds, squared errors
   compared row by row). Reported: the left-out error against the table's noise, and the fitted
   model against the law on rows without noise.
2. *The cores.* For each input, the space of functions its core spans, set beside the space the
   law itself spans as a function of that input (the law is read for this, only to score). A
   reading of the core is possible if the first holds the second and does not hold everything:
   reported, how much of the law's functions the core misses, and how many of 200 other functions
   of the grammar's blocks it misses by no more.
3. *The reading, where the rank is one.* Each core is then one function of one input, up to a
   factor. ``apedp_dyn/proposers/block_search.py`` proposes laws for it; the one taken is the
   simplest whose whole (that law in place of the core, the other cores as fitted) misses the real
   rows by no more than ``MARGIN`` times what the spline model misses rows left out of its fit.
   The product of the laws taken is refitted, every constant free, on the real rows, and the
   benchmark's own check says whether it is accurate and whether it is the law itself.

**What it is not.** No proposer, no judge, nothing in force, no comparison with the split search at
equal seconds (that is the test the proposal asks for). Ranks above one are fitted and scored but
not read as formulas: a core of rank two holds its functions only up to a mixing of its entries,
and unmixing them is the part of the proposal that is not written.

    python scripts/study_symbolic_tensor_train.py --noise 0.01,0.05 --out DIR
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
import warnings
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp
from scipy.interpolate import BSpline
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

TAG = "DEV-SYMBOLIC-TENSOR-TRAIN-2026-11"
ROWS = 700                 #: the rows a proposer has (as in ``study_split_search.py``)
KNOTS_INSIDE, DEGREE = 10, 3
LAMBDA = 1.0e-3            #: weight of the smoothness penalty (second differences of the spline's coefficients)
RANK_MAX = 3
FOLDS, STARTS, SWEEPS = 5, 3, 60
Z_BETTER = 2.0             #: a higher rank is taken only if it is better on left-out rows by this many standard errors
AT_THE_FLOOR = 1.5         #: left-out error over the table's noise below which a fit is said to reach the noise
SPACE_TOL = 0.02           #: directions of a function space smaller than this share of the largest are not counted
GRID, DECOYS = 200, 200
MARGIN = 1.02


def basis(x: np.ndarray, low: float, high: float) -> np.ndarray:
    """Cubic B-splines on ``[low, high]``: one column a spline, one row a value of the input."""
    knots = np.r_[[low] * DEGREE, np.linspace(low, high, KNOTS_INSIDE + 2), [high] * DEGREE]
    return np.asarray(BSpline.design_matrix(np.clip(x, low, high), knots, DEGREE).toarray())


N_BASIS = KNOTS_INSIDE + DEGREE + 1
_D2 = np.diff(np.eye(N_BASIS), 2, axis=0)
_PENALTY = _D2.T @ _D2


class Train:
    """A tensor train whose cores hold splines of one input: ``cores[i][a, j, b]`` is the weight of spline ``j`` in entry ``(a, b)``."""

    def __init__(self, inputs: int, ranks: Sequence[int], seed: int = 0) -> None:
        self.inputs, self.ranks = inputs, [1, *ranks, 1]
        rng = np.random.default_rng(seed)
        self.cores = []
        for i in range(inputs):
            core = 0.3 * rng.standard_normal((self.ranks[i], N_BASIS, self.ranks[i + 1]))
            core[0, :, 0] += 1.0                      # B-splines sum to one: the constant function
            self.cores.append(core)

    def _matrices(self, bases: Sequence[np.ndarray], i: int) -> np.ndarray:
        return np.einsum("ajb,nj->nab", self.cores[i], bases[i])

    def predict(self, bases: Sequence[np.ndarray]) -> np.ndarray:
        left = np.ones((len(bases[0]), 1))
        for i in range(self.inputs):
            left = np.einsum("na,nab->nb", left, self._matrices(bases, i))
        return np.asarray(left[:, 0])

    def fit(self, bases: Sequence[np.ndarray], y: np.ndarray, sweeps: int = 40, tol: float = 1.0e-9) -> Train:
        """Alternating least squares from the cores as they are: one core at a time, the others held."""
        n, last = len(y), np.inf
        order = [*range(self.inputs), *range(self.inputs - 2, 0, -1)]
        for _ in range(sweeps):
            for i in order:
                left = np.ones((n, 1))
                for k in range(i):
                    left = np.einsum("na,nab->nb", left, self._matrices(bases, k))
                right = np.ones((n, 1))
                for k in range(self.inputs - 1, i, -1):
                    right = np.einsum("nab,nb->na", self._matrices(bases, k), right)
                ra, rb = self.ranks[i], self.ranks[i + 1]
                design = np.einsum("na,nj,nb->najb", left, bases[i], right).reshape(n, ra * N_BASIS * rb)
                scale = float(np.sqrt(np.mean(design ** 2))) + 1.0e-300
                design = design / scale
                penalty = np.kron(np.kron(np.eye(ra), _PENALTY), np.eye(rb))
                gram = design.T @ design / n + LAMBDA * penalty + 1.0e-10 * np.eye(design.shape[1])
                self.cores[i] = (np.linalg.solve(gram, design.T @ y / n) / scale).reshape(ra, N_BASIS, rb)
                # the whole is unchanged when a factor moves from one core to its neighbour: keep the cores of one size
                size = float(np.sqrt(np.mean(self.cores[i] ** 2))) + 1.0e-300
                neighbour = i + 1 if i + 1 < self.inputs else i - 1
                self.cores[i] /= size
                self.cores[neighbour] *= size
            error = float(np.mean((self.predict(bases) - y) ** 2))
            if abs(last - error) < tol * max(error, 1.0e-300):
                break
            last = error
        return self


def add(first: Train, second: Train) -> Train:
    """The sum of two tensor trains is a tensor train: the cores side by side, joined at the two ends."""
    out = Train(first.inputs, [a + b for a, b in zip(first.ranks[1:-1], second.ranks[1:-1], strict=True)])
    for i in range(first.inputs):
        one, two = first.cores[i], second.cores[i]
        if i == 0:
            core = np.concatenate([one, two], axis=2)
        elif i == first.inputs - 1:
            core = np.concatenate([one, two], axis=0)
        else:
            core = np.zeros((one.shape[0] + two.shape[0], N_BASIS, one.shape[2] + two.shape[2]))
            core[: one.shape[0], :, : one.shape[2]] = one
            core[one.shape[0]:, :, one.shape[2]:] = two
        out.cores[i] = core
    return out


def _mse(model: Train, bases: Sequence[np.ndarray], y: np.ndarray) -> float:
    return float(np.mean((model.predict(bases) - y) ** 2))


def grown(bases: Sequence[np.ndarray], y: np.ndarray, rank_max: int = RANK_MAX, starts: int = STARTS) -> list[Train]:
    """Models of rank 1 to ``rank_max`` at every cut: one more product fitted to what is left, then every core refined."""
    inputs, out, current, left = len(bases), [], None, y.copy()
    for _ in range(rank_max):
        one = min((Train(inputs, [1] * (inputs - 1), seed=s).fit(bases, left) for s in range(starts)),
                  key=lambda m: _mse(m, bases, left))
        current = one if current is None else add(current, one)
        current.fit(bases, y, sweeps=SWEEPS)
        out.append(copy.deepcopy(current))
        left = y - current.predict(bases)
    return out


def choose(bases: Sequence[np.ndarray], y: np.ndarray, rank_max: int = RANK_MAX, folds: int = FOLDS,
           seed: int = 1) -> dict[str, Any]:
    """The smallest rank that no higher rank beats on rows left out of the fit; the model of that rank fitted on all rows."""
    n = len(y)
    size = float(np.sqrt(np.mean(y ** 2)))
    order = np.random.default_rng(seed).permutation(n)
    squared = np.zeros((rank_max, n))
    for fold in range(folds):
        test = order[fold::folds]
        train = np.setdiff1d(order, test)
        for r, model in enumerate(grown([b[train] for b in bases], y[train] / size, rank_max)):
            squared[r, test] = (size * model.predict([b[test] for b in bases]) - y[test]) ** 2
    rank = 0
    for higher in range(1, rank_max):
        gain = squared[rank] - squared[higher]
        if gain.mean() / (gain.std() / np.sqrt(n) + 1.0e-300) > Z_BETTER:
            rank = higher
    model = grown(bases, y / size, rank + 1)[rank]
    model.cores[0] = model.cores[0] * size
    return {"rank": rank + 1, "model": model, "left_out_rmse": float(np.sqrt(squared[rank].mean())),
            "left_out_rmse_by_rank": [float(np.sqrt(row.mean())) for row in squared]}


def _space(columns: np.ndarray, tol: float = SPACE_TOL) -> np.ndarray:
    u, s, _ = np.linalg.svd(columns, full_matrices=False)
    return np.asarray(u[:, : int(np.sum(s > tol * s[0]))])


def core_spaces(model: Train, grid: np.ndarray, low: float, high: float) -> list[np.ndarray]:
    """For each input, an orthonormal basis (values on ``grid``) of the functions its core spans, the constant added."""
    on_grid = basis(grid, low, high)
    out = []
    for core in model.cores:
        functions = on_grid @ core.transpose(1, 0, 2).reshape(N_BASIS, -1)
        out.append(np.linalg.qr(np.c_[_space(functions), np.ones(len(grid))])[0])
    return out


def law_space(law: Callable[..., Any], inputs: int, i: int, grid: np.ndarray, low: float, high: float,
              rng: np.random.Generator, draws: int = 40) -> np.ndarray:
    """The functions of input ``i`` the law itself spans: the law along ``grid`` with the other inputs held, many times."""
    columns = []
    for _ in range(draws):
        rows = np.tile(rng.uniform(low, high, inputs), (len(grid), 1))
        rows[:, i] = grid
        columns.append(np.asarray(law(*rows.T), dtype=float) * np.ones(len(grid)))
    return _space(np.array(columns).T)


def miss(space: np.ndarray, function: np.ndarray) -> float:
    """The share of a function (values on the grid) that a space does not hold: 0 inside it, 1 at right angles to it."""
    unit = function / (float(np.linalg.norm(function)) + 1.0e-300)
    return float(np.linalg.norm(unit - space @ (space.T @ unit)))


def decoys(grid: np.ndarray, constants: Sequence[float], rng: np.random.Generator, count: int = DECOYS) -> list[np.ndarray]:
    """Other functions of one input, of the kinds the grammar's blocks are, with constants of the grammar."""
    x = grid
    families: list[Callable[[float], np.ndarray]] = [
        lambda a: np.sin(a * x), lambda a: np.exp(-a * x), lambda a: np.log(a * x + 1.0), lambda a: np.sqrt(x + a),
        lambda a: 1.0 / (x + a), lambda a: np.tanh(a * x), lambda a: x / (x + a), lambda a: x ** 2 + 0.0 * a,
        lambda a: x ** 3 + 0.0 * a, lambda a: 1.0 / x + 0.0 * a, lambda a: x + 0.0 * a,
        lambda a: 1.0 / (a + np.sin(1.3 * a * x) + 1.0)]
    out = []
    for _ in range(count):
        values = families[int(rng.integers(len(families)))](float(rng.choice(constants)))
        out.append(values - values.mean())
    return out


def stand_in(scored: Sequence[tuple[float, int, bool, str]], yard: float, margin: float = MARGIN) -> tuple[str | None, bool]:
    """Of ``(error on the real rows, blocks, has an additive constant, law)``: the simplest within the yardstick, else the best fit.

    Simplest: fewest blocks, then no additive constant, then the smaller error.
    """
    passing = [item for item in scored if item[0] <= margin * yard]
    if passing:
        return min(passing, key=lambda item: (item[1], item[2], item[0]))[3], True
    return (min(scored, key=lambda item: item[0])[3], False) if scored else (None, False)


def read_core(i: int, X: np.ndarray, y: np.ndarray, model: Train, bases: Sequence[np.ndarray], grid: np.ndarray,
              low: float, high: float, yard: float) -> tuple[str | None, bool]:
    """A law of one input in place of core ``i`` of a rank-one model, the other cores as fitted, judged on the real rows."""
    from apedp_dyn.proposers import block_search

    others = np.ones(len(y))
    for k in range(model.inputs):
        if k != i:
            others = others * (bases[k] @ model.cores[k][0, :, 0])
    on_grid = basis(grid, low, high) @ model.cores[i][0, :, 0]
    name = sp.Symbol(f"x{i}")
    scored = []
    for _, law, blocks in block_search.search(grid, on_grid / float(np.sqrt(np.mean(on_grid ** 2))), name=str(name)):
        with np.errstate(all="ignore"):
            values = np.asarray(sp.lambdify(name, sp.sympify(law, locals={str(name): name}), "numpy")(X[:, i]),
                                dtype=float) * np.ones(len(y))
        if not np.all(np.isfinite(values)):
            continue
        whole = others * values
        factor = float(np.dot(whole, y) / (np.dot(whole, whole) + 1.0e-300))
        scored.append((float(np.sqrt(np.mean((factor * whole - y) ** 2))), int(blocks), not law.startswith("("), law))
    return stand_in(scored, yard)


def polish(expr: sp.Expr, names: Sequence[str], X: np.ndarray, y: np.ndarray) -> tuple[sp.Expr, float]:
    """Every constant of ``expr`` but the exponents refitted on the rows given."""
    exponents = {e for power in expr.atoms(sp.Pow) for e in power.exp.atoms(sp.Float)}
    floats = sorted(set(expr.atoms(sp.Float)) - exponents, key=str)
    params = [sp.Symbol(f"_p{k}") for k in range(len(floats))]
    template = expr.xreplace(dict(zip(floats, params, strict=True)))
    fn = sp.lambdify([*params, *[sp.Symbol(n) for n in names]], template, "numpy")
    size = float(np.sqrt(np.mean(y ** 2)))

    def residual(theta: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"):
            values = np.asarray(fn(*theta, *X.T), dtype=float) * np.ones(len(y))
        return np.asarray(np.where(np.isfinite(values), values, 1.0e6 * size) - y)

    fit = least_squares(residual, np.array([float(f) for f in floats]), max_nfev=400)
    return (template.xreplace({p: sp.Float(float(v)) for p, v in zip(params, fit.x, strict=True)}),
            float(np.sqrt(np.mean(fit.fun ** 2))))


def table(job: dict[str, Any]) -> dict[str, Any]:
    """The table ``run_wider_conditions.one`` reads for this job: the same rows and noise, and its rows without noise."""
    import run_wider_conditions as wide

    from apedp_dyn.recursion import curriculum

    names = [f"x{i}" for i in range(job["inputs"])]
    symbols = [sp.Symbol(n) for n in names]
    truth = sp.sympify(job["text"], locals=dict(zip(names, symbols, strict=True)))
    key = hashlib.sha256(f"wider-conditions|{job['index']}".encode()).digest()[:8]
    rng = np.random.default_rng(int.from_bytes(key, "big"))
    X = rng.uniform(curriculum.LOW, curriculum.HIGH, (wide.ROWS + wide.TEST_ROWS, job["inputs"]))
    law = sp.lambdify(symbols, truth, "numpy")
    clean = np.asarray(law(*X.T), dtype=float) * np.ones(len(X))
    sigma = job["noise"] * float(np.sqrt(np.mean(clean[: wide.ROWS] ** 2)))
    y = clean[: wide.ROWS] + sigma * rng.standard_normal(wide.ROWS)
    return {"names": names, "truth": truth, "law": law, "X": X[:ROWS], "y": y[:ROWS], "sigma": sigma,
            "X_clean": X[wide.ROWS:], "y_clean": clean[wide.ROWS:], "low": curriculum.LOW, "high": curriculum.HIGH,
            "constants": [float(c) for c in curriculum.CONSTANTS]}


def one(job: dict[str, Any]) -> dict[str, Any]:
    import run_srbench_baseline as base

    warnings.filterwarnings("ignore")
    started = time.perf_counter()
    t = table(job)
    X, y, low, high, inputs = t["X"], t["y"], t["low"], t["high"], job["inputs"]
    bases = [basis(X[:, i], low, high) for i in range(inputs)]
    chosen = choose(bases, y)
    model = chosen["model"]
    fitted = model.predict([basis(t["X_clean"][:, i], low, high) for i in range(inputs)])
    r2 = base._r2(t["y_clean"], fitted)
    row: dict[str, Any] = {k: job[k] for k in ("index", "law", "inputs", "noise")}
    row.update({"rank": chosen["rank"], "left_out_error_over_noise": round(chosen["left_out_rmse"] / t["sigma"], 3),
                "left_out_error_over_noise_by_rank": [round(v / t["sigma"], 2) for v in chosen["left_out_rmse_by_rank"]],
                "spline_model_r2_on_rows_without_noise": None if r2 is None else round(float(r2), 6),
                "spline_model_accurate": bool(r2 is not None and r2 >= base.R2_ACCURACY)})
    row["reaches_the_noise"] = bool(row["left_out_error_over_noise"] <= AT_THE_FLOOR)
    fit_seconds = time.perf_counter() - started

    grid = np.linspace(low, high, GRID)
    rng = np.random.default_rng(job["index"])
    others = decoys(grid, t["constants"], rng)
    per_input = []
    for i, space in enumerate(core_spaces(model, grid, low, high)):
        own = law_space(t["law"], inputs, i, grid, low, high, rng)
        worst = max(miss(space, own[:, k]) for k in range(own.shape[1]))
        misses = np.array([miss(space, g) for g in others])
        per_input.append({"functions_in_the_core": int(space.shape[1] - 1), "functions_in_the_law": int(own.shape[1]),
                          "law_missed": round(worst, 4), "decoys_missed_median": round(float(np.median(misses)), 4),
                          "decoys_missed_no_more_than_the_law": round(float(np.mean(misses <= max(worst, 1.0e-12))), 3)})
    row["cores"] = per_input
    row["every_core_holds_the_law_within_5_percent"] = bool(max(p["law_missed"] for p in per_input) <= 0.05)

    if chosen["rank"] == 1 and row["reaches_the_noise"]:
        parts = [read_core(i, X, y, model, bases, grid, low, high, chosen["left_out_rmse"]) for i in range(inputs)]
        row["read"] = {"stand_ins_within_the_yardstick": [bool(p[1]) for p in parts]}
        if all(p[0] for p in parts):
            symbols = {n: sp.Symbol(n) for n in t["names"]}
            whole: sp.Expr = sp.Integer(1)
            for law, _ in parts:
                whole = whole * sp.sympify(law, locals=symbols)
            read, error = polish(whole, t["names"], X, y)
            with np.errstate(all="ignore"):
                values = np.asarray(sp.lambdify(list(symbols.values()), read, "numpy")(*t["X_clean"].T), dtype=float)
            score = base._r2(t["y_clean"], values * np.ones(len(t["y_clean"])))
            row["read"].update({
                "law_read": str(read), "truth": job["text"], "refit_error_over_noise": round(error / t["sigma"], 3),
                "accurate": bool(score is not None and score >= base.R2_ACCURACY),
                "the_law_itself": bool(base.symbolic_recovery(read, t["truth"], t["names"], t["X_clean"])["recovered"])})
    row["seconds_fit"], row["seconds"] = round(fit_seconds, 1), round(time.perf_counter() - started, 1)
    return row


def summarise(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for noise in sorted({r["noise"] for r in rows}):
        cell = [r for r in rows if r["noise"] == noise]
        at_floor = [r for r in cell if r["reaches_the_noise"]]
        read = [r for r in cell if "law_read" in r.get("read", {})]
        out[f"noise {noise:g}"] = {
            "tables": len(cell),
            "spline_model_reaches_the_noise": len(at_floor),
            "spline_model_accurate": sum(r["spline_model_accurate"] for r in cell),
            "ranks_chosen": {str(k): sum(r["rank"] == k for r in cell) for k in range(1, RANK_MAX + 1)},
            "of_those_at_the_noise_every_core_holds_the_law_within_5_percent":
                sum(r["every_core_holds_the_law_within_5_percent"] for r in at_floor),
            "rank_one_and_read_as_a_formula": len(read),
            "read_accurate": sum(r["read"]["accurate"] for r in read),
            "read_the_law_itself": sum(r["read"]["the_law_itself"] for r in read),
            "median_seconds_fit": float(np.median([r["seconds_fit"] for r in cell])) if cell else None,
            "median_seconds": float(np.median([r["seconds"] for r in cell])) if cell else None,
        }
    return out


def jobs(noises: Sequence[float], inputs: Sequence[int]) -> list[dict[str, Any]]:
    import run_wider_conditions as wide

    return [job for job in wide.draw_laws() if job["inputs"] in inputs and job["noise"] in noises]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--noise", default="0.01,0.05")
    parser.add_argument("--inputs", default="4,5")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    todo = jobs([float(v) for v in args.noise.split(",")], [int(v) for v in args.inputs.split(",")])
    if args.limit:
        todo = todo[: args.limit]
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            rows = list(pool.map(one, todo))
    else:
        rows = [one(job) for job in todo]
    record = {
        "tag": TAG, "development_not_a_result": True,
        "what": "development: a tensor train of one-input splines fitted to the real rows of the wide development "
                "tables; whether it reaches the noise, whether its cores hold the law's one-input functions, and, where "
                "the rank is one, the product read as a formula; a feasibility check for a proposal, nothing in force",
        "settings": {"rows": ROWS, "splines_per_input": N_BASIS, "penalty": LAMBDA, "rank_max": RANK_MAX, "folds": FOLDS,
                     "starts": STARTS, "z_better": Z_BETTER, "at_the_floor": AT_THE_FLOOR, "margin": MARGIN,
                     "space_tol": SPACE_TOL, "decoys": DECOYS},
        "tables_of_the_set": "scripts/run_wider_conditions.py, seed 2026112402, four and five inputs (development laws)",
        "summary": summarise(rows), "rows": rows,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "record.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    print(json.dumps(record["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
