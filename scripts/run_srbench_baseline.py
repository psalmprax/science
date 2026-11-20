"""SRBench ground-truth baseline for the repository's own engines (PREREG-SRBENCH-BASELINE-2026-11).

    python scripts/run_srbench_baseline.py run --shard 3/32 --out <dir>   # one shard
    python scripts/run_srbench_baseline.py collect --out <dir>            # merge shards, summarise

Every engine gets the same problems, rows, noise and time limit, and is scored
the same way. The problems are SRBench's ground-truth track (119 Feynman
equations, 14 Strogatz ODE right-hand sides) from PMLB, pinned by SHA-256 in
``benchmarks/srbench/manifest.json``.

Protocol (fixed by the preregistration):

- rows: 1,000 training and 500 test rows drawn without replacement, seeded by the
  problem name; a problem with fewer than 1,500 rows is split 75/25.
- noise: Gaussian on the training target only, standard deviation
  ``noise * RMS(y_train)``, at ``NOISE_LEVELS``. Test targets are clean.
- variables are renamed ``x0..xk`` for every engine; L2 also receives their units.
- one process per run, killed at ``TIME_LIMIT_S``.
- **symbolic recovery**: every numeric constant in the returned model except
  exponents, plus an outer scale and offset, is refitted by least squares to the
  *clean* truth on the test rows; recovered if the refitted model's RMS error is at
  most ``SYMBOLIC_TOL`` of the truth's RMS and no non-constant additive term was
  refitted to nothing (a superset of the truth does not pass by zeroing its extra
  terms). This is "the right structure up to its constants", the question SRBench's
  symbolic-solution check asks.
- **accuracy solution**: test R^2 of the engine's own predictions >= ``R2_ACCURACY``.
- for L2, which adjudicates its own candidates, the referee's verdict is scored
  too: a promotion of a recovered structure, a promotion of anything else (a false
  promotion), or no promotion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp
from scipy.optimize import least_squares
from scipy.stats import beta as beta_dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import fetch_srbench_ground_truth as srb  # noqa: E402

PREREGISTRATION_ID = "PREREG-SRBENCH-BASELINE-2026-11"
DIGEST = ROOT / "preregistration" / "DIGEST-SRBENCH-BASELINE.json"
FROZEN = ("preregistration/PREREG-SRBENCH-BASELINE-2026-11.md", "scripts/run_srbench_baseline.py",
          "scripts/fetch_srbench_ground_truth.py", "benchmarks/srbench/manifest.json")

ENGINES = ("l2", "rime", "rime_ad", "hierarchical_extended")
NOISE_LEVELS = (0.0, 0.001, 0.01, 0.1)
SEED = 0
N_TRAIN, N_TEST = 1000, 500
TIME_LIMIT_S = 120.0
SYMBOLIC_TOL = 1e-8
R2_ACCURACY = 0.999
FUNCTIONS = {"arcsin": sp.asin, "arccos": sp.acos, "arctan": sp.atan, "ln": sp.log}


# ---------------------------------------------------------------- problems and rows
def load_manifest() -> list[dict[str, Any]]:
    return list(json.loads(srb.MANIFEST.read_text(encoding="utf-8"))["problems"])


def truth_expr(entry: dict[str, Any]) -> sp.Expr:
    """The problem's formula over x0..xk (in the order of its features)."""
    local = {name: sp.Symbol(f"x{i}") for i, name in enumerate(entry["features"])}
    return sp.sympify(entry["formula"], locals={**FUNCTIONS, **local})


def _seed(*parts: Any) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big")


def make_split(entry: dict[str, Any], frame: Any, noise: float, seed: int = SEED) -> dict[str, Any]:
    n = len(frame)
    rng = np.random.default_rng(_seed(PREREGISTRATION_ID, entry["name"], seed))
    if n >= N_TRAIN + N_TEST:
        order = rng.choice(n, N_TRAIN + N_TEST, replace=False)
        tr, te = order[:N_TRAIN], order[N_TRAIN:]
    else:
        order = rng.permutation(n)
        cut = round(0.75 * n)
        tr, te = order[:cut], order[cut:]
    names = [f"x{i}" for i in range(len(entry["features"]))]
    X = frame[entry["features"]].to_numpy(dtype=float)
    y = frame["target"].to_numpy(dtype=float)
    y_tr = y[tr].copy()
    if noise > 0:
        noise_rng = np.random.default_rng(_seed(PREREGISTRATION_ID, entry["name"], seed, noise))
        y_tr = y_tr + noise_rng.standard_normal(len(y_tr)) * noise * float(np.sqrt(np.mean(y_tr ** 2)))
    return {"names": names, "X_train": X[tr], "y_train": y_tr, "X_test": X[te], "y_test": y[te]}


# ---------------------------------------------------------------- scoring
def _r2(y: np.ndarray, pred: np.ndarray) -> float | None:
    pred = np.asarray(pred, dtype=float).ravel()
    if pred.shape != y.shape or not np.all(np.isfinite(pred)):
        return None
    sst = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - float(np.sum((y - pred) ** 2)) / sst if sst > 0 else None


def symbolic_recovery(model: sp.Expr | None, truth: sp.Expr, names: list[str], X: np.ndarray) -> dict[str, Any]:
    """Refit every constant of ``model`` (plus outer scale and offset) to the clean truth."""
    if model is None:
        return {"recovered": False, "rel_rmse": None, "reason": "no model"}
    symbols = [sp.Symbol(n) for n in names]
    if not model.free_symbols <= set(symbols):
        return {"recovered": False, "rel_rmse": None, "reason": f"unknown symbols {model.free_symbols - set(symbols)}"}
    # Exponents are not refitted: x**1.02 is a different law from x, as SRBench's check would say.
    exponents = {e for pw in model.atoms(sp.Pow) for e in pw.exp.atoms(sp.Float)}
    floats = sorted(set(model.atoms(sp.Float)) - exponents, key=str)
    params = [sp.Symbol(f"_p{i}") for i in range(len(floats))]
    scale, offset = sp.Symbol("_scale"), sp.Symbol("_offset")
    templ = scale * model.xreplace(dict(zip(floats, params, strict=True))) + offset
    target = np.asarray(sp.lambdify(symbols, truth, "numpy", dummify=True)(*X.T), dtype=float) * np.ones(len(X))
    rms = float(np.sqrt(np.mean(target ** 2)))
    if not np.all(np.isfinite(target)) or rms == 0:
        return {"recovered": False, "rel_rmse": None, "reason": "truth not finite on test rows"}
    fn = sp.lambdify([*params, scale, offset, *symbols], templ, "numpy", dummify=True)

    def resid(theta: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"):
            value = np.asarray(fn(*theta, *X.T), dtype=complex) * np.ones(len(X))
        out = np.where(np.isfinite(value.real) & (np.abs(value.imag) < 1e-12), value.real, 1e6 * rms) - target
        return out / rms

    x0 = np.array([float(f) for f in floats] + [1.0, 0.0])
    try:
        fit = least_squares(resid, x0, method="lm" if len(x0) <= len(X) else "trf", xtol=1e-15, ftol=1e-15,
                            gtol=1e-15, max_nfev=2000)
        rel = float(np.sqrt(np.mean(fit.fun ** 2)))
    except Exception as exc:  # noqa: BLE001 - a model sympy/numpy cannot evaluate is scored as not recovered, with the reason
        return {"recovered": False, "rel_rmse": None, "reason": f"refit failed: {type(exc).__name__}"}
    if rel > SYMBOLIC_TOL:
        return {"recovered": False, "rel_rmse": rel, "reason": ""}
    # The refit may not delete terms: a model that contains the truth plus extra
    # terms would otherwise pass by fitting their coefficients to zero. Every
    # non-constant additive term must still carry signal after the refit
    # (a stray additive constant is allowed: SRBench accepts a constant difference).
    fitted = dict(zip(params, fit.x[:len(params)], strict=True))
    for term in sp.Add.make_args(model.xreplace(dict(zip(floats, params, strict=True)))):
        if not term.free_symbols & set(symbols):
            continue
        value = sp.lambdify(symbols, term.xreplace(fitted), "numpy", dummify=True)
        with np.errstate(all="ignore"):
            size = np.asarray(value(*X.T), dtype=complex) * np.ones(len(X)) * fit.x[-2]
        if not np.isfinite(size).all() or float(np.sqrt(np.mean(np.abs(size) ** 2))) < 1e-6 * rms:
            return {"recovered": False, "rel_rmse": rel, "reason": f"superfluous term {term}"}
    return {"recovered": True, "rel_rmse": rel, "reason": ""}


def parse_formula(text: str | None) -> sp.Expr | None:
    """A RIME / RIME-AD / hierarchical formula string as sympy, or None if it is not y = f(x)."""
    if not text or "=" in text or "'" in text or "[t-" in text:
        return None
    x = sp.Symbol("x")
    transforms = {
        "sq": sp.Lambda(x, x ** 2), "cub": sp.Lambda(x, x ** 3),
        "ssqrt": sp.Lambda(x, sp.sign(x) * sp.sqrt(sp.Abs(x))), "slog": sp.Lambda(x, sp.sign(x) * sp.log(1 + sp.Abs(x))),
    }
    try:
        # RIME-AD writes its intercept column as ``const``.
        return sp.sympify(text.replace("^", "**"), locals={**FUNCTIONS, **transforms, "const": sp.Integer(1)})
    except Exception:  # noqa: BLE001 - an unparseable formula is reported as such, not guessed at
        return None


# ---------------------------------------------------------------- engines
def run_l2(split: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    from apedp_dyn.discovery_l2.dimensional_types import Dimension
    from apedp_dyn.discovery_l2.l2_engine import L2DiscoveryEngine
    from apedp_dyn.epistemic.adjudicate import EpistemicStatus

    names = split["names"]
    X = {n: split["X_train"][:, i] for i, n in enumerate(names)}
    dims = {n: srb_dimension(entry["units"][f], Dimension) for n, f in zip(names, entry["features"], strict=True)}
    result = L2DiscoveryEngine().discover(
        X=X, Y=split["y_train"], dimensions=dims, target_dimension=srb_dimension(entry["target_units"], Dimension),
        target_name="y", seed=SEED,
    )
    adj = result.adjudication

    def structure(scored: Any) -> sp.Expr | None:
        node = scored.candidate.metadata.get("ast") if scored is not None else None
        return node.to_sympy() if node is not None else None

    top = structure(adj.ranking[0]) if adj.ranking else None
    promoted = structure(adj.promoted) if adj.status is EpistemicStatus.UNIQUE_SUPPORT else None
    model, pred = _refit_terms(top, names, split) if top is not None else (None, None)
    return {"model": model, "pred": pred, "status": adj.status.name, "promoted": promoted,
            "reason": str(adj.reason)[:300], "generated": len(result.generated_nodes)}


def srb_dimension(units: dict[str, str], cls: Any) -> Any:
    return cls(**units)


def _refit_terms(structure: sp.Expr, names: list[str], split: dict[str, Any]) -> tuple[sp.Expr, np.ndarray]:
    """L2 ranks structures with one coefficient per additive term; refit those on the training rows."""
    symbols = [sp.Symbol(n) for n in names]
    terms = [t.as_coeff_Mul()[1] for t in sp.Add.make_args(sp.expand(structure))]
    cols_tr = [np.asarray(sp.lambdify(symbols, t, "numpy", dummify=True)(*split["X_train"].T), float) * np.ones(len(split["X_train"])) for t in terms]
    cols_te = [np.asarray(sp.lambdify(symbols, t, "numpy", dummify=True)(*split["X_test"].T), float) * np.ones(len(split["X_test"])) for t in terms]
    A = np.column_stack(cols_tr)
    ok = np.all(np.isfinite(A), axis=1)
    coef = np.linalg.lstsq(A[ok], split["y_train"][ok], rcond=None)[0]
    model = sum(sp.Float(float(c)) * t for c, t in zip(coef, terms, strict=True))
    return model, np.column_stack(cols_te) @ coef


def run_rime(split: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    from apedp_dyn.proposers.rime_proposer import RimeProposer

    return _run_proposer(RimeProposer(target_col="target", static_mode=True), split)


def run_rime_ad(split: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    from apedp_dyn.proposers.rime_ad_proposer import RimeAdProposer

    return _run_proposer(RimeAdProposer(target_col="target"), split)


def _run_proposer(proposer: Any, split: dict[str, Any]) -> dict[str, Any]:
    import pandas as pd

    names = split["names"]
    train = pd.DataFrame({**{n: split["X_train"][:, i] for i, n in enumerate(names)}, "target": split["y_train"]})
    test = pd.DataFrame({n: split["X_test"][:, i] for i, n in enumerate(names)})
    proposals = list(proposer.fit_and_propose(train))
    if not proposals:
        return {"model": None, "pred": None, "formula": None}
    top = proposals[0]
    return {"model": parse_formula(top.formula), "pred": np.asarray(top.compute_score(test), dtype=float),
            "formula": top.formula}


def run_hierarchical_extended(split: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    from apedp.advanced_search import HierarchicalSearch

    names = split["names"]
    candidates = HierarchicalSearch(template_set="extended").search(
        split["X_train"], split["y_train"], budget=120, variable_names=names)
    if not candidates:
        return {"model": None, "pred": None, "formula": None}
    expr = parse_formula(candidates[0]["expression"])
    if expr is None:
        return {"model": None, "pred": None, "formula": candidates[0]["expression"]}
    symbols = [sp.Symbol(n) for n in names]
    f = sp.lambdify(symbols, expr, "numpy", dummify=True)
    with np.errstate(all="ignore"):
        tr = np.asarray(f(*split["X_train"].T), float) * np.ones(len(split["X_train"]))
        te = np.asarray(f(*split["X_test"].T), float) * np.ones(len(split["X_test"]))
    ok = np.isfinite(tr)
    a, b = np.linalg.lstsq(np.column_stack([tr[ok], np.ones(ok.sum())]), split["y_train"][ok], rcond=None)[0]
    return {"model": sp.Float(float(a)) * expr + sp.Float(float(b)), "pred": a * te + b,
            "formula": candidates[0]["expression"]}


RUNNERS = {"l2": run_l2, "rime": run_rime, "rime_ad": run_rime_ad, "hierarchical_extended": run_hierarchical_extended}


# ---------------------------------------------------------------- one run, isolated and timed
def score_run(engine: str, entry: dict[str, Any], split: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    truth = truth_expr(entry)
    sym = symbolic_recovery(raw.get("model"), truth, split["names"], split["X_test"])
    r2 = _r2(split["y_test"], raw["pred"]) if raw.get("pred") is not None else None
    row = {"symbolic": sym["recovered"], "symbolic_rel_rmse": sym["rel_rmse"], "symbolic_note": sym["reason"],
           "test_r2": r2, "accuracy_solution": bool(r2 is not None and r2 >= R2_ACCURACY),
           "model": str(raw.get("model")) if raw.get("model") is not None else None,
           "formula": raw.get("formula")}
    if engine == "l2":
        promoted = raw.get("promoted")
        promoted_ok = symbolic_recovery(promoted, truth, split["names"], split["X_test"])["recovered"] if promoted is not None else False
        row.update({"l2_status": raw["status"], "l2_promoted": str(promoted) if promoted is not None else None,
                    "l2_outcome": ("promoted_correct" if promoted_ok else "false_promotion") if promoted is not None else "no_promotion",
                    "l2_reason": raw["reason"], "l2_generated": raw["generated"]})
    return row


def _child(conn: Any, engine: str, entry: dict[str, Any], split: dict[str, Any]) -> None:
    try:
        raw = RUNNERS[engine](split, entry)
        conn.send(("ok", score_run(engine, entry, split, raw)))
    except Exception as exc:  # noqa: BLE001 - an engine crash is a scored outcome, reported with its class and message
        conn.send(("error", {"error": f"{type(exc).__name__}: {exc}"[:300], "trace": traceback.format_exc()[-1500:]}))
    finally:
        conn.close()


def run_one(job: dict[str, Any], entries: dict[str, dict[str, Any]], time_limit: float = TIME_LIMIT_S) -> dict[str, Any]:
    entry = entries[job["problem"]]
    frame = srb.load_dataset(entry)
    split = make_split(entry, frame, job["noise"])
    base = {**job, "family": entry["family"], "n_train": len(split["y_train"]), "n_test": len(split["y_test"])}
    ctx = mp.get_context("fork")
    parent, child = ctx.Pipe(duplex=False)
    started = time.perf_counter()
    proc = ctx.Process(target=_child, args=(child, job["engine"], entry, split))
    proc.start()
    child.close()
    outcome: tuple[str, dict[str, Any]] = ("timeout", {})
    if parent.poll(time_limit):
        try:
            outcome = parent.recv()
        except EOFError:
            outcome = ("error", {"error": "engine process died without a result"})
    proc.join(1.0)
    if proc.is_alive():
        proc.kill()
        proc.join()
    return {**base, "status": outcome[0], "seconds": round(time.perf_counter() - started, 2), **outcome[1]}


# ---------------------------------------------------------------- jobs, shards, summary
def all_jobs(problems: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    problems = problems if problems is not None else load_manifest()
    return [{"problem": p["name"], "noise": noise, "seed": SEED, "engine": engine}
            for p in problems for noise in NOISE_LEVELS for engine in ENGINES]


def shard(jobs: list[dict[str, Any]], k: int, n: int) -> list[dict[str, Any]]:
    """Shard by problem, so each shard downloads only its own datasets."""
    problems = sorted({job["problem"] for job in jobs})
    mine = {name for i, name in enumerate(problems) if i % n == k}
    return [job for job in jobs if job["problem"] in mine]


def clopper_pearson(k: int, n: int) -> list[float]:
    if n == 0:
        return [0.0, 1.0]
    return [0.0 if k == 0 else float(beta_dist.ppf(0.025, k, n - k + 1)),
            1.0 if k == n else float(beta_dist.ppf(0.975, k + 1, n - k))]


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
                cell: dict[str, Any] = {
                    "n": n, "symbolic": sym, "symbolic_rate": sym / n, "symbolic_interval_95": clopper_pearson(sym, n),
                    "accuracy_solution": acc, "accuracy_rate": acc / n,
                    "timeouts": sum(r["status"] == "timeout" for r in sel),
                    "errors": sum(r["status"] == "error" for r in sel),
                    "median_seconds": float(np.median([r["seconds"] for r in sel])),
                }
                if engine == "l2":
                    for outcome in ("promoted_correct", "false_promotion", "no_promotion"):
                        cell[outcome] = sum(r.get("l2_outcome") == outcome for r in sel)
                    promoted = cell["promoted_correct"] + cell["false_promotion"]
                    cell["false_promotion_share_of_promotions"] = cell["false_promotion"] / promoted if promoted else None
                out[f"{engine}|{family}|{noise}"] = cell
    return out


def frozen_digests(root: Path = ROOT) -> dict[str, str]:
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in FROZEN}


def check_frozen(root: Path = ROOT, digest_path: Path = DIGEST) -> None:
    if json.loads(digest_path.read_text(encoding="utf-8"))["sha256"] != frozen_digests(root):
        raise SystemExit(f"{PREREGISTRATION_ID} changed since it was frozen")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("command", choices=("run", "collect"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard", default="0/1", help="k/N: run jobs whose index is k modulo N")
    args = parser.parse_args(argv)
    check_frozen()
    entries = {p["name"]: p for p in load_manifest()}
    jobs = all_jobs(list(entries.values()))
    if args.command == "run":
        k, n = (int(v) for v in args.shard.split("/"))
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / f"runs-shard-{k:03d}-of-{n:03d}.jsonl"
        if path.exists():
            raise SystemExit(f"{path} exists; each shard runs once")
        with path.open("w", encoding="utf-8") as fh:
            for job in shard(jobs, k, n):
                row = run_one(job, entries)
                fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")
                fh.flush()
                print(f"{row['problem']:24s} {row['engine']:22s} noise={row['noise']:<6} {row['status']:8s} "
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
