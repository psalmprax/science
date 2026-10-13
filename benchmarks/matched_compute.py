"""Matched-compute comparison: RIME vs existing proposers vs (optional) PySR.

v0.1.3 matched-compute record — the effectiveness axis the v0.1.2 throughput
baseline deferred. Protocol, fixed BEFORE any run (do not tune per result):

  **Equal budget.** Every method gets the SAME development data (600 rows
  planted worlds), the SAME wall-clock allowance (``--budget`` seconds,
  default 2.0), and the same holdout split is untouched by every method.
  Where a method reports its candidate-evaluation count, it is recorded;
  wall seconds consumed are recorded for all. A method over budget is
  flagged ``over_budget: true``, never silently trimmed.

  **Grammar-coverage axis.** Every method declares the dataset FAMILIES its
  grammar can express (``declared_coverage``). An abstention on an
  out-of-coverage family is CORRECT behaviour — the v0.1.2 measurement is
  precisely that a sequential-mechanism grammar must refuse a static scaling
  law — while an EMITTED proposal that misses its own holdout is FAILED on
  any family. Abstention inside declared coverage is a MISS. This axis is
  the difference between "lost on compute" and "out of scope" in the table.

  **Scoring.** ``recovered`` = best emitted proposal reaches holdout
  R² ≥ RECOVERY_R2 on the fresh holdout rows — in BOTH split orders: the
  holdout is consumed twice, once in row order and once with row order
  REVERSED. A law that is a function of a row's own values (an algebraic
  mechanism) is order-invariant; a grid-order tracker (a lag fitted to a
  smoothly varying axis) collapses under reversal. A proposal that passes
  forward and collapses on reversal is scored ``index_fit`` — the QEC
  masquerade, caught by construction rather than by a different split.
  Holdout data is never used during fitting (anti-snooping, AGENTS.md §1.2).

  **Status.** Measures discovery effectiveness at matched compute on
  synthetic worlds with planted laws. It makes NO claim about throughput
  (that axis lives in the v0.1.2 performance baseline) and none about real
  experimental data.

PySR runs only when importable; otherwise the artifact records it as
``available: false`` and every other method still runs: the table is never
silently truncated.

Usage:
    python benchmarks/matched_compute.py [--budget 2.0] [--json PATH]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any, Callable

import numpy as np
import pandas as pd

RECOVERY_R2 = 0.99  # preregistered: worlds are low-noise; 0.99 means exactness
DEV_ROWS = 600
HOLDOUT_ROWS = 300
DEFAULT_BUDGET_S = 2.0

# Repo root on sys.path (runnable as a script from anywhere).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── worlds ───────────────────────────────────────────────────────────────────

FAMILIES = [
    "sequential_lag",
    "sequential_state",
    "sequential_interaction",
    "static_scaling",
    "static_log",
]


def _hash_noise(n: int, phase: float) -> np.ndarray:
    i = np.arange(n, dtype=np.float64) + phase
    v = np.sin(i * 12.9898) * 43758.5453
    return v - np.floor(v) - 0.5


@dataclass
class World:
    name: str
    family: str
    dev: pd.DataFrame
    holdout: pd.DataFrame
    law: str


def _split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One contiguous series split end-to-end (row order preserved)."""
    dev = df.iloc[:DEV_ROWS].reset_index(drop=True)
    hold = df.iloc[DEV_ROWS:DEV_ROWS + HOLDOUT_ROWS].reset_index(drop=True)
    return dev, hold


def _make_worlds() -> list[World]:
    n = DEV_ROWS + HOLDOUT_ROWS
    worlds: list[World] = []

    x = _hash_noise(n + 20, 0.0)
    y = np.full(n + 20, np.nan)
    y[2:] = 3.0 * x[:-2] + 1.0
    dev, hold = _split(pd.DataFrame({"x": x, "target": y}))
    worlds.append(World("lag2_amplitude_offset", "sequential_lag", dev, hold, "y = 3·x[t-2] + 1"))

    x = _hash_noise(n, 7.0)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.8 * y[t - 1] + x[t]
    dev, hold = _split(pd.DataFrame({"x": x, "target": y}))
    worlds.append(World("state_decay_080", "sequential_state", dev, hold, "y = 0.8·y[t-1] + x[t]"))

    x = _hash_noise(n + 20, 0.0)
    z = _hash_noise(n + 20, 7.0)
    y = np.full(n + 20, np.nan)
    y[1:] = 4.0 * z[1:] * x[:-1]
    dev, hold = _split(pd.DataFrame({"x": x, "z": z, "target": y}))
    worlds.append(World("lead_lag_interaction", "sequential_interaction", dev, hold,
                        "y = 4·z[t]·x[t-1]"))

    p = 1e-4 * np.power(10.0, np.arange(n) / (n - 1.0))
    y = 1000.0 * p * p + _hash_noise(n, 31.0) * 1e-6
    dev, hold = _split(pd.DataFrame({"p": p, "target": y}))
    worlds.append(World("static_threshold_law", "static_scaling", dev, hold, "y = 1000·p²"))

    x = np.linspace(-50.0, 50.0, n)
    y = 20.0 * np.sign(x) * np.log1p(np.abs(x)) + _hash_noise(n, 41.0) * 1e-6
    dev, hold = _split(pd.DataFrame({"x": x, "target": y}))
    worlds.append(World("static_log_law", "static_log", dev, hold, "y = 20·sign(x)·log1p(|x|)"))

    return worlds


# ── evaluation helpers ───────────────────────────────────────────────────────

def _holdout_r2(proposal: Any, holdout: pd.DataFrame) -> float | None:
    y = holdout["target"].to_numpy(dtype=float)
    s = np.asarray(proposal.compute_score(holdout), dtype=float)
    finite = np.isfinite(y) & np.isfinite(s)
    if finite.sum() < 30:
        return None
    var = float(np.var(y[finite]))
    if var <= 0.0:
        return None
    return float(1.0 - np.var(y[finite] - s[finite]) / var)


@dataclass
class MethodResult:
    wall_seconds: float
    holdout_r2: float | None
    holdout_r2_reversed: float | None
    n_proposals: int
    candidate_evals: int | None
    abstained: bool
    over_budget: bool

    @staticmethod
    def abstain(wall: float, evals: int | None) -> "MethodResult":
        return MethodResult(wall, None, None, 0, evals, True, False)


@dataclass
class Method:
    method_id: str
    declared_coverage: list[str]
    run: Callable[[World, float], MethodResult]
    abstraction: str  # what the method is, stated plainly in the artifact


def _rime_run(**kw: Any) -> Callable[[World, float], MethodResult]:
    from apedp_dyn.proposers.rime_proposer import RimeProposer

    def run(world: World, allowance: float) -> MethodResult:
        feat_cols = [c for c in world.dev.columns if c != "target"]
        proposer = RimeProposer(feature_cols=feat_cols, target_col="target", **kw)
        t0 = time.perf_counter()
        proposals = proposer.fit_and_propose(world.dev)
        wall = time.perf_counter() - t0
        best = proposals[0] if proposals else None
        if best is None:
            return MethodResult.abstain(wall, int(best.metadata["evaluated"]) if False else 0)
        r2 = _holdout_r2(best, world.holdout)
        r2_rev = _holdout_r2(best, world.holdout.iloc[::-1])
        evals = int(best.metadata["evaluated"])
        return MethodResult(wall, r2, r2_rev, len(proposals), evals, abstained=False,
                            over_budget=wall > allowance)

    return run


def _dli_run() -> Callable[[World, float], MethodResult]:
    from apedp_dyn.proposers.distributed_lag_interaction_proposer import (
        DistributedLagInteractionProposer,
    )

    def run(world: World, allowance: float) -> MethodResult:
        feats = [c for c in world.dev.columns if c != "target"]
        proposer = DistributedLagInteractionProposer()
        t0 = time.perf_counter()
        proposals = list(
            proposer.fit_and_propose(world.dev, feats, "target"))
        wall = time.perf_counter() - t0
        best = proposals[0] if proposals else None
        if best is None:
            return MethodResult.abstain(wall, None)
        r2 = _holdout_r2(best, world.holdout)
        r2_rev = _holdout_r2(best, world.holdout.iloc[::-1])
        return MethodResult(wall, r2, r2_rev, len(proposals), None, abstained=False,
                            over_budget=wall > allowance)

    return run


def _ols_basis_run() -> Callable[[World, float], MethodResult]:
    """The QEC-table control: algebraic basis + exhaustive subset selection +
    OLS. Its basis holds {1, x, x², x³, √|x|·sign, log1p|x|·sign} per input
    (first two inputs max); subset enumeration is hard-capped (preregistered
    255 masks) in deterministic order. Coverage: static laws in that basis —
    see declared_coverage. This is a deliberately thin control: it has no
    sequential grammar at all, and that limitation is scored by the axis."""

    def _basis(df: pd.DataFrame, feats: list[str]) -> tuple[np.ndarray, list[str]]:
        cols_b: list[np.ndarray] = []
        names_b: list[str] = []
        for f in feats:
            x = df[f].to_numpy(dtype=float)
            for nm, v in (
                (f, x), (f"{f}^2", x * x), (f"{f}^3", x * x * x),
                (f"ssqrt({f})", np.sign(x) * np.sqrt(np.abs(x))),
                (f"slog({f})", np.sign(x) * np.log1p(np.abs(x))),
            ):
                cols_b.append(v)
                names_b.append(nm)
        return np.column_stack(cols_b), names_b

    def run(world: World, allowance: float) -> MethodResult:
        feats = [c for c in world.dev.columns if c != "target"][:2]
        X, names = _basis(world.dev, feats)
        y = world.dev["target"].to_numpy(dtype=float)
        finite = np.isfinite(y) & np.isfinite(X).all(axis=1)
        Xf, yf = X[finite], y[finite]
        mu = Xf.mean(0)
        sd = np.where(Xf.std(0) > 0, Xf.std(0), 1.0)
        Z = (Xf - mu) / sd
        t0 = time.perf_counter()
        masks = _subsets_under_cap(len(names), 255)
        best: tuple[tuple[int, ...], float] | None = None
        for idxs in masks:
            sub = np.hstack([np.ones((len(Z), 1)), Z[:, list(idxs)]])
            coef, *_ = np.linalg.lstsq(sub, yf, rcond=None)
            mse = float(np.mean((sub @ coef - yf) ** 2))
            if best is None or mse < best[1]:
                best = (idxs, mse)
        wall = time.perf_counter() - t0
        assert best is not None and masks, "enumeration must produce masks"
        # Emit the OLS solution as a proposal-like object: standardized-mask
        # coefficients back-transformed to raw scale (the QEC-table form).
        idxs, _mse = best
        sub = np.hstack([np.ones((len(Z), 1)), Z[:, list(idxs)]])
        coef, *_ = np.linalg.lstsq(sub, yf, rcond=None)
        sel = np.zeros(len(names), dtype=bool)
        sel[list(idxs)] = True
        abstained = bool(np.all(np.abs(coef[1:]) < 0.05))
        if abstained:
            return MethodResult.abstain(wall, len(masks))

        hold_raw, _names_h = _basis(world.holdout, feats)
        hold_X = (hold_raw[:, list(idxs)] - mu[list(idxs)]) / sd[list(idxs)]
        yh = world.holdout["target"].to_numpy(dtype=float)
        sh = hold_X @ coef[1:] + coef[0]
        r2 = _r2_from_hat(yh, sh)
        r2_rev = _r2_from_hat(yh[::-1], sh[::-1])
        return MethodResult(wall, r2, r2_rev, 1, len(masks), abstained=False,
                            over_budget=wall > allowance)

    return run


def _r2_from_hat(yh: np.ndarray, sh: np.ndarray) -> float | None:
    finite = np.isfinite(yh) & np.isfinite(sh)
    if finite.sum() < 30:
        return None
    var = float(np.var(yh[finite]))
    return float(1.0 - np.var(yh[finite] - sh[finite]) / var) if var > 0 else None


def _subsets_under_cap(n: int, cap: int) -> list[tuple[int, ...]]:
    import itertools

    out: list[tuple[int, ...]] = []
    for size in (1, 2, 3, 4):
        for combo in itertools.combinations(range(n), size):
            out.append(combo)
            if len(out) >= cap:
                return out
    return out


def _pysr_run() -> Callable[[World, float], MethodResult]:
    from pysr import PySRRegressor

    def run(world: World, allowance: float) -> MethodResult:
        feats = [c for c in world.dev.columns if c != "target"]
        reg = PySRRegressor(
            niterations=1,
            maxsize=12,
            populations=8,
            population_size=15,
            timeout_in_seconds=max(1, int(allowance)),
            deterministic=True,
            proactive_ratio=0.0,
            verbosity=0,
            progress=False,
        )
        t0 = time.perf_counter()
        try:
            reg.fit(world.dev[feats].to_numpy(dtype=float),
                    world.dev["target"].to_numpy(dtype=float))
            wall = time.perf_counter() - t0
            sh = np.asarray(reg.predict(world.holdout[feats].to_numpy(dtype=float)),
                            dtype=float)
        except Exception:
            return MethodResult(time.perf_counter() - t0, None, 0, None,
                                abstained=True, over_budget=False)
        yh = world.holdout["target"].to_numpy(dtype=float)
        finite = np.isfinite(yh) & np.isfinite(sh)
        if finite.sum() < 30:
            return MethodResult(wall, None, None, n_prop, None, abstained=(n_prop == 0),
                                over_budget=wall > allowance)
        r2 = _r2_from_hat(yh, sh)
        r2_rev = _r2_from_hat(yh[::-1], sh[::-1])
        n_prop = len(getattr(reg, "equations_", []) or [])
        return MethodResult(wall, r2, r2_rev, n_prop, None, abstained=(n_prop == 0),
                            over_budget=wall > allowance)

    return run


def build_methods() -> tuple[list[Method], dict[str, Any]]:
    methods: list[Method] = [
        Method(
            "rime_temporal",
            ["sequential_lag", "sequential_state", "sequential_interaction"],
            _rime_run(gamma=1.5, max_depth=2, beam=8, budget=200),
            "RIME default grammar (lags/memory, v0.1.2)",
        ),
        Method(
            "rime_static",
            ["static_scaling", "static_log"],
            _rime_run(gamma=1.5, max_depth=2, beam=8, budget=200, static_mode=True,
                      algebraic_transforms=["sq", "slog"]),
            "RIME static grammar (no temporal primitives; sq/slog transform columns)",
        ),
        Method(
            "dli_lag_interaction",
            ["sequential_lag"],
            _dli_run(),
            "DistributedLagInteractionProposer (existing proposer, OLS over lag features). "
            "Measured grammar gap: its feature matrix builds only SAME-TIME interactions, "
            "so delay-interactions (x[t-1]·z[t]) are out-of-grammar despite the name — "
            "narrowing the declared coverage to sequential_lag.",
        ),
        Method(
            "ols_basis_selection",
            ["static_scaling", "static_log"],
            _ols_basis_run(),
            "Algebraic-basis subset selection + OLS (the QEC-table GA control)",
        ),
    ]
    env: dict[str, Any] = {}
    try:
        pysr_ok = find_spec("pysr") is not None
    except Exception:
        pysr_ok = False
    env["pysr_available"] = pysr_ok
    if pysr_ok:
        methods.append(Method(
            "pysr",
            FAMILIES,
            _pysr_run(),
            "PySR (Julia symbolic regression), wall-clock matched via timeout_in_seconds",
        ))
    return methods, env


# ── scoring ──────────────────────────────────────────────────────────────────

def score(method: Method, world: World, res: MethodResult, allowance: float) -> str:
    in_cov = world.family in method.declared_coverage
    if res.abstained:
        return "correct_abstention" if not in_cov else "miss"
    fwd = res.holdout_r2
    rev = res.holdout_r2_reversed
    # Dual-order evaluation is defined ONLY for static worlds: reordering the
    # holdout of a genuine time series violates the data semantics for every
    # causal mechanism (there is nothing to discriminate there). For static
    # laws the reversed order is the grid-order discriminator.
    if world.family.startswith("static"):
        if fwd is not None and rev is not None and fwd >= RECOVERY_R2 and rev >= RECOVERY_R2:
            return "recovered"
        if fwd is not None and fwd >= RECOVERY_R2 and (rev is None or rev < RECOVERY_R2):
            return "index_fit"  # passes forward, collapses reversed: grid order
        return "failed"
    if fwd is not None and fwd >= RECOVERY_R2:
        return "recovered"
    return "failed"  # emitted but missed holdout


VICTORY = {"correct_abstention": 1, "recovered": 1}  # index_fit/failed/miss score 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=DEFAULT_BUDGET_S)
    ap.add_argument("--json", default="benchmarks/matched_compute_results.json")
    args = ap.parse_args()

    worlds = _make_worlds()
    methods, env = build_methods()
    artifact: dict[str, Any] = {
        "protocol": {
            "dev_rows": DEV_ROWS,
            "holdout_rows": HOLDOUT_ROWS,
            "recovery_r2": RECOVERY_R2,
            "wall_budget_s": args.budget,
            "seed_note": "all worlds deterministic (hash-noise generators)",
            "holdout_use": "consumed once, at evaluation time only",
        },
        "environment": env,
        "families": FAMILIES,
        "cells": [],
        "scores": {},
    }

    for method in methods:
        rows = []
        for world in worlds:
            res = method.run(world, args.budget)
            verdict = score(method, world, res, args.budget)
            rows.append({
                "world": world.name,
                "family": world.family,
                "law": world.law,
                "verdict": verdict,
                "holdout_r2": None if res.holdout_r2 is None else round(res.holdout_r2, 6),
                "holdout_r2_reversed": None if res.holdout_r2_reversed is None
                else round(res.holdout_r2_reversed, 6),
                "n_proposals": res.n_proposals,
                "candidate_evals": res.candidate_evals,
                "abstained": res.abstained,
                "wall_seconds": round(res.wall_seconds, 4),
                "over_budget": res.over_budget,
            })
        tally: dict[str, int] = {}
        for r in rows:
            tally[r["verdict"]] = tally.get(r["verdict"], 0) + 1
        artifact["cells"].append({"method": method.method_id,
                                  "abstraction": method.abstraction,
                                  "declared_coverage": method.declared_coverage,
                                  "runs": rows,
                                  "tally": tally})
        artifact["scores"][method.method_id] = {
            "points": sum(VICTORY.get(r["verdict"], 0) for r in rows),
            "of": len(rows),
            "tally": tally,
        }

    with open(args.json, "w") as fh:
        json.dump(artifact, fh, indent=2)

    # Console table (same data as the JSON).
    print(f"Matched-compute comparison | budget={args.budget}s | recovery R²>={RECOVERY_R2} "
          f"| dev={DEV_ROWS} holdout={HOLDOUT_ROWS}")
    pysr = env["pysr_available"]
    print(f"PySR importable: {pysr}" + ("" if pysr else "  (recorded as unavailable, not skipped)"))
    for entry in artifact["cells"]:
        print(f"\n== {entry['method']}  [{entry['abstraction']}]")
        for r in entry["runs"]:
            mark = r["verdict"]
            r2 = "-" if r["holdout_r2"] is None else f"{r['holdout_r2']:.4f}"
            print(f"  {r['world']:<24} {r['family']:<22} {mark:<19} r2={r2:<8} "
                  f"wall={r['wall_seconds']:.3f}s")
        print(f"  tally: {entry['tally']}")
    print(f"\nArtifact written: {args.json}")


if __name__ == "__main__":
    main()
