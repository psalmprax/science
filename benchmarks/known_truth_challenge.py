"""Known-truth recovery challenge (v0.1.3, reviewer-directed).

The question the QEC experiment left open: can RIME recover the RIGHT
structure when it is identifiable, and abstain when the evidence cannot
identify it? Planted, fully controlled worlds — not a prediction leaderboard.

Protocol (preregistered before any run):

  **Blind discovery.** Methods see only the generated frame (p, y) — never
  the exponent, coefficient, grid choice or formula. Deterministic
  hash-noise worlds; every run is reproducible.

  **Independent axes.** Exponent d ∈ {2, 3, 6}, grid geometry (log / uniform
  / irregular with varying ratios), noise level (1×, 3×), coefficient C and
  an offset variant vary independently. Out-of-grammar negative control:
  exponential growth (exp) on the same axis.

  **In-grammar mapping (declared, verifiable).** With transform columns
  {p, p², p³} at the input layer, RIME's own frozen seed catalogue +
  interaction family reaches p²…p⁶ (e.g. cub·cub = p⁶, sq·cub = p⁵): every
  monomial world here IS in grammar; exp() is out.

  **Verdicts per (method, world).**
    recovered_structure       — predictive (holdout R² ≥ 0.99 in BOTH row
        orders; all worlds here are static so reversal-invariance applies)
        AND effective exponent matches the planted one (best single-power
        fit over p¹…p⁶ against the frozen prediction) with |α−C|/C ≤ 5%;
    recovered_predictive_only — passes the fit, fails the structure test
        (observational alias; recorded, never counted as structure);
    failed                    — emitted, missed the fit;
    abstained_correct         — no proposal on an out-of-grammar world.
  Abstention NEVER counts toward the recovery rate (a review-agreed rule).

  **Reported rates.** correct-recovery rate and false-promotion rate per
  method; false promotion = emitted proposal that is at best
  predictive-only or failed on an in-grammar world.

  **Transfer axis.** Each recovered frozen proposal is additionally
  evaluated on the same-d holdouts of the OTHER grid geometries:
  cross-configuration transfer R².

  **v0.1.4 extensions (spec §16d).**
    (1) Observation-noise seed variation: ``noise_seed`` shifts the noise
    phase deterministically (same design points, new draws); seed 0
    reproduces the pinned v0.1.3 single run exactly.
    (2) ``--repeats N``: N full passes over independently-noised draws of
    the same registered worlds -> per-world R²/coef stats and recovery
    bounds (noise-driven uncertainty only; structural axes unchanged).
    (3) ``--adjudicated``: every EMITTED proposal is adjudicated through
    ``apedp_dyn/epistemic/scope_refusal.py`` — the referee fits preregistered
    family-distinct probes (exp, log) from the data alone and promotes only
    UNIQUE_SUPPORT over all of them. The referee never sees the world flag;
    out-of-grammar emissions get REFUSED (adjudicated) rather than counted
    as raw leakage. Abstention still never counts toward recovery.

Status: synthetic known-truth challenge; no real-data claim derives from
it. Artifacts: benchmarks/known_truth_results.json (single run),
benchmarks/known_truth_results_repeats.json (repeat bounds).

Usage:
    python benchmarks/known_truth_challenge.py [--json PATH]
    python benchmarks/known_truth_challenge.py --adjudicated
    python benchmarks/known_truth_challenge.py --repeats 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apedp.reproducibility import write_provenance_artifact

RECOVERY_R2 = 0.99  # preregistered recovery threshold (both row orders)
COEF_TOL = 0.05     # preregistered coefficient tolerance
DEV_ROWS = 600
HOLDOUT_ROWS = 300
POWERS = tuple(range(1, 7))  # effective-exponent basis: p¹ … p⁶


# ── worlds ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class World:
    name: str
    d: int              # planted exponent
    C: float
    grid: str           # log | uniform | irregular
    noise_mult: float
    offset: float
    in_grammar: bool

    @property
    def formula(self) -> str:
        if not self.in_grammar:
            return "y = 5·exp(12·normalized(p))"
        return ("y = %.6g·p^%d" % (self.C, self.d)
                + (f" + {self.offset:g}" if self.offset else ""))


def _hash_noise(n: int, phase: float) -> np.ndarray:
    i = np.arange(n, dtype=np.float64) + phase
    v = np.sin(i * 12.9898) * 43758.5453
    return v - np.floor(v) - 0.5


def _unit_noise(n: int, phase: float) -> np.ndarray:
    """Deterministic unit-standard-deviation noise stream (hash-noise /0.28868)."""
    return _hash_noise(n, phase) * (3.4641016151377546)


def _axis(n: int, grid: str) -> np.ndarray:
    p0, p1 = 1e-4, 1e-3
    if grid == "log":
        return p0 * np.power(p1 / p0, np.arange(n) / (n - 1.0))
    if grid == "uniform":
        return np.linspace(p0, p1, n)
    if grid == "irregular":
        base = p0 * np.power(p1 / p0, np.arange(n) / (n - 1.0))
        jitter = 0.8 + 0.4 * _hash_noise(n, 13.0)  # varying ratios 0.8–1.2
        return base * jitter
    raise ValueError(grid)


@dataclass
class BuiltWorld:
    world: World
    dev: pd.DataFrame
    hold: pd.DataFrame
    sigma: float   # declared observation-noise scale (measurement record)


def build_worlds(noise_seed: int = 0) -> list[BuiltWorld]:
    specs: list[tuple[str, int, float, str, float, float, bool]] = [
        ("d2_C1000_log_1x", 2, 1000.0, "log", 1.0, 0.0, True),
        ("d2_C1000_log_3x", 2, 1000.0, "log", 3.0, 0.0, True),
        ("d2_C1000_uniform_1x", 2, 1000.0, "uniform", 1.0, 0.0, True),
        ("d2_C1000_irregular_1x", 2, 1000.0, "irregular", 1.0, 0.0, True),
        ("d2_C250_log_1x", 2, 250.0, "log", 1.0, 0.0, True),
        ("d2_offset_log_1x", 2, 1000.0, "log", 1.0, 0.002, True),
        ("d3_C100_log_1x", 3, 100.0, "log", 1.0, 0.0, True),
        ("d3_C100_uniform_1x", 3, 100.0, "uniform", 1.0, 0.0, True),
        ("d3_C100_irregular_1x", 3, 100.0, "irregular", 1.0, 0.0, True),
        ("d6_C1e12_log_1x", 6, 1e12, "log", 1.0, 0.0, True),
        ("d6_C1e12_irregular_1x", 6, 1e12, "irregular", 1.0, 0.0, True),
        ("neg_exp_log_1x", 2, 5.0, "log", 1.0, 0.0, False),
    ]
    n = DEV_ROWS + HOLDOUT_ROWS
    built: list[BuiltWorld] = []
    for name, d, C, grid, noise, offset, ig in specs:
        p = _axis(n, grid)
        # Noise is RELATIVE to each world's own signal (1.0× = σ = 1% of the
        # truth's std; 3.0× = 3%). The first cut used an absolute 2e-6 sigma,
        # which at C·p³ ~ 1e-7 was noise-DOMINATED for the d≥3 worlds — a
        # design flaw caught at audit, before any conclusion was cited. With
        # a relative definition the SNR is comparable across exponents and
        # abstention there is a genuine (gated) miss, not sound evidence
        # insufficiency.
        if ig:
            truth = C * p ** d + offset
        else:
            truth = 5.0 * np.exp(12.0 * (p - p.min()) / (p.max() - p.min()))
        # Noise sigma is relative to the DEV-slice truth std (the data the
        # methods actually see). The full-axis std is dominated by the tail
        # of the log axis, which lies entirely in the holdout — computing it
        # there made the dev SNR collapse for steep exponents (a measured
        # design flaw, fixed before any conclusion was cited).
        sigma = noise * 0.01 * float(np.std(truth[:DEV_ROWS]))
        # Observation-noise seed: the phase shifts deterministically per
        # repeat (same design points, new noise draws). Seed 0 reproduces
        # the pinned v0.1.3 single-run artifact exactly.
        phase = 23.0 + 1009.0 * float(noise_seed)
        y = truth + _unit_noise(n, phase) * sigma
        df = pd.DataFrame({"p": p, "target": y})
        built.append(BuiltWorld(
            World(name, d, C, grid, noise, offset, ig),
            df.iloc[:DEV_ROWS].reset_index(drop=True),
            df.iloc[DEV_ROWS:].reset_index(drop=True),
            float(sigma),
        ))
    return built


# ── methods (blind; frozen read-only closures returned) ─────────────────────

@dataclass
class Frozen:
    predict: Callable[[pd.DataFrame], np.ndarray]
    wall_seconds: float
    n_proposals: int


def _rime_propose(dev: pd.DataFrame) -> Frozen | None:
    from apedp_dyn.proposers.rime_proposer import RimeProposer

    proposer = RimeProposer(
        feature_cols=["p"], target_col="target", gamma=1.5,
        max_depth=2, beam=8, budget=200,
        static_mode=True, algebraic_transforms=["sq", "cub"],
        max_proposals=3,
    )
    t0 = time.perf_counter()
    proposals = list(proposer.fit_and_propose(dev))
    wall = time.perf_counter() - t0
    if not proposals:
        return Frozen(lambda fd: np.zeros(len(fd)), wall, 0)
    best = proposals[0]
    scorer = best.compute_score
    return Frozen(lambda fd: np.asarray(scorer(fd), dtype=float), wall, len(proposals))


import itertools


def _ols_propose(dev: pd.DataFrame) -> Frozen:
    """Correctly specified baseline: BIC-selected sparse OLS over {p¹…p⁶, 1}.

    Full standardized design was tried first and is recorded as a FAILURE
    mode specific to extrapolating grids: free high-power terms fit dev
    noise, and a degree-6 polynomial extrapolated beyond the training range
    explodes (holdout R² −33 at 2× the training axis). BIC selection over
    the same monomial set is the repo's own discipline (spec §10 class),
    keeps the correctly-specified form, and extrapolates like the QEC-table
    OLS. Fast (63 subsets, deterministic).
    """
    x = dev["p"].to_numpy(dtype=float)
    y = dev["target"].to_numpy(dtype=float)
    ok = np.isfinite(y) & np.isfinite(x)
    Xr = np.column_stack([x[ok] ** k for k in POWERS])
    mu = Xr.mean(axis=0)
    sd = np.where(Xr.std(axis=0) > 0, Xr.std(axis=0), 1.0)
    Z = (Xr - mu) / sd
    n = int(ok.sum())
    yv = y[ok]

    best: tuple[float, list[int], np.ndarray] | None = None
    for size in range(0, len(POWERS) + 1):
        for idxs in itertools.combinations(range(len(POWERS)), size):
            sub = np.hstack([Z[:, list(idxs)] if idxs else
                             np.zeros((n, 0)), np.ones((n, 1))])
            w, *_ = np.linalg.lstsq(sub, yv, rcond=None)
            mse = float(np.mean((sub @ w - yv) ** 2))
            k_params = size + 1
            bic = n * np.log(max(mse, 1e-300)) + k_params * np.log(n)
            if best is None or bic < best[0]:
                best = (bic, list(idxs), w)
    assert best is not None
    _, idxs, w = best

    def predict(frame: pd.DataFrame) -> np.ndarray:
        xf = frame["p"].to_numpy(dtype=float)
        Xf = np.column_stack([xf ** k for k in POWERS])
        Zf = (Xf - mu) / sd
        sub = np.hstack([Zf[:, list(idxs)] if idxs else
                         np.zeros((len(xf), 0)), np.ones((len(xf), 1))])
        return sub @ w

    return Frozen(predict, 0.0, 1)


# ── evaluation ───────────────────────────────────────────────────────────────

def _r2(yhat: np.ndarray, ytrue: np.ndarray) -> float:
    m = np.isfinite(yhat) & np.isfinite(ytrue)
    if int(m.sum()) < 30:
        return float("nan")
    var = float(np.var(ytrue[m]))
    return float(1.0 - np.var(ytrue[m] - yhat[m]) / var) if var > 0 else float("nan")


def _structure(pred: Callable[[pd.DataFrame], np.ndarray], hold: pd.DataFrame,
               world: World) -> tuple[int, float]:
    """(effective exponent k*, |α−C|/C) from the frozen prediction.

    Selection by STANDARDIZED conditional residual (fit {z_k, 1} with
    z_k = (p^k − μ_k)/σ_k; the raw axis collapses at p⁶ ~ 1e-20 — the same
    conditioning lesson the OLS control taught). The raw coefficient is
    recovered exactly as α = w₁/σ_k, so tolerance checking stays in world
    units.
    """
    p = hold["p"].to_numpy(dtype=float)
    s = np.asarray(pred(hold), dtype=float)
    m = np.isfinite(s)
    p, s = p[m], s[m]
    best_k, best_alpha, best_res = 0, float("nan"), float("inf")
    for k in POWERS:
        col = p ** k
        mu_k, sd_k = float(col.mean()), float(col.std())
        if sd_k <= 0:
            continue
        A = np.column_stack([(col - mu_k) / sd_k, np.ones(len(p))])
        coef, *_ = np.linalg.lstsq(A, s, rcond=None)
        res = float(np.mean((A @ coef - s) ** 2))
        if res < best_res:
            best_k = k
            best_alpha = float(coef[0]) / sd_k
            best_res = res
    rel = abs(best_alpha - world.C) / abs(world.C)
    return best_k, float(rel)


def _verdict(f: Frozen | None, world: World, hold: pd.DataFrame) -> dict[str, Any]:
    row: dict[str, Any] = {}
    if f is None or f.n_proposals == 0:
        row.update({"verdict": "abstained",
                    "holdout_r2": None, "holdout_r2_reversed": None,
                    "structure": None, "coef_relerr": None,
                    "n_proposals": 0, "wall_seconds": round(f.wall_seconds, 4) if f else None})
        return row
    s_fwd = f.predict(hold)
    hold_rev = hold.iloc[::-1]
    s_rev = f.predict(hold_rev)
    y = hold["target"].to_numpy(dtype=float)
    r2 = _r2(s_fwd, y)
    r2_rev = _r2(s_rev, y[::-1])
    pred_ok = (r2 == r2 and r2_rev == r2_rev
               and r2 >= RECOVERY_R2 and r2_rev >= RECOVERY_R2)
    row.update({
        "holdout_r2": None if r2 != r2 else round(r2, 6),
        "holdout_r2_reversed": None if r2_rev != r2_rev else round(r2_rev, 6),
        "n_proposals": f.n_proposals, "wall_seconds": round(f.wall_seconds, 4),
    })
    if world.in_grammar:
        k, rel = _structure(f.predict, hold, world)
        struct_ok = (k == world.d) and (rel <= COEF_TOL)
        row.update({"structure": k, "coef_relerr": round(rel, 6)})
        row["verdict"] = ("recovered_structure" if pred_ok and struct_ok
                          else "recovered_predictive_only" if pred_ok else "failed")
    else:
        # Out of grammar: correct = did not propose. Emitted-but-missed is a
        # FAILURE (emitting a structure claim it cannot back), never counted
        # as a correct abstention.
        row.update({"structure": None, "coef_relerr": None})
        row["verdict"] = "failed"
    return row


def _scope_block(f: Frozen | None, dev: pd.DataFrame, hold: pd.DataFrame,
                 sigma: float) -> dict[str, Any]:
    """Referee-side out-of-grammar refusal pass (spec §16d), blind.

    The referee sees only (dev, holdout) frames and the declared noise
    scale — never the world flag, never the planted truth. Probes come from
    apedp_dyn/epistemic/scope_refusal.py; nothing here can turn refusal
    into acceptance by construction (adjudicate() owns the verdict).
    """
    from apedp_dyn.epistemic.scope_refusal import (
        adjudicate_scope,
        build_family_probes,
        scored_from_predict,
    )
    if f is None or f.n_proposals == 0:
        return {"scope_verdict": "not_applicable", "probes": [],
                "reason": "nothing emitted"}
    hold_y = hold["target"].to_numpy(dtype=float)
    emitted = scored_from_predict("emitted", f.predict, dev, hold,
                                  noise_sigma=sigma)
    probes = build_family_probes(dev, hold, noise_sigma=sigma)
    res = adjudicate_scope(emitted, probes, holdout_scale={
        "obs_rms": float(np.sqrt(np.mean(hold_y ** 2))),
        "noise_sigma": sigma, "n_samples": HOLDOUT_ROWS})
    return res.to_dict()


def run_challenge(json_path: str = "benchmarks/known_truth_results.json",
                  noise_seed: int = 0, adjudicated: bool = False) -> dict[str, Any]:
    built = build_worlds(noise_seed=noise_seed)
    artifact = _run_core(built, adjudicated=adjudicated,
                         noise_seed=noise_seed)
    with open(json_path, "w") as fh:
        json.dump(artifact, fh, indent=2)
    dataset = {
        world.world.name: {
            "development_csv": world.dev.to_csv(index=False),
            "holdout_csv": world.hold.to_csv(index=False),
        }
        for world in built
    }
    write_provenance_artifact(
        Path(json_path).with_suffix(Path(json_path).suffix + ".provenance.json"),
        dataset=dataset,
        config={
            "seed": noise_seed,
            "noise_seed": noise_seed,
            "deterministic": True,
            "adjudicated": adjudicated,
            "recovery_r2": RECOVERY_R2,
            "coefficient_relative_error_max": COEF_TOL,
        },
        seed=noise_seed,
        deterministic=True,
        metrics=artifact["rates"],
        results={"result_artifact": str(json_path), "protocol": artifact["protocol"]},
    )
    return artifact


def _run_core(built: list[BuiltWorld], adjudicated: bool = False,
              noise_seed: int = 0) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    frozen_store: dict[tuple[str, str], Frozen] = {}

    for bw in built:
        for method in ("rime_static", "ols_monomial"):
            if method == "rime_static":
                f = _rime_propose(bw.dev)
            else:
                f = _ols_propose(bw.dev)
            frozen_store[(method, bw.world.name)] = f
            row = _verdict(f, bw.world, bw.hold)
            row.update({"world": bw.world.name, "method": method,
                        "family": "in_grammar" if bw.world.in_grammar else "out_of_grammar",
                        "d_true": bw.world.d if bw.world.in_grammar else None,
                        "grid": bw.world.grid, "noise_mult": bw.world.noise_mult,
                        "formula": bw.world.formula})
            if adjudicated:
                row["scope"] = _scope_block(f, bw.dev, bw.hold, bw.sigma)
            cells.append(row)

    # Transfer axis: frozen proposals re-evaluated on same-d OTHER-grid holdouts.
    transfer: list[dict[str, Any]] = []
    for src in cells:
        if src["family"] != "in_grammar":
            continue
        if src["verdict"] not in ("recovered_structure", "recovered_predictive_only"):
            continue
        f = frozen_store[(src["method"], src["world"])]
        for bw in built:
            if (bw.world.d != src["d_true"] or bw.world.grid == src["grid"]
                    or not bw.world.in_grammar):
                continue
            r2 = _r2(f.predict(bw.hold), bw.hold["target"].to_numpy(dtype=float))
            transfer.append({"from": src["world"], "method": src["method"],
                             "to": bw.world.name, "r2": None if r2 != r2 else round(r2, 6)})

    artifact = {
        "protocol": {
            "dev_rows": DEV_ROWS, "holdout_rows": HOLDOUT_ROWS,
            "recovery_r2": RECOVERY_R2, "coef_tol": COEF_TOL,
            "powers_basis": list(POWERS),
            "blind": "methods never see d, C, grid geometry, offset or the formula",
            "abstention": "never counts toward recovery; correct ONLY on out-of-grammar "
                          "(abstained verdict there); on in-grammar worlds abstention is "
                          "recorded as a miss, per the preregistered rule",
            "in_grammar_note": "RIME reaches p^2..p^6 via static grammar + sq/cub "
                               "transform columns and its own interaction family",
            "scope_refusal": ("referee-side adjudicated refusal via "
                              "apedp_dyn/epistemic/scope_refusal.py; referee "
                              "sees only (dev, holdout) + declared noise "
                              "scale, never the world flag (spec §16d)")
                              if adjudicated else "disabled (raw emission scoring)",
            "noise_seed": noise_seed,
            "noise_phase_rule": "phase = 23.0 + 1009.0*seed",
        },
        "cells": cells,
        "transfer": transfer,
        "rates": _rates(cells, adjudicated=adjudicated),
    }
    return artifact


def _stats(values: list[Any]) -> dict[str, float] | None:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n if n > 1 else 0.0
    return {"mean": round(mean, 6), "std": round(var ** 0.5, 6),
            "min": round(min(vals), 6), "max": round(max(vals), 6)}


def _rates(cells: list[dict[str, Any]], adjudicated: bool = False) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for method in ("rime_static", "ols_monomial"):
        rows = [c for c in cells if c["method"] == method]
        ig = [r for r in rows if r["family"] == "in_grammar"]
        oog = [r for r in rows if r["family"] == "out_of_grammar"]
        rec = sum(r["verdict"] == "recovered_structure" for r in ig)
        alias = sum(r["verdict"] == "recovered_predictive_only" for r in ig)
        failed = sum(r["verdict"] == "failed" for r in ig)
        abstain_ig = sum(r["verdict"] == "abstained" for r in ig)
        out[method] = {
            "correct_recovery": rec,
            "in_grammar_worlds": len(ig),
            "recovery_rate": round(rec / len(ig), 4) if ig else None,
            "false_promotion": alias,
            "failures": failed,
            "abstained_in_grammar": abstain_ig,
            "out_of_grammar_abstain": sum(r["verdict"] == "abstained" for r in oog),
            "out_of_grammar_emitted": sum(r["verdict"] == "failed" for r in oog),
        }
        if adjudicated:
            oog_scope = [r["scope"]["scope_verdict"] for r in oog if r.get("scope")]
            ig_scope = [r["scope"]["scope_verdict"] for r in ig if r.get("scope")]
            out[method].update({
                # An out-of-grammar emission that the referee DECLINED is
                # contained: not a false promotion, no scope leak.
                "out_of_grammar_emitted_declined":
                    sum(v == "declined" for v in oog_scope),
                "out_of_grammar_emitted_promoted":
                    sum(v == "promoted" for v in oog_scope),
                # Honest diagnostic: true recoveries the referee refused —
                # any nonzero value must be examined, never tuned away.
                "in_grammar_scope_declined": sum(v == "declined" for v in ig_scope),
            })
    return out


def run_challenge_repeats(json_path: str = "benchmarks/known_truth_results_repeats.json",
                          seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
                          adjudicated: bool = True) -> dict[str, Any]:
    """Preregistered uncertainty pass (§16d): full worlds under noise-seed
    variation.

    Each repeat draws fresh observation noise on the SAME registered design
    points, so the recorded bounds are noise-driven uncertainty of the
    discovery procedure only. ``adjudicated=True`` (default) runs every
    emission through the referee-side scope refusal, so the bounds cover
    the adjudication layer too.
    """
    per_seed: list[tuple[int, dict[str, Any]]] = []
    datasets: dict[str, Any] = {}
    for seed in seeds:
        built = build_worlds(noise_seed=seed)
        datasets[str(seed)] = {
            world.world.name: {
                "development_csv": world.dev.to_csv(index=False),
                "holdout_csv": world.hold.to_csv(index=False),
            }
            for world in built
        }
        per_seed.append((seed, _run_core(built, adjudicated=adjudicated,
                                         noise_seed=seed)))

    cells_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for seed, art in per_seed:
        for row in art["cells"]:
            cells_by_key.setdefault((row["method"], row["world"]), []).append(row)

    per_world: dict[str, Any] = {}
    for (method, world_name), rows in sorted(cells_by_key.items()):
        verdict_counts: dict[str, int] = {}
        scope_counts: dict[str, int] = {}
        for r in rows:
            verdict_counts[r["verdict"]] = verdict_counts.get(r["verdict"], 0) + 1
            s = r.get("scope")
            if s:
                scope_counts[s["scope_verdict"]] = scope_counts.get(
                    s["scope_verdict"], 0) + 1
        entry = {
            "method": method, "world": world_name, "repeats": len(rows),
            "verdict_counts": verdict_counts,
            "holdout_r2": _stats([r["holdout_r2"] for r in rows]),
            "coef_relerr": _stats([r.get("coef_relerr") for r in rows]),
            "structure_counts": {
                str(k): sum(1 for r in rows if r.get("structure") == k)
                for k in {r.get("structure") for r in rows}},
        }
        if scope_counts:
            entry["scope_verdict_counts"] = scope_counts
        per_world[f"{method}|{world_name}"] = entry

    rates = _rates_repeats(per_seed)
    artifact = {
        "protocol": {
            "dev_rows": DEV_ROWS, "holdout_rows": HOLDOUT_ROWS,
            "recovery_r2": RECOVERY_R2, "coef_tol": COEF_TOL,
            "n_repeats": len(seeds), "seeds": list(seeds),
            "noise_seed_rule": "phase = 23.0 + 1009.0*seed; same design "
                               "points, fresh observation-noise draws; "
                               "structural axes unchanged per world",
            "adjudicated": adjudicated,
            "scope_refusal": ("every emission adjudicated per spec §16d"
                              if adjudicated else "disabled"),
            "blind": "methods never see d, C, grid geometry, offset or the formula; "
                     "the referee never sees the world flag either",
        },
        "per_world": per_world,
        "rates": rates,
    }
    with open(json_path, "w") as fh:
        json.dump(artifact, fh, indent=2)
    write_provenance_artifact(
        Path(json_path).with_suffix(Path(json_path).suffix + ".provenance.json"),
        dataset=datasets,
        config={
            "seed": seeds[0] if seeds else 0,
            "seeds": list(seeds),
            "deterministic": True,
            "adjudicated": adjudicated,
            "recovery_r2": RECOVERY_R2,
            "coefficient_relative_error_max": COEF_TOL,
        },
        seed=seeds[0] if seeds else 0,
        deterministic=True,
        metrics=artifact["rates"],
        results={"result_artifact": str(json_path), "protocol": artifact["protocol"]},
    )
    return artifact


def _rates_repeats(per_seed: list[tuple[int, dict[str, Any]]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for method in ("rime_static", "ols_monomial"):
        per_seed_rates: list[float] = []
        oog_declined = oog_promoted = ig_scope_declined = 0
        for seed, art in per_seed:
            rs = _rates(art["cells"], adjudicated=True)[method]
            if rs["recovery_rate"] is not None:
                per_seed_rates.append(rs["recovery_rate"])
            oog_declined += rs.get("out_of_grammar_emitted_declined", 0)
            oog_promoted += rs.get("out_of_grammar_emitted_promoted", 0)
            ig_scope_declined += rs.get("in_grammar_scope_declined", 0)
        out[method] = {
            "recovery_rate_per_seed": per_seed_rates,
            "recovery_rate": _stats(per_seed_rates),
            "correct_recovery_in_repeats":
                "{}/{}".format(sum(1 for v in per_seed_rates if v >= 1.0),
                               len(per_seed_rates)),
            "in_grammar_worlds_per_seed": len(
                [c for c in per_seed[0][1]["cells"]
                 if c["method"] == method and c["family"] == "in_grammar"]),
            "out_of_grammar_emitted_declined": oog_declined,
            "out_of_grammar_emitted_promoted": oog_promoted,
            "in_grammar_scope_declined_across_repeats": ig_scope_declined,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="benchmarks/known_truth_results.json")
    ap.add_argument("--repeats", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--adjudicated", action="store_true")
    args = ap.parse_args()
    if args.repeats > 0:
        seeds = tuple(args.seeds) if args.seeds else tuple(range(args.repeats))
        path = (args.json if "repeats" in args.json
                else args.json.replace(".json", "_repeats.json"))
        artifact = run_challenge_repeats(path, seeds=seeds, adjudicated=True)
        print(f"Known-truth repeats pass | {len(seeds)} noise seeds {list(seeds)} "
              f"| adjudicated scope refusal ON")
        for key, pw in artifact["per_world"].items():
            r2 = "-" if pw["holdout_r2"] is None else (
                f"{pw['holdout_r2']['mean']:.4f}"
                f"±{pw['holdout_r2']['std']:.2g}"
                f" [{pw['holdout_r2']['min']:.3f},{pw['holdout_r2']['max']:.3f}]")
            vc = pw["verdict_counts"]
            print(f"  {key:<45} rec={vc.get('recovered_structure', 0):>2}/{pw['repeats']}"
                  f" R²={r2}")
        print("rates:", json.dumps(artifact["rates"], indent=2))
        print(f"artifact: {path}")
        return
    artifact = run_challenge(args.json, adjudicated=args.adjudicated)
    print(f"Known-truth recovery challenge | recovery R² ≥ {RECOVERY_R2} (both orders)"
          f" | coefficient tolerance {COEF_TOL:.0%}"
          f" | adjudicated={args.adjudicated}")
    for row in artifact["cells"]:
        r2 = "-" if row["holdout_r2"] is None else f"{row['holdout_r2']:.4f}"
        extra = "-" if row.get("structure") is None else (
            f"(k*={row['structure']}, |C-hat-C|/C={row['coef_relerr']})")
        if row.get("scope"):
            extra += f" [scope:{row['scope']['scope_verdict']}]"
        print(f"  {row['method']:<14} {row['world']:<26} {row['verdict']:<28}"
              f" R^2={r2:<9} {extra}")
    print("rates:", json.dumps(artifact["rates"], indent=2))
    print(f"artifact: {args.json}")


if __name__ == "__main__":
    main()
