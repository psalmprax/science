"""Noise-vs-rediscovery benchmark: planted physics-law worlds measured blind.

Answers the audit question that clean-data anecdote runs leave open
(``docs/ASSESSMENT.md`` §3 fix-order #1): *how often do the discovery engines
recover the right structure as observation noise grows, and is the answer a
rewarded lucky fit or an honest abstention?*

Protocol (preregistered in this docstring before any run, mirroring the
conventions of ``benchmarks/known_truth_challenge.py`` and
``benchmarks/rational_noise_sensitivity_study.py``):

  **Blind discovery.** Engines see only the generated training frame — never
  the formula, the parameter values, the validation split or the noise
  level. Mapping engine output back onto the clean ground truth for scoring
  is the harness's job only.

  **Worlds.** A frozen table of static scalar laws ``y = f(x_1..x_k)``:
  13 embedded worlds spanning the language families the shipped engines
  declare (monomials, reciprocals, exponentials, saturating Langmuir/Hill,
  trigonometric oscillation, products of variables) plus the 5 curated
  Feynman cases imported from ``apedp.ingestion.external_benchmarks``.
  Every evaluation lambda is closed-form numpy; nothing is fitted.

  **Noise axis.** Observation noise is additive Gaussian on the *target
  only*, scaled by the population standard deviation of the clean target on
  the train design (the same relative-sigma convention as
  ``FeynmanBenchmarkCase.generate_data``): sigma_rel in
  {0.0, 0.01, 0.02, 0.05, 0.10}. Inputs are noise-free. Every
  (world, sigma, rep) cell draws from its own seeded
  ``numpy.random.default_rng(seed == hash-of-world + sigma index + rep)``
  stream, so runs are exactly reproducible.

  **Holdout.** A separate fixed-seed design split: train points get the
  noise draw; validation points are scored against their CLEAN targets
  (noise-free), the same convention as ``rational_noise_sensitivity_study``.
  Engines never see validation rows.

  **Verdicts per trial** (abstention never counts toward recovery, the
  review-agreed rule from the known-truth challenge):
    recovered_structure    — predictive AND structure-consistent
    recovered_predictive_only — passes the fit, fails the structure probe
    failed                 — emitted, missed the fit
    abstained              — no usable proposal
  Rates are reported per (engine, sigma): structural recovery rate,
  predictive-only rate, failure rate, abstention rate and mean clean-holdout
  R². An engine that cannot receive a multivariate world is marked
  ``skipped_not_applicable`` and excluded from that engine's denominator.

  **Thresholds** (preregistered here, harness-local; deliberately not
  imported from the canonical promotion spec, whose thresholds govern
  referee state machines rather than benchmark scoring):
    PREDICTIVE_R2_MIN    = 0.95  clean-holdout R² gate for "predictive"
    STRUCTURE_R2_MIN     = 0.90  R² of the engine's predictions against the
      clean truth on a denser probe grid spanning the declared world ranges
      (defaults to the train ranges; a world may declare safer probe ranges
      for singular forms). This exposes train-range aliases that look
      predictive in-range but are not the functional form.
    MULTIVARIATE_SCREENS: proposal count, model count etc. are sanity caps
      on emitted output, not decisions.

  **Engines (adapters).** Each adapter is blind and pluggable:
    synthesis — ``apedp.model_synthesis.AutonomousModelSynthesizer``
      (single-variable worlds only, by its univariate ``sp.Symbol("x")`` API);
    rime_ad — ``apedp_dyn.proposers.rime_ad_proposer.RimeAdProposer``
      (multivariate fan-out; requires the native ``apedp_kernel`` and is
      skipped with an explicit warning when the wheel is absent, mirroring
      the ``NativeSearchUnavailableError`` fail-loud philosophy).
  Adding an engine means adding one adapter function; nothing else changes.

Artifacts: ``benchmarks/feynman_noise_rediscovery_results.json``
(regenerated on each run; the file carries a full config + version stamp so
comparisons across engine upgrades are pinned to the harness text above).

Usage:
    python benchmarks/run_feynman_noise_rediscovery.py
    python benchmarks/run_feynman_noise_rediscovery.py --engine rime_ad --reps 3
    python benchmarks/run_feynman_noise_rediscovery.py --case-substr langmuir \\
        --sigmas 0.0 0.05 --engine synthesis
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Preregistered harness-local scoring constants (see docstring protocol).
# ---------------------------------------------------------------------------
PREDICTIVE_R2_MIN = 0.95
STRUCTURE_R2_MIN = 0.90
PROBE_GRID_POINTS = 128
DEFAULT_SIGMAS: tuple[float, ...] = (0.0, 0.01, 0.02, 0.05, 0.10)
DEFAULT_REPS = 3
DEFAULT_N_TRAIN = 200
DEFAULT_N_VAL = 50
TARGET_COLUMN = "target"

ALLOWED_VERDICTS = (
    "recovered_structure",
    "recovered_predictive_only",
    "failed",
    "abstained",
    "skipped_not_applicable",
)

HARNESS_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# World table (frozen procedural cases, deterministic evaluation lambdas)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EquationWorld:
    """A static scalar law with a closed-form numpy evaluator."""

    equation_id: str
    formula_str: str
    variables: tuple[str, ...]
    variable_ranges: dict[str, tuple[float, float]]
    evaluation_lambda: Callable[..., float]
    probe_ranges: dict[str, tuple[float, float]] | None = None
    description: str = ""

    def eval(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the clean law on an (n_vars,) row of design points."""
        if self.variables == ("x",):
            return np.asarray(
                [self.evaluation_lambda(row[0]) for row in x], dtype=float
            )
        return np.asarray(
            [self.evaluation_lambda(*row) for row in x], dtype=float
        )


def _feynman_worlds() -> list[EquationWorld]:
    """Adapter from the curated ingestion suite to benchmark worlds."""
    from apedp.ingestion.external_benchmarks import AIFeynmanBenchmarkSuite

    worlds: list[EquationWorld] = []
    for case_id, case in sorted(AIFeynmanBenchmarkSuite.CASES.items()):
        worlds.append(
            EquationWorld(
                equation_id=f"feynman_{case_id}",
                formula_str=case.formula_str,
                variables=tuple(case.variables),
                variable_ranges=dict(case.variable_ranges),
                evaluation_lambda=case.evaluation_lambda,
                description=f"curated AI Feynman case {case_id} ({case.description})",
            )
        )
    return worlds


_EMBEDDED_WORLDS: list[EquationWorld] = [
    EquationWorld(
        "mono_power_15",
        "2.0 * x**1.5",
        ("x",),
        {"x": (0.5, 8.0)},
        lambda x: 2.0 * float(np.power(x, 1.5)),
        description="plain monomial power law",
    ),
    EquationWorld(
        "inverse_square",
        "4.0 / x**2",
        ("x",),
        {"x": (0.5, 5.0)},
        lambda x: 4.0 / float(np.power(x, 2)),
        description="reciprocal-square (Coulomb/gravity shape)",
    ),
    EquationWorld(
        "affine_two_var",
        "1.5 * x1 - 0.7 * x2 + 2.0",
        ("x1", "x2"),
        {"x1": (-3.0, 3.0), "x2": (-3.0, 3.0)},
        lambda x1, x2: 1.5 * x1 - 0.7 * x2 + 2.0,
        description="multivariate linear with offset",
    ),
    EquationWorld(
        "quadratic_constant",
        "3.0 * x**2 - 2.0 * x + 1.0",
        ("x",),
        {"x": (-3.0, 3.0)},
        lambda x: 3.0 * x**2 - 2.0 * x + 1.0,
        description="signed quadratic (exercises the v**p fix)",
    ),
    EquationWorld(
        "exponential_decay",
        "5.0 * exp(-0.3 * x)",
        ("x",),
        {"x": (0.0, 10.0)},
        lambda x: 5.0 * float(np.exp(-0.3 * x)),
        description="exponential decay law",
    ),
    EquationWorld(
        "saturated_growth",
        "10.0 * (1 - exp(-0.5 * x))",
        ("x",),
        {"x": (0.0, 12.0)},
        lambda x: 10.0 * (1.0 - float(np.exp(-0.5 * x))),
        description="saturating exponential growth",
    ),
    EquationWorld(
        "langmuir_saturation",
        "8.0 * x / (2.0 + x)",
        ("x",),
        {"x": (0.1, 20.0)},
        lambda x: 8.0 * x / (2.0 + x),
        description="Langmuir/Michaelis-Menten saturating law (fitted half-saturation)",
    ),
    EquationWorld(
        "hill2_cooperative",
        "8.0 * x**2 / (2.0**2 + x**2)",
        ("x",),
        {"x": (0.1, 20.0)},
        lambda x: 8.0 * x**2 / (4.0 + x**2),
        description="cooperative Hill n=2 saturation",
    ),
    EquationWorld(
        "product_two_var",
        "0.5 * x1 * x2",
        ("x1", "x2"),
        {"x1": (0.5, 5.0), "x2": (0.5, 5.0)},
        lambda x1, x2: 0.5 * x1 * x2,
        description="multiplicative interaction of two variables",
    ),
    EquationWorld(
        "sinusoid",
        "3.0 * sin(2.0 * pi * 0.5 * x)",
        ("x",),
        {"x": (0.0, 4.0)},
        lambda x: 3.0 * float(np.sin(2.0 * np.pi * 0.5 * x)),
        description="single-frequency oscillation",
    ),
    EquationWorld(
        "sqrt_law",
        "4.0 * sqrt(x)",
        ("x",),
        {"x": (0.05, 9.0)},
        lambda x: 4.0 * float(np.sqrt(x)),
        probe_ranges={"x": (0.05, 12.0)},
        description="square-root response (pendulum period shape)",
    ),
    EquationWorld(
        "rational_transition",
        "2.0 * x**1.3 / (1.0 + 0.7 * x**2.1)",
        ("x",),
        {"x": (0.5, 4.5)},
        lambda x: (2.0 * float(np.power(x, 1.3))) / (1.0 + 0.7 * float(np.power(x, 2.1))),
        description="off-grid continuous rational transition (pinned rational reference law)",
    ),
]


def all_worlds() -> list[EquationWorld]:
    return _EMBEDDED_WORLDS + _feynman_worlds()


# ---------------------------------------------------------------------------
# Design generation (deterministic per world + sigma + rep)
# ---------------------------------------------------------------------------


def _design_seed(world_id: str, rep: int) -> int:
    """Design-point seed: depends on the world and replica only, never sigma.

    Same design points, new noise draws across the noise axis (the
    ``known_truth_challenge`` v0.1.4 'noise phase' convention), so a cell's
    recovery change is attributable to noise rather than to reshuffled inputs.
    """
    digest = hashlib.sha256(f"{world_id}|design|{rep}".encode()).digest()
    return int.from_bytes(digest[:6], "big")


def _noise_seed(world_id: str, sigma: float, rep: int) -> int:
    """Noise-phase seed: varies with sigma under a fixed design."""
    digest = hashlib.sha256(f"{world_id}|{sigma:.6f}|{rep}".encode()).digest()
    return int.from_bytes(digest[:6], "big")


def build_design(
    world: EquationWorld,
    sigma_rel: float,
    n_train: int,
    n_val: int,
    rep: int,
) -> dict[str, np.ndarray]:
    """Generate one seeded trial: noisy train frame + clean holdout design."""
    design_rng = np.random.default_rng(_design_seed(world.equation_id, rep))
    x_train = np.column_stack(
        [
            design_rng.uniform(world.variable_ranges[v][0], world.variable_ranges[v][1], size=n_train)
            for v in world.variables
        ]
    )
    y_clean_train = world.eval(x_train)
    sigma_abs = sigma_rel * float(np.std(y_clean_train))
    noise_rng = np.random.default_rng(_noise_seed(world.equation_id, sigma_rel, rep))
    y_train = y_clean_train + noise_rng.normal(0.0, sigma_abs, size=n_train) if sigma_abs > 0 else y_clean_train.copy()
    val_rng = np.random.default_rng(_design_seed(world.equation_id + "|val", rep))
    if world.variables == ("x",):
        x_val = np.sort(
            val_rng.uniform(world.variable_ranges["x"][0], world.variable_ranges["x"][1], size=n_val)
        ).reshape(-1, 1)
    else:
        x_val = val_rng.uniform(
            [world.variable_ranges[v][0] for v in world.variables],
            [world.variable_ranges[v][1] for v in world.variables],
            size=(n_val, len(world.variables)),
        )
    y_val = world.eval(x_val)
    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val_clean": y_val,
    }


def probe_design(world: EquationWorld, n_points: int = PROBE_GRID_POINTS) -> np.ndarray:
    """Dense grid over each variable's declared probe range for the structure test."""
    ranges = world.probe_ranges if world.probe_ranges is not None else world.variable_ranges
    axes = [
        np.linspace(ranges[v][0], ranges[v][1], n_points) for v in world.variables
    ]
    if len(axes) == 1:
        return axes[0][:, None]
    grid = np.meshgrid(*axes, indexing="ij")
    return np.column_stack([g.ravel() for g in grid])


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    resid = y_true - y_pred
    ss_res = float(np.sum(np.square(resid)))
    ss_tot = float(np.sum(np.square(y_true - np.mean(y_true))))
    if ss_tot <= 1e-18:
        return 0.0
    return 1.0 - ss_res / ss_tot


def score_trial(
    y_hat_val: np.ndarray,
    y_hat_probe: np.ndarray,
    y_val_clean: np.ndarray,
    y_probe_clean: np.ndarray,
) -> dict[str, Any]:
    """Apply the preregistered predictive/structure gates to engine output."""
    finite_val = np.isfinite(y_hat_val) & np.isfinite(y_val_clean)
    finite_probe = np.isfinite(y_hat_probe) & np.isfinite(y_probe_clean)
    val_frac = float(np.mean(finite_val))
    probe_frac = float(np.mean(finite_probe))
    val_r2 = r_squared(y_val_clean[finite_val], y_hat_val[finite_val]) if val_frac > 0 else float("-inf")
    probe_r2 = r_squared(y_probe_clean[finite_probe], y_hat_probe[finite_probe]) if probe_frac > 0 else float("-inf")
    if val_frac < 0.5 or probe_frac < 0.5:
        return {"verdict": "failed", "val_r2": val_r2, "probe_r2": probe_r2,
                "finite_val_fraction": val_frac, "finite_probe_fraction": probe_frac}
    if val_r2 >= PREDICTIVE_R2_MIN and probe_r2 >= STRUCTURE_R2_MIN:
        verdict = "recovered_structure"
    elif val_r2 >= PREDICTIVE_R2_MIN:
        verdict = "recovered_predictive_only"
    else:
        verdict = "failed"
    return {"verdict": verdict, "val_r2": val_r2, "probe_r2": probe_r2,
            "finite_val_fraction": val_frac, "finite_probe_fraction": probe_frac}


# ---------------------------------------------------------------------------
# Engine adapters (each one is blind: receives only train data)
# ---------------------------------------------------------------------------


@dataclass
class AdapterOutput:
    """Validation predictions from one adapter, or an abstention."""

    abstained: bool
    abstain_reason: str = ""
    y_val_hat: np.ndarray | None = None
    y_probe_hat: np.ndarray | None = None
    detail: dict[str, Any] = field(default_factory=dict)


def _synthesize_predict(model: Any, x_1d: np.ndarray) -> np.ndarray:
    """Evaluate a ``SynthesizedModel`` at 1-D design points via its sympy form.

    ``SynthesizedModel`` exposes no ``predict`` method; its fitted state is
    ``sympy_expr`` plus a parameters mapping, the same state the synthesis
    engine itself lambdifies (``model_synthesis.py`` line ~1737). Constants,
    not data points, are substituted.
    """
    import sympy as sp

    expr = model.sympy_expr
    params = getattr(model, "parameters", {}) or {}
    subs: dict[sp.Symbol, float] = {}
    for name, value in params.items():
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue  # predictive-status strings are not numeric coefficients
        subs[sp.Symbol(str(name))] = value
    if subs:
        expr = expr.subs(subs)
    if not expr.free_symbols:
        return np.full(x_1d.shape, float(expr), dtype=float)
    fn = sp.lambdify(list(expr.free_symbols), expr, modules=["numpy"])
    args = {sym.name: x_1d for sym in expr.free_symbols}
    if len(args) == 1:
        return np.asarray(fn(next(iter(args.values()))), dtype=float).ravel()
    return np.asarray(fn(*[args[k] for k in sorted(args)]), dtype=float).ravel()


def adapter_synthesis(
    world: EquationWorld, design: dict[str, np.ndarray], target_col: str
) -> AdapterOutput:
    """Univariate adapter over ``AutonomousModelSynthesizer`` (sp.Symbol('x') API)."""
    if world.variables != ("x",):
        return AdapterOutput(abstained=True, abstain_reason="not_applicable_multivariate")
    from apedp.model_synthesis import AutonomousModelSynthesizer

    synth = AutonomousModelSynthesizer()
    x_train_1d = design["x_train"][:, 0]
    x_val_1d = design["x_val"][:, 0]
    models = synth.synthesize(x_train_1d, design["y_train"], x_val_1d, design["y_val_clean"])
    if not models:
        return AdapterOutput(abstained=True, abstain_reason="empty_candidate_set")
    top = models[0]
    if getattr(top, "abstention_reason", None) is not None:
        return AdapterOutput(abstained=True, abstain_reason=f"engine_abstained:{top.abstention_reason}")
    try:
        y_val_hat = _synthesize_predict(top, x_val_1d)
        y_probe_hat = _synthesize_predict(top, probe_design(world)[:, 0])
    except Exception as exc:  # noqa: BLE001 - evaluation failure is trial-level, recorded not raised
        return AdapterOutput(abstained=True, abstain_reason=f"evaluation_failed:{type(exc).__name__}")
    return AdapterOutput(
        abstained=False,
        y_val_hat=y_val_hat,
        y_probe_hat=y_probe_hat,
        detail={"top_expression": str(top.expression_str), "epistemic_state": str(top.epistemic_state)},
    )


_KERNEL_AVAILABLE: bool | None = None


def _kernel_available() -> bool:
    global _KERNEL_AVAILABLE
    if _KERNEL_AVAILABLE is None:
        try:
            import apedp_kernel  # noqa: F401
        except ImportError:
            _KERNEL_AVAILABLE = False
        else:
            _KERNEL_AVAILABLE = True
    return _KERNEL_AVAILABLE


def adapter_rime_ad(
    world: EquationWorld, design: dict[str, np.ndarray], target_col: str
) -> AdapterOutput:
    """Multivariate adapter over the RIME-AD fan-out (ranks on dev data only)."""
    if not _kernel_available():
        return AdapterOutput(abstained=True, abstain_reason="apedp_kernel_unavailable")
    from apedp_dyn.proposers.rime_ad_proposer import RimeAdProposer

    df_train = pd.DataFrame(
        design["x_train"], columns=list(world.variables)
    )
    df_train[target_col] = design["y_train"]
    proposer = RimeAdProposer(target_col=target_col)
    proposals = proposer.fit_and_propose(df_train, target_col=target_col)
    if not proposals:
        return AdapterOutput(abstained=True, abstain_reason="empty_proposal_set")
    top = proposals[0]
    df_val = pd.DataFrame(design["x_val"], columns=list(world.variables))
    df_probe = pd.DataFrame(probe_design(world), columns=list(world.variables))
    try:
        y_val_hat = np.asarray(pd.Series(top.compute_score(df_val)).to_numpy(), dtype=float)
        y_probe_hat = np.asarray(pd.Series(top.compute_score(df_probe)).to_numpy(), dtype=float)
    except Exception as exc:  # noqa: BLE001 - scoring failure is trial-level, recorded not raised
        return AdapterOutput(abstained=True, abstain_reason=f"score_failed:{exc}")
    return AdapterOutput(
        abstained=False,
        y_val_hat=y_val_hat,
        y_probe_hat=y_probe_hat,
        detail={
            "top_formula": getattr(top, "formula", ""),
            "candidate_id": getattr(top, "candidate_id", ""),
        },
    )


ADAPTERS: dict[str, Callable[[EquationWorld, dict[str, np.ndarray], str], AdapterOutput]] = {
    "synthesis": adapter_synthesis,
    "rime_ad": adapter_rime_ad,
}


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def summarise(trials: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {v: 0 for v in ALLOWED_VERDICTS}
    for t in trials:
        counts[t["verdict"]] += 1
    denom = [x for x in trials if x["verdict"] != "skipped_not_applicable"]
    n = max(len(denom), 1)
    struct_like = [x["val_r2"] for x in denom if x["val_r2"] is not None and np.isfinite(x["val_r2"])]
    return {
        "n_trials": len(trials),
        "n_scored": len(denom),
        "counts": counts,
        "structure_recovery_rate": counts["recovered_structure"] / n,
        "predictive_only_rate": counts["recovered_predictive_only"] / n,
        "failed_rate": counts["failed"] / n,
        "abstain_rate": counts["abstained"] / n,
        "mean_val_r2": float(np.mean(struct_like)) if struct_like else None,
    }


def _cell_key(trial: dict[str, Any]) -> str:
    """Stable key for one trial cell used by the resume checkpoint."""
    return f"{trial['world']}|{trial['engine']}|{trial['sigma_rel']:.6f}|{trial['rep']}"


def run_benchmark(
    engines: list[str],
    sigmas: list[float],
    reps: int,
    n_train: int,
    n_val: int,
    case_substr: str | None = None,
    output_path: Path | str | None = None,
    state_path: Path | str | None = None,
) -> dict[str, Any]:
    """Run the sweep. With ``state_path`` the run is crash-resumable.

    The state file records every completed trial keyed by
    ``(world, engine, sigma, rep)``; a re-invocation with the same state
    skips completed cells and appends only the missing ones. The state file
    is rewritten after EVERY cell, so a kill between cells costs at most one
    re-run (cells are deterministic, so replays reproduce byte-identical
    verdicts and R² values per the harness test). Resume rejects a state
    file whose recorded protocol cells conflict with the requested ones
    rather than silently mixing two protocols.
    """
    worlds = [w for w in all_worlds() if case_substr is None or case_substr in w.equation_id]
    if not worlds:
        raise ValueError(f"No worlds match --case-substr {case_substr!r}")
    state_file = Path(state_path) if state_path is not None else None
    trials: list[dict[str, Any]] = []
    done: set[str] = set()
    if state_file is not None and state_file.exists():
        try:
            prior = json.loads(state_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Unreadable resume state file {state_file}: {exc}") from exc
        prior_cfg = prior.get("config", {})
        conflicts = [k for k in ("engines", "sigmas", "reps", "n_train", "n_val", "case_substr")
                     if k in prior_cfg]
        requested = {"engines": engines, "sigmas": [f"{s:.6f}" for s in sigmas],
                     "reps": reps, "n_train": n_train, "n_val": n_val, "case_substr": case_substr}
        prior_norm = {k: ([f"{s:.6f}" for s in prior_cfg["sigmas"]] if k == "sigmas" else prior_cfg.get(k))
                      for k in conflicts}
        if any(prior_norm[k] != requested[k] for k in conflicts if prior_cfg.get(k) is not None):
            raise RuntimeError(
                f"Resume state {state_file} was recorded with config {prior_norm}, "
                f"conflicting with the requested {requested}; refusing to mix protocols."
            )
        for trial in prior.get("trials", []):
            done.add(_cell_key(trial))
            trials.append(trial)
        print(f"Resume: {len(trials)} completed trials loaded from {state_file}", flush=True)
    t_start = time.time()
    cells_total = len(worlds) * len(engines) * len(sigmas) * reps
    cells_run = 0
    for world in worlds:
        for engine in engines:
            for sigma in sigmas:
                for rep in range(reps):
                    if state_file is not None and _cell_key(
                        {"world": world.equation_id, "engine": engine,
                         "sigma_rel": sigma, "rep": rep}
                    ) in done:
                        continue
                    cells_run += 1
                    design = build_design(world, sigma, n_train, n_val, rep)
                    t0 = time.time()
                    try:
                        out = ADAPTERS[engine](world, design, TARGET_COLUMN)
                    except Exception as exc:  # noqa: BLE001 - adapter crash is a recorded trial, not a run abort
                        out = AdapterOutput(abstained=True, abstain_reason=f"adapter_error:{type(exc).__name__}:{exc}")
                    wall = time.time() - t0
                    trial: dict[str, Any] = {
                        "world": world.equation_id,
                        "engine": engine,
                        "sigma_rel": sigma,
                        "rep": rep,
                        "wall_seconds": round(wall, 3),
                        "abstained": out.abstained,
                        "abstain_reason": out.abstain_reason,
                    }
                    if out.abstained:
                        trial["verdict"] = "skipped_not_applicable" if out.abstain_reason == "not_applicable_multivariate" else "abstained"
                        trial["val_r2"] = None
                        trial["probe_r2"] = None
                    else:
                        assert out.y_val_hat is not None and out.y_probe_hat is not None
                        y_probe = world.eval(probe_design(world))
                        s = score_trial(out.y_val_hat, out.y_probe_hat, design["y_val_clean"], y_probe)
                        trial.update(s)
                        trial["detail"] = out.detail
                    trials.append(trial)
                    if state_file is not None:
                        state_file.parent.mkdir(parents=True, exist_ok=True)
                        state_file.write_text(
                            json.dumps({"harness_version": HARNESS_VERSION, "trials": trials,
                                        "config": {"engines": engines, "sigmas": [float(s) for s in sigmas],
                                                   "reps": reps, "n_train": n_train, "n_val": n_val,
                                                   "case_substr": case_substr}},
                                       indent=2, sort_keys=True), encoding="utf-8")
                    print(
                        f"[{world.equation_id:>28s}] {engine:>9s} sigma={sigma:.3f} rep={rep} "
                        f"-> {trial['verdict']:<26s} val_r2={trial['val_r2']} wall={trial['wall_seconds']}s",
                        flush=True,
                    )

    by_engine_sigma: dict[str, Any] = {}
    for engine in engines:
        by_engine_sigma[engine] = {}
        for sigma in sigmas:
            subset = [t for t in trials if t["engine"] == engine and t["sigma_rel"] == sigma]
            by_engine_sigma[engine][f"{sigma:.3f}"] = summarise(subset)
    overalls = {engine: summarise([t for t in trials if t["engine"] == engine]) for engine in engines}

    report = {
        "harness_version": HARNESS_VERSION,
        "protocol": {
            "PREDICTIVE_R2_MIN": PREDICTIVE_R2_MIN,
            "STRUCTURE_R2_MIN": STRUCTURE_R2_MIN,
            "sigmas_rel": list(sigmas),
            "reps": reps,
            "n_train": n_train,
            "n_val": n_val,
            "abstention_counts_toward_recovery": False,
            "holdout_scoring": "R2 against clean (noise-free) validation targets; engines see train rows only",
            "structure_test": "dense probe-grid R2 over declared world probe ranges",
        },
        "config": {"engines": engines, "case_substr": case_substr, "worlds": [w.equation_id for w in worlds]},
        "wall_seconds_total": round(time.time() - t_start, 2),
        "cells_scoped": cells_total,
        "cells_executed_this_invocation": cells_run,
        "overall": overalls,
        "by_engine_sigma": by_engine_sigma,
        "trials": trials,
    }
    if output_path is not None:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nReport written to {out_file}")
    return report


def _print_summary(report: dict[str, Any]) -> None:
    print("\n=== Noise-vs-rediscovery summary (structure recovery rate) ===")
    for engine, per_sigma in report["by_engine_sigma"].items():
        row = " | ".join(
            f"sigma={sig:6s}: {s['structure_recovery_rate']:.0%} (PO {s['predictive_only_rate']:.0%}, "
            f"fail {s['failed_rate']:.0%}, abstain {s['abstain_rate']:.0%})"
            for sig, s in sorted(per_sigma.items())
        )
        print(f"{engine:>10s}: {row}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--engine", choices=["all", *sorted(ADAPTERS)], default="all")
    parser.add_argument("--sigmas", type=float, nargs="+", default=list(DEFAULT_SIGMAS))
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS)
    parser.add_argument("--n-train", type=int, default=DEFAULT_N_TRAIN)
    parser.add_argument("--n-val", type=int, default=DEFAULT_N_VAL)
    parser.add_argument("--case-substr", default=None,
                        help="restrict worlds by substring, e.g. langmuir")
    parser.add_argument("--output",
                        default=str(Path(__file__).parent / "feynman_noise_rediscovery_results.json"))
    parser.add_argument("--state",
                        default=str(Path(__file__).parent / "feynman_noise_rediscovery_state.json"),
                        help="crash-resumable checkpoint (per-cell); empty string disables")
    args = parser.parse_args()
    engines = sorted(ADAPTERS) if args.engine == "all" else [args.engine]
    report = run_benchmark(
        engines=engines,
        sigmas=args.sigmas,
        reps=args.reps,
        n_train=args.n_train,
        n_val=args.n_val,
        case_substr=args.case_substr,
        output_path=args.output,
        state_path=args.state or None,
    )
    _print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
