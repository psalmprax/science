"""Competitor adapters for PREREG-APEDP-EXTBENCH-60-2026-09-26 (frozen protocol).

Adapter contract (§4): input = public CSVs + budget; output = AdapterResult with
(a) expression string, (b) predictions on train/val/near/far grids,
(c) wall-clock used, (d) abstention flag. File-based grids keep the blind
IndependentEvaluator reusable for systems with no Python API.

Environment notes (recorded 2026-09-26, post-freeze provisioning):
- APEDP: worktree state at freeze (see protocol §2); no pip changes.
- PySINDy 2.1.0 installed into .venv (competitor side only; APEDP pins unchanged).
- PySR 2.5.0: NOT installed (requires Julia toolchain) — adapter is defined and
  import-guarded; provision a container with Julia + pysr==2.5.0 at execution.
- AI-Feynman @ a05bc4a: NOT installed (GitHub repo, not a pip package) — adapter
  is defined and import-guarded; clone at the pinned commit at execution.

Scope honesty: PySINDy discovers dynamical systems from trajectories. It competes
only on block O (sorted trajectory + finite-difference derivatives, default STLSQ
optimizer and library); on all other blocks it returns OUT_OF_SCOPE abstention
rather than a misapplied fit. Forcing it onto static regression would
misrepresent the method.
"""
from __future__ import annotations

import csv
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AdapterResult:
    """Per-problem competitor output (§4 contract)."""

    competitor: str
    problem_id: str
    expression: str
    abstained: bool
    abstain_reason: str = ""
    predictions: dict[str, list[float]] = field(default_factory=dict)
    wall_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "competitor": self.competitor,
            "problem_id": self.problem_id,
            "expression": self.expression,
            "abstained": self.abstained,
            "abstain_reason": self.abstain_reason,
            "predictions": {k: [float(v) for v in vals] for k, vals in self.predictions.items()},
            "wall_seconds": float(self.wall_seconds),
            "metadata": dict(self.metadata),
        }


def load_public_grids(corpus_root: str | Path, problem_id: str) -> dict[str, np.ndarray]:
    """Reads public CSVs for one problem. Never touches hidden/ ground truth."""
    root = Path(corpus_root) / "public" / problem_id

    def read(name: str) -> np.ndarray:
        with open(root / name, encoding="utf-8") as f:
            rows = list(csv.reader(f))
        return np.array([[float(v) for v in r] for r in rows[1:]])

    out = {}
    for name in ("train.csv", "validation.csv", "near_ood.csv", "far_ood.csv"):
        mat = read(name)
        out[name] = mat
    return out


class APEDPAdapter:
    """APEDP competitor: AutonomousModelSynthesizer on train/val, lambdify on grids."""

    name = "apedp"

    #: The frozen protocol's own selection constraint, pinned at the adapter
    #: rather than inherited from an engine global.
    #:
    #: PREREG-APEDP-EXTBENCH-60 permits val-MSE selection only, and near-OOD
    #: may disqualify a more complex candidate but never promote one. That was
    #: previously enforced by ``model_synthesis._NEAR_OOD_MAY_PROMOTE``, a
    #: module constant that every caller in the repo shares -- so the round's
    #: protocol was really a statement about the engine, and any other caller
    #: flipping it silently rewrote a frozen round. Keeping the two apart means
    #: this round's guarantee is enforced where the promise is made, and a
    #: later round can promote without asking this file's permission.
    #:
    #: Set it to True to run a round whose protocol allows promotion. Doing so
    #: is a protocol change and needs its own preregistered round (§6.3); it is
    #: not a tuning knob.
    allow_near_ood_promote: bool = False

    def __init__(self, allow_near_ood_promote: bool | None = None) -> None:
        if allow_near_ood_promote is not None:
            self.allow_near_ood_promote = bool(allow_near_ood_promote)

    def run(self, corpus_root: str | Path, problem_id: str) -> AdapterResult:
        from apedp.model_synthesis import AutonomousModelSynthesizer

        t0 = time.time()
        grids = load_public_grids(corpus_root, problem_id)
        tr, va = grids["train.csv"], grids["validation.csv"]
        near = grids["near_ood.csv"]
        # near-OOD is passed in, far-OOD deliberately is not. Both are public
        # splits, so the contract permits either, but far-OOD MSE is the
        # primary metric for block X (protocol §5.3) and block X *is* a
        # re-split of P/E/R. Handing the selector the split it is scored on
        # would make that number a training target. near-OOD is a weaker
        # version of the same contamination -- metric §5.3 also reports
        # near-OOD MSE -- so it is bounded and stated rather than hidden: it
        # only ever breaks ties between candidates already inside the
        # noise floor, and it can only disqualify a *more* complex model
        # (parsimony tiebreak, model_synthesis.py:1660-1670). It never
        # promotes anything -- enforced by ``allow_near_ood_promote = False``
        # above, not by an engine global, so that this round's guarantee
        # survives a change anywhere else in the tree.
        #
        # Passing it is not cosmetic. With near_ood_mse=None on every
        # candidate, three consumers were running on a permanently-absent
        # input: the parsimony tiebreak's OOD guard could never fire, the
        # Stage C near-OOD trade-off probe could never record a trade-off,
        # and `ood_tested` was False for all 60 problems, so every
        # UNIQUE_SUPPORT rationale in the round opened by declaring
        # extrapolation untested -- which was true, but was true of a grid
        # the adapter had already loaded. The stage-D2 evidence report's
        # OOD penalty likewise never applied.
        models = AutonomousModelSynthesizer().synthesize(
            tr[:, 0], tr[:, 1], va[:, 0], va[:, 1],
            near[:, 0], near[:, 1],
            allow_near_ood_promote=self.allow_near_ood_promote,
        )
        top = models[0]
        epi = top.epistemic_state.value if top.epistemic_state else ""
        abstained = epi in {"INSUFFICIENT_EVIDENCE", "NO_VALID_CANDIDATE"}
        preds: dict[str, list[float]] = {}
        # An abstaining model must not also ship a numeric curve. When the
        # engine declines to identify a structure it has no warrant for any
        # value, and emitting one anyway puts a confident-looking number
        # beside an explicit "I cannot answer this" -- two contradictory claims
        # about the same problem, with the number the one a scorer will pick up.
        # Measured on EXT-BENCH-60: the abstained block's predictions ran a
        # median NRMSE of 1.26 (75% of splits above 0.2) against non-promoted
        # problems' 0.005. None of that was model error; it was the output of
        # models that had already said they had nothing to say.
        #
        # This is a contract rule, not a quality filter: abstention is not
        # rewarded for happening to predict well.
        if not abstained:
            try:
                import sympy as sp

                f = sp.lambdify(sp.Symbol("x"), top.sympy_expr, modules=["numpy"])
                for key in ("train.csv", "validation.csv", "near_ood.csv", "far_ood.csv"):
                    with np.errstate(all="ignore"):
                        p = np.asarray(f(grids[key][:, 0]), dtype=float).ravel()
                    preds[key] = [float(v) if np.isfinite(v) else float("nan") for v in p]
            except (ArithmeticError, TypeError, ValueError):
                logger.debug("APEDP prediction eval failed for %s", problem_id, exc_info=True)
        return AdapterResult(
            competitor=self.name, problem_id=problem_id,
            expression=top.expression_str, abstained=abstained,
            abstain_reason="" if not abstained else f"epistemic={epi}",
            predictions=preds, wall_seconds=time.time() - t0,
            metadata={"family": top.structural_family, "val_mse": top.val_mse,
                      # The synthesizer's own OOD score, kept separate from
                      # the grid MSE the blind evaluator computes. These two
                      # should agree to within fit-vs-truth error; where they
                      # do not, the adapter and the engine have diverged,
                      # and this field is what makes that visible.
                      "near_ood_mse": top.near_ood_mse,
                      "far_ood_mse": top.far_ood_mse,
                      "epistemic_state": epi, "proof_claim": "P0"},
        )


class PySINDyAdapter:
    """PySINDy competitor, block O (trajectories) only; OUT_OF_SCOPE elsewhere."""

    name = "pysindy"

    def run(self, corpus_root: str | Path, problem_id: str) -> AdapterResult:
        import pysindy as ps

        t0 = time.time()
        grids = load_public_grids(corpus_root, problem_id)
        if not problem_id.startswith("O"):
            return AdapterResult(
                competitor=self.name, problem_id=problem_id, expression="OUT_OF_SCOPE",
                abstained=True,
                abstain_reason="SINDy discovers dynamical systems from trajectories; "
                               "static-regression blocks are out of method scope",
                wall_seconds=time.time() - t0)
        tr = grids["train.csv"]
        order = np.argsort(tr[:, 0])
        t_sorted = tr[order, 0]
        x_sorted = tr[order, 1]
        dt = np.median(np.diff(t_sorted))
        # Default optimizer + library (author-recommended defaults per protocol).
        model = ps.SINDy()
        model.fit(x_sorted.reshape(-1, 1), t=dt)
        equation = str(model.equations()[0])
        preds: dict[str, list[float]] = {}
        try:
            for key in ("train.csv", "validation.csv", "near_ood.csv", "far_ood.csv"):
                g = grids[key]
                o = np.argsort(g[:, 0])
                sim = model.simulate(g[o, 1][:1], t=np.ascontiguousarray(g[o, 0] - g[o, 0][0]))
                back = np.empty_like(sim)
                back[o] = sim.ravel()[: len(back)]
                preds[key] = [float(v) for v in back]
        except (ArithmeticError, TypeError, ValueError):
            logger.debug("SINDy simulation failed for %s", problem_id, exc_info=True)
        return AdapterResult(
            competitor=self.name, problem_id=problem_id, expression=equation,
            abstained=False, predictions=preds, wall_seconds=time.time() - t0)


class PySRAdapter:
    """PySR competitor (pysr==2.5.0 + Julia toolchain). Not provisioned this turn."""

    name = "pysr"

    def run(self, corpus_root: str | Path, problem_id: str) -> AdapterResult:
        try:
            import pysr  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "PySR not installed: provision a container with the Julia toolchain "
                "and pysr==2.5.0 (frozen pin), then rerun."
            ) from e
        raise NotImplementedError("PySR run path binds at execution time (default settings, 600 s budget).")


class AIFeynmanAdapter:
    """AI-Feynman competitor (repo @ a05bc4a). Not provisioned this turn."""

    name = "ai_feynman"

    def run(self, corpus_root: str | Path, problem_id: str) -> AdapterResult:
        raise RuntimeError(
            "AI-Feynman not installed: clone https://github.com/SJ001/AI-Feynman "
            "at a05bc4a5be23d6eb3e1d0b2f7eb1ab5b78a920ad (frozen pin), then rerun."
        )


class RimeADNativeAdapter:
    """RIME-AD native engine (`evolve_native`) as an adapter.

    Introduced 2026-10-16 for *future* rounds only. The frozen round
    (PREREG-APEDP-EXTBENCH-60-2026-09-26) measured the full
    `AutonomousModelSynthesizer` stack, not the native RIME-AD search.
    Scoring this engine against the same hidden grids would extend a frozen
    record with a second method on already-consumed holdout -- that is a new
    round under protocol section 6.3 with its own preregistration, seed and
    checksummed predictions. Until such a round exists, use scratch
    (`--out-dir`) runs and never write into `tournament_results_external_60/`.

    Contract: reads public CSVs only (`load_public_grids`), fits on train,
    evaluates the winning expression on all four public grids through
    `RustKernelBridge.evaluate_expression`. Expression-eval failure is an
    abstention, and abstentions never carry numeric predictions.
    """

    name = "rime_ad_native"

    def __init__(
        self,
        population_size: int = 300,
        generations: int = 50,
        islands: int = 4,
        seed: int = 42,
    ) -> None:
        self.population_size = population_size
        self.generations = generations
        self.islands = islands
        self.seed = seed

    def run(self, corpus_root: str | Path, problem_id: str) -> AdapterResult:
        from apedp.rust_bridge import RustKernelBridge

        t0 = time.time()
        grids = load_public_grids(corpus_root, problem_id)
        tr = grids["train.csv"]
        bridge = RustKernelBridge()
        pareto, evals = bridge.evolve_native(
            [np.ascontiguousarray(tr[:, 0])],
            ["x"],
            np.ascontiguousarray(tr[:, 1]),
            population_size=self.population_size,
            generations=self.generations,
            islands=self.islands,
            seed=self.seed,
        )
        if not pareto:
            return AdapterResult(
                competitor=self.name, problem_id=problem_id, expression="",
                abstained=True, abstain_reason="native search returned no candidates",
                wall_seconds=time.time() - t0,
            )
        best = min(pareto, key=lambda p: p["mse"])
        expr = str(best["expression"])
        preds: dict[str, list[float]] = {}
        try:
            for key in ("train.csv", "validation.csv", "near_ood.csv", "far_ood.csv"):
                with np.errstate(all="ignore"):
                    p = np.asarray(
                        bridge.evaluate_expression(expr, ["x"], [grids[key][:, 0]]),
                        dtype=float,
                    ).ravel()
                preds[key] = [float(v) if np.isfinite(v) else float("nan") for v in p]
        except (ArithmeticError, TypeError, ValueError):
            logger.debug("RIME-AD native eval failed for %s", problem_id, exc_info=True)
            return AdapterResult(
                competitor=self.name, problem_id=problem_id, expression=expr,
                abstained=True,
                abstain_reason="expression evaluation failed on a public grid",
                wall_seconds=time.time() - t0,
                metadata={"n_evaluations": int(evals), "train_mse": float(best["mse"])},
            )
        return AdapterResult(
            competitor=self.name, problem_id=problem_id, expression=expr,
            abstained=False, predictions=preds, wall_seconds=time.time() - t0,
            metadata={
                "n_evaluations": int(evals),
                "train_mse": float(best["mse"]),
                "population_size": self.population_size,
                "generations": self.generations,
                "islands": self.islands,
                "seed": self.seed,
            },
        )


__all__ = [
    "AIFeynmanAdapter",
    "APEDPAdapter",
    "AdapterResult",
    "PySINDyAdapter",
    "PySRAdapter",
    "RimeADNativeAdapter",
    "load_public_grids",
]
