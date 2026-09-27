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

    def run(self, corpus_root: str | Path, problem_id: str) -> AdapterResult:
        from apedp.model_synthesis import AutonomousModelSynthesizer

        t0 = time.time()
        grids = load_public_grids(corpus_root, problem_id)
        tr, va = grids["train.csv"], grids["validation.csv"]
        models = AutonomousModelSynthesizer().synthesize(tr[:, 0], tr[:, 1], va[:, 0], va[:, 1])
        top = models[0]
        epi = top.epistemic_state.value if top.epistemic_state else ""
        abstained = epi in {"INSUFFICIENT_EVIDENCE", "NO_VALID_CANDIDATE"}
        preds: dict[str, list[float]] = {}
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


__all__ = [
    "AIFeynmanAdapter",
    "APEDPAdapter",
    "AdapterResult",
    "PySINDyAdapter",
    "PySRAdapter",
    "load_public_grids",
]
