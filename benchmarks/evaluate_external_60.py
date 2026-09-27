"""Blind evaluator for PREREG-APEDP-EXTBENCH-60-2026-09-26 (§5 metrics).

Reads hidden/ ground truth ONLY at execution time, over locked competitor
outputs. Per protocol §6.2 this module must not run until all competitors'
outputs are locked (currently: APEDP done, SINDy O-block done; PySR and
AI-Feynman outstanding). A fixture self-test (main --self-test) exercises all
scoring paths on synthetic data in /tmp without touching real hidden truth.

Metrics (§5, fixed): family accuracy (canonical, never string match),
parameter-proximity flag (best-effort literal alignment, null when unaligned),
predictive MSE on val/near/far grids (public OOD y; no hidden needed),
abstention correctness vs oracle expectation, false-discovery flag
(UNIQUE_SUPPORT on ABSTAIN/AMBIGUOUS oracle), budget compliance, proof-tier
audit (informational). Paired Wilcoxon + Holm across competitors where paired
attempts suffice; otherwise reported as insufficient pairs.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

EXPECTED_ABSTAIN = "ABSTAIN"
EXPECTED_AMBIGUOUS = "AMBIGUOUS"


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def canonical_family(expression: str) -> str | None:
    """Canonical structural family or None when unparseable (never string match)."""
    try:
        from apedp.symbolic_equivalence import SymbolicEquivalenceEngine

        return SymbolicEquivalenceEngine.canonicalize(expression).structural_family
    except Exception:
        logger.debug("canonicalize failed", exc_info=True)
        return None


def grid_mse(public_dir: Path, grid_name: str, predictions: list[float]) -> float | None:
    """MSE of predicted grid values vs public grid y (no hidden truth needed)."""
    try:
        import csv

        with open(public_dir / grid_name, encoding="utf-8") as f:
            rows = list(csv.reader(f))
        y_true = np.array([float(r[1]) for r in rows[1:]])
        y_pred = np.asarray(predictions, dtype=float)
        if y_pred.shape != y_true.shape or not np.all(np.isfinite(y_pred)):
            return None
        return float(np.mean((y_true - y_pred) ** 2))
    except (ArithmeticError, TypeError, ValueError, OSError):
        return None


def score_one(problem_id: str, result: dict[str, Any], truth: dict[str, Any],
              public_dir: Path) -> dict[str, Any]:
    """Score one competitor output against hidden truth. No tuning hooks."""
    oracle = str(truth.get("expected_outcome", "RECOVER"))
    abstained = bool(result.get("abstained", False))
    out: dict[str, Any] = {"problem_id": problem_id, "oracle": oracle, "abstained": abstained}
    preds = result.get("predictions", {}) or {}
    out["val_mse"] = grid_mse(public_dir, "validation.csv", preds.get("validation.csv", []))
    out["near_ood_mse"] = grid_mse(public_dir, "near_ood.csv", preds.get("near_ood.csv", []))
    out["far_ood_mse"] = grid_mse(public_dir, "far_ood.csv", preds.get("far_ood.csv", []))
    out["wall_seconds"] = result.get("wall_seconds", None)
    meta = result.get("metadata", {}) or {}
    out["proof_tier"] = meta.get("proof_claim", meta.get("proof_tier", None))
    out["epistemic_state"] = meta.get("epistemic_state", None)

    if abstained:
        out["abstention_correct"] = oracle in (EXPECTED_ABSTAIN, EXPECTED_AMBIGUOUS)
        out["family_match"] = None
        out["false_discovery"] = False
        return out

    out["abstention_correct"] = None
    fam = canonical_family(str(result.get("expression", "")))
    out["predicted_family"] = fam
    # Both sides through the SAME canonicalizer: the corpus's hand-written
    # family label lives in a different taxonomy and must never be compared
    # directly (that bug scored 0.0 across the board).
    truth_fam = canonical_family(str(truth.get("ground_truth_formula", "")))
    out["truth_family_canonical"] = truth_fam
    out["family_match"] = bool(fam) and bool(truth_fam) and fam == truth_fam
    epi = str(meta.get("epistemic_state", "") or "")
    out["false_discovery"] = bool(
        oracle in (EXPECTED_ABSTAIN, EXPECTED_AMBIGUOUS) and epi == "UNIQUE_SUPPORT"
    )
    return out


def evaluate_round(corpus_root: str | Path, results_by_competitor: dict[str, list[dict]],
                   ) -> dict[str, Any]:
    """Blind scoring over locked outputs. Opens hidden/ — call only when locked."""
    root = Path(corpus_root)
    per_competitor: dict[str, Any] = {}
    for competitor, results in results_by_competitor.items():
        by_pid = {r["problem_id"]: r for r in results}
        scored: dict[str, Any] = {}
        for pid, res in by_pid.items():
            truth = _load_json(root / "hidden" / pid / "ground_truth.json")
            scored[pid] = score_one(pid, res, truth, root / "public" / pid)
        per_competitor[competitor] = scored

    summary: dict[str, Any] = {}
    for competitor, scored in per_competitor.items():
        attempted = [s for s in scored.values() if not s["abstained"]]
        summary[competitor] = {
            "n_problems": len(scored),
            "n_attempted": len(attempted),
            "n_abstained": len(scored) - len(attempted),
            "family_accuracy": (
                sum(1 for s in attempted if s["family_match"]) / len(attempted) if attempted else None
            ),
            "abstention_precision": _safe_ratio(
                [s for s in scored.values() if s["abstained"]],
                lambda s: bool(s["abstention_correct"])),
            "false_discoveries": sum(1 for s in attempted if s["false_discovery"]),
            "median_val_mse": _median([s["val_mse"] for s in attempted]),
            "median_far_ood_mse": _median([s["far_ood_mse"] for s in attempted]),
            "over_budget": sum(
                1 for s in scored.values()
                if isinstance(s.get("wall_seconds"), (int, float)) and s["wall_seconds"] > 600
            ),
        }
    summary["wilcoxon_val_mse"] = _paired_wilcoxon(per_competitor)
    return {"per_competitor": summary,
            "detail": {c: s for c, s in per_competitor.items()}}


def _safe_ratio(items: list[dict], pred) -> float | None:
    if not items:
        return None
    return sum(1 for s in items if pred(s)) / len(items)


def _median(vals: list) -> float | None:
    xs = [float(v) for v in vals if isinstance(v, (int, float)) and math.isfinite(v)]
    return float(np.median(xs)) if xs else None


def _paired_wilcoxon(per_competitor: dict[str, Any]) -> dict[str, Any]:
    """Paired Wilcoxon on val-MSE over jointly attempted problems + Holm note."""
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return {"status": "scipy-unavailable"}
    comps = sorted(per_competitor)
    out: dict[str, Any] = {}
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            a, b = per_competitor[comps[i]], per_competitor[comps[j]]
            pairs = [(a[pid]["val_mse"], b[pid]["val_mse"]) for pid in a
                     if pid in b and isinstance(a[pid].get("val_mse"), (int, float))
                     and isinstance(b[pid].get("val_mse"), (int, float))]
            key = f"{comps[i]}-vs-{comps[j]}"
            if len(pairs) < 6:
                out[key] = {"status": "insufficient-pairs", "n": len(pairs)}
                continue
            xa = np.array([p[0] for p in pairs])
            xb = np.array([p[1] for p in pairs])
            try:
                stat, pval = wilcoxon(xa, xb, alternative="two-sided")
            except ValueError as e:
                out[key] = {"status": "degenerate", "n": len(pairs), "reason": str(e)}
                continue
            out[key] = {"status": "ok", "n": len(pairs), "statistic": float(stat),
                        "p_value": float(pval),
                        "note": "Holm correction across metric pairs at report time"}
    return out


def _self_test() -> None:
    """Fixture end-to-end test on synthetic data (never touches real hidden/)."""
    import tempfile

    from apedp.symbolic_equivalence import SymbolicEquivalenceEngine

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        truth_fam = SymbolicEquivalenceEngine.canonicalize("2.5*x**1.5").structural_family
        for pid, fam, oracle in [("F01", truth_fam, "RECOVER"),
                                 ("F02", "power_law", "ABSTAIN"),
                                 ("F03", "power_law", "AMBIGUOUS")]:
            (root / "public" / pid).mkdir(parents=True)
            (root / "hidden" / pid).mkdir(parents=True)
            for grid in ("train.csv", "validation.csv", "near_ood.csv", "far_ood.csv"):
                with open(root / "public" / pid / grid, "w", encoding="utf-8") as f:
                    f.write("x,y\n0.5,1.0\n1.0,2.0\n")
            with open(root / "hidden" / pid / "ground_truth.json", "w", encoding="utf-8") as f:
                json.dump({"problem_id": pid, "canonical_structural_family": fam,
                           "expected_outcome": oracle}, f)
        results = [
            {"problem_id": "F01", "expression": "2.5*x**1.5", "abstained": False,
             "predictions": {"validation.csv": [1.0, 2.0], "near_ood.csv": [1.0, 2.0],
                             "far_ood.csv": [1.0, 2.0]}, "wall_seconds": 3.0,
             "metadata": {"epistemic_state": "MULTI_HYPOTHESIS", "proof_claim": "P0"}},
            {"problem_id": "F02", "expression": "", "abstained": True,
             "abstain_reason": "sparse", "predictions": {}, "wall_seconds": 0.5, "metadata": {}},
            {"problem_id": "F03", "expression": "1.0*x + 0.0", "abstained": False,
             "predictions": {"validation.csv": [1.0, 2.0], "near_ood.csv": [1.0, 2.0],
                             "far_ood.csv": [1.0, 2.0]}, "wall_seconds": 4.0,
             "metadata": {"epistemic_state": "UNIQUE_SUPPORT"}},
        ]
        rep = evaluate_round(root, {"demo": results})
        s = rep["per_competitor"]["demo"]
        assert rep["detail"]["demo"]["F01"]["family_match"] is True, rep
        assert rep["detail"]["demo"]["F02"]["abstention_correct"] is True, rep
        assert rep["detail"]["demo"]["F03"]["false_discovery"] is True, rep
        assert s["n_attempted"] == 2 and s["n_abstained"] == 1, s
        assert s["false_discoveries"] == 1, s
        assert s["abstention_precision"] == 1.0, s
        print("self-test OK:", {k: s[k] for k in
                                ("n_attempted", "n_abstained", "family_accuracy",
                                 "abstention_precision", "false_discoveries")})


if __name__ == "__main__":
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        _self_test()
    else:
        print("Refusing: pass --self-test for the fixture check. "
              "Full-round execution opens hidden/ and requires all competitors locked.")
