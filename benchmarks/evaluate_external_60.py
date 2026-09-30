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

Additionally reported, alongside and never instead of the protocol's MSE:
``far_ood_nrmse`` and a ``far_ood_worse_than_mean`` count. Absolute MSE is not
comparable across problems whose y scales differ, and on this corpus it is not
-- 29 of 60 problems have a smaller mean |y| on the far grid than on
validation, so the absolute median is partly a measure of how small the numbers
are. NRMSE fixes the scale, making 1.0 the "no better than the grid mean" line.
The protocol metric is unchanged and still primary; this only makes it legible.
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


#: A grid whose standard deviation is below this fraction of its own mean|y|
#: is saturated -- the signal is flat, so a normalized score measures float
#: noise rather than model quality. A60's far grid sits at 4.6e-05.
_MIN_SPREAD_FRACTION = 1e-2


def grid_nrmse_with_spread(
    public_dir: Path, grid_name: str, predictions: list[float],
) -> tuple[float, float] | None:
    """``(nrmse, coefficient_of_variation)`` or None. See :func:`grid_nrmse`."""
    mse = grid_mse(public_dir, grid_name, predictions)
    if mse is None:
        return None
    try:
        import csv

        with open(public_dir / grid_name, encoding="utf-8") as f:
            rows = list(csv.reader(f))
        y_true = np.array([float(r[1]) for r in rows[1:]])
        spread = float(np.std(y_true))
        if not math.isfinite(spread) or spread <= 0.0:
            return None
        level = float(np.mean(np.abs(y_true)))
        cv = spread / level if level > 0.0 else 0.0
        return float(math.sqrt(mse) / spread), cv
    except (ArithmeticError, TypeError, ValueError, OSError):
        return None


def grid_nrmse(public_dir: Path, grid_name: str, predictions: list[float]) -> float | None:
    """Root-mean-square error normalized by the spread of the true y.

    ``grid_mse`` is an *absolute* error, so it is not comparable across
    problems: on a grid where the signal has decayed toward zero, an absolute
    MSE of 1e-10 is arithmetically excellent and evidentially empty. Measured
    on EXT-BENCH-60, mean |y| on the far grid is below validation's on 29 of 60
    problems (E15: 1.8e-6 vs 7.1e-2), so the absolute far-OOD median rewards
    problems for having small numbers rather than for being right.

    Dividing by the standard deviation of the *true* y removes the scale and
    makes 1.0 the meaningful line: a constant predictor equal to the grid mean
    scores exactly 1.0, so ``nrmse > 1.0`` means "worse than predicting the
    mean". Two problems on EXT-BENCH-60 sit above that line for real reasons --
    C44 (NRMSE 13.2) and E16 (8.8) -- while their absolute MSE ranks them 14th
    and 53rd of 56, invisible until normalized.

    The normalizer is std(y), not mean|y|, and on a *saturated* grid that
    denominator is float noise: A60's far grid is y == 10 to within 4.6e-05 of
    its own level, so std is 4.6e-04 and a model that is right to six figures
    scores 6.7. Normalizing by the level rather than the spread would make that
    case read ~1e-6 and stop it being called a failure, but it would break the
    exact 1.0-is-the-mean-predictor property that makes the number interpretable
    at all. So the saturated case is *reported*, not silently folded away:
    ``grid_nrmse_with_spread`` returns the score plus the grid's own
    coefficient of variation, and ``far_ood_worse_than_mean`` excludes grids
    whose spread is under ``_MIN_SPREAD_FRACTION`` of their own level, with the
    count of exclusions reported alongside. A check that quietly drops the
    awkward cases is the same failure as one that reports nothing.

    Returns None when the grid is constant (zero spread leaves nothing to
    normalize by) or when the prediction is unusable, so a degenerate grid can
    never masquerade as a perfect or a failed score.
    """
    scored = grid_nrmse_with_spread(public_dir, grid_name, predictions)
    return None if scored is None else scored[0]


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
    # Reported *alongside* the absolute MSE, never instead of it. The protocol
    # fixes MSE as the metric, so this does not restate the headline; it makes
    # the headline interpretable across problems whose y scales differ by
    # orders of magnitude.
    out["far_ood_nrmse"] = grid_nrmse(public_dir, "far_ood.csv", preds.get("far_ood.csv", []))
    scored = grid_nrmse_with_spread(public_dir, "far_ood.csv", preds.get("far_ood.csv", []))
    # The grid's own signal-to-noise, so a large NRMSE can be read as "the
    # model is wrong" rather than "the grid is flat and the denominator is
    # noise". Recorded whether or not the score is meaningful.
    out["far_ood_cv"] = None if scored is None else scored[1]
    out["far_ood_grid_saturated"] = bool(
        scored is not None and scored[1] < _MIN_SPREAD_FRACTION
    )
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
            # Scale-free companion to the line above. Read the two together:
            # a low median_far_ood_mse with problems above 1.0 here means the
            # absolute number is being carried by problems whose y happens to
            # be small, not by problems that extrapolate well.
            "median_far_ood_nrmse": _median([s.get("far_ood_nrmse") for s in attempted]),
            # The count that the absolute median cannot surface: attempts worse
            # than the constant mean predictor. Each one is a real failure that
            # a small absolute MSE hides. Saturated grids are excluded because
            # their spread is float noise and the ratio is an artifact of the
            # denominator -- the exclusions are counted, never dropped quietly.
            "far_ood_worse_than_mean": sum(
                1 for s in attempted
                if isinstance(s.get("far_ood_nrmse"), (int, float))
                and math.isfinite(s["far_ood_nrmse"]) and s["far_ood_nrmse"] > 1.0
                and not s.get("far_ood_grid_saturated")),
            "far_ood_saturated_grids": sum(
                1 for s in attempted if s.get("far_ood_grid_saturated")),
            "over_budget": sum(
                1 for s in scored.values()
                if isinstance(s.get("wall_seconds"), (int, float)) and s["wall_seconds"] > 600
            ),
        }
    summary["wilcoxon_val_mse"] = _paired_wilcoxon(per_competitor)
    return {"per_competitor": summary,
            "detail": dict(per_competitor)}


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

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        # The fixture stores a ground-truth *formula*, which is what the real
        # corpus stores and what score_one reads. It previously wrote
        # ``canonical_structural_family`` instead, which the scorer never
        # looks at: truth_fam canonicalized to None, family_match was False for
        # every problem, and this self-test asserted True and so had been
        # failing at HEAD. A fixture whose keys drift from the scorer's is a
        # check that reports the wrong thing, which is worse than no check.
        #
        # The family is derived here through the same canonicalizer the scorer
        # uses, so the two cannot drift apart again, and the expected value is
        # asserted below rather than assumed.
        fixtures = [("F01", "2.5*x**1.5", "RECOVER"),
                    ("F02", "3.0*x**0.5", "ABSTAIN"),
                    ("F03", "1.0*x + 0.0", "AMBIGUOUS")]
        for pid, formula, oracle in fixtures:
            (root / "public" / pid).mkdir(parents=True)
            (root / "hidden" / pid).mkdir(parents=True)
            for grid in ("train.csv", "validation.csv", "near_ood.csv", "far_ood.csv"):
                with open(root / "public" / pid / grid, "w", encoding="utf-8") as f:
                    f.write("x,y\n0.5,1.0\n1.0,2.0\n")
            with open(root / "hidden" / pid / "ground_truth.json", "w", encoding="utf-8") as f:
                json.dump({"problem_id": pid, "ground_truth_formula": formula,
                           "canonical_structural_family": canonical_family(formula),
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
        # The scorer must be able to read the truth it is given. If this is
        # None the fixture and the scorer have drifted, and every family
        # assertion below is vacuous -- so it is checked first, on its own.
        # F02 abstains, so score_one returns before setting the key; an
        # abstention carries no family verdict by design.
        for pid, formula, _ in fixtures:
            if rep["detail"]["demo"][pid]["abstained"]:
                assert "truth_family_canonical" not in rep["detail"]["demo"][pid], (
                    f"{pid} abstained but still carries a family verdict"
                )
                continue
            got = rep["detail"]["demo"][pid]["truth_family_canonical"]
            assert got == canonical_family(formula), (
                f"{pid}: scorer read truth as {got!r}, expected "
                f"{canonical_family(formula)!r} -- fixture/scorer key drift"
            )
        assert rep["detail"]["demo"]["F01"]["family_match"] is True, rep
        assert rep["detail"]["demo"]["F02"]["abstention_correct"] is True, rep
        assert rep["detail"]["demo"]["F03"]["false_discovery"] is True, rep
        # F03's expression is linear against a linear truth, so it matches the
        # family and still counts as a false discovery: UNIQUE_SUPPORT on an
        # AMBIGUOUS oracle. Matching the family does not excuse the claim.
        assert rep["detail"]["demo"]["F03"]["family_match"] is True, rep
        assert s["n_attempted"] == 2 and s["n_abstained"] == 1, s
        assert s["false_discoveries"] == 1, s
        assert s["abstention_precision"] == 1.0, s
        # The fixture predicts [1.0, 2.0] against a true [1.0, 2.0]: exact, so
        # NRMSE is 0.0 and nothing counts as worse-than-mean.
        assert s["median_far_ood_nrmse"] == 0.0, s
        assert s["far_ood_worse_than_mean"] == 0, s
        assert rep["detail"]["demo"]["F01"]["far_ood_nrmse"] == 0.0, rep
        # An abstention has no predictions, so it carries no NRMSE at all --
        # absent, not zero. A check that reports nothing must not be
        # indistinguishable from a check that passed.
        assert rep["detail"]["demo"]["F02"].get("far_ood_nrmse") is None, rep
        print("self-test OK:", {k: s[k] for k in
                                ("n_attempted", "n_abstained", "family_accuracy",
                                 "abstention_precision", "false_discoveries",
                                 "median_far_ood_nrmse", "far_ood_worse_than_mean")})

    _nrmse_unit_checks()


def _nrmse_unit_checks() -> None:
    """grid_nrmse boundary behaviour: mean predictor, better, worse, constant."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)

        def write(ys: list[float]) -> None:
            with open(d / "g.csv", "w", encoding="utf-8") as f:
                f.write("x,y\n")
                for i, y in enumerate(ys):
                    f.write(f"{float(i)},{y!r}\n")

        def nrmse(preds: list[float]) -> float:
            """Assert-present so a None can never be silently compared."""
            got = grid_nrmse(d, "g.csv", preds)
            assert got is not None, f"expected a score for {preds}, got None"
            return got

        base = [1.0, 3.0, 5.0]
        mean_pred = [3.0, 3.0, 3.0]

        write(base)
        # Predicting the grid mean is the 1.0 line by construction. This is the
        # anchor the whole metric rests on, so it is asserted exactly.
        assert nrmse(mean_pred) == 1.0
        # A perfect predictor is 0.0.
        assert nrmse(base) == 0.0
        # Linear in RMSE: doubling the error doubles the score.
        near, far = nrmse([1.5, 3.0, 4.5]), nrmse([2.0, 3.0, 4.0])
        assert 0.0 < near < far
        assert abs(2.0 * near - far) < 1e-9, (near, far)
        # A constant prediction far from the data is worse than the mean.
        assert nrmse([100.0, 100.0, 100.0]) > 1.0
        # The property the absolute metric lacks: NRMSE is invariant to the
        # scale of y, so a decaying problem and a growing one are comparable.
        # The *prediction* has to scale with the grid -- predicting 3.0 against
        # a grid of ~1e-6 is genuinely terrible, and NRMSE says so.
        scale = 1e-6
        write([y * scale for y in base])
        tiny = nrmse([v * scale for v in mean_pred])
        tiny_abs = grid_mse(d, "g.csv", [v * scale for v in mean_pred])
        write(base)
        unit = nrmse(mean_pred)
        unit_abs = grid_mse(d, "g.csv", mean_pred)
        assert abs(tiny - unit) < 1e-9, (tiny, unit)
        assert tiny_abs is not None and unit_abs is not None
        # The absolute MSEs differ by 1e12 for the identical relative error --
        # the reason it cannot rank problems against each other.
        assert abs(unit_abs / tiny_abs - 1e12) < 1e-3, (tiny_abs, unit_abs)
        # And the rescaling is not a free pass: the same wrong prediction is
        # correctly punished on the small grid.
        write([y * scale for y in base])
        assert nrmse(mean_pred) > 1e5

        # A constant grid has zero spread: nothing to normalize by, so the
        # honest answer is None -- not 0.0 (perfect) and not inf (hopeless).
        write([2.0, 2.0, 2.0])
        assert grid_nrmse(d, "g.csv", [2.0, 2.0, 2.0]) is None
        assert grid_nrmse(d, "g.csv", [1.0, 2.0, 3.0]) is None
        # An unusable prediction is None, not a number.
        assert grid_nrmse(d, "g.csv", [1.0, float("nan"), 3.0]) is None
        assert grid_nrmse(d, "g.csv", [1.0]) is None
        assert grid_nrmse(d, "missing.csv", [1.0, 2.0, 3.0]) is None

        # A *saturated* grid -- the A60 case. y is 10 to within 4.6e-05 of its
        # own level, so std is float noise. NRMSE is arithmetically finite and
        # enormous (6.7) while the model is right to six figures, which is why
        # the flag must be qualified rather than trusted raw.
        write([10.0, 10.0 + 4e-4, 10.0 - 4e-4, 10.0 + 2e-4])
        sat = grid_nrmse_with_spread(d, "g.csv", [10.0, 10.0, 10.0, 10.0])
        assert sat is not None
        sat_nrmse, sat_cv = sat
        assert sat_cv < _MIN_SPREAD_FRACTION, sat_cv
        assert sat_nrmse > 1.0, sat_nrmse
        # The inflation is entirely the tiny denominator: predicting the
        # grid's own mean still scores 1.0, because std(y_true) is exactly
        # what the mean predictor's RMSE equals. (That grid is not symmetric
        # about its mean, so 1.0 is the property to check, not a hand value.)
        y_sat = [10.0, 10.0 + 4e-4, 10.0 - 4e-4, 10.0 + 2e-4]
        write(y_sat)
        mean_sat = sum(y_sat) / len(y_sat)
        assert abs(nrmse([mean_sat] * 4) - 1.0) < 1e-9
        # A healthy grid is not flagged.
        write([1.0, 3.0, 5.0, 7.0])
        healthy = grid_nrmse_with_spread(d, "g.csv", [1.0, 3.0, 5.0, 7.0])
        assert healthy is not None and healthy[1] >= _MIN_SPREAD_FRACTION, healthy
    print("grid_nrmse unit checks OK")


if __name__ == "__main__":
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        _self_test()
    else:
        print("Refusing: pass --self-test for the fixture check. "
              "Full-round execution opens hidden/ and requires all competitors locked.")
