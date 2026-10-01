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
    # Why the engine declined, carried through from the runner's
    # ``abstain_reason``. This used to be dropped on the floor, so the report
    # could not tell "declined for lack of evidence" apart from "declined and
    # nobody recorded why" -- and on a report whose whole purpose is honest
    # accounting, those two are the same claim as far as a reader is concerned.
    # SynthesizedModel.abstention_reason carries the same contract upstream:
    # None means the model was NOT withheld, so a non-abstained record
    # deliberately reports None here rather than a placeholder string.
    out["abstention_reason"] = (result.get("abstain_reason") or None) if abstained else None

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


def _wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation because the interesting buckets
    here are tiny: at n=4 with 4 successes, a normal interval would claim
    [0.51, 1.49] and clamp it to a point estimate of 100% with no visible
    uncertainty at all. Wilson stays inside [0, 1] and keeps the width visible,
    which is the entire point of attaching an interval to a 4-sample bucket.
    """
    if n <= 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def epistemic_calibration(scored: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Family accuracy per epistemic state, with intervals and an overlap flag.

    This exists because the state tally reads like a verdict and is not one.
    On EXT-BENCH-60 the confident bucket (UNIQUE_SUPPORT) holds 4 of 56
    attempts, so its accuracy carries an interval roughly [0.5, 1.0] -- wide
    enough to sit entirely inside the cautious bucket's. Reporting the two
    rates side by side without that would invite reading "confident is 100%
    accurate" off a four-sample bucket, and would equally invite reading a
    real effect out of the fact that the confident bucket contains the round's
    single false discovery.

    ``intervals_overlap`` is the honest summary and is reported explicitly
    rather than left for the reader to work out: when it is true, the round
    does not measure whether the state predicts correctness. It does not mean
    the states are equivalent -- it means this corpus cannot tell.
    """
    buckets: dict[str, dict[str, Any]] = {}
    for record in scored.values():
        if record.get("abstained"):
            continue
        state = str(record.get("epistemic_state") or "UNSTATED")
        bucket = buckets.setdefault(state, {"n": 0, "family_right": 0,
                                            "false_discoveries": 0})
        bucket["n"] += 1
        if record.get("family_match") is True:
            bucket["family_right"] += 1
        if record.get("false_discovery"):
            bucket["false_discoveries"] += 1

    for bucket in buckets.values():
        n, right = bucket["n"], bucket["family_right"]
        lo, hi = _wilson_interval(right, n)
        bucket["family_accuracy"] = (right / n) if n else None
        bucket["family_accuracy_ci95"] = [lo, hi]
        # An explicit sample-size warning: a rate read off a handful of cases
        # should not be quoted without its n, so n travels with the rate.
        bucket["too_small_to_rank"] = n < 10

    overlap = None
    rates = [b for b in buckets.values() if b["n"] > 0]
    if len(rates) >= 2:
        overlap = all(
            not (a["family_accuracy_ci95"][1] < b["family_accuracy_ci95"][0]
                 or b["family_accuracy_ci95"][1] < a["family_accuracy_ci95"][0])
            for i, a in enumerate(rates) for b in rates[i + 1:]
        )
    return {
        "by_state": buckets,
        "intervals_overlap": overlap,
        # n_attempted is reported so a reader can tell "one state, nothing to
        # compare" from "nothing was scored". Those are very different claims
        # and the verdict text below used to make the second one while
        # claiming the first: on an empty round this returned "single state
        # present", naming a state that did not exist.
        "n_attempted": sum(b["n"] for b in buckets.values()),
        "verdict": (
            "nothing scored -- there is no calibration to report"
            if not buckets else
            ("not measurable on this corpus -- the per-state intervals overlap, "
             "so no ordering of states by accuracy is supported"
             if overlap else
             ("single state present; nothing to compare" if len(rates) < 2
              else "intervals are disjoint; the ordering is supported by this corpus"))
        ),
    }


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
    expected = _expected_problem_ids(root)
    not_fully_scored: dict[bool | None, list[str]] = {False: [], None: []}
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
            # Whether the epistemic state predicts anything, stated with the
            # uncertainty attached instead of left to a reader who will not
            # compute it. See epistemic_calibration.
            "epistemic_calibration": epistemic_calibration(scored),
            # Coverage of the round itself. Without this, a run that scored
            # 3 of 60 problems emits a summary structurally identical to a
            # complete one -- every headline metric null or 0, nothing saying
            # three.
            #
            # ``problems_missing`` is None when the manifest is unreadable,
            # because the specific ids cannot be known. ``round_complete`` is
            # still decidable in one case without it: a competitor that scored
            # nothing has provably not returned every problem, whatever the
            # corpus says, so that is False rather than None. Collapsing the
            # two would let an empty competitor hide behind a missing file.
            "problems_missing": (None if expected is None
                                 else sorted(expected - set(scored))),
            "round_complete": (
                False if not scored else
                None if expected is None else
                not (expected - set(scored))),
        }
        # Recorded while the summary dict is in hand. Reading it back out of
        # ``summary`` later would have to exclude the non-competitor keys this
        # dict also carries (wilcoxon, round_integrity), and getting that
        # exclusion wrong reports a phantom partial round.
        complete = summary[competitor]["round_complete"]
        if complete is not True:
            not_fully_scored[complete].append(competitor)
    summary["wilcoxon_val_mse"] = _paired_wilcoxon(per_competitor)
    # Top-level, so it cannot be missed by a reader who only looks at the
    # aggregate. A check that scores nothing must say so where it will be read.
    #
    # "Short" and "could not be checked" are kept apart. Collapsing them into
    # one list would make an unreadable manifest read as a partial round --
    # wrong, but at least alarming -- and the reverse mistake, rendering
    # unknown as fine, is the silent one this whole block exists to prevent.
    short = sorted(not_fully_scored[False])
    unchecked = sorted(not_fully_scored[None])
    problems: list[str] = []
    if short:
        problems.append(
            f"{len(short)} competitor(s) did not return every problem "
            f"({', '.join(short)}); their headline numbers describe a subset")
    if unchecked:
        problems.append(
            f"{len(unchecked)} competitor(s) could not be checked for "
            f"completeness ({', '.join(unchecked)}) because the corpus "
            "manifest is unreadable")
    summary["round_integrity"] = {
        "n_competitors": len(per_competitor),
        "n_expected_problems": None if expected is None else len(expected),
        "competitors_scoring_a_subset": short,
        "competitors_with_unknown_completeness": unchecked,
        "verdict": (
            "no competitor produced any result -- nothing was scored"
            if not any(per_competitor.values()) else
            ("every competitor scored the full corpus" if not problems else
             "NOT A COMPLETE ROUND: " + "; ".join(problems))
        ),
    }
    return {"per_competitor": summary,
            "detail": dict(per_competitor)}


def _expected_problem_ids(root: Path) -> set[str] | None:
    """Problem ids the corpus manifest defines, or None if it cannot be read.

    Returns None rather than an empty set on a missing or malformed manifest.
    The difference is the point: an empty set would make every round look
    complete, and "I could not check" must never render as "nothing to check".
    """
    manifest = root / "manifest.json"
    try:
        problems = json.loads(manifest.read_text(encoding="utf-8"))["problems"]
        ids = {p["problem_id"] for p in problems}
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return ids or None


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
        # The report must preserve WHY an abstention happened. The fixture
        # writes abstain_reason="sparse"; score_one used to discard it, so
        # every abstention in the report read the same whether or not the
        # engine had an actual reason. Asserted directly so a future rename on
        # either side fails here instead of quietly flattening every
        # abstention in the corpus to the same unexplained blank.
        assert rep["detail"]["demo"]["F02"]["abstention_reason"] == "sparse", rep        # Non-abstained records report None, not a placeholder: "not withheld"
        # and "withheld for an unknown reason" must stay distinguishable.
        for pid in ("F01", "F03"):
            assert rep["detail"]["demo"][pid]["abstention_reason"] is None, (
                f"{pid} did not abstain but carries an abstention reason"
            )
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
    _calibration_unit_checks()


def _calibration_unit_checks() -> None:
    """The calibration block must refuse to over-read a small bucket."""
    # Uses the module's own globals rather than re-importing by package path:
    # importing "benchmarks.evaluate_external_60" from inside the same module
    # makes mypy see the file under two module names and refuse to check it.
    _wilson_interval, epistemic_calibration = (
        globals()["_wilson_interval"], globals()["epistemic_calibration"])

    # Wilson stays inside [0,1] where the normal approximation would not: at
    # n=4 with 4 successes the normal interval reaches 1.49 before clamping.
    lo, hi = _wilson_interval(4, 4)
    assert 0.0 <= lo < 1.0 and hi <= 1.0, (lo, hi)
    assert hi - lo > 0.4, (lo, hi)  # uncertainty must stay visible
    assert _wilson_interval(0, 0) == (0.0, 0.0)

    def rec(state, match, false=False, abstained=False):
        return {"abstained": abstained, "epistemic_state": state,
                "family_match": match, "false_discovery": false}

    # Two tiny buckets, both perfect-looking, different sizes: the intervals
    # must overlap, so the verdict must say "not measurable" rather than
    # promoting the smaller or the larger.
    out = epistemic_calibration({
        "a": rec("UNIQUE_SUPPORT", True), "b": rec("UNIQUE_SUPPORT", True),
        "c": rec("MULTI_HYPOTHESIS", True), "d": rec("MULTI_HYPOTHESIS", False),
    })
    assert out["intervals_overlap"] is True, out
    assert "not measurable" in out["verdict"], out["verdict"]
    assert out["by_state"]["UNIQUE_SUPPORT"]["n"] == 2, out
    assert out["by_state"]["UNIQUE_SUPPORT"]["too_small_to_rank"] is True, out

    # A large, cleanly separated bucket is reportable. 100 vs 25 over 20 and
    # 20 cases is far enough apart that the intervals do not touch.
    big = {f"u{i}": rec("UNIQUE_SUPPORT", True) for i in range(20)}
    big.update({f"m{i}": rec("MULTI_HYPOTHESIS", i >= 15) for i in range(20)})
    out2 = epistemic_calibration(big)
    assert out2["intervals_overlap"] is False, out2
    assert out2["by_state"]["UNIQUE_SUPPORT"]["too_small_to_rank"] is False, out2

    # Abstentions are excluded: they have no family verdict to score, and
    # counting them as wrong would penalize the engine for declining.
    out3 = epistemic_calibration({"a": rec("MULTI_HYPOTHESIS", True),
                                  "b": rec("MULTI_HYPOTHESIS", None, abstained=True)})
    assert out3["by_state"]["MULTI_HYPOTHESIS"]["n"] == 1, out3
    print("epistemic_calibration unit checks OK")


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
