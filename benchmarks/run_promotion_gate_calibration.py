"""Measure what the promotion gates actually let through.

A gate is a statistical test whether or not it was designed as one, so it has a
false-pass rate on worlds with nothing in them and a power on worlds with
something. This harness measures both for the evidence-taking entry points of
``PromotionTracker`` by simulation, on data whose truth is known because it was
generated here:

  stage 2   in-sample recovery: false-pass rate on pure noise and power on a
            planted information coefficient, by sample size; for the gate as it
            stands (threshold and five positive folds) and for the bare
            threshold, which is what the gate was before folds were computed
  stage 4   out-of-sample transfer of a candidate with no real signal that
            passed stage 2 by luck, including overlapping targets scored with
            and without the declared horizon
  stage 5   economic validity of the best of K zero-edge strategies, K = 1, 20,
            200, with the search journaled in a persistent trial ledger; and
            power on strategies with a true Sharpe ratio
  pipeline  stages 2-5 end to end on a null world and on a planted one

Every rate is reported with a 95% Wilson interval. Nothing here changes a
threshold: the thresholds are the canonical specification's, and this is the
measurement of what they do.

Writes benchmarks/promotion_gate_calibration.json (full run) or the path given
with --json. ``--smoke`` runs a few simulations per cell for wiring tests.

    python benchmarks/run_promotion_gate_calibration.py
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import shutil
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

warnings.filterwarnings("ignore", category=RuntimeWarning)

from apedp_dyn.conf import canonical_spec
from apedp_dyn.epistemic.agnostic_adapter import InvariantKind
from apedp_dyn.epistemic.experiment_ledger import FullLineageExperimentLedger
from apedp_dyn.epistemic.lean_universal_bridge import (
    LeanTheoremVerification,
    LeanVerificationResult,
)
from apedp_dyn.epistemic.promotion_lifecycle import (
    PromotionTracker,
    information_coefficient,
)
from apedp_dyn.epistemic.proof_claim import ProofClaimStatus
from apedp_dyn.epistemic.trial_ledger import CandidateTrialLedger
from apedp_dyn.market.universe import UniverseAudit

DEFAULT_JSON = REPO_ROOT / "benchmarks" / "promotion_gate_calibration.json"
SEED = 20261115
SINGLE = UniverseAudit.single_instrument("SIMULATED")


def wilson(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def rate(successes: int, n: int) -> dict[str, Any]:
    low, high = wilson(successes, n)
    return {"passed": successes, "simulations": n, "rate": successes / n if n else None,
            "wilson95": [round(low, 5), round(high, 5)]}


def _lean_ok(candidate_id: str) -> LeanVerificationResult:
    """Stage 1 is not under test here; every simulated candidate clears it."""
    theorems = [
        LeanTheoremVerification(
            invariant_name=kind.value, invariant_kind=kind, theorem_identifier=kind.value,
            lean_theorem_statement="x = x", lean_proof_script="rfl", is_formally_proven=True,
            proof_method="lean4_kernel", diagnostic_message="",
        )
        for kind in (InvariantKind.SCALE_HOMOGENEITY, InvariantKind.BOUND_POSITIVITY)
    ]
    return LeanVerificationResult(
        candidate_id=candidate_id, formula_repr="f", all_theorems_proven=True,
        proof_claim_status=ProofClaimStatus.P2_VERIFIED, theorems=theorems,
        lean_source_code="theorem t (x : Nat) : x = x := rfl", kernel_exit_code=0,
    )


def _tracker(candidate_id: str = "sim") -> PromotionTracker:
    tracker = PromotionTracker(candidate_id)
    tracker.verify_stage_1_from_lean(_lean_ok(candidate_id))
    return tracker


def _pair(rng: np.random.Generator, n: int, ic: float) -> tuple[np.ndarray, np.ndarray]:
    signal = rng.normal(size=n)
    return signal, ic * signal + math.sqrt(max(1.0 - ic * ic, 0.0)) * rng.normal(size=n)


def _days(start: str, periods: int) -> np.ndarray:
    return np.busday_offset(np.datetime64(start, "D"), np.arange(periods), roll="forward")


# ---------------------------------------------------------------------------
# Stage 2
# ---------------------------------------------------------------------------


def calibrate_stage_2(rng: np.random.Generator, sims: int) -> list[dict[str, Any]]:
    rows = []
    for n in (250, 600, 2000):
        for true_ic in (0.0, 0.02, 0.05, 0.10):
            gate = bare = 0
            for _ in range(sims):
                predictions, targets = _pair(rng, n, true_ic)
                gate += _tracker().verify_stage_2_from_predictions(predictions, targets).passed
                bare += information_coefficient(predictions, targets)[0] >= canonical_spec.STAGE2_MIN_IC
            rows.append({
                "n": n, "true_ic": true_ic, "world": "null" if true_ic == 0.0 else "planted",
                "gate_threshold_and_five_positive_folds": rate(gate, sims),
                "bare_threshold_only": rate(int(bare), sims),
            })
    return rows


# ---------------------------------------------------------------------------
# Stage 4
# ---------------------------------------------------------------------------


def _lucky_stage_2(rng: np.random.Generator, n: int, workdir: Path, tag: str) -> tuple[PromotionTracker, FullLineageExperimentLedger]:
    """A no-signal candidate that got through stages 2 and 3, as a selected one has."""
    while True:
        predictions, targets = _pair(rng, n, 0.0)
        tracker = _tracker(tag)
        if tracker.verify_stage_2_from_predictions(predictions, targets).passed:
            break
    # Stage 3 is not under test in this cell: give it evidence that clears it.
    signal = rng.normal(size=600)
    literature = {"A": rng.normal(size=600), "B": rng.normal(size=600)}
    tracker.verify_stage_3_from_forecasts(
        signal + 0.5 * rng.normal(size=600), signal, {k: 0.1 * v for k, v in literature.items()},
        signal, literature,
    )
    ledger = FullLineageExperimentLedger(persistence_path=workdir / f"{tag}.jsonl")
    ledger.register_candidate(tag, "sim", "f")
    return tracker, ledger


def calibrate_stage_4(rng: np.random.Generator, sims: int, workdir: Path) -> dict[str, Any]:
    independent = 0
    for i in range(sims):
        tracker, ledger = _lucky_stage_2(rng, 600, workdir, f"s4_{i}")
        predictions, targets = _pair(rng, 400, 0.0)
        independent += tracker.verify_stage_4_from_holdout(
            predictions, targets, experiment_ledger=ledger, dataset_id="sim", slice_identifier=f"h{i}"
        ).passed

    # Overlapping 21-period targets and a slow signal, with no relation between
    # them. Scored as if the 400 observations were independent, then with the
    # horizon declared.
    horizon = 21
    naive = declared = 0
    for i in range(sims):
        steps = rng.normal(size=400 + horizon)
        targets = np.convolve(steps, np.ones(horizon), mode="valid")[:400]
        slow = np.convolve(rng.normal(size=400 + horizon), np.ones(horizon), mode="valid")[:400]
        for label, h in (("naive", 1), ("declared", horizon)):
            tracker, ledger = _lucky_stage_2(rng, 600, workdir, f"s4o_{label}_{i}")
            passed = tracker.verify_stage_4_from_holdout(
                slow, targets, experiment_ledger=ledger, dataset_id="sim",
                slice_identifier=f"o_{label}_{i}", horizon=h,
            ).passed
            if label == "naive":
                naive += passed
            else:
                declared += passed
    return {
        "null_candidate_independent_targets": rate(independent, sims),
        "null_candidate_overlapping_targets_horizon_not_declared": rate(naive, sims),
        "null_candidate_overlapping_targets_horizon_declared": rate(declared, sims),
        "holdout_n": 400, "overlap_horizon": horizon,
    }


# ---------------------------------------------------------------------------
# Stage 5
# ---------------------------------------------------------------------------


def _through_stage_4(rng: np.random.Generator, workdir: Path, tag: str) -> PromotionTracker:
    """A candidate with strong evidence for stages 2-4, so stage 5 is what is measured."""
    attempt = 0
    while True:
        name = f"{tag}_{attempt}"
        attempt += 1
        tracker = _tracker(name)
        if not tracker.verify_stage_2_from_predictions(*_pair(rng, 600, 0.30)).passed:
            continue
        signal = rng.normal(size=600)
        literature = {"A": rng.normal(size=600), "B": rng.normal(size=600)}
        if not tracker.verify_stage_3_from_forecasts(
            signal + 0.5 * rng.normal(size=600), signal, {k: 0.1 * v for k, v in literature.items()},
            signal, literature,
        ).passed:
            continue
        ledger = FullLineageExperimentLedger(persistence_path=workdir / f"{name}_e.jsonl")
        ledger.register_candidate(name, "sim", "f")
        if tracker.verify_stage_4_from_holdout(
            *_pair(rng, 400, 0.28), experiment_ledger=ledger, dataset_id="sim", slice_identifier=name
        ).passed:
            return tracker


def calibrate_stage_5(rng: np.random.Generator, sims: int, workdir: Path) -> list[dict[str, Any]]:
    periods, daily_vol = 504, 0.01
    index = _days("2022-01-03", periods)
    turnover = np.full(periods, 0.2)
    rows = []
    for true_sharpe in (0.0, 1.0, 2.0, 3.0):
        for searched in (1, 20, 200):
            passed = 0
            for i in range(sims):
                drift = true_sharpe * daily_vol / math.sqrt(252.0)
                ledger = CandidateTrialLedger(persistence_path=workdir / f"t_{true_sharpe}_{searched}_{i}.jsonl")
                best, best_sharpe = None, -math.inf
                for k in range(searched):
                    returns = rng.normal(drift, daily_vol, periods)
                    sharpe = float(returns.mean() / returns.std(ddof=1) * math.sqrt(252.0))
                    ledger.record_trial(f"k{k}", "sim", "f", "TACTICAL", sharpe, returns=returns,
                                        metadata={"search_id": "sim"})
                    if sharpe > best_sharpe:
                        best, best_sharpe = returns, sharpe
                assert best is not None
                tracker = _through_stage_4(rng, workdir, f"s5_{true_sharpe}_{searched}_{i}")
                passed += tracker.verify_stage_5_from_returns(
                    best, turnover, index, trial_ledger=ledger, universe=SINGLE
                ).passed
            rows.append({
                "true_annual_sharpe": true_sharpe, "strategies_searched": searched,
                "world": "null" if true_sharpe == 0.0 else "planted",
                "best_of_search_passes_stage_5": rate(passed, sims),
                "observations": periods,
            })
    return rows


# ---------------------------------------------------------------------------
# Pipeline, stages 2-5
# ---------------------------------------------------------------------------


def calibrate_pipeline(rng: np.random.Generator, sims: int, workdir: Path) -> list[dict[str, Any]]:
    index = _days("2022-01-03", 504)
    rows = []
    for world, ic in (("null", 0.0), ("planted", 0.10), ("planted_strong", 0.25)):
        reached = dict.fromkeys((2, 3, 4, 5), 0)
        for i in range(sims):
            tag = f"p_{world}_{i}"
            tracker = _tracker(tag)
            # One signal series drives every stage, as it would for a real candidate.
            signal, target = _pair(rng, 2000, ic)
            dev, hold = slice(0, 1200), slice(1200, 2000)
            if not tracker.verify_stage_2_from_predictions(signal[dev], target[dev]).passed:
                continue
            reached[2] += 1
            literature = {"A": rng.normal(size=1200), "B": rng.normal(size=1200)}
            incumbents = {name: np.zeros(1200) for name in literature}  # the "no forecast" incumbent
            if not tracker.verify_stage_3_from_forecasts(
                target[dev], ic * signal[dev], incumbents, signal[dev], literature
            ).passed:
                continue
            reached[3] += 1
            ledger = FullLineageExperimentLedger(persistence_path=workdir / f"{tag}_e.jsonl")
            ledger.register_candidate(tag, "sim", "f")
            if not tracker.verify_stage_4_from_holdout(
                signal[hold], target[hold], experiment_ledger=ledger, dataset_id="sim", slice_identifier=tag
            ).passed:
                continue
            reached[4] += 1
            # Trade the sign of the signal on the holdout; 1% daily volatility.
            position = np.sign(signal[hold][:504])
            gross = position * target[hold][:504] * 0.01
            turnover = np.abs(np.diff(position, prepend=0.0))
            trials = CandidateTrialLedger(persistence_path=workdir / f"{tag}_t.jsonl")
            trials.record_trial(tag, "sim", "f", "TACTICAL",
                                float(gross.mean() / gross.std(ddof=1) * math.sqrt(252.0)), returns=gross)
            if tracker.verify_stage_5_from_returns(
                gross, turnover, index, trial_ledger=trials, universe=SINGLE
            ).passed:
                reached[5] += 1
        rows.append({
            "world": world, "true_ic": ic,
            "development_n": 1200, "holdout_n": 800,
            **{f"reached_stage_{stage}": rate(count, sims) for stage, count in reached.items()},
        })
    return rows


def run(json_path: Path | None, sims: int, mode: str) -> dict[str, Any]:
    started = time.time()
    rng = np.random.default_rng(SEED)
    workdir = Path(tempfile.mkdtemp(prefix="gate_calibration_"))
    try:
        payload: dict[str, Any] = {
            "harness": "benchmarks/run_promotion_gate_calibration.py",
            "mode": mode,
            "seed": SEED,
            "simulations_per_cell": sims,
            "canonical_spec_sha256": canonical_spec.PINNED_SPEC_SHA256,
            "thresholds": {
                "stage2_min_ic": canonical_spec.STAGE2_MIN_IC,
                "stage4_min_oos_ic": canonical_spec.STAGE4_MIN_OOS_IC,
                "stage4_oos_alpha": canonical_spec.STAGE4_OOS_ALPHA,
                "min_oos_transfer_retention": canonical_spec.MIN_OOS_TRANSFER_RETENTION,
                "min_net_sharpe": canonical_spec.MIN_NET_SHARPE,
                "dsr_threshold": canonical_spec.DSR_THRESHOLD,
                "stage5_min_fee_bps": canonical_spec.STAGE5_MIN_FEE_BPS,
            },
            "stage_2": calibrate_stage_2(rng, sims),
            "stage_4": calibrate_stage_4(rng, max(sims // 2, 2), workdir),
            "stage_5": calibrate_stage_5(rng, max(sims // 5, 2), workdir),
            "pipeline_stages_2_to_5": calibrate_pipeline(rng, max(sims // 2, 2), workdir),
        }
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    payload["elapsed_seconds"] = round(time.time() - started, 1)
    payload["recorded_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    if json_path is not None:
        json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _fmt(cell: dict[str, Any]) -> str:
    return f"{cell['rate']:.3f} [{cell['wilson95'][0]:.3f}, {cell['wilson95'][1]:.3f}]"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--sims", type=int, default=1000, help="simulations per stage-2 cell")
    parser.add_argument("--smoke", action="store_true", help="a few simulations per cell")
    args = parser.parse_args(argv)
    sims = 10 if args.smoke else args.sims
    payload = run(args.json, sims, "smoke" if args.smoke else "full")
    print(f"stage 2 (n, true IC): gate | bare threshold   [{sims} sims]")
    for row in payload["stage_2"]:
        print(f"  n={row['n']:<5} ic={row['true_ic']:<5} {_fmt(row['gate_threshold_and_five_positive_folds'])} | "
              f"{_fmt(row['bare_threshold_only'])}")
    print("stage 4, candidate with no signal:")
    for key, cell in payload["stage_4"].items():
        if isinstance(cell, dict):
            print(f"  {key}: {_fmt(cell)}")
    print("stage 5 (true Sharpe, strategies searched): best passes")
    for row in payload["stage_5"]:
        print(f"  sr={row['true_annual_sharpe']:<4} K={row['strategies_searched']:<4} "
              f"{_fmt(row['best_of_search_passes_stage_5'])}")
    print("pipeline, stages 2-5:")
    for row in payload["pipeline_stages_2_to_5"]:
        print(f"  {row['world']:<8}", {k: row[k]["rate"] for k in row if k.startswith("reached_stage_")})
    print(f"wrote {args.json}  ({payload['elapsed_seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
