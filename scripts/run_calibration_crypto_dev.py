"""The calibration frozen by preregistration/PREREG-CRYPTO-CAL-2026-11.md.

How often do the referee's in-sample gates (stages 2 and 3) pass something
they should not, and how often do they pass something they should, on real
crypto returns? Run on the development panel of PREREG-CRYPTO-PIT-2026-11
(2021-2025) only: this script is given that one file and never opens the
holdout.

    python scripts/run_calibration_crypto_dev.py \\
        --development-panel <dir>/panel_development.csv.gz --out <new dir>

Three families, each through a fresh ``PromotionTracker``:

- **placebos** (``N_PLACEBOS``): a score drawn from a generator seeded by the
  candidate's id alone, so it knows nothing about markets. Any pass is false.
- **injected signals** (``N_PER_STRENGTH`` at each of ``STRENGTHS``): within
  each day, ``rho * z + sqrt(1 - rho**2) * noise``, where ``z`` is the
  standardized rank of the next day's return. Each has a true effect of known
  size; the pass rate against ``rho`` is the gate's power curve. The measured
  development IC of every candidate is recorded beside it.
- **literature factors**: one-day reversal and twenty-day momentum
  themselves. Whatever their IC, they are not new; if one passes stage 2 it
  must be refused at stage 3.

Pairs, universe and literature features are the frozen analysis's
(``run_prereg_crypto_pit``), on the same development period.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_prereg_crypto_pit as pit  # noqa: E402  (the frozen analysis's pairing and features)

from apedp_dyn.epistemic.promotion_lifecycle import PromotionStage, PromotionTracker  # noqa: E402
from apedp_dyn.market.universe import point_in_time_membership  # noqa: E402

PREREGISTRATION_ID = "PREREG-CRYPTO-CAL-2026-11"
DIGEST = ROOT / "preregistration" / "DIGEST-CAL.json"
FROZEN = ("preregistration/PREREG-CRYPTO-CAL-2026-11.md", "scripts/run_calibration_crypto_dev.py")

N_PLACEBOS = 500
STRENGTHS = (0.01, 0.015, 0.02, 0.025, 0.03, 0.05)
N_PER_STRENGTH = 100
LITERATURE = ("reversal", "momentum")
#: The development period ends before the holdout of PREREG-CRYPTO-PIT-2026-11.
LAST_DEVELOPMENT_DAY = "2025-12-31"


def frozen_digests(root: Path = ROOT) -> dict[str, str]:
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in FROZEN}


def check_frozen(root: Path = ROOT, digest_path: Path = DIGEST) -> None:
    recorded = json.loads(digest_path.read_text(encoding="utf-8"))["sha256"]
    now = frozen_digests(root)
    if recorded != now:
        raise SystemExit(f"{PREREGISTRATION_ID} changed since it was frozen: recorded {recorded}, now {now}")


def seed_of(candidate_id: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{PREREGISTRATION_ID}|{candidate_id}".encode()).digest()[:8], "big")


def within_day_centred_rank(day: np.ndarray, values: np.ndarray) -> np.ndarray:
    """The frozen analysis's centred rank in [-0.5, 0.5], within each day."""
    frame = pd.DataFrame({"day": day, "v": values})
    rank = frame.groupby("day")["v"].rank(method="first").to_numpy() - 1.0
    size = frame.groupby("day")["v"].transform("size").to_numpy()
    return rank / (size - 1.0) - 0.5


def standardized(day: np.ndarray, values: np.ndarray) -> np.ndarray:
    frame = pd.DataFrame({"day": day, "v": values})
    grouped = frame.groupby("day")["v"]
    return ((frame["v"] - grouped.transform("mean")) / grouped.transform("std", ddof=0)).to_numpy()


def judge(candidate_id: str, pred: np.ndarray, base: dict[str, np.ndarray], journal: Path) -> dict[str, Any]:
    """Stages 1-3 exactly as the frozen analysis runs them, on the development pairs."""
    tracker = PromotionTracker(candidate_id, state_path=journal, allow_reported_metrics=True)
    pit._stage1(tracker)
    rec2 = tracker.verify_stage_2_from_predictions(pred, base["target"])
    out: dict[str, Any] = {"candidate": candidate_id, "ic": rec2.metrics["in_sample_ic"],
                           "cv_stable": rec2.metrics["cv_stable"], "stage2": rec2.passed, "stage3": None}
    if rec2.passed:
        beta = float(np.dot(pred, base["target"]) / np.dot(pred, pred))
        rec3 = tracker.verify_stage_3_from_forecasts(
            base["target"], beta * pred, {"zero": np.zeros_like(base["target"])},
            pred, {name: base[name] for name in LITERATURE},
        )
        out["stage3"] = rec3.passed
        out["dm_p_value"] = rec3.metrics.get("dm_p_value")
        out["max_literature_r2"] = rec3.metrics.get("max_literature_r2")
    out["highest_stage_passed"] = PromotionStage(tracker.highest_stage_passed).name
    return out


def clopper_pearson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    a = (1.0 - level) / 2.0
    lo = 0.0 if k == 0 else float(beta_dist.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(beta_dist.ppf(1.0 - a, k + 1, n - k))
    return lo, hi


def run(development_panel: Path, out: Path, n_placebos: int = N_PLACEBOS,
        strengths: tuple[float, ...] = STRENGTHS, n_per_strength: int = N_PER_STRENGTH) -> dict[str, Any]:
    check_frozen()
    out.mkdir(parents=True, exist_ok=False)
    journal = out / "promotion_journal.jsonl"
    panel = pd.read_csv(io.BytesIO(gzip.decompress(development_panel.read_bytes())))
    panel = panel.sort_values("observation_date").reset_index(drop=True)
    dates = panel["observation_date"].astype(str).tolist()
    if dates[-1] > LAST_DEVELOPMENT_DAY:
        raise SystemExit(f"the panel runs to {dates[-1]}, past the development period: refusing")
    members = point_in_time_membership(panel, top_n=pit.TOP_N, lookback_bars=pit.LOOKBACK,
                                       rebalance_every=pit.REBALANCE).to_numpy(dtype=bool)
    feat = pit.features(panel)
    # The pair set every synthetic candidate is judged on: the frozen pairing of
    # an always-finite predictor (the frozen placebo) with the next-day return,
    # literature features finite. Synthetic predictors are re-ranked on it.
    base = pit.pairs(feat, members, "placebo", "next_return", dates[0], dates[-1], 1.0, LITERATURE)
    day = base["date"]
    z = standardized(day, base["target"])

    placebos = []
    for i in range(n_placebos):
        cid = f"CAL-PLAC-{i:04d}"
        draw = np.random.default_rng(seed_of(cid)).random(len(day))
        placebos.append(judge(cid, within_day_centred_rank(day, draw), base, journal))

    injected = []
    for rho in strengths:
        for i in range(n_per_strength):
            cid = f"CAL-INJ-{rho:.3f}-{i:03d}"
            noise = np.random.default_rng(seed_of(cid)).standard_normal(len(day))
            score = rho * z + math.sqrt(1.0 - rho * rho) * noise
            row = judge(cid, within_day_centred_rank(day, score), base, journal)
            row["rho"] = rho
            injected.append(row)

    literature = []
    for name in LITERATURE:
        own = pit.pairs(feat, members, name, "next_return", dates[0], dates[-1], 1.0, LITERATURE)
        literature.append(judge(f"CAL-LIT-{name}", own["pred"], own, journal))

    false_passes = sum(1 for r in placebos if r["stage2"])
    false_through_3 = sum(1 for r in placebos if r["stage3"])
    power = []
    for rho in strengths:
        rows = [r for r in injected if r["rho"] == rho]
        k2 = sum(1 for r in rows if r["stage2"])
        k3 = sum(1 for r in rows if r["stage3"])
        power.append({"rho": rho, "n": len(rows), "stage2_passes": k2, "stage3_passes": k3,
                      "stage2_rate": k2 / len(rows), "stage2_interval_95": clopper_pearson(k2, len(rows)),
                      "mean_measured_ic": float(np.mean([r["ic"] for r in rows]))})
    results = {
        "preregistration": PREREGISTRATION_ID,
        "development_period": [dates[0], dates[-1]],
        "pairs": len(day), "days": len(set(day)),
        "placebo": {"n": len(placebos), "stage2_false_passes": false_passes,
                    "stage3_false_passes": false_through_3,
                    "stage2_rate_interval_95": clopper_pearson(false_passes, len(placebos)),
                    "ic_mean": float(np.mean([r["ic"] for r in placebos])),
                    "ic_sd": float(np.std([r["ic"] for r in placebos], ddof=1)) if len(placebos) > 1 else 0.0,
                    "ic_max_abs": float(max(abs(r["ic"]) for r in placebos)) if placebos else 0.0},
        "power": power,
        "literature": literature,
        "frozen": frozen_digests(),
    }
    (out / "results.json").write_text(json.dumps(results, indent=1, sort_keys=True, default=float) + "\n",
                                      encoding="utf-8")
    with (out / "candidates.jsonl").open("w", encoding="utf-8") as fh:
        for row in [*placebos, *injected, *literature]:
            fh.write(json.dumps(row, sort_keys=True, default=float) + "\n")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--development-panel", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    results = run(args.development_panel, args.out)
    p = results["placebo"]
    print(f"placebos: {p['stage2_false_passes']}/{p['n']} passed stage 2, {p['stage3_false_passes']} passed stage 3")
    for row in results["power"]:
        print(f"rho={row['rho']:.3f}: stage 2 {row['stage2_passes']}/{row['n']}, stage 3 {row['stage3_passes']}/{row['n']}, "
              f"mean IC {row['mean_measured_ic']:.4f}")
    for row in results["literature"]:
        print(f"{row['candidate']}: IC {row['ic']:.4f}, stage 2 {row['stage2']}, stage 3 {row['stage3']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
