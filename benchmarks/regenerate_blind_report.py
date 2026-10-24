"""Regenerate the EXT-BENCH-60 blind report from the frozen record.

Writes to a NEW file by default. The stored
``apedp_blind_report_INTERNAL.json`` is a frozen-round artifact and is never
overwritten in place by this script; pass --in-place only when you have decided
the stale value should be replaced, and it writes a .bak first.

Why this exists: the stored report disagrees with the record it sits beside on
exactly one field. Re-scoring ``apedp_[AC-EOPRX].json`` with the project's own
``evaluate_round`` reproduces every summary field except ``family_accuracy``
(0.8929 stored vs 0.9107 recomputed). The disagreement traces to A60: the
report was scored from an older A60 whose expression canonicalized to
``polynomial_deg_4`` with far-OOD MSE 67699, while the record holds a
``logistic_saturation`` expression scoring 9.4e-06. A60 reproduces
bit-identically on demand (3/3 runs), so the record is sound and the report is
the stale artifact.

Scoring is read-only with respect to the round: it opens ``hidden/`` truth and
writes only the report.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.evaluate_external_60 import evaluate_round

CORPUS = ROOT / "benchmark_corpus_external_60"
RESULTS = ROOT / "tournament_results_external_60"
STORED = RESULTS / "apedp_blind_report_INTERNAL.json"
BLOCKS = ("A", "C", "E", "O", "P", "R", "X")


def load_record() -> dict[str, list[dict]]:
    """Every apedp result in the frozen round that has hidden truth on disk."""
    out: dict[str, list[dict]] = {}
    results: list[dict] = []
    for block in BLOCKS:
        path = RESULTS / f"apedp_{block}.json"
        if not path.is_file():
            print(f"  missing {path.name}, skipping")
            continue
        blob = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(blob, dict) or "results" not in blob:
            print(f"  {path.name} is not a block file, skipping")
            continue
        for rec in blob["results"]:
            if (CORPUS / "public" / rec["problem_id"]).is_dir():
                results.append(rec)
    out["apedp"] = results
    return out


def diff_against_stored(fresh: dict) -> list[tuple[str, object, object]]:
    """Fields where a regeneration disagrees with what is on disk."""
    if not STORED.is_file():
        return []
    stored = json.loads(STORED.read_text(encoding="utf-8")).get("per_competitor", {}).get("apedp", {})
    rows = []
    for key in sorted(set(fresh) | set(stored)):
        new, old = fresh.get(key), stored.get(key)
        if isinstance(new, float) or isinstance(old, float):
            same = (isinstance(new, (int, float)) and isinstance(old, (int, float))
                    and abs(new - old) <= 1e-9 * max(1.0, abs(old)))
        else:
            same = new == old
        if not same:
            rows.append((key, old, new))
    return rows


def main(in_place: bool) -> int:
    record = load_record()
    n = len(record["apedp"])
    if n == 0:
        print("No frozen results found; nothing to score.")
        return 1
    print(f"scoring {n} locked results against hidden/ truth")

    report = evaluate_round(CORPUS, record)
    summary = report["per_competitor"]["apedp"]

    print("\nsummary")
    for key in ("n_problems", "n_attempted", "n_abstained", "family_accuracy",
                "abstention_precision", "false_discoveries",
                "median_val_mse", "median_far_ood_mse",
                "median_far_ood_nrmse", "far_ood_worse_than_mean"):
        if key in summary:
            print(f"  {key:26s} {summary[key]}")

    rows = diff_against_stored(summary)
    if rows:
        print("\ndiffers from the stored report:")
        for key, old, new in rows:
            print(f"  {key}: stored {old!r} -> recomputed {new!r}")
        worse = [k for k, _, _ in rows if k in
                 ("family_accuracy", "abstention_precision")]
        for k in worse:
            print(f"\n  NOTE: {k} moves UP. The stored value was produced by an "
                  f"older A60, not by this record.")
    else:
        print("\nidentical to the stored report on every field")

    if in_place:
        backup = STORED.with_suffix(STORED.suffix + ".bak")
        backup.write_text(STORED.read_text(encoding="utf-8"), encoding="utf-8")
        STORED.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        print(f"\nwrote {STORED.name} (previous copy at {backup.name})")
    else:
        dest = RESULTS / "apedp_blind_report_REGENERATED.json"
        dest.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        print(f"\nwrote {dest.name}")
        print(f"the stored {STORED.name} is untouched")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-place", action="store_true",
                    help="replace the stored report (backs it up first)")
    raise SystemExit(main(ap.parse_args().in_place))
