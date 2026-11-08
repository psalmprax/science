#!/usr/bin/env python3
"""Audit official DQI sweep receipts without replacing the official score."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def close_enough(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=5e-12, abs_tol=1e-5)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path, help="results.json from sweep_dqi_eea_bound.py")
    parser.add_argument("--baseline-iters", type=int, default=222)
    parser.add_argument(
        "--pinned-reference-cost",
        type=float,
        default=24_439_473_914_136.67,
        help="Pinned ledger reference cost; used only for reconciliation, not to overwrite the receipt.",
    )
    args = parser.parse_args()
    rows = json.loads(args.results.read_text())
    if not isinstance(rows, list):
        raise SystemExit("results.json must contain a list of sweep rows")

    clean: dict[int, dict[str, Any]] = {}
    print("n_iters,status,Q,T_mean,epsilon_plus,R,score,score_formula_check,sha256")
    for row in rows:
        n = int(row["n_iters"])
        report = row.get("verification") or {}
        status = row.get("status", "UNKNOWN")
        q = report.get("num_qubits", row.get("num_qubits"))
        t_mean = report.get("toffoli_count")
        eps = report.get("epsilon_plus", report.get("eps_plus", report.get("effective_error_rate")))
        rep = report.get("repetition_factor", report.get("rep_factor"))
        score = report.get("score", row.get("measured_cost"))
        formula_status = "MISSING_FIELDS"
        if all(isinstance(x, (int, float)) for x in (q, t_mean, rep, score)):
            assert isinstance(q, (int, float)) and isinstance(t_mean, (int, float))
            recomputed = float(rep) * (int(q) ** 2) * float(t_mean)
            formula_status = "PASS" if close_enough(float(score), recomputed) else f"MISMATCH({recomputed:.12g})"
        sha = row.get("artifact_sha256", "")
        if status == "VERIFIED" and formula_status == "PASS" and score is not None:
            clean[n] = {"row": row, "score": float(score), "q": q, "t_mean": t_mean, "rep": rep, "eps": eps}
        print(",".join([
            str(n), str(status), str(q if q is not None else ""),
            str(t_mean if t_mean is not None else ""), str(eps if eps is not None else ""),
            str(rep if rep is not None else ""), str(score if score is not None else ""),
            formula_status, sha,
        ]))

    baseline = clean.get(args.baseline_iters)
    if baseline is None:
        print(f"BASELINE NOT ADJUDICATED: n_iters={args.baseline_iters} lacks a PASS receipt with formula-consistent fields.")
        return 2

    baseline_score = baseline["score"]
    print(f"\nBaseline n_iters={args.baseline_iters}: score={baseline_score:.12f}")
    print(f"Pinned ledger reference cost: {args.pinned_reference_cost:.12f}")
    print(f"Rebuilt-baseline / pinned-reference = {baseline_score / args.pinned_reference_cost:.12f}")
    print("\nVerified cost comparisons (improvement > 1 means candidate is better):")
    for n in sorted(clean):
        cand = clean[n]
        gain = baseline_score / cand["score"]
        print(f"n_iters={n}: gain={gain:.9f}x, delta={(gain - 1) * 100:+.5f}%")
    best_n = min(clean, key=lambda k: clean[k]["score"])
    print(f"\nBest among formula-consistent PASS receipts: n_iters={best_n}, score={clean[best_n]['score']:.12f}")
    print("NOTE: PASS receipts establish verifier acceptance under the run's seed/configuration; they do not by themselves establish unseen-seed generalization.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
