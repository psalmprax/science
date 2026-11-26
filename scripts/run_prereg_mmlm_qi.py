#!/usr/bin/env python3
"""PREREG-MMLM-QI-2026-11: trained quantum-inspired byte models against a classical automaton of their size, once.

Status: audit-only

``arm`` trains one model of ``apedp_mmlm/quantum_inspired.py``: a short learning-rate sweep (the same
five rates for every arm, each for ``--sweep-seconds``, the rate kept being the one best on a fixed
slice of the validation split), then a fresh model at that rate for ``--seconds``, keeping the
parameters best on that slice, then scores the scored split. ``baselines`` scores interpolated
Kneser-Ney with 0 to 4 bytes of history on it. ``collect`` reads every arm and the baselines, applies
the expectations of ``preregistration/PREREG-MMLM-QI-2026-11.md`` and writes ``results.json``.

``--mode final`` (the preregistered run) checks the digest first, selects on the whole validation
split's first 300,000 bytes and scores the test split; ``--mode dev`` selects on the same slice of
``valid_a`` and scores ``valid_b``, and never reads the test split.

    python scripts/run_prereg_mmlm_qi.py arm --arm pure-complex-16 --mode final --out OUT
    python scripts/run_prereg_mmlm_qi.py baselines --mode final --out OUT
    python scripts/run_prereg_mmlm_qi.py collect --mode final --out OUT
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from apedp_mmlm import quantum_inspired as qi  # noqa: E402
from apedp_mmlm.text_corpus import split_bytes  # noqa: E402

PREREGISTRATION_ID = "PREREG-MMLM-QI-2026-11"
DIGEST = ROOT / "preregistration" / "DIGEST-MMLM-QI.json"
FROZEN = ("preregistration/PREREG-MMLM-QI-2026-11.md", "scripts/run_prereg_mmlm_qi.py",
          "apedp_mmlm/quantum_inspired.py", "apedp_mmlm/text_corpus.py", ".github/workflows/mmlm-qi.yml")
ARMS = ("pure-complex-16", "pure-complex-32", "pure-complex-64",
        "pure-real-16", "pure-real-32", "pure-real-64",
        "pure-real-23", "pure-real-45", "pure-real-90",
        "mixed-complex-16-2", "mixed-complex-32-2", "mixed-real-16-2", "mixed-real-32-2",
        "hmm-256", "hmm-1024",
        "pure-complex-23", "pure-complex-45",                   #: about the parameters of mixed-complex-16-2 and -32-2
        "pure-complex-16@1", "hmm-256@1")                       #: the last two: a second seed, for the noise
RATES = (3e-3, 1e-2, 3e-2, 1e-1, 3e-1)
BATCH, SEGMENT, SELECT_BYTES, CHUNKS = 64, 128, 300_000, 32
HYBRID_ORDER = 4                                               #: the counting model the hybrid mixes with
SAME_PARAMETERS = {16: 23, 32: 45, 64: 90}                     #: real D' with about the parameters of complex D
MIXED_SAME_PARAMETERS = {16: 23, 32: 45}                       #: pure complex D' with about the parameters of mixed D, r = 2
REFERENCES = {"MMLM-CM, all experts": 1.45485, "PPMd order 16, primed with the training text": 1.51287,
              "MMLM-CM's LSTM expert alone": 1.77240, "MMLM-CM's order-4 counts alone": 1.94505}   #: RESULT-MMLM-CM-2026-11


def frozen_digests() -> dict[str, str]:
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in FROZEN}


def check_frozen() -> None:
    if json.loads(DIGEST.read_text(encoding="utf-8"))["sha256"] != frozen_digests():
        raise SystemExit(f"{PREREGISTRATION_ID} changed since it was frozen")


def data(mode: str) -> tuple[bytes, bytes, bytes]:
    """(training bytes, selection slice, scored bytes)."""
    if mode == "final":
        return split_bytes("train"), split_bytes("valid")[:SELECT_BYTES], split_bytes("test")
    return split_bytes("train"), split_bytes("valid_a")[:SELECT_BYTES], split_bytes("valid_b")


def _arm(name: str) -> tuple[str, int]:
    arm, _, seed = name.partition("@")
    return arm, int(seed or 0)


def run_arm(args: argparse.Namespace) -> int:
    if args.mode == "final":
        check_frozen()
    torch.set_num_threads(args.threads)
    arm, seed = _arm(args.arm)
    train_bytes, select, scored = data(args.mode)
    sweep = {}
    for rate in RATES:
        torch.manual_seed(seed)
        result = qi.train(qi.build(arm, seed), train_bytes, select, args.sweep_seconds, BATCH, SEGMENT, rate,
                          evaluate_every=10**9, valid_chunks=CHUNKS)
        sweep[str(rate)] = {"selection_bpb": round(result.best_valid_bpb, 4), "bytes_seen": result.bytes_seen}
        print(f"{args.arm} sweep lr={rate}: {result.best_valid_bpb:.4f}", flush=True)
    rate = min(RATES, key=lambda r: sweep[str(r)]["selection_bpb"])
    torch.manual_seed(seed)
    model = qi.build(arm, seed)
    result = qi.train(model, train_bytes, select, args.seconds, BATCH, SEGMENT, rate, evaluate_every=args.evaluate_every,
                      valid_chunks=CHUNKS, log=lambda line: print(f"{args.arm} {line}", flush=True))
    t0 = time.monotonic()
    p_scored = qi.byte_probabilities(model, scored, CHUNKS)
    score = float(-np.log2(p_scored).mean())
    took = time.monotonic() - t0
    # the hybrid: the model mixed byte by byte with Kneser-Ney (4 bytes of history), the weight chosen on the
    # selection slice; both are scored on the same bytes
    p_select = qi.byte_probabilities(model, select, CHUNKS)
    kn_select = qi.kneser_ney_probabilities(train_bytes, select, HYBRID_ORDER)[: len(p_select)]
    kn_scored = qi.kneser_ney_probabilities(train_bytes, scored, HYBRID_ORDER)[: len(p_scored)]
    weight = qi.best_mixture_weight(p_select, kn_select)
    hybrid = {"counting_model": f"Kneser-Ney, {HYBRID_ORDER} bytes of history", "weight_on_model": weight,
              "bits_per_byte": round(qi.mixture(p_scored, kn_scored, weight), 4),
              "counting_alone_same_bytes": round(float(-np.log2(kn_scored).mean()), 4)}
    row: dict[str, Any] = {
        "arm": args.arm, "model": arm, "seed": seed, "mode": args.mode, "parameters": model.parameter_count(),
        "sweep_seconds_each": args.sweep_seconds, "sweep": sweep, "rate": rate, "seconds": result.seconds,
        "steps": result.steps, "bytes_seen": result.bytes_seen, "selection_bpb": round(result.best_valid_bpb, 4),
        "curve": result.curve, "scored_bytes": len(scored), "bits_per_byte": round(score, 4),
        "scoring_bytes_per_second": round(len(scored) / took), "hybrid_with_counts": hybrid, "threads": args.threads,
        "torch": torch.__version__, "python": platform.python_version(), "processor": platform.processor()}
    if args.mode == "final":
        row["frozen"] = frozen_digests()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"arm_{args.arm.replace('@', '_seed')}.json").write_text(json.dumps(row, indent=1) + "\n")
    print(json.dumps({k: v for k, v in row.items() if k not in ("curve", "frozen")}), flush=True)
    return 0


def run_baselines(args: argparse.Namespace) -> int:
    if args.mode == "final":
        check_frozen()
    train_bytes, _, scored = data(args.mode)
    rows = {f"Kneser-Ney, {k} bytes of history": round(qi.kneser_ney_bits(train_bytes, scored, k), 4) for k in range(5)}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "baselines.json").write_text(json.dumps({"mode": args.mode, "kneser_ney": rows}, indent=1) + "\n")
    print(json.dumps(rows, indent=1))
    return 0


def expectations(bpb: dict[str, float], kn: dict[str, float], hybrid: dict[str, float],
                 counting_alone: float) -> dict[str, Any]:
    """The seven expectations of the preregistration, read from bits per byte on the scored split; ``hybrid``
    is each arm mixed with Kneser-Ney (4 bytes of history), ``counting_alone`` that model on the same bytes."""
    e7_counts = {d: round(counting_alone - hybrid[f"pure-complex-{d}"], 4) for d in (16, 32)}
    e7_hmm = {d: round(hybrid[f"hmm-{d * d}"] - hybrid[f"pure-complex-{d}"], 4) for d in (16, 32)}
    qi_arms = {a: v for a, v in bpb.items() if not a.startswith("hmm") and "@" not in a}
    best_arm = min(qi_arms, key=lambda a: qi_arms[a])
    spread = {"pure-complex-16": abs(bpb["pure-complex-16"] - bpb["pure-complex-16@1"]),
              "hmm-256": abs(bpb["hmm-256"] - bpb["hmm-256@1"])}
    e2 = {d: round(bpb[f"hmm-{d * d}"] - bpb[f"pure-complex-{d}"], 4) for d in (16, 32)}
    e3 = {d: round(bpb[f"pure-real-{SAME_PARAMETERS[d]}"] - bpb[f"pure-complex-{d}"], 4) for d in (16, 32, 64)}
    e4 = {d: round(bpb[f"pure-complex-{MIXED_SAME_PARAMETERS[d]}"] - bpb[f"mixed-complex-{d}-2"], 4) for d in (16, 32)}
    curve = [bpb[f"pure-complex-{d}"] for d in (16, 32, 64)]
    kn2, kn4 = kn["Kneser-Ney, 2 bytes of history"], kn["Kneser-Ney, 4 bytes of history"]
    return {
        "best_quantum_inspired_arm": best_arm, "best_quantum_inspired_bpb": qi_arms[best_arm],
        "seed_spread": {k: round(v, 4) for k, v in spread.items()},
        "E1 learns more than three-byte statistics": {"held": qi_arms[best_arm] < kn2, "kneser_ney_2": kn2},
        "E2 beats a hidden Markov model with D^2 states by 0.05": {"held": all(v >= 0.05 for v in e2.values()),
                                                                  "margin_by_D": e2},
        "E3 complex beats real of the same size by 0.02": {"held": all(v >= 0.02 for v in e3.values()), "margin_by_D": e3,
                                                           "same_D_real_minus_complex": {
                                                               d: round(bpb[f"pure-real-{d}"] - bpb[f"pure-complex-{d}"], 4)
                                                               for d in (16, 32, 64)}},
        "E4 a mixed state beats a pure one of the same size by 0.02": {
                                                       "held": all(v >= 0.02 for v in e4.values()), "margin_by_D": e4,
                                                       "same_D_pure_minus_mixed": {
                                                           d: round(bpb[f"pure-complex-{d}"] - bpb[f"mixed-complex-{d}-2"], 4)
                                                           for d in (16, 32)},
                                                       "real_mixed_minus_complex_mixed": {
                                                           d: round(bpb[f"mixed-real-{d}-2"] - bpb[f"mixed-complex-{d}-2"], 4)
                                                           for d in (16, 32)}},
        "E5 better with a larger state": {"held": curve[0] > curve[1] > curve[2], "pure_complex_16_32_64": curve},
        "E6 beats the strongest Kneser-Ney": {"held": qi_arms[best_arm] < kn4, "kneser_ney_4": kn4},
        "E7 mixed with counting it adds what an automaton of its size does not": {
            "held": all(e7_counts[d] >= 0.02 and e7_hmm[d] >= 0.02 for d in (16, 32)),
            "counting_alone_minus_hybrid_by_D": e7_counts, "hmm_hybrid_minus_hybrid_by_D": e7_hmm},
        "hybrids_without_criterion": {a: h for a, h in hybrid.items() if "@" not in a},
        "references_without_criterion": {k: round(qi_arms[best_arm] - v, 4) for k, v in REFERENCES.items()},
        "margins_within_seed_spread": {"E2": any(abs(v) <= max(spread.values()) for v in e2.values()),
                                       "E3": any(abs(v) <= max(spread.values()) for v in e3.values()),
                                       "E4": any(abs(v) <= max(spread.values()) for v in e4.values())},
    }


def collect(args: argparse.Namespace) -> int:
    if args.mode == "final":
        check_frozen()
    rows = {json.loads(p.read_text())["arm"]: json.loads(p.read_text()) for p in sorted(args.out.glob("arm_*.json"))}
    kn = json.loads((args.out / "baselines.json").read_text())["kneser_ney"]
    missing = [a for a in ARMS if a not in rows]
    record: dict[str, Any] = {"preregistration": PREREGISTRATION_ID, "mode": args.mode, "missing_arms": missing,
                              "arms": {a: {k: v for k, v in rows[a].items() if k != "curve"} for a in ARMS if a in rows},
                              "curves": {a: rows[a]["curve"] for a in ARMS if a in rows}, "kneser_ney": kn,
                              "references_from_RESULT-MMLM-CM-2026-11": REFERENCES}
    if not missing:
        record["expectations"] = expectations({a: rows[a]["bits_per_byte"] for a in ARMS}, kn,
                                              {a: rows[a]["hybrid_with_counts"]["bits_per_byte"] for a in ARMS},
                                              rows["pure-complex-16"]["hybrid_with_counts"]["counting_alone_same_bytes"])
    if args.mode == "final":
        record["frozen"] = frozen_digests()
    (args.out / "results.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k in ("missing_arms", "expectations")}, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("command", choices=("arm", "baselines", "collect"))
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--mode", choices=("dev", "final"), required=True)
    parser.add_argument("--sweep-seconds", type=float, default=150.0)
    parser.add_argument("--seconds", type=float, default=2700.0)
    parser.add_argument("--evaluate-every", type=float, default=300.0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "arm":
        if not args.arm:
            parser.error("--arm is required")
        return run_arm(args)
    return run_baselines(args) if args.command == "baselines" else collect(args)


if __name__ == "__main__":
    raise SystemExit(main())
