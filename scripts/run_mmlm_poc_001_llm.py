#!/usr/bin/env python3
"""MMLM-POC-001 with a real language model: who chooses the moves, compared from the same start with the same number of moves.

**Development, not a result.** ``scripts/run_mmlm_poc_001.py`` has an arm that was labelled an LLM and is
hand-written algebra (see its correction). Here a real language model chooses each move
(``apedp_dyn/quantum/llm_decomposition_arm.py``) and is compared with that hand-written heuristic and with the
random walk. What the first runner got wrong is done differently:

* **the same start for every arm:** the first 50 products of the 54-product decomposition;
* **the same number of moves:** each arm may have ``--steps`` candidates checked by the exact checker. The
  language model's are its calls; a reply that is not a valid move costs one;
* **real accounting:** tokens are the service's own counts. The other arms use no tokens;
* **Gate 0 stops the run.** The experiment's claim was a decomposition shorter than the published 51 products.
  The repository holds no copy of that decomposition, so the gate cannot be cleared and nothing here can be
  called a result against it. Without ``--development`` the script stops at the gate. With it, the arms are
  compared by how far each lowers the exact count of wrong entries, and the record says development;
* **several runs, paired:** for each seed the three arms run from the same start; the language model's runs
  differ by sampling (temperature above zero). A sign test compares it with the heuristic.

    NVIDIA_API_KEY=... python scripts/run_mmlm_poc_001_llm.py --development --out DIR [--seeds 3 --steps 60 --model NAME]
    NVIDIA_API_KEY=... python scripts/run_mmlm_poc_001_llm.py --models --out DIR     # what the key reaches, and which is used
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_mmlm_poc_001 as poc  # noqa: E402

from apedp_dyn.quantum import llm_decomposition_arm as llm_arm  # noqa: E402
from apedp_dyn.quantum.field_tensor_search import (  # noqa: E402
    build_54_reference,
    build_multiplication_tensor,
    load_published_51_reference,
    verify_tensor_decomposition,
)

TARGET_RANK = 50
ARM_SECONDS = 600.0          # the hand-written arms' wall clock; their step count is what binds
#: strongest first; the first that answers is used (apedp_dyn.proposers.llm_clients.pick_model)
MODELS = ("nvidia/nemotron-3-ultra-550b-a55b", "nvidia/nemotron-3.5-lightning-30b-a3b", "minimaxai/minimax-m3.1-flash")
TEMPERATURE = 0.7


def gate_zero(T: np.ndarray) -> dict[str, Any]:
    """Whether the published 51-product decomposition is in the repository and verifies exactly."""
    reference = load_published_51_reference()
    valid, mismatches = verify_tensor_decomposition(T, reference)
    cleared = bool(valid and len(mismatches) == 0)
    return {"cleared": cleared, "mismatches": len(mismatches),
            "why": "the published 51 products verify exactly" if cleared else
                   "the repository holds no copy of the published 51-product decomposition (its loader returns zeros), "
                   "so no decomposition found here can be compared with it"}


def start_products(target_rank: int = TARGET_RANK) -> tuple[np.ndarray, np.ndarray]:
    """The products every arm starts from: the first ``target_rank`` of the 54-product decomposition."""
    reference = build_54_reference()
    return reference.U[:target_rank].copy(), reference.V[:target_rank].copy()


def hand_written(run: Callable[..., dict[str, Any]], label: str, T: np.ndarray, start: tuple[np.ndarray, np.ndarray], steps: int,
                 seed: int) -> dict[str, Any]:
    budget = poc.BudgetAccountant(wall_clock_limit_sec=ARM_SECONDS, cpu_time_limit_sec=ARM_SECONDS, max_evaluations=steps + 1,
                                  max_verifier_calls=steps + 1, arm_name=label)
    got = run(T, len(start[0]), ARM_SECONDS, steps, seed, budget, start)
    return {"arm": label, "best_defect": int(got["best_defect"]), "solved": bool(got["solved"]), "steps": int(got["steps_completed"]),
            "tokens": None}


def compare(client: Callable[[str], str], seeds: int, steps: int, target_rank: int = TARGET_RANK,
            progress: Callable[[list[dict[str, Any]]], None] | None = None) -> dict[str, Any]:
    """Each seed: the language model, the diagnostic-guided heuristic and the random walk, from the same products."""
    T = build_multiplication_tensor()
    start = start_products(target_rank)
    runs = []
    for seed in range(seeds):
        model = llm_arm.run_llm_arm(T, client, start[0], start[1], max_steps=steps)
        runs.append({"seed": seed, "language_model": model,
                     "heuristic": hand_written(poc.run_arm_b_mmlm, "diagnostic-guided heuristic (hand-written)", T, start, steps, seed),
                     "random_walk": hand_written(poc.run_arm_a_conventional, "random walk", T, start, steps, seed)})
        if progress is not None:
            progress(runs)
    return {"start_defect": runs[0]["language_model"]["start_defect"] if runs else None, "runs": runs, "summary": summarise(runs)}


def summarise(runs: list[dict[str, Any]]) -> dict[str, Any]:
    def defects(arm: str) -> list[int]:
        return [int(r[arm]["best_defect"]) for r in runs]

    model, heuristic, walk = defects("language_model"), defects("heuristic"), defects("random_walk")
    better = sum(m < h for m, h in zip(model, heuristic, strict=True))
    worse = sum(m > h for m, h in zip(model, heuristic, strict=True))
    tokens: dict[str, int] = {}
    for run in runs:
        for key, value in (run["language_model"]["tokens"] or {}).items():
            tokens[key] = tokens.get(key, 0) + int(value)
    return {"runs": len(runs), "wrong_entries_left": {"language_model": model, "heuristic": heuristic, "random_walk": walk},
            "solved": {arm: sum(bool(r[arm]["solved"]) for r in runs) for arm in ("language_model", "heuristic", "random_walk")},
            "language_model_against_heuristic": {
                "better": better, "worse": worse, "same": len(runs) - better - worse,
                "sign_test_p_one_sided": float(binomtest(better, better + worse, 0.5, alternative="greater").pvalue) if better + worse else None},
            "language_model_calls": sum(r["language_model"]["calls"] for r in runs),
            "language_model_moves_kept": sum(r["language_model"]["moves_kept"] for r in runs),
            "language_model_invalid_replies": sum(r["language_model"]["invalid_replies"] for r in runs),
            "language_model_stopped_early": [r["language_model"]["stopped"] for r in runs if r["language_model"]["stopped"]],
            "language_model_tokens": tokens or None}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--model", default="")
    parser.add_argument("--development", action="store_true", help="compare the arms although Gate 0 is not cleared")
    parser.add_argument("--models", action="store_true", help="list the models the key reaches and the one that would be used")
    args = parser.parse_args(argv)
    if args.models:
        from apedp_dyn.proposers import llm_clients, llm_https

        listed = llm_https.available_models()
        picked = llm_clients.pick_model(MODELS, fall_back_to_listed=False)
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "models.json").write_text(json.dumps({"listed": listed, "preferred": list(MODELS), "picked": picked}, indent=1) + "\n",
                                              encoding="utf-8")
        print(json.dumps({"models_listed": len(listed), "preferred": list(MODELS), "picked": picked}, indent=1))
        return 0
    gate = gate_zero(build_multiplication_tensor())
    if not gate["cleared"] and not args.development:
        raise SystemExit(f"Gate 0 is not cleared: {gate['why']}. Nothing was run. (--development compares the arms without a claim.)")
    from apedp_dyn.proposers import llm_clients, llm_https

    name = args.model or llm_clients.pick_model(MODELS, fall_back_to_listed=False)
    client = llm_https.nvidia_client(name, temperature=TEMPERATURE, max_tokens=300)
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "what": "development: MMLM-POC-001 with a real language model choosing the moves; not a result, and no claim "
                "against the published 51-product decomposition", "protocol": "MMLM-POC-001", "development": True,
        "gate_0": gate, "model": name, "temperature": TEMPERATURE, "target_rank": TARGET_RANK, "moves_per_arm": args.steps}

    def write(runs: list[dict[str, Any]]) -> None:                # after every run, so that a job that dies keeps what it had
        record.update({"runs": runs, "summary": summarise(runs), "seconds": round(time.perf_counter() - started, 1)})
        (args.out / "mmlm_poc_001_llm.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")

    record.update(compare(client, args.seeds, args.steps, progress=write))
    record["seconds"] = round(time.perf_counter() - started, 1)
    (args.out / "mmlm_poc_001_llm.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"model": name, "gate_0_cleared": gate["cleared"], "start_defect": record["start_defect"], **record["summary"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
