"""APEDP-side execution for PREREG-APEDP-EXTBENCH-60-2026-09-26.

Reads PUBLIC data only (never hidden/ ground truth). Runs APEDPAdapter per
problem, writes per-block JSON to tournament_results_external_60/.
Blind evaluation against hidden truth happens separately via
IndependentEvaluator. Usage: run_external_60_apedp.py [BLOCK ...]
(default: all blocks). APEDP code is frozen for this round: this runner
changes no library code or constants.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.external_60_adapters import AdapterResult, APEDPAdapter

CORPUS = ROOT / "benchmark_corpus_external_60"
OUTDIR = ROOT / "tournament_results_external_60"


def main(blocks: list[str]) -> None:
    manifest = json.loads((CORPUS / "manifest.json").read_text())
    adapter = APEDPAdapter()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for block in blocks:
        pids = [p["problem_id"] for p in manifest["problems"] if p["block"] == block]
        results = []
        for pid in pids:
            try:
                res = adapter.run(CORPUS, pid)
            except Exception as e:  # noqa: BLE001 - one failure must not kill the block
                traceback.print_exc()
                res = AdapterResult(
                    competitor="apedp", problem_id=pid, expression="RUNNER_ERROR",
                    abstained=True, abstain_reason=f"runner exception: {type(e).__name__}",
                ).to_dict()
            else:
                res = res.to_dict()
            results.append(res)
            done = len(results)
            print(f"[{block}] {done}/{len(pids)} {pid} "
                  f"{'ABSTAIN' if res['abstained'] else res['expression'][:50]}", flush=True)
        with open(OUTDIR / f"apedp_{block}.json", "w", encoding="utf-8") as f:
            json.dump({"block": block, "results": results}, f, indent=2)
        print(f"[{block}] wrote apedp_{block}.json ({len(results)} results)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["P", "E", "R", "O", "C", "X", "A"])
