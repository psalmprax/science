#!/usr/bin/env python3
"""The independent-trials hybrids on the same simulated searches and real prices as the two earlier studies.

Status: deprecated (retired 27 Nov 2026 on the owner's word; kept to replicate its record)

**Development, not a result.** Runs ``scripts/study_independent_trials.py``'s simulated worlds and
``scripts/study_independent_trials_real.py``'s strategies on real prices exactly as they ran (same seeds,
same worlds, same rules), with ``apedp_dyn/epistemic/independent_trials_hybrid.py``'s seven ways of
counting in place of three, so the three earlier ways come out as recorded and the four hybrids beside
them, world for world.

    python scripts/study_independent_trials_hybrid.py --out benchmarks/referee_trials/independent_trials_hybrid_dev.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import study_independent_trials as simulated  # noqa: E402
import study_independent_trials_real as real  # noqa: E402

from apedp_dyn.epistemic import independent_trials as it  # noqa: E402
from apedp_dyn.epistemic import independent_trials_hybrid as hybrid  # noqa: E402

TAG = "DEV-INDEPENDENT-TRIALS-HYBRID-2026-11"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--null", type=int, default=300)
    parser.add_argument("--alt", type=int, default=150)
    parser.add_argument("--reps", type=int, default=200)
    args = parser.parse_args(argv)
    shim = SimpleNamespace(deflated=hybrid.deflated, sharpe=it.sharpe)
    for study in (simulated, real):
        study.it = shim                      # type: ignore[attr-defined]
        study.METHODS = hybrid.METHODS       # type: ignore[attr-defined]
    t0 = time.time()
    sim = simulated.run(args.null, args.alt)
    rl = real.run(args.reps)
    record = {"tag": TAG, "development_not_a_result": True, "methods": list(hybrid.METHODS),
              "simulated": {"seed": simulated.SEED, "null_reps": args.null, "alt_reps": args.alt, "cells": sim},
              "real": {"seed": real.SEED, "reps": args.reps, "series": rl}, "seconds": round(time.time() - t0, 1)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
