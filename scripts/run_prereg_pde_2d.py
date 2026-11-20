"""PREREG-PDE-2D-2026-11: the 2-D weak form on simulated fields with known equations.

    python scripts/run_prereg_pde_2d.py --out preregistration/RESULT-PDE-2D-2026-11

Runs once. Refuses to run if a frozen file changed, or if the output directory exists.
Protocol and expectations: ``preregistration/PREREG-PDE-2D-2026-11.md``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pde_2d_worlds as worlds  # noqa: E402

from apedp_dyn.pde import weak_form_2d as wf  # noqa: E402

PREREGISTRATION_ID = "PREREG-PDE-2D-2026-11"
DIGEST = ROOT / "preregistration" / "DIGEST-PDE-2D.json"
FROZEN = ("preregistration/PREREG-PDE-2D-2026-11.md", "scripts/run_prereg_pde_2d.py", "scripts/pde_2d_worlds.py",
          "apedp_dyn/pde/weak_form_2d.py", "apedp_dyn/pde/weak_form.py")
SEEDS = (2000, 2001, 2002, 2003, 2004)
NOISE_LEVELS = (0.0, 0.01, 0.05, 0.10)
RULES = ("evidence", "threshold")


def one(world: str, seed: int, noise: float) -> dict[str, Any]:
    sim = worlds.simulate(world, seed)
    u = worlds.with_noise(sim["u"], noise, seed + 100_000)
    truth = sim["truth"]
    row: dict[str, Any] = {"world": world, "seed": seed, "noise": noise}
    for rule in RULES:
        started = time.perf_counter()
        found = wf.identify(u, sim["dt"], sim["dx"], sim["dy"], select=rule)
        row[rule] = {"exact": set(found) == set(truth), "extra": sorted(set(found) - set(truth)),
                     "missing": sorted(set(truth) - set(found)), "coefficients": found,
                     "max_relative_error": max(abs(found.get(k, 0.0) - v) / abs(v) for k, v in truth.items()),
                     "seconds": time.perf_counter() - started}
    return row


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for rule in RULES:
        for noise in (*NOISE_LEVELS, "up_to_0.05", "all"):
            sel = [r for r in rows if noise == "all" or (noise == "up_to_0.05" and r["noise"] <= 0.05) or r["noise"] == noise]
            exact = [r for r in sel if r[rule]["exact"]]
            out[f"{rule}|{noise}"] = {
                "runs": len(sel), "exact": len(exact), "with_extra_terms": sum(bool(r[rule]["extra"]) for r in sel),
                "with_missing_terms": sum(bool(r[rule]["missing"]) for r in sel),
                "median_max_relative_error_when_exact": float(np.median([r[rule]["max_relative_error"] for r in exact])) if exact else None,
                "worst_max_relative_error_when_exact": float(max(r[rule]["max_relative_error"] for r in exact)) if exact else None,
            }
        for world in worlds.WORLDS:
            sel = [r for r in rows if r["world"] == world]
            out[f"{rule}|world|{world}"] = {"runs": len(sel), "exact": sum(r[rule]["exact"] for r in sel)}
    return out


def frozen_digests() -> dict[str, str]:
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in FROZEN}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if json.loads(DIGEST.read_text(encoding="utf-8"))["sha256"] != frozen_digests():
        raise SystemExit(f"{PREREGISTRATION_ID} changed since it was frozen")
    args.out.mkdir(parents=True, exist_ok=False)
    rows = [one(world, seed, noise) for world in worlds.WORLDS for seed in SEEDS for noise in NOISE_LEVELS]
    results = {"preregistration": PREREGISTRATION_ID, "frozen": frozen_digests(), "summary": summarise(rows)}
    (args.out / "results.json").write_text(json.dumps(results, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    (args.out / "runs.json").write_text(json.dumps(rows, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(results["summary"], indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
