"""Development study of contrast confirmation on laws the blind benchmark already used.

    python scripts/study_contrast_confirmation_dev.py

NOT a preregistered result. The eight two-term invented laws of
PREREG-BLIND-L2-2026-11 were already seen, and the procedure was changed three
times while looking at them (coefficient uncertainty, a wider candidate list,
noise-weighted fitting). The numbers say the idea is worth a preregistered test
on fresh laws; they are not that test. The comparison with the referee is also
not at equal data: the referee sees 120 rows, contrast confirmation reads 30 to
about 200 more, chosen by design.

Each run: L2 searches 120 rows; its twelve best distinct candidates by internal
holdout error go to ``contrast_confirm`` with a pool of 5,000 unlabelled rows.
Second arm: the same with the true law removed from the candidates.
"""
from __future__ import annotations

import collections
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_blind_discovery_benchmark as blind  # noqa: E402

from apedp_dyn.discovery_l2 import contrast_confirmation as cc  # noqa: E402
from apedp_dyn.discovery_l2.l2_engine import L2DiscoveryEngine  # noqa: E402

OUT = ROOT / "benchmarks" / "contrast_confirmation_dev_study.json"
NOISE_LEVELS = (0.01, 0.05, 0.10)
SEEDS = (0, 1)


def fitted(expr: sp.Expr, names: list[str], dev_X: np.ndarray, dev_y: np.ndarray) -> cc.FittedCandidate:
    symbols = [sp.Symbol(n) for n in names]
    fns = [sp.lambdify(symbols, t.as_coeff_Mul()[1], "numpy") for t in sp.Add.make_args(sp.expand(expr))]

    def basis(X: np.ndarray) -> np.ndarray:
        return np.column_stack([np.asarray(f(*X.T), float) * np.ones(len(X)) for f in fns])

    return cc.fit_candidate(str(expr), basis, dev_X, dev_y)


def one(law: blind.Law, noise: float, seed: int, engine: L2DiscoveryEngine, drop_truth: bool) -> dict[str, Any]:
    key = hashlib.sha256(f"CCDEV|{law.law_id}|{noise}|{seed}".encode()).digest()[:8]
    rng = np.random.default_rng(int.from_bytes(key, "big"))
    distractor, ddim = blind._distractor(law, rng)
    names = [*law.dimensions, distractor]
    data = {n: rng.uniform(1, 10, blind.N_SAMPLES) for n in names}
    fn = sp.lambdify([sp.Symbol(n) for n in law.dimensions], law.sympy(), "numpy")

    def truth(X: np.ndarray) -> np.ndarray:
        return np.asarray(fn(*X.T[:len(law.dimensions)]), float)

    dev_X = np.column_stack([data[n] for n in names])
    y = truth(dev_X) * (1 + noise * rng.standard_normal(len(dev_X)))
    result = engine.discover(X=data, Y=y, dimensions={**law.dimensions, distractor: ddim},
                             target_dimension=law.target_dimension, target_name="y",
                             noise_sigma=noise * float(np.mean(np.abs(y))), seed=seed)
    target = blind.monomials(law.sympy())
    scored = sorted((s for s in result.candidates if s.holdout_rmse is not None and np.isfinite(s.holdout_rmse)),
                    key=lambda s: s.holdout_rmse)
    ranked: list[sp.Expr] = []
    seen = set()
    for item in scored:
        expr = item.candidate.metadata["ast"].to_sympy()
        try:
            mono = blind.monomials(expr)
        except ValueError:
            continue
        if mono in seen or (drop_truth and mono == target):
            continue
        seen.add(mono)
        ranked.append(expr)
        if len(ranked) == 12:
            break
    row = {"law": law.law_id, "noise": noise, "seed": seed, "truth_removed": drop_truth,
           "referee_status": result.adjudication.status.name, "candidates": len(ranked)}
    if not ranked:
        return {**row, "verdict": "no_candidates", "survivor_is_truth": False, "rows_read": 0}
    pool = rng.uniform(1, 10, (5000, len(names)))
    res = cc.contrast_confirm([fitted(e, names, dev_X, y) for e in ranked], pool,
                              lambda rows: truth(pool[rows]) * (1 + noise * rng.standard_normal(len(rows))))
    right = res.survivor is not None and blind.monomials(sp.sympify(res.survivor)) == target
    return {**row, "verdict": res.verdict, "survivor": res.survivor, "survivor_is_truth": bool(right),
            "equivalent": res.equivalent, "rows_read": res.rows_read}


def main() -> int:
    engine = L2DiscoveryEngine(operators=blind.OPERATORS)
    laws = [law for law in blind.invented_laws() if "+" in law.expression]
    rows = [one(law, noise, seed, engine, drop) for drop in (False, True) for law in laws
            for noise in (NOISE_LEVELS if not drop else (0.01,)) for seed in SEEDS]
    with_truth = [r for r in rows if not r["truth_removed"]]
    summary = {
        "status": "development study, not preregistered; see the module docstring",
        "with_truth": dict(collections.Counter(
            ("truth " if r["survivor_is_truth"] else "not truth ") + r["verdict"] for r in with_truth)),
        "referee_promoted_on_the_same_runs": sum(r["referee_status"] == "UNIQUE_SUPPORT" for r in with_truth),
        "runs_with_truth": len(with_truth),
        "truth_removed": dict(collections.Counter(r["verdict"] for r in rows if r["truth_removed"])),
        "median_rows_read": float(np.median([r["rows_read"] for r in with_truth])),
    }
    OUT.write_text(json.dumps({"summary": summary, "runs": rows}, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
