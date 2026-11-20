"""The preregistered test of contrast confirmation (PREREG-CONTRAST-2026-11).

    python scripts/run_prereg_contrast.py --out <new dir>

Fresh two-term laws (a generator seed never used before), each run three ways on
identical data: contrast confirmation, the same procedure with random rows
(control, equal row budget per comparison), and the referee's own verdict on the
120 development rows. A third arm removes the true law from the candidates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_blind_discovery_benchmark as blind  # noqa: E402
import study_contrast_confirmation_dev as study  # noqa: E402

from apedp_dyn.discovery_l2 import contrast_confirmation as cc  # noqa: E402
from apedp_dyn.discovery_l2.l2_engine import L2DiscoveryEngine  # noqa: E402

PREREGISTRATION_ID = "PREREG-CONTRAST-2026-11"
DIGEST = ROOT / "preregistration" / "DIGEST-CONTRAST.json"
FROZEN = ("preregistration/PREREG-CONTRAST-2026-11.md", "scripts/run_prereg_contrast.py",
          "scripts/study_contrast_confirmation_dev.py", "scripts/run_blind_discovery_benchmark.py",
          "apedp_dyn/discovery_l2/contrast_confirmation.py")
LAW_SEED, N_GENERATED = 20261120, 120   # shapes rotate, so 40 of the 120 are two-term laws
NOISE_LEVELS = (0.01, 0.05)
POOL_ROWS, MAX_CANDIDATES = 5000, 12


def fresh_laws() -> list[blind.Law]:
    return [law for law in blind.invented_laws(N_GENERATED, LAW_SEED) if "+" in law.expression]


def one(law: blind.Law, noise: float, engine: L2DiscoveryEngine) -> dict[str, Any]:
    key = hashlib.sha256(f"{PREREGISTRATION_ID}|{law.law_id}|{noise}".encode()).digest()[:8]
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
                             noise_sigma=noise * float(np.mean(np.abs(y))), seed=0)
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
        if mono not in seen:
            seen.add(mono)
            ranked.append(expr)
        if len(ranked) == MAX_CANDIDATES:
            break
    pool = rng.uniform(1, 10, (POOL_ROWS, len(names)))
    pool_noise = rng.standard_normal(POOL_ROWS)  # one noise draw per pool row, shared by every arm

    def measure(rows: np.ndarray) -> np.ndarray:
        return truth(pool[rows]) * (1 + noise * pool_noise[rows])

    def is_truth(text: str | None) -> bool:
        return text is not None and blind.monomials(sp.sympify(text)) == target

    def arm(exprs: list[sp.Expr], design: str) -> dict[str, Any]:
        if not exprs:
            return {"verdict": "no_candidates", "survivor_is_truth": False, "truth_in_class": False, "rows_read": 0}
        res = cc.contrast_confirm([study.fitted(e, names, dev_X, y) for e in exprs], pool, measure, rows_by=design)
        return {"verdict": res.verdict, "survivor": res.survivor, "equivalent": res.equivalent,
                "survivor_is_truth": is_truth(res.survivor),
                "truth_in_class": is_truth(res.survivor) or any(is_truth(e) for e in res.equivalent),
                "rows_read": res.rows_read}

    promoted = result.adjudication.promoted
    return {
        "law": law.law_id, "expression": law.expression, "noise": noise,
        "truth_among_candidates": any(blind.monomials(e) == target for e in ranked),
        "referee_status": result.adjudication.status.name,
        "referee_promoted_truth": bool(promoted is not None
                                       and blind.monomials(promoted.candidate.metadata["ast"].to_sympy()) == target),
        "contrast": arm(ranked, "contrast"), "random": arm(ranked, "random"),
        "truth_removed": arm([e for e in ranked if blind.monomials(e) != target], "contrast"),
    }


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def count(arm: str, test: Any) -> int:
        return sum(bool(test(r[arm])) for r in rows)

    def unique_truth(a: dict[str, Any]) -> bool:
        return a["verdict"] == cc.UNIQUE and a["survivor_is_truth"]

    def wrong_single(a: dict[str, Any]) -> bool:
        return a["verdict"] in (cc.UNIQUE, cc.UNCHALLENGED) and not a["survivor_is_truth"]

    only_contrast = sum(unique_truth(r["contrast"]) and not unique_truth(r["random"]) for r in rows)
    only_random = sum(unique_truth(r["random"]) and not unique_truth(r["contrast"]) for r in rows)
    discordant = only_contrast + only_random
    return {
        "runs": len(rows),
        "truth_among_candidates": sum(r["truth_among_candidates"] for r in rows),
        "referee_promoted_truth": sum(r["referee_promoted_truth"] for r in rows),
        **{f"{arm}_{name}": count(arm, test) for arm in ("contrast", "random") for name, test in (
            ("unique_truth", unique_truth), ("truth_in_class", lambda a: a["truth_in_class"]),
            ("wrong_law_alone", wrong_single), ("no_survivor", lambda a: a["verdict"] == cc.NO_SURVIVOR))},
        "paired_only_contrast": only_contrast, "paired_only_random": only_random,
        "paired_sign_test_p": float(binomtest(only_contrast, discordant, 0.5, alternative="greater").pvalue) if discordant else 1.0,
        "truth_removed_no_survivor": count("truth_removed", lambda a: a["verdict"] == cc.NO_SURVIVOR),
        "truth_removed_wrong_law_alone": count("truth_removed", wrong_single),
        "truth_removed_equivalent_class": count("truth_removed", lambda a: a["verdict"] == cc.EQUIVALENT_CLASS),
        "median_rows_read": {arm: float(np.median([r[arm]["rows_read"] for r in rows])) for arm in ("contrast", "random")},
    }


def frozen_digests() -> dict[str, str]:
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in FROZEN}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if json.loads(DIGEST.read_text(encoding="utf-8"))["sha256"] != frozen_digests():
        raise SystemExit(f"{PREREGISTRATION_ID} changed since it was frozen")
    args.out.mkdir(parents=True, exist_ok=False)
    engine = L2DiscoveryEngine(operators=blind.OPERATORS)
    rows = []
    with (args.out / "runs.jsonl").open("w", encoding="utf-8") as fh:
        for law in fresh_laws():
            for noise in NOISE_LEVELS:
                row = one(law, noise, engine)
                rows.append(row)
                fh.write(json.dumps(row, sort_keys=True) + "\n")
                fh.flush()
    results = {"preregistration": PREREGISTRATION_ID, "frozen": frozen_digests(), "summary": summarise(rows),
               "by_noise": {str(n): summarise([r for r in rows if r["noise"] == n]) for n in NOISE_LEVELS}}
    (args.out / "results.json").write_text(json.dumps(results, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(results["summary"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
