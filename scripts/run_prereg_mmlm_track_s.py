"""The preregistered test of MMLM Track S (PREREG-MMLM-S-2026-11).

    python scripts/run_prereg_mmlm_track_s.py --out <new dir>

Part A: does a model trained on SRBench's formulas help the search on laws from
elsewhere? Part B: does the self-adjusting model get better as it is fed laws?
"""
from __future__ import annotations

import argparse
import copy
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
import run_srbench_baseline as bench  # noqa: E402

from apedp_dyn.discovery_l2 import exhaustive_search as es  # noqa: E402
from apedp_mmlm import equation_language as el  # noqa: E402

PREREGISTRATION_ID = "PREREG-MMLM-S-2026-11"
DIGEST = ROOT / "preregistration" / "DIGEST-MMLM-S.json"
FROZEN = ("preregistration/PREREG-MMLM-S-2026-11.md", "scripts/run_prereg_mmlm_track_s.py",
          "apedp_mmlm/equation_language.py", "apedp_dyn/discovery_l2/exhaustive_search.py",
          "scripts/run_srbench_baseline.py", "scripts/run_blind_discovery_benchmark.py",
          "benchmarks/srbench/manifest.json")
STREAM_SEED, INVENTED_A_SEED = 20261121, 20261122
N_ROWS, NOISE = 300, 0.01

#: Laws from outside SRBench's ground-truth track (chemistry, biology, engineering, finance,
#: geometry), written before the model was run on any of them. Constants are dropped: the search
#: fits scale and offset.
OTHER_FIELD_LAWS: dict[str, str] = {
    "saturation_fraction": "x0/(x0 + x1)",          # Michaelis-Menten / Langmuir occupancy
    "arrhenius_factor": "exp(-x0/x1)",
    "ideal_gas_pressure": "x0*x1/x2",
    "radiated_power": "x0*x1**4",                   # Stefan-Boltzmann, area times T^4
    "kepler_period": "x0*sqrt(x0/x1)",
    "beer_lambert": "x0/exp(x1*x2)",
    "logistic_growth_rate": "x0*(x1 - x0)",
    "nernst_potential": "x0*log(x1/x2)",
    "compound_growth": "x0*exp(x1*x2)",
    "geometric_mean_output": "sqrt(x0*x1)",         # Cobb-Douglas with equal shares
    "drag_force": "x0*x1**2*x2",
    "reynolds_number": "x0*x1/x2",
    "parallel_resistance": "1/(1/x0 + 1/x1)",
    "hypotenuse": "sqrt(x0**2 + x1**2)",
    "escape_speed": "sqrt(x0/x1)",
    "boltzmann_ratio": "exp((x0 - x1)/x2)",
    "entropy_term": "x0*log(x0)",
    "body_mass_index": "x0/x1**2",
    "radioactive_decay": "x0/exp(x1/x2)",
    "sharpe_ratio": "(x0 - x1)/x2",
    "kelly_fraction": "x0 - x1/x2",
    "cylinder_volume": "x0**2*x1",
    "tidal_force": "x0/x1**3",
    "log_return": "log(x0/x1)",
}


def _law(name: str, text: str) -> dict[str, Any]:
    expr = sp.sympify(text)
    n = len(expr.free_symbols)
    return {"name": name, "expr": expr, "n_vars": n, "tree": el.to_tree(expr)}


def other_field_laws() -> list[dict[str, Any]]:
    return [_law(name, text) for name, text in OTHER_FIELD_LAWS.items()]


def invented(seed: int) -> list[dict[str, Any]]:
    """The blind benchmark's generator, variables renamed x0, x1 in order of appearance."""
    out = []
    for law in blind.invented_laws(120, seed):
        expr = law.sympy()
        names = sorted(expr.free_symbols, key=lambda s: list(law.dimensions).index(str(s)))
        expr = expr.xreplace({s: sp.Symbol(f"x{i}") for i, s in enumerate(names)})
        out.append({"name": law.law_id, "expr": expr, "n_vars": len(names), "tree": el.to_tree(expr)})
    return out


def srbench_trees() -> list[el.Tree]:
    return [el.to_tree(bench.truth_expr(entry)) for entry in bench.load_manifest()]


_ENUM: dict[int, Any] = {}


def candidates(n_vars: int) -> tuple[list[Any], dict[bytes, int], int]:
    if n_vars not in _ENUM:
        limit = 6 if n_vars <= 3 else 5
        nodes = es.enumerate_expressions(n_vars, limit)
        probes = es._probes(n_vars)
        _ENUM[n_vars] = (nodes, {es._fingerprint(es.evaluate(node, probes)): i for i, (node, _c) in enumerate(nodes)}, limit)
    return _ENUM[n_vars]


def position(law: dict[str, Any]) -> int | None:
    _nodes, index, _limit = candidates(law["n_vars"])
    fn = sp.lambdify([sp.Symbol(f"x{i}") for i in range(law["n_vars"])], law["expr"], "numpy")
    with np.errstate(all="ignore"):
        key = es._fingerprint(np.asarray(fn(*es._probes(law["n_vars"]).T), dtype=float) * np.ones(es.N_PROBE))
    return index.get(key) if key is not None else None


def part_a(model: el.SelfAdjustingEquationModel, laws: list[dict[str, Any]], with_data: bool) -> list[dict[str, Any]]:
    prior = el.search_prior(model)
    rows = []
    for law in laws:
        where = position(law)
        if where is None:
            rows.append({"name": law["name"], "reachable": False})
            continue
        nodes, _index, limit = candidates(law["n_vars"])
        scores = np.array([prior(node) for node, _c in nodes])
        mine = scores[where]
        row: dict[str, Any] = {"name": law["name"], "reachable": True, "candidates": len(nodes), "size_order": where + 1,
                               "model_order": int(np.sum(scores < mine) + np.sum((scores == mine)[:where]) + 1)}
        if with_data:
            seed = int.from_bytes(hashlib.sha256(f"{PREREGISTRATION_ID}|{law['name']}".encode()).digest()[:8], "big")
            rng = np.random.default_rng(seed)
            names = [f"x{i}" for i in range(law["n_vars"])]
            X = rng.uniform(1.0, 5.0, (N_ROWS + 200, law["n_vars"]))
            fn = sp.lambdify([sp.Symbol(n) for n in names], law["expr"], "numpy")
            clean = np.asarray(fn(*X.T), dtype=float)
            y = clean[:N_ROWS] * (1 + NOISE * rng.standard_normal(N_ROWS))
            for label, bits in (("plain", None), ("with_model", prior)):
                best = es.search(X[:N_ROWS], y, names, limit, structure_bits=bits)[0]
                found = sp.sympify(es.model_string(best))
                row[f"recovered_{label}"] = bool(bench.symbolic_recovery(found, law["expr"], names, X[N_ROWS:])["recovered"])
                row[f"selected_{label}"] = best.expression
        rows.append(row)
    return rows


def part_b(seed_trees: list[el.Tree], stream: list[dict[str, Any]]) -> dict[str, Any]:
    adaptive = el.SelfAdjustingEquationModel().fit(seed_trees)
    frozen = copy.deepcopy(adaptive)
    costs = []
    for law in stream:
        costs.append({"name": law["name"], "frozen": frozen.bits(law["tree"]), "adaptive": adaptive.observe(law["tree"])})
    half = len(costs) // 2
    return {"per_law": costs, "final_state": adaptive.state(),
            "mean_bits_first_half": {k: float(np.mean([c[k] for c in costs[:half]])) for k in ("frozen", "adaptive")},
            "mean_bits_second_half": {k: float(np.mean([c[k] for c in costs[half:]])) for k in ("frozen", "adaptive")}}


def summarise_a(rows: list[dict[str, Any]]) -> dict[str, Any]:
    hit = [r for r in rows if r["reachable"]]
    earlier = sum(r["model_order"] < r["size_order"] for r in hit)
    later = sum(r["model_order"] > r["size_order"] for r in hit)
    out: dict[str, Any] = {
        "laws": len(rows), "reachable": len(hit), "found_earlier": earlier, "found_later": later,
        "sign_test_p": float(binomtest(earlier, earlier + later, 0.5, alternative="greater").pvalue) if earlier + later else 1.0,
        "median_size_order": float(np.median([r["size_order"] for r in hit])) if hit else None,
        "median_model_order": float(np.median([r["model_order"] for r in hit])) if hit else None,
    }
    if hit and "recovered_plain" in hit[0]:
        out["recovered_plain"] = sum(r["recovered_plain"] for r in hit)
        out["recovered_with_model"] = sum(r["recovered_with_model"] for r in hit)
    return out


def frozen_digests() -> dict[str, str]:
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in FROZEN}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if json.loads(DIGEST.read_text(encoding="utf-8"))["sha256"] != frozen_digests():
        raise SystemExit(f"{PREREGISTRATION_ID} changed since it was frozen")
    args.out.mkdir(parents=True, exist_ok=False)
    seed_trees = srbench_trees()
    model = el.SelfAdjustingEquationModel().fit(seed_trees)
    a_other = part_a(model, other_field_laws(), with_data=True)
    seen_shapes = {tuple(el.tokens(tree)) for tree in seed_trees}
    for row, law in zip(a_other, other_field_laws(), strict=True):
        row["shape_in_training"] = tuple(el.tokens(law["tree"])) in seen_shapes
    a_invented = part_a(model, [law for law in invented(INVENTED_A_SEED) if law["n_vars"] <= 3], with_data=False)
    b_invented = part_b(seed_trees, invented(STREAM_SEED))
    b_other = part_b(seed_trees, other_field_laws())
    results = {
        "preregistration": PREREGISTRATION_ID, "frozen": frozen_digests(),
        "part_a_other_fields": summarise_a(a_other), "part_a_invented": summarise_a(a_invented),
        "part_a_other_fields_shape_not_in_training": summarise_a([r for r in a_other if not r["shape_in_training"]]),
        "part_b_invented_stream": {k: v for k, v in b_invented.items() if k != "per_law"},
        "part_b_other_field_stream": {k: v for k, v in b_other.items() if k != "per_law"},
    }
    (args.out / "results.json").write_text(json.dumps(results, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    (args.out / "runs.json").write_text(json.dumps(
        {"part_a_other_fields": a_other, "part_a_invented": a_invented,
         "part_b_invented_stream": b_invented["per_law"], "part_b_other_field_stream": b_other["per_law"]},
        indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
