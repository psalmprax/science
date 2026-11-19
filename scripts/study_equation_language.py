"""Development study of MMLM Track S: does a language model of equations know anything useful?

    python scripts/study_equation_language.py

NOT a preregistered result. Corpus: the 133 formulas of SRBench's ground-truth
track (119 Feynman, 14 Strogatz), converted to trees. Five-fold cross-validation
(seeded shuffle): every number is for equations the model was not trained on.

1. **Bits per held-out equation** under each engine, against a uniform code and
   a unigram code. Lower is better; this is the standard language-model measure.
2. **Search order.** For each held-out law that the exhaustive search can reach
   (it appears among the expressions of up to six nodes, five for more than
   three variables), its position when the candidates are tried in the search's
   own order (smallest first) and when they are tried in order of the model's
   code length. Fewer candidates before the law means less search.

Engines were run with the settings in ``apedp_mmlm/equation_language.py`` and
not tuned on these numbers.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_srbench_baseline as bench  # noqa: E402

from apedp_dyn.discovery_l2 import exhaustive_search as es  # noqa: E402
from apedp_mmlm import equation_language as el  # noqa: E402

OUT = ROOT / "benchmarks" / "equation_language_dev_study.json"
FOLDS, SEED = 5, 20261119
SEARCH_ENGINES = ("unigram", "tree", "automaton", "bayesian")


def corpus() -> list[dict[str, Any]]:
    rows = []
    for entry in bench.load_manifest():
        expr = bench.truth_expr(entry)
        rows.append({"name": entry["name"], "n_vars": len(entry["features"]), "expr": expr, "tree": el.to_tree(expr)})
    return rows


def reachable(row: dict[str, Any], enumerations: dict[int, Any]) -> int | None:
    """Index of the law among the search's candidates, or None if it is not among them."""
    n = row["n_vars"]
    if n not in enumerations:
        nodes = es.enumerate_expressions(n, 6 if n <= 3 else 5)
        probes = es._probes(n)
        enumerations[n] = (nodes, {es._fingerprint(es.evaluate(node, probes)): i for i, (node, _c) in enumerate(nodes)})
    _nodes, index = enumerations[n]
    fn = sp.lambdify([sp.Symbol(f"x{i}") for i in range(n)], row["expr"], "numpy")
    with np.errstate(all="ignore"):
        key = es._fingerprint(np.asarray(fn(*es._probes(n).T), dtype=float) * np.ones(es.N_PROBE))
    return index.get(key) if key is not None else None


def main() -> int:
    rows = corpus()
    order = np.random.default_rng(SEED).permutation(len(rows))
    fold_of = {int(i): k % FOLDS for k, i in enumerate(order)}
    enumerations: dict[int, Any] = {}
    positions = {i: reachable(row, enumerations) for i, row in enumerate(rows) if row["n_vars"] <= 5}
    bits: dict[str, list[float]] = {engine: [] for engine in el.ENGINES}
    ranks: list[dict[str, Any]] = []
    for fold in range(FOLDS):
        train = [rows[i]["tree"] for i in range(len(rows)) if fold_of[i] != fold]
        test = [i for i in range(len(rows)) if fold_of[i] == fold]
        models = {engine: el.EquationLanguageModel(engine).fit(train) for engine in el.ENGINES}
        for i in test:
            for engine, model in models.items():
                bits[engine].append(model.bits(rows[i]["tree"]))
        for i in test:
            position = positions.get(i)
            if position is None:
                continue
            nodes, _index = enumerations[rows[i]["n_vars"]]
            trees = [el.from_search_node(node) for node, _c in nodes]
            record: dict[str, Any] = {"name": rows[i]["name"], "candidates": len(nodes), "size_order": position + 1}
            for engine in SEARCH_ENGINES:
                scores = np.array([models[engine].bits(tree) for tree in trees])
                mine = scores[position]
                record[engine] = int(np.sum(scores < mine) + np.sum((scores == mine)[:position]) + 1)
            ranks.append(record)
    summary: dict[str, Any] = {
        "equations": len(rows),
        "mean_bits_per_held_out_equation": {engine: float(np.mean(v)) for engine, v in bits.items()},
        "reachable_by_the_search": len(ranks),
        "median_candidates_tried_before_the_law": {
            key: float(np.median([r[key] for r in ranks])) for key in ("size_order", *SEARCH_ENGINES)},
        "mean_log10_candidates_tried": {
            key: float(np.mean(np.log10([r[key] for r in ranks]))) for key in ("size_order", *SEARCH_ENGINES)},
        "laws_found_earlier_than_size_order": {
            engine: sum(r[engine] < r["size_order"] for r in ranks) for engine in SEARCH_ENGINES},
        "laws_found_later_than_size_order": {
            engine: sum(r[engine] > r["size_order"] for r in ranks) for engine in SEARCH_ENGINES},
    }
    OUT.write_text(json.dumps({"status": "development study, not preregistered", "summary": summary, "ranks": ranks},
                              indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
