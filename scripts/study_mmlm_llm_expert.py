#!/usr/bin/env python3
"""Development: does a real language model, as one of MMLM's experts, bring the true law forward in the search?

**Development, not a result.** MMLM's equation model orders the exhaustive search without looking at the data
(PREREG-MMLM-S: the true law is reached after about a twentieth of the candidates). ``apedp_mmlm.llm_expert`` adds
an expert that reads the table: a large language model shown rows with anonymous names, which replies with
shapes to try first. This script measures, law by law, where the true law stands in four orders of the same
candidates:

* ``size``: the search's own order, smallest first;
* ``mmlm``: by the equation model fitted on SRBench's formulas, as in PREREG-MMLM-S;
* ``llm_alone``: only the shapes the language model named, in its order (the law is there or it is not);
* ``mmlm_with_llm``: the named shapes first, then the rest by the equation model taught those shapes.

Laws: the blind benchmark's invented laws from a seed used nowhere else, and the other-field laws of
PREREG-MMLM-S, 300 rows each at 1 % noise; only laws the exhaustive search can express are scored. The language
model sees numbers only. The invented laws cannot have been read anywhere; the other-field laws are textbook
shapes and are reported apart.

    NVIDIA_API_KEY=... python scripts/study_mmlm_llm_expert.py --out DIR [--laws 60] [--model NAME]
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
import sympy as sp
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_prereg_mmlm_track_s as track_s  # noqa: E402

from apedp_mmlm import equation_language as el  # noqa: E402
from apedp_mmlm import llm_expert  # noqa: E402

SEED = 2026112490            # the invented laws' seed, used nowhere else
N_ROWS, NOISE = 300, 0.01
WITHIN = (1, 10, 100, 1000)
ORDERS = ("size", "mmlm", "llm_alone", "mmlm_with_llm")
#: strongest first; the first that answers is used (apedp_dyn.proposers.llm_clients.pick_model)
MODELS = ("nvidia/nemotron-3-ultra-550b-a55b", "nvidia/nemotron-3.5-lightning-30b-a3b", "minimaxai/minimax-m3.1-flash")


def laws(count: int) -> list[dict[str, Any]]:
    """Laws the search can express: invented ones first (``count`` of them), then the other-field laws."""
    invented = [{**law, "family": "invented"} for law in track_s.invented(SEED) if track_s.position(law) is not None][:count]
    other = [{**law, "family": "other fields"} for law in track_s.other_field_laws() if track_s.position(law) is not None]
    return [*invented, *other]


def table_of(law: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    seed = int.from_bytes(hashlib.sha256(f"mmlm-llm-expert|{law['name']}".encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    X = rng.uniform(1.0, 5.0, (N_ROWS, law["n_vars"]))
    fn = sp.lambdify([sp.Symbol(f"x{i}") for i in range(law["n_vars"])], law["expr"], "numpy")
    clean = np.asarray(fn(*X.T), dtype=float) * np.ones(N_ROWS)
    return X, clean * (1 + NOISE * rng.standard_normal(N_ROWS))


def one(law: dict[str, Any], model: el.SelfAdjustingEquationModel, base_scores: dict[int, np.ndarray], expert: llm_expert.LlmExpert
        ) -> dict[str, Any]:
    nodes, index, _limit = track_s.candidates(law["n_vars"])
    where = track_s.position(law)
    if where is None:
        raise ValueError(f"{law['name']} is not among the search's candidates")
    X, y = table_of(law)
    shapes, call = expert.shapes(X, y)
    named = [place for place in (llm_expert.place_in_search(shape, law["n_vars"], index) for shape in shapes) if place is not None]
    informed = el.search_prior(llm_expert.taught(model, shapes))
    informed_scores = np.array([informed(node) for node, _c in nodes])
    first = list(dict.fromkeys(named))
    return {"name": law["name"], "family": law["family"], "law": str(law["expr"]), "inputs": law["n_vars"], "candidates": len(nodes),
            "rank": {"size": where + 1, "mmlm": llm_expert.rank_by(base_scores[law["n_vars"]], where),
                     "llm_alone": first.index(where) + 1 if where in first else None,
                     "mmlm_with_llm": llm_expert.rank_with_expert(where, named, informed_scores)},
            "shapes_named": len(shapes), "shapes_in_the_search": len(first), "call": call}


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def block(sel: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {"laws": len(sel)}
        for order in ORDERS:
            ranks = [r["rank"][order] for r in sel]
            reached = [v for v in ranks if v is not None]
            out[order] = {"median_rank": float(np.median(reached)) if len(reached) == len(sel) and sel else None,
                          **{f"within_{n}": sum(v is not None and v <= n for v in ranks) for n in WITHIN}}
        better = sum(r["rank"]["mmlm_with_llm"] < r["rank"]["mmlm"] for r in sel)
        worse = sum(r["rank"]["mmlm_with_llm"] > r["rank"]["mmlm"] for r in sel)
        out["with_llm_against_mmlm"] = {"earlier": better, "later": worse, "same": len(sel) - better - worse,
                                        "sign_test_p_one_sided": float(binomtest(better, better + worse, 0.5, alternative="greater").pvalue)
                                        if better + worse else None}
        out["law_among_the_named_shapes"] = sum(r["rank"]["llm_alone"] is not None for r in sel)
        return out

    families = sorted({r["family"] for r in rows})
    usage: dict[str, int] = {}
    for row in rows:
        for key, value in (row["call"].get("usage") or {}).items():
            usage[key] = usage.get(key, 0) + int(value)
    return {"all": block(rows), **{family: block([r for r in rows if r["family"] == family]) for family in families},
            "calls": len(rows), "calls_failed": sum("error" in r["call"] for r in rows), "tokens": usage or None}


def run(expert: llm_expert.LlmExpert, count: int, progress: Any = None) -> dict[str, Any]:
    model = el.SelfAdjustingEquationModel().fit(track_s.srbench_trees())
    prior = el.search_prior(model)
    chosen = laws(count)
    base_scores = {n: np.array([prior(node) for node, _c in track_s.candidates(n)[0]]) for n in sorted({law["n_vars"] for law in chosen})}
    rows = []
    for law in chosen:
        rows.append(one(law, model, base_scores, expert))
        if progress is not None:
            progress(rows)
    return {"what": "development: a language model as one of MMLM's experts, by where the true law stands in the search; not a result",
            "laws_seed": SEED, "rows_per_table": N_ROWS, "noise": NOISE, "model": getattr(expert.client, "model", None),
            "summary": summarise(rows), "rows": rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--laws", type=int, default=60)
    parser.add_argument("--model", default="")
    args = parser.parse_args(argv)
    from apedp_dyn.proposers import llm_clients, llm_https

    name = args.model or llm_clients.pick_model(MODELS, fall_back_to_listed=False)
    client = llm_https.nvidia_client(name, temperature=0.0, max_tokens=400)
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    def progress(rows: list[dict[str, Any]]) -> None:
        (args.out / "mmlm_llm_expert_rows.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")

    record = run(llm_expert.LlmExpert(client), args.laws, progress)
    record["seconds"] = round(time.perf_counter() - started, 1)
    (args.out / "mmlm_llm_expert.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"model": record["model"], **record["summary"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
