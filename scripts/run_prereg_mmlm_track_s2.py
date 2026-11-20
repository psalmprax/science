"""The second preregistered test of MMLM Track S (PREREG-MMLM-S2-2026-11).

    python scripts/run_prereg_mmlm_track_s2.py --out <new dir>

Part 1: does the prior-ordered search with an early stop save work without losing laws?
Part 2: does a model trained on one field price another field's (longer, unchosen) laws?
Part 3: fed from the trial ledger, do per-domain models learn at least as well as one pooled model?
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_prereg_mmlm_track_s as s1  # noqa: E402
import run_srbench_baseline as bench  # noqa: E402

from apedp_dyn.discovery_l2 import exhaustive_search as es  # noqa: E402
from apedp_dyn.discovery_l2 import guided_search as gs  # noqa: E402
from apedp_mmlm import equation_language as el  # noqa: E402
from apedp_mmlm import ledger_learning as ll  # noqa: E402
from apedp_referee.trial_ledger import CandidateTrialLedger  # noqa: E402

PREREGISTRATION_ID = "PREREG-MMLM-S2-2026-11"
DIGEST = ROOT / "preregistration" / "DIGEST-MMLM-S2.json"
FROZEN = ("preregistration/PREREG-MMLM-S2-2026-11.md", "scripts/run_prereg_mmlm_track_s2.py",
          "scripts/run_prereg_mmlm_track_s.py", "apedp_mmlm/equation_language.py", "apedp_mmlm/ledger_learning.py",
          "apedp_dyn/discovery_l2/guided_search.py", "apedp_dyn/discovery_l2/exhaustive_search.py",
          "benchmarks/srbench/manifest.json")
INVENTED_SEARCH_SEED, INVENTED_STREAM_SEED = 20261123, 20261124
N_INVENTED_SEARCH, N_STREAM = 40, 60
NOISE_LEVELS = (0.01, 0.05)
N_ROWS = 300


def part_search(model: el.SelfAdjustingEquationModel) -> tuple[list[dict[str, Any]], float]:
    laws = [("other_field", law) for law in s1.other_field_laws()]
    fresh = [law for law in s1.invented(INVENTED_SEARCH_SEED) if law["n_vars"] <= 3 and s1.position(law) is not None]
    laws += [("invented", law) for law in fresh[:N_INVENTED_SEARCH]]
    guided = gs.GuidedSearch(el.search_prior(model))
    started = time.perf_counter()
    for n_vars in sorted({law["n_vars"] for _set, law in laws}):
        guided.order(n_vars, 6 if n_vars <= 3 else 5)
    ordering_seconds = time.perf_counter() - started
    rows = []
    for which, law in laws:
        names = [f"x{i}" for i in range(law["n_vars"])]
        limit = 6 if law["n_vars"] <= 3 else 5
        fn = sp.lambdify([sp.Symbol(n) for n in names], law["expr"], "numpy")
        for noise in NOISE_LEVELS:
            key = hashlib.sha256(f"{PREREGISTRATION_ID}|{law['name']}|{noise}".encode()).digest()[:8]
            rng = np.random.default_rng(int.from_bytes(key, "big"))
            X = rng.uniform(1.0, 5.0, (N_ROWS + 200, law["n_vars"]))
            clean = np.asarray(fn(*X.T), dtype=float) * np.ones(len(X))
            y = clean[:N_ROWS] * (1 + noise * rng.standard_normal(N_ROWS))

            def recovered(best: es.Scored, law: dict[str, Any] = law, names: list[str] = names, X: np.ndarray = X) -> bool:
                found = sp.sympify(es.model_string(best))
                return bool(bench.symbolic_recovery(found, law["expr"], names, X[N_ROWS:])["recovered"])

            t0 = time.perf_counter()
            g = guided.search(X[:N_ROWS], y, names, limit)
            t1 = time.perf_counter()
            full = es.search(X[:N_ROWS], y, names, limit)[0]
            t2 = time.perf_counter()
            rows.append({"set": which, "name": law["name"], "noise": noise, "candidates": g.candidates,
                         "evaluated": g.evaluated, "stopped_early": g.stopped_early,
                         "guided_recovered": recovered(g.best), "full_recovered": recovered(full),
                         "guided_selected": g.best.expression, "full_selected": full.expression,
                         "guided_seconds": t1 - t0, "full_seconds": t2 - t1})
    return rows, ordering_seconds


def summarise_search(rows: list[dict[str, Any]], ordering_seconds: float) -> dict[str, Any]:
    def cell(sel: list[dict[str, Any]]) -> dict[str, Any]:
        return {"runs": len(sel), "guided_recovered": sum(r["guided_recovered"] for r in sel),
                "full_recovered": sum(r["full_recovered"] for r in sel),
                "stopped_early": sum(r["stopped_early"] for r in sel),
                "wrong_early_stops": sum(r["stopped_early"] and not r["guided_recovered"] for r in sel),
                "median_share_of_candidates_fitted": float(np.median([r["evaluated"] / r["candidates"] for r in sel])),
                "guided_seconds": float(sum(r["guided_seconds"] for r in sel)),
                "full_seconds": float(sum(r["full_seconds"] for r in sel))}

    out = {"all": cell(rows), "one_time_ordering_seconds": ordering_seconds}
    for which in ("other_field", "invented"):
        for noise in NOISE_LEVELS:
            out[f"{which}|{noise}"] = cell([r for r in rows if r["set"] == which and r["noise"] == noise])
    return out


def part_transfer() -> dict[str, Any]:
    """Train on the Feynman formulas only; price the 14 Strogatz right-hand sides."""
    entries = bench.load_manifest()
    trees = {entry["name"]: el.to_tree(bench.truth_expr(entry)) for entry in entries}
    train = [trees[e["name"]] for e in entries if e["family"] == "feynman"]
    test = [trees[e["name"]] for e in entries if e["family"] == "strogatz"]
    models: dict[str, Any] = {name: el.EquationLanguageModel(name).fit(train) for name in ("uniform", "unigram")}
    models["mixture"] = el.SelfAdjustingEquationModel().fit(train)
    tokens = sum(len(el.tokens(t)) for t in test)
    return {"test_laws": len(test), "test_tokens": tokens, "mean_tokens_per_law": tokens / len(test),
            "bits_per_token": {name: float(sum(m.bits(t) for t in test)) / tokens for name, m in models.items()}}


def part_domains() -> dict[str, Any]:
    """Two domains interleaved through a real trial ledger: per-domain models against one pooled model."""
    entries = bench.load_manifest()
    seed = [el.to_tree(bench.truth_expr(e)) for e in entries if e["family"] == "feynman"]
    streams = {"dynamics": [{"name": e["name"], "expr": bench.truth_expr(e)} for e in entries if e["family"] == "strogatz"],
               "invented": s1.invented(INVENTED_STREAM_SEED)[:N_STREAM]}
    order = [("invented", i) for i in range(N_STREAM)]
    step = N_STREAM // len(streams["dynamics"])
    for k in range(len(streams["dynamics"])):
        order.insert(k * (step + 1) + step, ("dynamics", k))
    ledger = CandidateTrialLedger()
    learner = ll.DomainLearner(seed)
    pooled = el.SelfAdjustingEquationModel().fit(seed)
    frozen = el.SelfAdjustingEquationModel().fit(seed)
    costs: list[dict[str, Any]] = []
    for domain, index in order:
        law = streams[domain][index]
        tree = ll.parse_law(str(law["expr"]))
        if tree is None:
            costs.append({"domain": domain, "name": law["name"], "parsed": False})
            continue
        row = {"domain": domain, "name": law["name"], "parsed": True, "frozen": frozen.bits(tree),
               "per_domain": learner.model(domain).bits(tree), "pooled": pooled.observe(tree)}
        ledger.record_trial(candidate_id=f"{domain}:{law['name']}", proposer_id="contrast_resolution",
                            formula=str(law["expr"]), strategy_archetype="equation_discovery", in_sample_sharpe=0.0,
                            status="confirmed", metadata={"domain": domain, "resolved_by": "contrast"})
        row["sync"] = learner.sync(ledger)
        costs.append(row)

    def mean(domain: str, key: str, second_half: bool) -> float | None:
        sel = [c for c in costs if c["domain"] == domain and c["parsed"]]
        sel = sel[len(sel) // 2:] if second_half else sel
        return float(np.mean([c[key] for c in sel])) if sel else None

    return {"per_law": costs, "unparsed": sum(not c["parsed"] for c in costs),
            "second_half_mean_bits": {d: {k: mean(d, k, True) for k in ("frozen", "pooled", "per_domain")} for d in streams},
            "all_mean_bits": {d: {k: mean(d, k, False) for k in ("frozen", "pooled", "per_domain")} for d in streams},
            "final_state": learner.state()}


def frozen_digests() -> dict[str, str]:
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in FROZEN}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if json.loads(DIGEST.read_text(encoding="utf-8"))["sha256"] != frozen_digests():
        raise SystemExit(f"{PREREGISTRATION_ID} changed since it was frozen")
    args.out.mkdir(parents=True, exist_ok=False)
    model = el.SelfAdjustingEquationModel().fit(s1.srbench_trees())
    search_rows, ordering_seconds = part_search(model)
    domains = part_domains()
    results = {"preregistration": PREREGISTRATION_ID, "frozen": frozen_digests(),
               "part_1_search": summarise_search(search_rows, ordering_seconds),
               "part_2_transfer": part_transfer(),
               "part_3_domains": {k: v for k, v in domains.items() if k != "per_law"}}
    (args.out / "results.json").write_text(json.dumps(results, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    (args.out / "runs.json").write_text(json.dumps({"part_1_search": search_rows, "part_3_domains": domains["per_law"]},
                                                   indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
