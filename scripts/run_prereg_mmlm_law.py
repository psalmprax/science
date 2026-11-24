"""PREREG-MMLM-L-2026-11: the scientific model (laws) on laws it has not seen, and as a search prior.

    python scripts/run_prereg_mmlm_law.py --out preregistration/RESULT-MMLM-L-2026-11

Runs once. Refuses to run if a frozen file changed, or if the output directory exists.
Protocol and expectations: ``preregistration/PREREG-MMLM-L-2026-11.md``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_prereg_mmlm_track_s as s1  # noqa: E402
import run_srbench_baseline as bench  # noqa: E402

from apedp_dyn.discovery_l2 import exhaustive_search as es  # noqa: E402
from apedp_dyn.discovery_l2 import guided_search as gs  # noqa: E402
from apedp_dyn.discovery_l2.confirmed_stop import search_confirmed  # noqa: E402
from apedp_mmlm import equation_language as el  # noqa: E402
from apedp_mmlm import law_corpora  # noqa: E402
from apedp_mmlm import law_language as ll  # noqa: E402

PREREGISTRATION_ID = "PREREG-MMLM-L-2026-11"
DIGEST = ROOT / "preregistration" / "DIGEST-MMLM-L.json"
FROZEN = ("preregistration/PREREG-MMLM-L-2026-11.md", "scripts/run_prereg_mmlm_law.py",
          "apedp_mmlm/law_language.py", "apedp_mmlm/law_corpora.py", "apedp_mmlm/equation_language.py",
          "apedp_dyn/discovery_l2/exhaustive_search.py", "apedp_dyn/discovery_l2/guided_search.py",
          "apedp_dyn/discovery_l2/confirmed_stop.py", "benchmarks/srbench/manifest.json",
          "scripts/run_prereg_mmlm_track_s.py", "scripts/run_srbench_baseline.py")
SEARCH_SEED = 20261125
N_INVENTED = 40
NOISE_LEVELS = (0.01, 0.05)
N_ROWS = 300
DYNAMICS = "dynamics"


def training() -> tuple[list[el.Tree], list[str]]:
    """SRBench's 133 laws with a domain each: the Feynman volume, or ``dynamics`` for the Strogatz systems."""
    trees, domains = [], []
    for entry in bench.load_manifest():
        trees.append(el.to_tree(bench.truth_expr(entry)))
        domains.append(DYNAMICS if entry["family"] == "strogatz" else "feynman_" + entry["name"].split("_")[1])
    return trees, domains


def fresh_laws(trained: list[el.Tree]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """ODEBench components, less those whose canonical tokens equal a training law's or an earlier component's."""
    seen = {tuple(ll.law_tokens(tree)) for tree in trained}
    kept, in_training, repeated = [], 0, 0
    for law in law_corpora.odebench():
        law["tree"] = el.to_tree(law["expr"])
        key = tuple(ll.law_tokens(law["tree"]))
        if key in seen:
            if any(key == tuple(ll.law_tokens(tree)) for tree in trained):
                in_training += 1
            else:
                repeated += 1
            continue
        seen.add(key)
        kept.append(law)
    return kept, {"components": len(kept) + in_training + repeated, "same_as_a_training_law": in_training,
                  "same_as_an_earlier_component": repeated, "kept": len(kept)}


def sign_test(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    """Paired sign test that ``a`` is lower than ``b``; ties are dropped."""
    wins, losses = int((a < b).sum()), int((a > b).sum())
    p = float(binomtest(wins, wins + losses, 0.5).pvalue) if wins + losses else 1.0
    return {"lower": wins, "higher": losses, "p_value": p}


def prequential_unigram(seed: list[el.Tree], stream: list[el.Tree]) -> list[float]:
    known, out = list(seed), []
    for tree in stream:
        out.append(float(el.EquationLanguageModel("unigram").fit(known).bits(tree)))
        known.append(tree)
    return out


def part_language(trees: list[el.Tree], domains: list[str], fresh: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    stream = [law["tree"] for law in fresh]
    columns: dict[str, list[float]] = {}
    for label, keep in (("all", [True] * len(trees)), ("feynman_only", [d != DYNAMICS for d in domains])):
        seed = [t for t, k in zip(trees, keep, strict=True) if k]
        seed_domains = [d for d, k in zip(domains, keep, strict=True) if k]
        law = ll.LawLanguageModel().fit(seed, seed_domains)
        old = el.SelfAdjustingEquationModel().fit(seed)
        unigram = el.EquationLanguageModel("unigram").fit(seed)
        columns[f"{label}|frozen|law_model"] = [law.bits(t, DYNAMICS) for t in stream]
        columns[f"{label}|frozen|earlier_model"] = [float(old.bits(t)) for t in stream]
        columns[f"{label}|frozen|unigram"] = [float(unigram.bits(t)) for t in stream]
        columns[f"{label}|learning|law_model"] = [law.observe(t, DYNAMICS) for t in stream]
        columns[f"{label}|learning|earlier_model"] = [float(old.observe(t)) for t in stream]
        columns[f"{label}|learning|unigram"] = prequential_unigram(seed, stream)
        if label == "all":
            final_state = law.state()
    arrays = {k: np.array(v) for k, v in columns.items()}
    summary: dict[str, Any] = {"laws": len(stream), "mean_bits_per_law": {k: float(v.mean()) for k, v in arrays.items()},
                               "law_model_state_after_learning": final_state}
    for label in ("all", "feynman_only"):
        for mode in ("frozen", "learning"):
            for rival in ("earlier_model", "unigram"):
                summary[f"sign_test|{label}|{mode}|law_model_vs_{rival}"] = sign_test(
                    arrays[f"{label}|{mode}|law_model"], arrays[f"{label}|{mode}|{rival}"])
    per_law = [{"name": law["name"], "tokens": " ".join(ll.law_tokens(law["tree"])), **{k: v[i] for k, v in columns.items()}}
               for i, law in enumerate(fresh)]
    return summary, per_law


def part_search(trees: list[el.Tree], domains: list[str], fresh: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    invented = [law for law in s1.invented(SEARCH_SEED) if law["n_vars"] <= 3 and s1.position(law) is not None][:N_INVENTED]
    reachable = [law for law in fresh if 1 <= law["n_vars"] <= 3 and s1.position(law) is not None]
    laws = [("invented", None, law) for law in invented] + [("odebench", DYNAMICS, law) for law in reachable]
    model = ll.LawLanguageModel().fit(trees, domains)
    old = gs.GuidedSearch(el.search_prior(el.SelfAdjustingEquationModel().fit(trees)))
    new = {None: gs.GuidedSearch(ll.search_prior(model)), DYNAMICS: gs.GuidedSearch(ll.search_prior(model, DYNAMICS))}
    rows = []
    for which, domain, law in laws:
        names = [f"x{i}" for i in range(law["n_vars"])]
        limit = 6 if law["n_vars"] <= 3 else 5
        fn = sp.lambdify([sp.Symbol(n) for n in names], law["expr"], "numpy")
        truth = s1.position(law)
        ranks = {}
        for tag, guided in (("earlier_prior", old), ("law_prior", new[domain])):
            order = guided.order(law["n_vars"], limit)[2]
            ranks[tag] = int(np.where(order == truth)[0][0]) + 1
        for noise in NOISE_LEVELS:
            key = hashlib.sha256(f"{PREREGISTRATION_ID}|{law['name']}|{noise}".encode()).digest()[:8]
            rng = np.random.default_rng(int.from_bytes(key, "big"))
            X = rng.uniform(1.0, 5.0, (N_ROWS + 200, law["n_vars"]))
            clean = np.asarray(fn(*X.T), dtype=float) * np.ones(len(X))
            y = clean[:N_ROWS] * (1 + noise * rng.standard_normal(N_ROWS))

            def recovered(best: es.Scored, law: dict[str, Any] = law, names: list[str] = names, X: np.ndarray = X) -> bool:
                found = sp.sympify(es.model_string(best))
                return bool(bench.symbolic_recovery(found, law["expr"], names, X[N_ROWS:])["recovered"])

            row: dict[str, Any] = {"set": which, "name": law["name"], "noise": noise, "n_vars": law["n_vars"],
                                   "rank_earlier_prior": ranks["earlier_prior"], "rank_law_prior": ranks["law_prior"]}
            t0 = time.perf_counter()
            full = es.search(X[:N_ROWS], y, names, limit)[0]
            row.update(full_recovered=recovered(full), full_seconds=time.perf_counter() - t0, full_selected=full.expression)
            arms: dict[str, Callable[[], gs.GuidedResult]] = {"earlier_prior_first_stop": lambda: old.search(X[:N_ROWS], y, names, limit),  # noqa: B023
                    "earlier_prior_confirmed": lambda: search_confirmed(old, X[:N_ROWS], y, names, limit),  # noqa: B023
                    "law_prior_first_stop": lambda: new[domain].search(X[:N_ROWS], y, names, limit),  # noqa: B023
                    "law_prior_confirmed": lambda: search_confirmed(new[domain], X[:N_ROWS], y, names, limit)}  # noqa: B023
            for arm, run in arms.items():
                t0 = time.perf_counter()
                result = run()
                row[arm] = {"seconds": time.perf_counter() - t0, "evaluated": result.evaluated, "candidates": result.candidates,
                            "stopped_early": bool(result.stopped_early), "recovered": recovered(result.best),
                            "selected": result.best.expression}
            rows.append(row)
    return summarise_search(rows), rows


ARMS = ("earlier_prior_first_stop", "earlier_prior_confirmed", "law_prior_first_stop", "law_prior_confirmed")


def summarise_search(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def cell(sel: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {"runs": len(sel), "full_recovered": sum(r["full_recovered"] for r in sel),
                               "full_seconds": float(sum(r["full_seconds"] for r in sel))}
        for arm in ARMS:
            out[arm] = {"recovered": sum(r[arm]["recovered"] for r in sel),
                        "stopped_early": sum(r[arm]["stopped_early"] for r in sel),
                        "wrong_early_stops": sum(r[arm]["stopped_early"] and not r[arm]["recovered"] for r in sel),
                        "median_share_of_candidates_fitted": float(np.median([r[arm]["evaluated"] / r[arm]["candidates"] for r in sel])) if sel else None,
                        "seconds": float(sum(r[arm]["seconds"] for r in sel))}
        return out

    out = {"all": cell(rows)}
    for which in ("invented", "odebench"):
        out[which] = cell([r for r in rows if r["set"] == which])
    laws = {r["name"]: r for r in rows}.values()  # one row per law for the rank comparison
    a = np.array([r["rank_law_prior"] for r in laws])
    b = np.array([r["rank_earlier_prior"] for r in laws])
    out["ranks"] = {"laws": len(a), "median_rank_law_prior": float(np.median(a)) if len(a) else None,
                    "median_rank_earlier_prior": float(np.median(b)) if len(b) else None, "sign_test": sign_test(a, b)}
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
    trees, domains = training()
    fresh, counts = fresh_laws(trees)
    language, per_law = part_language(trees, domains, fresh)
    search, search_rows = part_search(trees, domains, fresh)
    results = {"preregistration": PREREGISTRATION_ID, "frozen": frozen_digests(), "fresh_laws": counts,
               "odebench_sha256": law_corpora.ODEBENCH_SHA256, "part_1_language": language, "part_2_search": search}
    (args.out / "results.json").write_text(json.dumps(results, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    (args.out / "runs.json").write_text(json.dumps({"part_1_language": per_law, "part_2_search": search_rows},
                                                   indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
