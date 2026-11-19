"""The language-model proposer with a real model, through the main search path.

    python scripts/run_llm_proposer_benchmark.py models                 # what the key can call
    python scripts/run_llm_proposer_benchmark.py smoke                  # each preferred model on one toy law
    python scripts/run_llm_proposer_benchmark.py run --shard 0/2 --out llm-out
    python scripts/run_llm_proposer_benchmark.py collect --out llm-out

The model is given numbers only: variables are called ``x0, x1, ...`` and no description of the
problem is passed, so it cannot answer by recalling a named law. Its proposals go where PySR's go in
``apedp_dyn.discovery_l2.main_search``: after the exhaustive first pass, before contrast confirmation
and the adequacy check. It needs ``NVIDIA_API_KEY`` in the environment and network access to NVIDIA's
API, so it runs on GitHub Actions (``.github/workflows/llm-proposer.yml``) with the key as a secret.

``run`` and ``collect`` are for a preregistered run: they refuse to work until a model has been fixed
in ``MODEL`` and the files are frozen by a digest. ``models`` and ``smoke`` are for finding out which
model answers and whether the plumbing works.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_srbench_baseline as base  # noqa: E402
import run_srbench_main_search as path_run  # noqa: E402

from apedp_dyn.discovery_l2 import main_search  # noqa: E402
from apedp_dyn.grammar.proposer_interface import CandidateProposal, ProposerBase  # noqa: E402
from apedp_dyn.proposers import llm_clients  # noqa: E402
from apedp_dyn.proposers.llm_proposer import LlmEquationProposer  # noqa: E402

PREREGISTRATION_ID = "PREREG-LLM-PROPOSER-2026-11"
DIGEST = ROOT / "preregistration" / "DIGEST-LLM-PROPOSER.json"
FROZEN = ("preregistration/PREREG-LLM-PROPOSER-2026-11.md", "scripts/run_llm_proposer_benchmark.py",
          "scripts/run_srbench_main_search.py", "scripts/run_srbench_baseline.py", "benchmarks/srbench/manifest.json",
          "apedp_dyn/proposers/llm_proposer.py", "apedp_dyn/proposers/llm_clients.py",
          "apedp_dyn/discovery_l2/main_search.py")
#: fixed before the preregistered run, from what ``models`` and ``smoke`` show; ``None`` until then
MODEL: str | None = None
NOISE_LEVELS = (0.0, 0.01)
ROUNDS, PER_ROUND, MAX_PROPOSALS = 3, 5, 8
INTERVAL_S = 3.2      # between requests of one shard: two shards stay under about 40 requests a minute
TIME_LIMIT_S = 900.0
ENGINE = "llm_path"


class Recording(ProposerBase):
    """Passes calls through to a proposer and keeps what it returned, for the run's record."""

    def __init__(self, inner: ProposerBase) -> None:
        self.inner = inner
        self.proposals: Sequence[CandidateProposal] = ()

    @property
    def proposer_id(self) -> str:
        return self.inner.proposer_id

    def fit_and_propose(self, *args: Any, **kwargs: Any) -> Sequence[CandidateProposal]:
        self.proposals = self.inner.fit_and_propose(*args, **kwargs)
        return self.proposals


def llm_path(X: dict[str, np.ndarray], y: np.ndarray, pool_X: np.ndarray, measure: Any, model: str,
             interval: float = INTERVAL_S, always_search: bool = False, rounds: int = ROUNDS,
             retries: int = 4) -> tuple[Any, dict[str, Any]]:
    client = llm_clients.nvidia_client(model, min_interval=interval, retries=retries)
    proposer = Recording(LlmEquationProposer(client, target_col="__target__", rounds=rounds, per_round=PER_ROUND,
                                             max_proposals=MAX_PROPOSALS))
    result = main_search.discover(X, y, pool_X, measure, proposer=proposer, noise="absolute", seed=base.SEED,
                                  always_search=always_search)
    seen = proposer.inner.last_run  # type: ignore[attr-defined]
    record = {"model": model, "calls": client.calls, "proposals": len(proposer.proposals),  # type: ignore[attr-defined]
              "recited_before_data": sum(bool(p.metadata.get("recited_before_data")) for p in proposer.proposals),
              "skeletons_tried": seen.get("skeletons_tried", 0), "replies_rejected": seen.get("replies_rejected", 0),
              "first_reply": seen.get("first_reply", ""), "formulas": [p.formula for p in proposer.proposals]}
    return result, record


def run_llm_path(split: dict[str, Any], entry: dict[str, Any], budget: float = 0.0) -> dict[str, Any]:
    """The same development rows and pool as PREREG-MAIN-SEARCH, with the language model in PySR's place."""
    if MODEL is None:
        raise SystemExit("no model has been fixed for the preregistered run")
    names, X, y = split["names"], split["X_train"], split["y_train"]
    key = hashlib.sha256(f"{path_run.PREREGISTRATION_ID}|{entry['name']}|pool".encode()).digest()[:8]
    order = np.random.default_rng(int.from_bytes(key, "big")).permutation(len(y))
    cut = round(path_run.DEV_SHARE * len(y))
    dev, pool = order[:cut], order[cut:]
    pool_y = y[pool]
    result, record = llm_path({n: X[dev, i] for i, n in enumerate(names)}, y[dev], X[pool], lambda rows: pool_y[rows], MODEL)
    law = sp.sympify(result.law, locals={n: sp.Symbol(n) for n in names}) if result.law else None
    extra = {"action": result.action, "found_by": result.found_by, "rejected_law": result.rejected_law,
             "adequacy": result.adequacy, "rows_read": result.rows_read, "first_pass": result.first_pass,
             "search": result.search, "candidates": result.candidates, "llm": record}
    return {"model": law, "pred": path_run._predict(law, names, split["X_test"]) if law is not None else None,
            "formula": result.law, "extra": extra}


path_run.RUNNERS[ENGINE] = run_llm_path  # run_one isolates, times and scores any registered engine


def all_jobs() -> list[dict[str, Any]]:
    return [{"problem": p["name"], "noise": noise, "seed": base.SEED, "engine": ENGINE}
            for p in base.load_manifest() for noise in NOISE_LEVELS]


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for family in ("feynman", "strogatz", "all"):
        for noise in (*NOISE_LEVELS, "all"):
            sel = [r for r in rows if family in (r["family"], "all") and noise in (r["noise"], "all")]
            ok = [r for r in sel if r["status"] == "ok"]
            reported = [r for r in ok if r.get("model") not in (None, "None")]
            asked = [r for r in ok if r.get("search", {}).get("ran")]
            out[f"{family}|{noise}"] = {
                "n": len(sel), "errors": sum(r["status"] == "error" for r in sel), "timeouts": sum(r["status"] == "timeout" for r in sel),
                "symbolic": sum(bool(r["symbolic"]) for r in ok), "accuracy_solution": sum(bool(r["accuracy_solution"]) for r in ok),
                "reported": len(reported), "reported_symbolic": sum(bool(r["symbolic"]) for r in reported),
                "reported_not_accurate": sum(not r["accuracy_solution"] for r in reported),
                "found_by": {s: sum(r.get("found_by") == s for r in reported) for s in ("first_pass", "search")},
                "model_asked": len(asked), "model_calls": sum(r["llm"]["calls"] for r in asked),
                "model_proposals": sum(r["llm"]["proposals"] for r in asked),
                "recited_before_data": sum(r["llm"]["recited_before_data"] for r in asked),
                "median_seconds": float(np.median([r["seconds"] for r in sel])) if sel else None,
            }
    return out


def frozen_digests() -> dict[str, str]:
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in FROZEN}


def check_frozen() -> None:
    if MODEL is None or not DIGEST.exists() or json.loads(DIGEST.read_text(encoding="utf-8"))["sha256"] != frozen_digests():
        raise SystemExit(f"{PREREGISTRATION_ID} is not frozen: fix MODEL and write the digest first")


def toy(text: str, seed: int) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, Any]:
    rng = np.random.default_rng(seed)
    fn = sp.lambdify([sp.Symbol("x0"), sp.Symbol("x1")], sp.sympify(text), "numpy")
    X, pool = rng.uniform(1.0, 5.0, (300, 2)), rng.uniform(1.0, 5.0, (600, 2))
    scale = 0.01 * float(np.sqrt(np.mean(fn(*X.T) ** 2)))
    y = fn(*X.T) + scale * rng.standard_normal(300)
    noise = np.random.default_rng(seed + 1)
    return {"x0": X[:, 0], "x1": X[:, 1]}, y, pool, lambda rows: fn(*pool[rows].T) + scale * noise.standard_normal(len(rows))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("command", choices=("models", "smoke", "run", "collect"))
    parser.add_argument("--out", type=Path, default=Path("llm-out"))
    parser.add_argument("--shard", default="0/1")
    args = parser.parse_args(argv)
    if args.command in ("models", "smoke"):
        listed = llm_clients.available_models()
        print(f"{len(listed)} models listed; preferred and listed: {[m for m in llm_clients.PREFERRED_MODELS if m in listed]}", flush=True)
        print("LISTED " + " | ".join(listed), flush=True)
        if args.command == "models":
            return 0
        # every preferred model the catalogue lists, on one toy law: which answer, how fast, and what they propose
        truth = "x0*exp(-0.37*x1)"
        X, y, pool, measure = toy(truth, seed=7)
        for model in [m for m in llm_clients.PREFERRED_MODELS if m in listed]:
            started = time.time()
            try:
                result, record = llm_path(X, y, pool, measure, model, interval=1.6, always_search=True, rounds=2, retries=1)
                print("SURVEY " + json.dumps({"model": model, "seconds": round(time.time() - started, 1), "action": result.action,
                                              "law": result.law, "found_by": result.found_by, **record}, default=str), flush=True)
            except Exception as exc:  # noqa: BLE001 - a model that fails is a survey result, not a crash
                print("SURVEY " + json.dumps({"model": model, "seconds": round(time.time() - started, 1),
                                              "failed": f"{type(exc).__name__}: {exc}"[:200]}), flush=True)
        return 0
    check_frozen()
    args.out.mkdir(parents=True, exist_ok=True)
    entries = {p["name"]: p for p in base.load_manifest()}
    if args.command == "run":
        k, n = (int(v) for v in args.shard.split("/"))
        with (args.out / f"runs-shard-{k:03d}-of-{n:03d}.jsonl").open("w", encoding="utf-8") as handle:
            for job in base.shard(all_jobs(), k, n):
                handle.write(json.dumps(path_run.run_one(job, entries, 0.0, TIME_LIMIT_S), sort_keys=True, default=str) + "\n")
                handle.flush()
        return 0
    rows = [json.loads(line) for file in sorted(args.out.glob("runs-shard-*.jsonl")) for line in file.read_text().splitlines()]
    if len(rows) != len(all_jobs()):
        raise SystemExit(f"expected {len(all_jobs())} runs, found {len(rows)}")
    (args.out / "runs.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    (args.out / "results.json").write_text(json.dumps(
        {"preregistration": PREREGISTRATION_ID, "model": MODEL, "frozen": frozen_digests(), "summary": summarise(rows)},
        indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"collected {len(rows)} runs into {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
