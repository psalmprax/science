#!/usr/bin/env python3
"""A smoke run of the language-model search controller with a real model and a real search. Toy laws only.

Needs ``NVIDIA_API_KEY`` in the environment and PySR. Prints, for each law, what the model asked for, what
was accepted after checking, and what the refined path concluded. Nothing is written to the repository.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from apedp_dyn.discovery_l2 import refined_search  # noqa: E402
from apedp_dyn.proposers import llm_clients  # noqa: E402
from apedp_dyn.proposers.llm_controller import ControlledSearchProposer  # noqa: E402
from apedp_dyn.proposers.pysr_proposer import PySRProposer  # noqa: E402

MODEL = os.environ.get("LLM_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")
BUDGET_S = float(os.environ.get("SMOKE_BUDGET_SECONDS", "30"))
LAWS = ("x0*exp(-0.37*x1)", "x0*x1/(x2 + 1.5)", "sqrt(x0**2 + 3*x1)")


def main() -> int:
    failed = 0
    for seed, text in enumerate(LAWS):
        truth = sp.sympify(text)
        names = sorted((str(s) for s in truth.free_symbols))
        fn = sp.lambdify([sp.Symbol(n) for n in names], truth, "numpy")
        rng = np.random.default_rng(100 + seed)
        X, pool = rng.uniform(1.0, 5.0, (300, len(names))), rng.uniform(1.0, 5.0, (600, len(names)))
        scale = 0.01 * float(np.sqrt(np.mean(fn(*X.T) ** 2)))
        y = fn(*X.T) + scale * rng.standard_normal(300)
        noise = np.random.default_rng(200 + seed)

        def measure(rows: np.ndarray, fn: Any = fn, pool: np.ndarray = pool, scale: float = scale,
                    noise: np.random.Generator = noise) -> np.ndarray:
            out: np.ndarray = fn(*pool[rows].T) + scale * noise.standard_normal(len(rows))
            return out

        client = llm_clients.nvidia_client(MODEL, thinking=True, reasoning_budget=2000, retries=2)
        controller = ControlledSearchProposer(client, lambda unary: PySRProposer(budget_seconds=BUDGET_S, unary=unary),
                                              target_col="__target__")
        started = time.time()
        try:
            result = refined_search.discover_refined({n: X[:, i] for i, n in enumerate(names)}, y, pool, measure,
                                                     proposer=controller, noise="absolute")
            print("CONTROLLER " + json.dumps({"truth": text, "seconds": round(time.time() - started, 1), "action": result.action,
                                              "law": result.law, "tentative_law": result.tentative_law,
                                              "controller": controller.last_run, "calls": client.calls,  # type: ignore[attr-defined]
                                              "failures": result.proposer_failures}, default=str), flush=True)
            failed += bool(controller.last_run.get("model_failure"))
        except Exception as exc:  # noqa: BLE001 - the smoke reports a failure and goes on to the next law
            print("CONTROLLER " + json.dumps({"truth": text, "failed": f"{type(exc).__name__}: {exc}"[:300]}), flush=True)
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
