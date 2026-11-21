#!/usr/bin/env python3
"""A smoke run of the agentic variation operator with a real model. Invented laws only.

Needs ``NVIDIA_API_KEY`` in the environment; PySR for the second arm. For each law it prints what the
agent did (steps, fits, what it submitted, why its session ended), the action lines of its replies,
and what the refined path concluded: first with the agent alone, then with the agent and a search
taking turns. Nothing is written to the repository, and nothing here is a measurement: it shows
whether the model can drive the loop.
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
from apedp_dyn.proposers import agentic_variation as av  # noqa: E402
from apedp_dyn.proposers import llm_clients  # noqa: E402

MODEL = os.environ.get("LLM_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")
BUDGET_S = float(os.environ.get("SMOKE_BUDGET_SECONDS", "60"))
SESSION_S = float(os.environ.get("SMOKE_SESSION_SECONDS", "240"))
ARMS = tuple(a for a in os.environ.get("SMOKE_ARMS", "agent,agent_and_search").split(",") if a)
#: Invented for this smoke; none is a textbook law. The last is a rate with a saturating loss term.
LAWS = ("x0*exp(-0.37*x1)", "x0*x1/(x2 + 1.5)", "sqrt(x0**2 + 3*x1)", "0.8*x0*(1 - x0/3.0) - 0.4*x0*x1/(1.0 + x0)")


class Logged:
    """The client, with every reply's action lines and the time it took kept for the report."""

    def __init__(self, client: Any) -> None:
        self._client = client
        self.calls: list[dict[str, Any]] = []

    def __call__(self, prompt: str) -> str:
        started = time.time()
        try:
            reply: str = self._client(prompt)
        except Exception as exc:
            self.calls.append({"seconds": round(time.time() - started, 1), "failed": f"{type(exc).__name__}: {exc}"[:160]})
            raise
        self.calls.append({"seconds": round(time.time() - started, 1), "prompt_chars": len(prompt),
                           "actions": [f"{kind} {text}"[:100] for kind, text in av.read_actions(reply)],
                           "reply_tail": reply[-160:] if not av.read_actions(reply) else ""})
        return reply


def search(budget: float) -> Any:
    from apedp_dyn.proposers.pysr_proposer import PySRProposer

    return PySRProposer(budget_seconds=budget)


def main() -> int:
    failed = 0
    for seed, text in enumerate(LAWS):
        truth = sp.sympify(text)
        names = sorted(str(s) for s in truth.free_symbols)
        fn = sp.lambdify([sp.Symbol(n) for n in names], truth, "numpy")
        rng = np.random.default_rng(300 + seed)
        X, pool = rng.uniform(1.0, 5.0, (300, len(names))), rng.uniform(1.0, 5.0, (600, len(names)))
        scale = 0.01 * float(np.sqrt(np.mean(fn(*X.T) ** 2)))
        y = fn(*X.T) + scale * rng.standard_normal(300)
        noise = np.random.default_rng(400 + seed)

        def measure(rows: np.ndarray, fn: Any = fn, pool: np.ndarray = pool, scale: float = scale,
                    noise: np.random.Generator = noise) -> np.ndarray:
            out: np.ndarray = fn(*pool[rows].T) + scale * noise.standard_normal(len(rows))
            return out

        for arm in ARMS:
            client = Logged(llm_clients.nvidia_client(MODEL, thinking=True, reasoning_budget=2000, retries=2))
            proposer = av.AgenticVariationProposer(
                client, search if arm == "agent_and_search" else None, budget_seconds=BUDGET_S,
                rounds=2 if arm == "agent_and_search" else 1, session_seconds=SESSION_S, target_col="__target__")
            started = time.time()
            try:
                result = refined_search.discover_refined({n: X[:, i] for i, n in enumerate(names)}, y, pool, measure,
                                                         proposer=proposer, noise="absolute")
                recovered = None
                if result.law:
                    found = sp.sympify(result.law)
                    probe = np.random.default_rng(500).uniform(1.0, 5.0, (200, len(names)))
                    with np.errstate(all="ignore"):
                        gap = sp.lambdify([sp.Symbol(n) for n in names], found, "numpy")(*probe.T) - fn(*probe.T)
                    recovered = bool(np.sqrt(np.mean(gap ** 2)) < 0.02 * np.sqrt(np.mean(fn(*probe.T) ** 2)))
                print("AGENTIC " + json.dumps({
                    "arm": arm, "truth": text, "seconds": round(time.time() - started, 1), "action": result.action,
                    "law": result.law, "tentative_law": result.tentative_law, "close_to_truth": recovered,
                    "run": {k: v for k, v in proposer.last_run.items() if k not in ("last_reply", "last_answer")},
                    "calls": client.calls, "failures": result.proposer_failures}, default=str), flush=True)
                sessions = proposer.last_run.get("sessions", [])
                failed += not any(s["fits"] for s in sessions)       # the model never drove a single fit
            except Exception as exc:  # noqa: BLE001 - the smoke reports a failure and goes on to the next law
                print("AGENTIC " + json.dumps({"arm": arm, "truth": text, "failed": f"{type(exc).__name__}: {exc}"[:300],
                                               "calls": client.calls}), flush=True)
                failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
