"""Simulated multivariate time series with a known lagged causal graph, for measuring graph discovery.

Every variable depends on its own previous value; each ordered pair of different variables is linked
with a probability that gives about one incoming cross-link per variable, at a lag of 1 to 3. A link
acts through a function: the identity in ``linear`` worlds; in ``nonlinear`` worlds one of identity,
``tanh``, a signed square (odd) or a centred square (even, which partial correlation cannot see).
``heavy_tail`` worlds are linear with Student-t noise (3 degrees of freedom), as in market returns.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

KINDS = ("linear", "nonlinear", "heavy_tail")
FUNCTIONS = {
    "identity": lambda v: v,
    "tanh": lambda v: 1.5 * np.tanh(v),
    "signed_square": lambda v: 0.7 * v * np.abs(v),
    "even_square": lambda v: 0.7 * (v * v - 1.0),
}


def make(kind: str, n_vars: int, length: int, seed: int, tau_max: int = 3) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    names = [f"x{i}" for i in range(n_vars)]
    links: list[dict[str, Any]] = [{"source": i, "lag": 1, "target": i, "coef": float(rng.uniform(0.3, 0.6)), "fn": "identity"}
                                   for i in range(n_vars)]
    for i in range(n_vars):
        for j in range(n_vars):
            if i != j and rng.random() < 1.0 / (n_vars - 1):
                fn = "identity" if kind != "nonlinear" else str(rng.choice(list(FUNCTIONS)))
                links.append({"source": i, "lag": int(rng.integers(1, tau_max + 1)), "target": j,
                              "coef": float(rng.choice([-1.0, 1.0]) * rng.uniform(0.3, 0.5)), "fn": fn})
    burn = 200
    for _attempt in range(20):
        x = np.zeros((length + burn, n_vars))
        noise = rng.standard_t(3, size=x.shape) / np.sqrt(3.0) if kind == "heavy_tail" else rng.standard_normal(x.shape)
        for t in range(tau_max, len(x)):
            value = noise[t].copy()
            for link in links:
                value[link["target"]] += link["coef"] * FUNCTIONS[link["fn"]](x[t - link["lag"], link["source"]])
            x[t] = value
        if np.all(np.isfinite(x)) and float(np.abs(x).max()) < 50.0:
            break
        for link in links:  # an unstable draw: weaken every link and simulate again
            link["coef"] *= 0.8
    else:
        raise FloatingPointError("no stable process found")
    frame = pd.DataFrame(x[burn:], columns=names)
    truth = {(names[link["source"]], link["lag"], names[link["target"]]) for link in links}
    kinds = {(names[link["source"]], link["lag"], names[link["target"]]): link["fn"] for link in links}
    return {"frame": frame, "truth": truth, "function_of": kinds}
