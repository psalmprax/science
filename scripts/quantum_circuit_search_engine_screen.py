#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Screen the direct-polynomial invsqrt configuration space across search engines.

This is a focused algorithmic comparison, not a full benchmark suite. It uses a
single shared objective — the verifier binding metric on the full invsqrt domain
— and runs the same search problem through several optimizer families so we can
compare whether a different search strategy finds a better direct-polynomial
configuration than the current baseline sweep.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import asdict
from typing import Any

import numpy as np

from apedp_dyn.quantum_compilation.contracts import load_contract
from apedp_dyn.quantum_compilation.invsqrt_direct_poly import evaluate_direct_poly_model, PolyPrecision
from apedp_dyn.quantum_compilation.invsqrt import subset_xs


def make_candidates(
    index_bits,          # int or iterable
    coef_bits,           # int or iterable
    delta_bits,          # int or iterable
):
    """Build the cartesian product of PolyPrecision candidates."""
    return [
        PolyPrecision(index_bits=k, coef_bits=wc, delta_bits=d)
        for k, wc, d in itertools.product(index_bits, coef_bits, delta_bits)
    ]


def evaluate_candidate(n_p: int, xs_sub: np.ndarray, prec: PolyPrecision):
    """Return (feasible, metric_dict) or (False, None) if the model rejected the config."""
    v_full, r_full, m_full = evaluate_direct_poly_model(n_p, prec)
    if not v_full:
        return False, None
    v_sub, r_sub, m_sub = evaluate_direct_poly_model(n_p, prec, xs=xs_sub)
    if not v_sub:
        return False, None
    row = dict(precision=asdict(prec))
    row.update({
        "full": m_full.get("worst_error_over_tolerance"),
        "sub": m_sub.get("worst_error_over_tolerance"),
        "worst_x_full": m_full.get("worst_x"),
        "worst_x_sub": m_sub.get("worst_x"),
    })
    return True, row


def random_search(candidates, n_p, xs_sub, trials):
    rng = np.random.default_rng(17)
    indices = np.arange(len(candidates))
    seen = set()
    results = []
    for _ in range(min(trials, len(candidates))):
        i = int(rng.choice(indices))
        if i in seen:
            continue
        seen.add(i)
        feasible, row = evaluate_candidate(n_p, xs_sub, candidates[i])
        if feasible:
            results.append(row)
    return results


def exhaustive_search(candidates, n_p, xs_sub):
    results = []
    for prec in candidates:
        feasible, row = evaluate_candidate(n_p, xs_sub, prec)
        if feasible:
            results.append(row)
    return results


def ga_search(candidates, n_p, xs_sub, population, generations,
              mutation_rate, tournament_size):
    """Very small scratch GA over the discrete precision grid.

    State lives in the index space of `candidates` so mutation/crossover stay
    on the same discrete set. This is intentionally simple and is used only as
    one more comparison point inside the single invsqrt hardware task.
    """
    indices = list(range(len(candidates)))
    pool = np.random.default_rng(20261104)
    pop = pool.choice(indices, size=population, replace=True).tolist()
    elite = []
    best_rows = []
    for gen in range(generations):
        fitness = []
        rows_for_gen = []
        for idx in pop:
            feasible, row = evaluate_candidate(n_p, xs_sub, candidates[idx])
            if feasible:
                score = -float(row["full"]) if row["full"] is not None else -np.inf
                fitness.append(score)
                rows_for_gen.append((idx, row))
            else:
                fitness.append(-1e18)
                rows_for_gen.append((idx, None))
        pop_sorted = [idx for idx, row in sorted(rows_for_gen, key=lambda t: t[1]["full"] if t[1] else 1e18)]
        if rows_for_gen:
            feasible_sorted = [r for r in rows_for_gen if r[1]]
            feasible_sorted.sort(key=lambda r: r[1]["full"])
            if feasible_sorted and (not elite or feasible_sorted[0][1]["full"] < elite[0][1]["full"]):
                elite = feasible_sorted[0:1]
            best_rows.extend(r for r in feasible_sorted)

        next_pop = [e[0] for e in elite]
        while len(next_pop) < population:
            a = pool.choice(pop_sorted)
            if pool.random() < mutation_rate:
                a = pool.choice(indices)
            b = pool.choice(pop_sorted)
            child = a if pool.random() < 0.5 else b
            next_pop.append(int(child))
        pop = next_pop[:population]
        if gen % 100 == 0:
            pass
    return best_rows


def main():
    contract = load_contract("invsqrt")
    n_p = int(contract.constants["N_P"])
    xs_sub = subset_xs(4096, prefix=64, stride=127)

    index_bits = [8, 9, 10, 11, 12]
    coef_bits = [28, 29, 30, 31, 32]
    delta_bits = [12, 14, 16, 18, 20, 22, 24]

    candidates = make_candidates(index_bits, coef_bits, delta_bits)

    engines = [
        ("exhaustive", lambda c: exhaustive_search(c, n_p, xs_sub)),
        ("random", lambda c: random_search(c, n_p, xs_sub, trials=min(2000, len(c)*3))),
        ("ga", lambda c: ga_search(c, n_p, xs_sub, population=64, generations=120,
                                    mutation_rate=0.18, tournament_size=4)),
    ]

    start = time.time()
    fused = {}
    for name, fn in engines:
        t0 = time.time()
        rows = fn(candidates)
        t1 = time.time()
        if not rows:
            fused[name] = {"n_valid": 0, "elapsed_s": t1 - t0, "best": None}
            continue
        best = min(rows, key=lambda r: r["full"])
        fused[name] = {
            "n_valid": len(rows),
            "elapsed_s": t1 - t0,
            "best": {
                "precision": best["precision"],
                "full": best["full"],
                "sub": best["sub"],
                "worst_x_full": best["worst_x_full"],
                "worst_x_sub": best["worst_x_sub"],
            },
        }
    fused["wall_time_s"] = time.time() - start
    print(json.dumps(fused, indent=2, default=str))


if __name__ == "__main__":
    main()
