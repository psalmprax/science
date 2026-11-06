#!/usr/bin/env python3
"""Compare the two integer-relation searchers in the native kernel on one input set.

`find_integer_relation` is a branch-and-bound over `[-max_coeff, max_coeff]^n`:
exact inside that box, exponential in `n`, and it refuses `n > 8`.
`find_integer_relation_lll` reduces `[ I | round(x * 10^digits) ]`: polynomial in
`n` for a fixed precision, no box, and a candidate is returned only after an exact
rational substitution on the `f64` inputs.

Each case declares what decides it:

* `specific` — the relation is known, and a method that answers must return it.
* `exact_relation` — the inputs satisfy the relation with exactly zero residual, so
  any shorter exact relation is also a true statement; the criterion is `found`,
  `verified`, zero residual, not a particular vector.
* `refusal` — the tolerance is below anything the inputs admit, so a search that
  answers has answered wrongly.

Each case also declares whether it is inside the branch-and-bound's scope. A box
search that refuses `n = 10` or a coefficient of 137 is doing what it documents, so
that outcome is recorded as `refused_out_of_scope`, not as an error. This is a
designed input set, not a distribution over problem space.
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Running this file puts `scripts/` on sys.path, which would import a stale
# installed `apedp_kernel` instead of the one built from this checkout.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "benchmarks" / "integer_relation_methods.json"

PRIMES_9 = [2, 3, 5, 7, 11, 13, 17, 19, 23]
PRIMES_11 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

#: Seed for the randomized block. Fixed so the record can be regenerated and the
#: case list is identical run to run; the generator's parameters are recorded beside
#: it, because a seed without them reproduces nothing.
RANDOM_SEED = 20261102
RANDOM_PER_FAMILY = 40


def _cases() -> list[dict]:
    root2 = math.sqrt(2.0)
    root3 = math.sqrt(3.0)
    root5 = math.sqrt(5.0)

    ten = [float(p) for p in PRIMES_9]
    ten = [*ten, sum(ten)]

    twelve = [float(p) for p in PRIMES_11]
    twelve = [*twelve, sum((2 * i + 1) * v for i, v in enumerate(twelve))]

    # Exact dyadic values, so 137:−89:−1 holds with zero residual. A relation of
    # that size that held only to f64 rounding would need
    # `n*log10(max|m|) + 16` digits — about 21 for n = 3 — and an f64 carries 17,
    # so it could not be certified at all. The case is about box reach, not float
    # resolution, and the inputs make that explicit. The denominators are large
    # odd mantissas so the only relation shorter than the intended one has
    # coefficients in the millions, which `max_coeff` removes from the question.
    a = 1_234_567 / 2**20
    b = 7_654_321 / 2**20
    combo = 137.0 * a - 89.0 * b

    return [
        {
            "case": "stoichiometry",
            "criterion": "specific",
            "bnb_scope": "in",
            "x": [2.016, 31.998, 18.015],
            "labels": ["H2", "O2", "H2O"],
            "expected": [2, 1, -2],
            "max_coeff": 5,
            "eps": 1e-2,
        },
        {
            "case": "harmonic_ratio",
            "criterion": "specific",
            "bnb_scope": "in",
            "x": [60.0, 90.0],
            "labels": ["w1", "w2"],
            "expected": [3, -2],
            "max_coeff": 5,
            "eps": 1e-9,
        },
        {
            "case": "planted_three_term",
            "criterion": "specific",
            "bnb_scope": "in",
            "x": [root2, root3, root2 + root3],
            "labels": ["a", "b", "a_plus_b"],
            "expected": [1, 1, -1],
            "max_coeff": 5,
            "eps": 1e-12,
        },
        {
            "case": "coefficients_outside_the_box",
            "criterion": "specific",
            "bnb_scope": "out_of_scope",
            "x": [a, b, combo],
            "labels": ["a", "b", "a_and_b"],
            "expected": [137, -89, -1],
            "max_coeff": 20,
            "eps": 1e-12,
        },
        {
            "case": "ten_inputs_exact_sum",
            "criterion": "exact_relation",
            "bnb_scope": "out_of_scope",
            "x": ten,
            "labels": [f"p{i}" for i in range(9)] + ["sum"],
            "expected": None,
            "max_coeff": 20,
            "eps": 1e-10,
        },
        {
            "case": "twelve_inputs_exact_weighted_sum",
            "criterion": "exact_relation",
            "bnb_scope": "out_of_scope",
            "x": twelve,
            "labels": [f"q{i}" for i in range(11)] + ["sum"],
            "expected": None,
            "max_coeff": 40,
            "eps": 1e-10,
        },
        {
            # The tolerance is below what any inexact relation can reach, and the
            # ladder stops at the input's own precision. Escalating past 17 digits
            # *does* find an exact relation among these three values — because an
            # f64 is a rational and any three rationals have one with coefficients
            # around 10^8. That is a fact about the rounded inputs, not about
            # sqrt(2), sqrt(3), sqrt(5), and it is recorded below as its own case.
            "case": "independent_irrationals",
            "criterion": "refusal",
            "bnb_scope": "in",
            "x": [root2, root3, root5],
            "labels": ["a", "b", "c"],
            "expected": None,
            "max_coeff": 50,
            "eps": 1e-300,
            "lll_max_digits": 17,
        },
        {
            # Same three values, ladder allowed past the input precision. The
            # criterion is `exact_relation`: it must be found, verified and have a
            # zero residual, and the record names it as a relation among the f64
            # values rather than among the constants.
            "case": "rounded_values_have_an_exact_relation",
            "criterion": "exact_relation",
            "bnb_scope": "out_of_scope",
            "x": [root2, root3, root5],
            "labels": ["a", "b", "c"],
            "expected": None,
            "max_coeff": 50,
            "eps": 1e-300,
            "lll_max_digits": 40,
        },
    ]


def _random_cases() -> list[dict]:
    """Three families of seeded random questions, each with a decidable answer.

    * `random_planted_in_box` -- a dyadic relation the box can hold. The last value
      is `-sign * sum(m_i x_i)` with `|m_i| <= 4`, which is exact in binary, so a
      zero residual is a property of the inputs and not of the arithmetic.
    * `random_planted_out_of_box` -- the same construction with `|m_i|` in
      `[20, 400]`, so the answer exists but cannot fit in `max_coeff = 12`.
    * `random_independent` -- continuous draws with `eps = 1e-300` and the ladder
      stopped at the input's precision, where no relation can exist. Both searchers
      must refuse.

    This is a distribution over *these* families, not over problem space: the two
    planted families always carry a relation, and the independent family never does.
    """
    rng = random.Random(RANDOM_SEED)
    cases: list[dict] = []

    def _dyadic(values: list[float], inner: list[int], sign: int) -> tuple[list[float], list[int]]:
        total = sum(m * v for m, v in zip(inner, values, strict=True))
        return [*values, -sign * total], [*inner, sign]

    for index in range(RANDOM_PER_FAMILY):
        n = rng.choice((3, 4, 5))
        values = [rng.randrange(1 << 20, 1 << 22) / float(1 << 20) for _ in range(n - 1)]
        inner = [rng.choice((-4, -3, -2, -1, 1, 2, 3, 4)) for _ in range(n - 1)]
        x, planted = _dyadic(values, inner, rng.choice((-1, 1)))
        cases.append(
            {
                "case": f"random_planted_in_box_{index:03d}",
                "family": "random_planted_in_box",
                "criterion": "relation_within_tolerance",
                "bnb_scope": "in",
                "x": x,
                "labels": [f"r{i}" for i in range(len(x))],
                "expected": planted,
                "max_coeff": 6,
                "eps": 1e-12,
            }
        )

    for index in range(RANDOM_PER_FAMILY):
        values = [rng.randrange(1 << 20, 1 << 22) / float(1 << 20) for _ in range(2)]
        inner = [rng.randrange(20, 401) * rng.choice((-1, 1)) for _ in range(2)]
        x, planted = _dyadic(values, inner, rng.choice((-1, 1)))
        cases.append(
            {
                "case": f"random_planted_out_of_box_{index:03d}",
                "family": "random_planted_out_of_box",
                "criterion": "relation_within_tolerance",
                "bnb_scope": "out_of_scope",
                "x": x,
                "labels": [f"s{i}" for i in range(len(x))],
                "expected": planted,
                "max_coeff": 12,
                "eps": 1e-12,
            }
        )

    for index in range(RANDOM_PER_FAMILY):
        n = rng.choice((3, 4, 5))
        x = [rng.uniform(0.5, 2.0) for _ in range(n)]
        cases.append(
            {
                "case": f"random_independent_{index:03d}",
                "family": "random_independent",
                "criterion": "refusal",
                "bnb_scope": "in",
                "x": x,
                "labels": [f"u{i}" for i in range(n)],
                "expected": None,
                "max_coeff": 6,
                "eps": 1e-300,
                "lll_max_digits": 17,
            }
        )
    return cases


def _run_branch_and_bound(kernel, case):
    started = time.perf_counter()
    payload = kernel.rime_number_theory_integer_relation_json(
        case["x"], case["max_coeff"], case["eps"], case["labels"]
    )
    elapsed = (time.perf_counter() - started) * 1000.0
    found = json.loads(payload)
    return {
        "found": found is not None,
        "coefficients": None if found is None else list(found["coefficients"]),
        "absolute_residual": None if found is None else found["residual"],
        "verified": None,
        "digits_held": None,
        "reason": None if found is not None else "searched the box, found nothing",
        "elapsed_ms": round(elapsed, 3),
    }


def _run_lll(kernel, case):
    started = time.perf_counter()
    # No coefficient bound on purpose: `max_coeff` in the box search *is* the
    # search space, and handing the same number to the reduction would recreate
    # the box it is meant to escape.
    payload = kernel.rime_number_theory_lll_relation_json(
        x=case["x"],
        max_coeff=0,
        eps=case["eps"],
        labels=case["labels"],
        start_digits=17,
        max_digits=case.get("lll_max_digits", 100),
        max_iterations=20_000,
    )
    elapsed = (time.perf_counter() - started) * 1000.0
    found = json.loads(payload)
    if found is None:
        return {
            "found": False,
            "coefficients": None,
            "absolute_residual": None,
            "verified": None,
            "digits_held": None,
            "reason": "no verified candidate at any precision on the ladder",
            "elapsed_ms": round(elapsed, 3),
        }
    return {
        "found": True,
        "coefficients": list(found["coefficients"]),
        "absolute_residual": found["residual"],
        "relative_residual": found["relative_residual"],
        "verified": bool(found["verified"]),
        "digits_held": found["digits_held"],
        "working_digits": found["working_digits"],
        "max_coefficient": found["max_coefficient"],
        "generic_coefficient_bound": found["generic_coefficient_bound"],
        "iterations": found["iterations"],
        "reason": None,
        "elapsed_ms": round(elapsed, 3),
    }


def _answer_is_right(case, result) -> bool:
    criterion = case["criterion"]
    if criterion == "specific":
        return result["coefficients"] == case["expected"]
    if criterion == "relation_within_tolerance":
        # The randomized families are refused only on a wrong answer, not on a
        # different vector: any relation within the tolerance answers the question
        # that was asked. The reduction additionally has to have certified it.
        if not result["found"] or result.get("verified") is False:
            return False
        return abs(result["absolute_residual"]) <= case["eps"]
    # exact_relation: any shorter exact relation is also a true statement here.
    return result["absolute_residual"] == 0.0


def _assess(case, result, method) -> str:
    criterion = case["criterion"]
    if criterion == "refusal":
        return "correct" if not result["found"] else "wrong"
    if result["found"]:
        return "correct" if _answer_is_right(case, result) else "wrong"
    if method == "branch_and_bound" and case["bnb_scope"] == "out_of_scope":
        return "refused_out_of_scope"
    return "wrong"


def _tally(rows, method) -> dict:
    keys = ("correct", "refused_out_of_scope", "wrong")
    counts = dict.fromkeys(keys, 0)
    for row in rows:
        counts[row[f"{method}_assessment"]] += 1
    return counts


def main() -> int:
    try:
        import apedp_kernel as kernel
    except ImportError:
        print("apedp_kernel is not built; run `make build-native` first")
        return 2
    if not hasattr(kernel, "rime_number_theory_lll_relation_json"):
        print("this build predates the LLL binding; rebuild the extension")
        return 2

    rows = []
    for case in [*_cases(), *_random_cases()]:
        bnb = _run_branch_and_bound(kernel, case)
        lll = _run_lll(kernel, case)
        rows.append(
            {
                "case": case["case"],
                "family": case.get("family", "designed"),
                "criterion": case["criterion"],
                "branch_and_bound_scope": case["bnb_scope"],
                "n": len(case["x"]),
                "branch_and_bound_max_coeff": case["max_coeff"],
                "lll_pslq_max_coeff": 0,
                "lll_pslq_max_digits": case.get("lll_max_digits", 100),
                "eps": case["eps"],
                "expected_coefficients": case["expected"],
                "branch_and_bound": bnb,
                "lll_pslq": lll,
                "agreement": bnb["coefficients"] == lll["coefficients"],
                "branch_and_bound_assessment": _assess(case, bnb, "branch_and_bound"),
                "lll_pslq_assessment": _assess(case, lll, "lll_pslq"),
            }
        )

    def _median(method):
        values = sorted(r[method]["elapsed_ms"] for r in rows)
        return values[len(values) // 2]

    bnb_tally = _tally(rows, "branch_and_bound")
    lll_tally = _tally(rows, "lll_pslq")
    summary = {
        "cases": len(rows),
        "agree": sum(1 for r in rows if r["agreement"]),
        "branch_and_bound": bnb_tally,
        "lll_pslq": lll_tally,
        "branch_and_bound_refusals": sum(1 for r in rows if not r["branch_and_bound"]["found"]),
        "lll_pslq_refusals": sum(1 for r in rows if not r["lll_pslq"]["found"]),
        "median_elapsed_ms": {
            "branch_and_bound": _median("branch_and_bound"),
            "lll_pslq": _median("lll_pslq"),
        },
        "slowest_elapsed_ms": {
            "branch_and_bound": max(r["branch_and_bound"]["elapsed_ms"] for r in rows),
            "lll_pslq": max(r["lll_pslq"]["elapsed_ms"] for r in rows),
        },
    }

    randomized_rows = [r for r in rows if r["family"] != "designed"]
    randomized = {
        "seed": RANDOM_SEED,
        "per_family": RANDOM_PER_FAMILY,
        "families": {},
    }
    for family in sorted({r["family"] for r in randomized_rows}):
        subset = [r for r in randomized_rows if r["family"] == family]
        entry = {"cases": len(subset), "branches": sorted({r["branch_and_bound_scope"] for r in subset})}
        for method in ("branch_and_bound", "lll_pslq"):
            counts = dict.fromkeys(("correct", "refused_out_of_scope", "wrong"), 0)
            for row in subset:
                counts[row[f"{method}_assessment"]] += 1
            counts["pass_rate"] = round(counts["correct"] / max(1, len(subset)), 4)
            entry[method] = counts
        entry["planted_vector_recovered_by_lll_pslq"] = sum(
            1
            for row in subset
            if row["expected_coefficients"] is not None
            and row["lll_pslq"]["coefficients"] == row["expected_coefficients"]
        )
        entry["planted_vector_recovered_by_branch_and_bound"] = sum(
            1
            for row in subset
            if row["expected_coefficients"] is not None
            and row["branch_and_bound"]["coefficients"] == row["expected_coefficients"]
        )
        randomized["families"][family] = entry
    summary["randomized"] = randomized

    record = {
        "generated_by": "scripts/compare_integer_relation_methods.py",
        "kernel": {
            "branch_and_bound": "rime_number_theory_integer_relation_json",
            "lll_pslq": "rime_number_theory_lll_relation_json",
        },
        "methods": {
            "branch_and_bound": {
                "constant": "brute_force_branch_and_bound",
                "cost": "O((2*max_coeff+1)^n), exact inside the box, refuses n > 8",
                "acceptance": "minimal-l1 vector inside the box with |m.x| <= eps",
                "scope": "coefficients inside max_coeff and at most eight inputs",
            },
            "lll_pslq": {
                "constant": "lll_integer_relation_pslq",
                "cost": "polynomial in n for fixed precision; coefficient range is not a box",
                "acceptance": (
                    "exact rational substitution, plus a size margin against the "
                    "generic 10^(digits/n) vector and a residual above the embedding "
                    "rounding floor unless the residual is exactly zero"
                ),
                "scope": "any n; coefficients bounded only by what the reduction reaches",
            },
        },
        "design": {
            "cases": (
                "eight hand-built cases with a declared criterion and a declared scope "
                "each, plus three seeded random families"
            ),
            "not_a_distribution": True,
            "not_a_distribution_why": (
                "the random families are produced by a fixed generator, not sampled from "
                "the questions real work asks: the two planted families always carry a "
                "relation and the independent family never does. Their pass rates say "
                "whether each searcher does what its contract says on that generator, "
                "and nothing about discovery-rate in general"
            ),
            "randomized": (
                "seed=20261102, 40 cases per family drawn by random.Random(seed) in "
                "family order, so re-running the script reproduces the case list; a "
                "seed without the generator parameters would reproduce nothing"
            ),
            "assessment_rule": (
                "a refusal inside a method's documented scope is `wrong`; a refusal "
                "outside it is `refused_out_of_scope`, which is not an error"
            ),
            "precision": (
                "start_digits=17 is what an f64 carries; escalation adds coefficient "
                "range, not information, and the residual guards reject relations the "
                "embedding would otherwise manufacture"
            ),
            "float_limit": (
                "a relation among f64 inputs must clear digits_of_residual + "
                "n*log10(max|m|) working digits; for n=3 and |m|~10^2 that is about "
                "21, which an f64 cannot supply. The outside-the-box case therefore "
                "uses exactly representable dyadic values, so its zero residual is a "
                "property of the inputs and not of the arithmetic"
            ),
            "timing": (
                "measured on one machine in a single process against an optimised "
                "build; the branch-and-bound cost is combinatorial in n and the box, "
                "so its small-case time says nothing about larger inputs"
            ),
            "independent_irrationals": (
                "eps=1e-300 is below any residual the f64 values admit, so both "
                "searchers must refuse: the case tests refusal, not reach"
            ),
        },
        "rows": rows,
        "summary": summary,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "record": str(OUT.relative_to(ROOT))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
