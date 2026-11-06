#!/usr/bin/env python3
"""Harvest integer-relation questions from recorded discovery runs and answer them.

The comparison in `scripts/compare_integer_relation_methods.py` uses hand-built
cases. This script asks the same two searchers the questions the repository's own
recorded runs imply: the numeric groups those runs actually emitted, with their
provenance, so the coefficient sizes that occur are observed rather than assumed.

Sources, each read from a committed artifact:

* `benchmarks/known_truth_results.json` — per-cell formula emissions from the
  known-truth worlds. Constants are the multiplicative coefficients and the
  additive constant of the emitted formula.
* `benchmarks/feynman_noise_rediscovery_state.json` — per-trial formula emissions
  from the Feynman rediscovery runs.
* `benchmarks/lorenz_public_benchmark_results.json` — per-row fitted coefficient
  vectors from the public Lorenz benchmark.
* across-replicate groups: the leading coefficient of the same (world, engine)
  recovered under different noise levels. An integer relation there means the
  replicates agreed exactly; nothing more is claimed about it.

A group is only asked when it holds at least two distinct finite values. Every row
carries the file it came from and the JSON path into it, because a relation with no
provenance is not evidence about anything.

The LLL ladder is deliberately bounded here (`start_digits=17`, `max_digits=27`) so
the harvest finishes: past 27 digits the reduction starts returning exact relations
among the 52-bit rationals, which say nothing about the physics. Rows that find
nothing at that bound say `not_found_at_bounded_precision`, not "no relation".
"""

from __future__ import annotations

import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "benchmarks" / "integer_relations_from_discoveries.json"

KNOWN_TRUTH = ROOT / "benchmarks" / "known_truth_results.json"
FEYNMAN = ROOT / "benchmarks" / "feynman_noise_rediscovery_state.json"
LORENZ = ROOT / "benchmarks" / "lorenz_public_benchmark_results.json"

BNB_MAX_COEFF = 12
EPS = 1e-9
LLL_START_DIGITS = 17
LLL_MAX_DIGITS = 27
LLL_MAX_ITERATIONS = 5_000
MAX_GROUPS = 400

# A multiplicative coefficient: a number followed by `*`, not the exponent after `^`
# and not part of an identifier. A trailing additive constant follows a `+`.
_TERM_COEFF = re.compile(r"(?<![\w^.])(-?\d+(?:\.\d+)?)\s*\*")
_ADDITIVE = re.compile(r"\+\s*(-?\d+(?:\.\d+)?)\s*$")


def _formula_constants(formula: str) -> list[float]:
    values: list[float] = []
    for raw in _TERM_COEFF.findall(formula):
        values.append(float(raw))
    match = _ADDITIVE.search(formula.strip())
    if match:
        values.append(float(match.group(1)))
    return [v for v in values if math.isfinite(v)]


def _group(source, pointer, values, labels, evidence, klass) -> dict:
    return {
        "source": source,
        "pointer": pointer,
        "values": values,
        "labels": labels,
        "evidence": evidence,
        "class": klass,
    }


def harvest() -> list[dict]:
    groups: list[dict] = []

    known = json.loads(KNOWN_TRUTH.read_text(encoding="utf-8"))
    for i, cell in enumerate(known.get("cells", [])):
        formula = cell.get("formula") or ""
        values = _formula_constants(formula)
        if len({round(v, 12) for v in values}) < 2:
            continue
        groups.append(
            _group(
                "benchmarks/known_truth_results.json",
                f"cells[{i}].formula",
                values,
                [f"c{k}" for k in range(len(values))],
                {"world": cell.get("world"), "method": cell.get("method"), "formula": formula},
                "within_emission",
            )
        )

    feynman = json.loads(FEYNMAN.read_text(encoding="utf-8"))
    trials = feynman.get("trials", [])
    for i, trial in enumerate(trials):
        detail = trial.get("detail") or {}
        formula = detail.get("top_formula") or ""
        values = _formula_constants(formula)
        if len({round(v, 12) for v in values}) < 2:
            continue
        groups.append(
            _group(
                "benchmarks/feynman_noise_rediscovery_state.json",
                f"trials[{i}].detail.top_formula",
                values,
                [f"k{k}" for k in range(len(values))],
                {
                    "world": trial.get("world"),
                    "engine": trial.get("engine"),
                    "sigma_rel": trial.get("sigma_rel"),
                    "formula": formula,
                },
                "within_emission",
            )
        )

    # Across replicates: the leading coefficient of the same (world, engine) at
    # different noise levels. A relation here means the replicates agreed exactly.
    replicates: dict[tuple, list[float]] = {}
    for trial in trials:
        detail = trial.get("detail") or {}
        formula = detail.get("top_formula") or ""
        values = _formula_constants(formula)
        if not values:
            continue
        key = (trial.get("world"), trial.get("engine"))
        replicates.setdefault(key, []).append(values[0])
    for (world, engine), values in sorted(replicates.items(), key=lambda kv: str(kv[0])):
        if len({round(v, 12) for v in values}) < 2:
            continue
        groups.append(
            _group(
                "benchmarks/feynman_noise_rediscovery_state.json",
                f"trials[world={world!r}, engine={engine!r}].detail.top_formula[0]",
                values,
                [f"rep{k}" for k in range(len(values))],
                {"world": world, "engine": engine, "replicates": len(values)},
                "across_replicates",
            )
        )

    lorenz = json.loads(LORENZ.read_text(encoding="utf-8"))
    for i, row in enumerate(lorenz.get("rows", [])):
        values = [float(v) for v in row.get("coefficients", []) if math.isfinite(float(v))]
        if len(values) < 2:
            continue
        groups.append(
            _group(
                "benchmarks/lorenz_public_benchmark_results.json",
                f"rows[{i}].coefficients",
                values,
                [f"c{k}" for k in range(len(values))],
                {"noise_frac": row.get("noise_frac"), "path": row.get("path")},
                "within_emission",
            )
        )

    return groups


def classify_relation(values, chosen) -> str:
    """Name what kind of thing was found, not just that something was.

    Four outcomes are separable and only the first two are structure in the emitted
    numbers:

    * `exact_small` / `exact_large` — the residual is exactly zero on the inputs as
      given, so the emitted constants really do satisfy an integer relation. The
      size split matters: a relation with coefficients in the hundreds inside 17
      digits is a real constraint, while one that only appears after the ladder
      leaves the input precision is a relation among the rounded rationals.
    * `approximate_small` — a near-relation at input precision with small
      coefficients; admissible, but a tolerance rather than an identity.
    * `rational_approximation` — any two floats admit a near-rational ratio with
      moderate coefficients, so a two-input approximate hit says nothing about
      structure. Counted separately for that reason.
    * `beyond_input_precision` — found only after the ladder passed the 17 digits an
      f64 carries: an exact relation among the 52-bit rationals.
    * `trivial_duplicate` — two equal emitted values with coefficients ±1.
    """
    coefficients = chosen["coefficients"]
    if _identical_pair(values, coefficients):
        return "trivial_duplicate"
    working = chosen.get("working_digits") or 0
    size = max(abs(c) for c in coefficients)
    if working > LLL_START_DIGITS:
        return "beyond_input_precision"
    if chosen["residual"] == 0.0:
        return "exact_small" if size <= 10 else "exact_large"
    if len(values) == 2:
        return "rational_approximation"
    return "approximate_small" if size <= 10 else "approximate_large"


def _identical_pair(values, coefficients) -> bool:
    """A relation whose whole support is two equal values with coefficients ±1."""
    support = [i for i, c in enumerate(coefficients) if c != 0]
    if len(support) != 2:
        return False
    a, b = support
    if sorted((coefficients[a], coefficients[b])) != [-1, 1]:
        return False
    return round(values[a], 12) == round(values[b], 12)


def _query(kernel, values, labels):
    """Ask both searchers one group, and time each so the cost is attributable."""
    started = time.perf_counter()
    bnb = json.loads(
        kernel.rime_number_theory_integer_relation_json(values, BNB_MAX_COEFF, EPS, labels)
    )
    bnb_ms = (time.perf_counter() - started) * 1000.0
    started = time.perf_counter()
    lll = json.loads(
        kernel.rime_number_theory_lll_relation_json(
            x=values,
            max_coeff=0,
            eps=EPS,
            labels=labels,
            start_digits=LLL_START_DIGITS,
            max_digits=LLL_MAX_DIGITS,
            max_iterations=LLL_MAX_ITERATIONS,
        )
    )
    lll_ms = (time.perf_counter() - started) * 1000.0
    return bnb, lll, bnb_ms, lll_ms


def profile(groups, kernel, top: int = 15) -> None:
    """Print where the run's seconds go, group by group.

    The harvest is 235 questions and takes minutes; naming the groups that dominate
    is the only way to know whether a future change is aimed at the right thing.
    """
    timings = []
    for group in groups:
        _bnb, _lll, bnb_ms, lll_ms = _query(kernel, group["values"], group["labels"])
        timings.append(
            {
                "pointer": group["pointer"],
                "source": group["source"],
                "n": len(group["values"]),
                "branch_and_bound_ms": round(bnb_ms, 3),
                "lll_pslq_ms": round(lll_ms, 3),
                "total_ms": round(bnb_ms + lll_ms, 3),
            }
        )
    total = sum(t["total_ms"] for t in timings)
    by_n = Counter(t["n"] for t in timings)
    time_by_n: Counter = Counter()
    for t in timings:
        time_by_n[t["n"]] += t["total_ms"]
    print(
        json.dumps(
            {
                "groups": len(timings),
                "total_ms": round(total, 1),
                "groups_by_inputs": dict(sorted(by_n.items())),
                "ms_by_inputs": {n: round(ms, 1) for n, ms in sorted(time_by_n.items())},
                "slowest": sorted(timings, key=lambda t: t["total_ms"], reverse=True)[:top],
                "time_by_source": {
                    source: round(sum(t["total_ms"] for t in timings if t["source"] == source), 1)
                    for source in sorted({t["source"] for t in timings})
                },
            },
            indent=2,
        )
    )


def main() -> int:
    try:
        import apedp_kernel as kernel
    except ImportError:
        print("apedp_kernel is not built; run `make build-native` first")
        return 2
    if not hasattr(kernel, "rime_number_theory_lll_relation_json"):
        print("this build predates the LLL binding; rebuild the extension")
        return 2

    groups = harvest()
    if "--profile" in sys.argv:
        profile(groups[:MAX_GROUPS], kernel)
        return 0
    skipped = max(0, len(groups) - MAX_GROUPS)
    groups = groups[:MAX_GROUPS]

    rows = []
    for group in groups:
        values = group["values"]
        labels = group["labels"]
        bnb, lll, _bnb_ms, _lll_ms = _query(kernel, values, labels)
        chosen = lll if lll is not None else bnb
        classification = None if chosen is None else classify_relation(values, chosen)
        rows.append(
            {
                "source": group["source"],
                "pointer": group["pointer"],
                "class": group["class"],
                "evidence": group["evidence"],
                "n": len(values),
                "branch_and_bound": {
                    "found": bnb is not None,
                    "coefficients": None if bnb is None else list(bnb["coefficients"]),
                    "out_of_scope": len(values) > 8,
                },
                "lll_pslq": {
                    "found": lll is not None,
                    "coefficients": None if lll is None else list(lll["coefficients"]),
                    "max_coefficient": None if lll is None else lll["max_coefficient"],
                    "absolute_residual": None if lll is None else lll["residual"],
                    "working_digits": None if lll is None else lll["working_digits"],
                },
                "relation_class": classification,
            }
        )

    found = [r for r in rows if r["relation_class"] is not None]
    sizes = [max(abs(c) for c in r["lll_pslq"]["coefficients"]) for r in found]
    buckets = Counter()
    for size in sizes:
        if size <= 2:
            buckets["1-2"] += 1
        elif size <= 10:
            buckets["3-10"] += 1
        elif size <= 100:
            buckets["11-100"] += 1
        else:
            buckets["over-100"] += 1

    summary = {
        "groups": len(rows),
        "groups_skipped_over_cap": skipped,
        "by_class": dict(Counter(r["class"] for r in rows)),
        "branch_and_bound_found": sum(1 for r in rows if r["branch_and_bound"]["found"]),
        "branch_and_bound_out_of_scope": sum(
            1 for r in rows if r["branch_and_bound"]["out_of_scope"]
        ),
        "lll_pslq_found": sum(1 for r in rows if r["lll_pslq"]["found"]),
        "relation_classes": dict(Counter(r["relation_class"] for r in found)),
        "coefficient_size_buckets": dict(buckets),
        "class_size_crosstab": {
            f"{cls}@{bucket}": count
            for (cls, bucket), count in sorted(
                Counter(
                    (
                        r["relation_class"],
                        (
                            "1-2"
                            if max(abs(c) for c in r["lll_pslq"]["coefficients"]) <= 2
                            else "3-10"
                            if max(abs(c) for c in r["lll_pslq"]["coefficients"]) <= 10
                            else "11-100"
                            if max(abs(c) for c in r["lll_pslq"]["coefficients"]) <= 100
                            else "over-100"
                        ),
                    )
                    for r in found
                ).items()
            )
        },
        "largest_coefficient": max(sizes) if sizes else 0,
        "inputs_histogram": dict(sorted(Counter(r["n"] for r in rows).items())),
        "agreement": sum(
            1
            for r in rows
            if r["branch_and_bound"]["coefficients"] == r["lll_pslq"]["coefficients"]
        ),
    }

    record = {
        "generated_by": "scripts/harvest_integer_relations_from_discoveries.py",
        "sources": [
            "benchmarks/known_truth_results.json",
            "benchmarks/feynman_noise_rediscovery_state.json",
            "benchmarks/lorenz_public_benchmark_results.json",
        ],
        "settings": {
            "branch_and_bound": {"max_coeff": BNB_MAX_COEFF, "eps": EPS},
            "lll_pslq": {
                "max_coeff": 0,
                "eps": EPS,
                "start_digits": LLL_START_DIGITS,
                "max_digits": LLL_MAX_DIGITS,
                "max_iterations": LLL_MAX_ITERATIONS,
            },
            "max_groups": MAX_GROUPS,
        },
        "design": {
            "grouping": "one question per emitted formula, per fitted coefficient vector, "
            "or per set of replicates of one quantity",
            "provenance": "every row names its file and JSON pointer",
            "bounded_ladder": (
                "the reduction stops at 27 digits on purpose: past the input precision "
                "it starts returning exact relations among 52-bit rationals"
            ),
            "not_a_claim_of_structure": (
                "finding a relation in a group says those emitted numbers have one; it "
                "does not say the discovery was right about the world"
            ),
            "skipped_groups": "groups beyond max_groups are counted, never sampled silently",
        },
        "rows": rows,
        "summary": summary,
    }
    OUT.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
