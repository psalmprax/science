#!/usr/bin/env python3
"""Development: does an order learned from the system's own problems bring the law forward on its lists?

Status: audit-only

**Development, not a result.** Reads the candidate lists kept by ``scripts/measure_order_score.py``
(each candidate with what the judge made of it and a few features) and scores other orders of the
same candidates with ``apedp_dyn/recursion/order_score.py``. Nothing is searched or fitted again.

For each arm (a memory taught so many laws):

* the list as the memory hands it over, best fit first;
* the rule that needs no learning: the shortest candidate within a tolerance of the best fit on
  rows the fit never saw (``learned_order.simplest_within``);
* the learned order (``learned_order.LearnedOrder``), every problem scored by a model trained on
  other problems only (five folds).

And for the reference arm: a learning curve (how the learned order does on problems held apart as
it is trained on 5 to 200 others), and whether an order learned on one arm's lists carries to
another's. An order moves candidates; it adds none, so "found" cannot change, only the place.

With ``--fresh`` (the pools of ``measure_order_score.py --set fresh``): an order learned on all of
an arm's lists of the first set is scored on that arm's lists of a second set of problems, drawn
with another seed and sharing no structure with it. The order is learned before the second set's
lists are read, and nothing about it is changed afterwards.

    python scripts/study_learned_order.py --pools benchmarks/order_score/dev_run1_pools.json.gz --out benchmarks/order_score/dev_learned_order.json
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from apedp_dyn.proposers import learned_order  # noqa: E402
from apedp_dyn.recursion import order_score  # noqa: E402

TAG = "DEV-LEARNED-ORDER-2026-11"
REFERENCE = "taught-80"
HELD_APART = 100                 # problems of the reference arm never trained on in the learning curve
CURVE = (5, 10, 25, 50, 100, 200)
REPEATS = 20
TOLERANCES = (0.3, 1.0, 3.0)
HANDED = 3


def read(path: Path) -> dict[str, list[dict[str, Any]]]:
    """The printed pools as rows of ``order_score``, by arm: each candidate a dictionary again, each list's places as handed over."""
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        packed = json.load(handle)
    fields = packed["fields"]
    arms: dict[str, list[dict[str, Any]]] = {}
    for arm, key, level, inputs, noise, truth_bits, pool in packed["rows"]:
        candidates = [{name: (bool(value) if name in ("accurate", "solved", "same_form") else value)
                       for name, value in zip(fields, item, strict=True)} for item in pool]
        as_handed = list(range(len(candidates)))
        arms.setdefault(arm, []).append({
            "arm": arm, "key": key, "level": level, "inputs": inputs, "noise": noise, "truth_bits": truth_bits,
            "candidates": candidates, "solved_place": learned_order.place_under(candidates, as_handed, "solved"),
            "form_place": learned_order.place_under(candidates, as_handed, "same_form"),
            "accurate_place": learned_order.place_under(candidates, as_handed, "accurate")})
    return {arm: sorted(rows, key=lambda row: row["key"]) for arm, rows in arms.items()}


def brief(rows: list[dict[str, Any]], what: str = "solved_place") -> dict[str, Any]:
    full = order_score.summarise(rows, what)
    return {"problems": full["problems"], "found": full["found"], "first": full["first"], "within_3": full["within_3"],
            "mean_reciprocal_place": full["mean_reciprocal_place"],
            "within_3_by_level": {level: block["within_3"] for level, block in full["by_level"].items()}}


def arm_study(rows: list[dict[str, Any]], seed: int = 0) -> dict[str, Any]:
    learned = learned_order.out_of_fold(rows, folds=5, seed=seed)
    rules = {f"tolerance {tolerance:g}": learned_order.reorder(rows, tolerance=tolerance) for tolerance in TOLERANCES}
    best_rule = max(rules, key=lambda name: order_score.summarise(rules[name])["mean_reciprocal_place"])
    return {"as_handed": brief(rows), "true_form_as_handed": brief(rows, "form_place"),
            "an_accurate_candidate_as_handed": brief(rows, "accurate_place"),
            "rule": {name: brief(found) for name, found in rules.items()},
            "rule_against_as_handed": {"which": best_rule, **order_score.compare(rows, rules[best_rule])},
            "learned": brief(learned), "true_form_learned": brief(learned, "form_place"),
            "learned_against_as_handed": order_score.compare(rows, learned),
            "learned_against_the_rule": {"which": best_rule, **order_score.compare(rules[best_rule], learned)}}


def learning_curve(rows: list[dict[str, Any]], seed: int = 0) -> dict[str, Any]:
    """The learned order on problems held apart, trained on more and more of the others; ``REPEATS`` draws of each size."""
    rng = np.random.default_rng(seed)
    order = [rows[index] for index in rng.permutation(len(rows))]
    apart, rest = order[:HELD_APART], order[HELD_APART:]
    out: dict[str, Any] = {"held_apart": len(apart), "as_handed": brief(apart), "repeats": REPEATS, "trained_on": {}}
    for size in CURVE:
        if size > len(rest):
            continue
        within, first, reciprocal = [], [], []
        for _ in range(REPEATS):
            chosen = [rest[index] for index in rng.choice(len(rest), size=size, replace=False)]
            model = learned_order.LearnedOrder().fit([row["candidates"] for row in chosen])
            scored = order_score.summarise(learned_order.reorder(apart, model))
            within.append(scored["within_3"])
            first.append(scored["first"])
            reciprocal.append(scored["mean_reciprocal_place"])
        out["trained_on"][str(size)] = {"within_3_mean": round(float(np.mean(within)), 2), "within_3_min": int(min(within)),
                                        "within_3_max": int(max(within)), "first_mean": round(float(np.mean(first)), 2),
                                        "mean_reciprocal_place": round(float(np.mean(reciprocal)), 4)}
    return out


def carried(arms: dict[str, list[dict[str, Any]]], source: str) -> dict[str, Any]:
    """An order learned on one arm's lists, scored on the other arms' lists of problems it was not trained on."""
    rows = arms[source]
    keys = sorted(row["key"] for row in rows)
    train_keys = set(keys[: len(keys) * 2 // 3])
    model = learned_order.LearnedOrder().fit([row["candidates"] for row in rows if row["key"] in train_keys])
    out: dict[str, Any] = {"learned_on": source, "problems_trained_on": len(train_keys), "model": model.as_dict(), "on": {}}
    for arm, found in arms.items():
        fresh = [row for row in found if row["key"] not in train_keys]
        if fresh and any(row["candidates"] for row in fresh):
            out["on"][arm] = {"as_handed": brief(fresh), "learned": brief(learned_order.reorder(fresh, model)),
                              "against_as_handed": order_score.compare(fresh, learned_order.reorder(fresh, model))}
    return out


def on_fresh(arms: dict[str, list[dict[str, Any]]], fresh: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Orders learned on all of an arm's lists of the first set, scored on an arm's lists of the second set.

    A second-set arm is scored with the order learned on the same arm of the first set where there is one, and, for a
    list built as a proposer builds it (``split-80``), also with the order learned on the first measurement's lists of
    the same memory (``taught-80``): whether an order learned one way carries to lists built the other way.
    """
    out: dict[str, Any] = {}
    for arm, found in fresh.items():
        sources = [name for name in dict.fromkeys([arm, arm.replace("split-", "taught-")]) if name in arms]
        if not sources or not any(row["candidates"] for row in found):
            continue
        rules = {f"tolerance {tolerance:g}": learned_order.reorder(found, tolerance=tolerance) for tolerance in TOLERANCES}
        block: dict[str, Any] = {"as_handed": brief(found), "true_form_as_handed": brief(found, "form_place"),
                                 "rule": {name: brief(rows) for name, rows in rules.items()}, "learned_on": {}}
        for source in sources:
            shared = {row["key"] for row in arms[source]} & {row["key"] for row in found}
            if shared:
                raise ValueError(f"{len(shared)} problems of the second set are in the first: it is not a second set")
            model = learned_order.LearnedOrder().fit([row["candidates"] for row in arms[source]])
            learned = learned_order.reorder(found, model)
            block["learned_on"][source] = {
                "trained_on": dict(model.trained_on), "model": model.as_dict(), "learned": brief(learned),
                "true_form_learned": brief(learned, "form_place"),
                "learned_against_as_handed": order_score.compare(found, learned),
                "learned_against_the_rule": {name: order_score.compare(rows, learned) for name, rows in rules.items()}}
        out[arm] = block
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pools", type=Path, nargs="+", required=True, help="pools of the first set, one file or several")
    parser.add_argument("--fresh", type=Path, nargs="+", help="pools of the second set (measure_order_score.py --set fresh)")
    parser.add_argument("--model-out", type=Path, help="write the order learned on all of one arm's lists of the first set")
    parser.add_argument("--model-arm", help="the arm whose order is written (default: the reference arm)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    arms: dict[str, list[dict[str, Any]]] = {}
    for path in args.pools:
        arms.update(read(path))
    reference = REFERENCE if REFERENCE in arms else max(arms, key=lambda arm: len(arms[arm]))
    record = {"tag": TAG, "development": True,
              "what": "development: other orders of the candidate lists the memory handed over, scored by the place of the "
                      "first candidate the judge counts solved; not a result, nothing was fixed beforehand",
              "pools": [str(path) for path in args.pools], "handed_to_the_referee": HANDED,
              "features": list(learned_order.FEATURES),
              "arms": {arm: arm_study(rows) for arm, rows in arms.items() if any(row["candidates"] for row in rows)},
              "learning_curve": {"arm": reference, **learning_curve(arms[reference])},
              "carried_to_other_arms": carried(arms, reference)}
    if args.fresh:
        second: dict[str, list[dict[str, Any]]] = {}
        for path in args.fresh:
            second.update(read(path))
        record["on_the_second_set"] = {"pools": [str(path) for path in args.fresh], **on_fresh(arms, second)}
    if args.model_out is not None:
        chosen = args.model_arm or reference
        model = learned_order.LearnedOrder().fit([row["candidates"] for row in arms[chosen]])
        args.model_out.parent.mkdir(parents=True, exist_ok=True)
        args.model_out.write_text(json.dumps({
            "tag": TAG, "development": True,
            "what": "an order of the memory's candidates learned on every list of one arm of the order-score run; development, "
                    "not in force; read by apedp_dyn.proposers.ranked_memory.load_order",
            "learned_on": {"pools": [str(path) for path in args.pools], "arm": chosen}, "model": model.as_dict()},
            indent=1, sort_keys=True) + "\n",
            encoding="utf-8")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    for arm, study in record["arms"].items():
        print(f"{arm:11s} within 3: as handed {study['as_handed']['within_3']:3d}, rule {max(r['within_3'] for r in study['rule'].values()):3d}, "
              f"learned {study['learned']['within_3']:3d} of {study['as_handed']['found']} found in {study['as_handed']['problems']} "
              f"(sign test p {study['learned_against_as_handed']['sign_test_p']})")
    for arm, block in record.get("on_the_second_set", {}).items():
        if not isinstance(block, dict):
            continue
        for source, scored in block["learned_on"].items():
            print(f"second set, {arm:11s} (order learned on {source}) within 3: as handed {block['as_handed']['within_3']:3d}, "
                  f"learned {scored['learned']['within_3']:3d} of {block['as_handed']['found']} found in "
                  f"{block['as_handed']['problems']} (sign test p {scored['learned_against_as_handed']['sign_test_p']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
