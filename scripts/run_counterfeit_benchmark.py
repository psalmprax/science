#!/usr/bin/env python3
"""The counterfeit-law benchmark: a method is scored on what it refuses as much as on what it finds.

    python scripts/run_counterfeit_benchmark.py build [--check]        # benchmarks/counterfeit_laws.json
    python scripts/run_counterfeit_benchmark.py smoke
    python scripts/run_counterfeit_benchmark.py forecast               # preregistration/FORECAST-COUNTERFEIT-2026-11.json
    python scripts/run_counterfeit_benchmark.py run --shard k/N --out DIR     # PREREG-COUNTERFEIT-2026-11, once
    python scripts/run_counterfeit_benchmark.py collect --out DIR

Benchmarks of law discovery ask how many laws a method finds. A method that reports its best fit
every time does well on that and cannot be relied on. This benchmark is built so that in half
its cases reporting the best fit is wrong: every case hands the method a law's *counterfeits*
(other forms fitted to the law itself, ``apedp_dyn.redteam.counterfeit``) as candidates.

**A case** is a world of ``apedp_dyn.redteam.worlds``: rows drawn uniformly on the law's domain
with independent Gaussian noise (the conditions of the preregistered runs), split into
development rows and a pool the method may measure, and a list of candidates. For each law there
are ``len(NOISE_LEVELS)`` noise levels times two cases:

* *offered*: the law beside one counterfeit from each band of distance (two of the bands are
  near enough to count as accurate, the third is not). The right answer is the law.
* *withheld*: no law, and only counterfeits that are wrong by the benchmarks' own line (their
  best fit to the law differs from it by at least 5 % of its spread). The right answer is to
  report nothing.

A first design withheld the law and left every counterfeit in. Each of its 60 laws had a
counterfeit near enough to count as accurate, so reporting the best fit was never wrong there and
the benchmark could not tell a careful method from a careless one. It was replaced before any
case was frozen; its file was never run beyond four cases of its first law in a smoke test.

**The score.** A law reported and right counts one. A reported law that is wrong (not the law,
and less accurate than R^2 0.999 on fresh rows without noise) costs ``(1 - a) / a`` with ``a`` the
referee's permitted share of false reports (canonical ``REFEREE_MAX_FPR``): nineteen right
answers. So a method whose reports are wrong at exactly the permitted rate scores nothing. A
report that is accurate without being the law counts nothing either way, and so does saying
nothing. The total is divided by the number of cases in which the law was offered, so 1 is
perfect and a method that reports whatever fits best goes far below zero.

**Four methods are scored**: the route in force; the same route with the checks of
``discovery_l2/vouching.py`` on what it is about to report (``vouched``: built from what the red
team found, and what this benchmark is the first run on fresh laws for); the candidate that fits
best; and the best fit confirmed on thirty fresh rows.

Laws are drawn here with their own seed and leave out every structure of the invented-laws
benchmark (by fingerprint), of the red team's laws, of the laws the memory was developed on and
of the laws it is tested on.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import build_invented_laws as grammar  # noqa: E402
import run_memory_laws as memory_run  # noqa: E402
import run_red_team as red_team  # noqa: E402
import run_search_evolution as development  # noqa: E402
import run_srbench_main_search as path_run  # noqa: E402

from apedp_dyn.conf.canonical_spec import REFEREE_MAX_FPR, REFEREE_MIN_VRR  # noqa: E402
from apedp_dyn.discovery_l2 import vouching  # noqa: E402
from apedp_dyn.redteam import counterfeit  # noqa: E402
from apedp_dyn.redteam import worlds as rt  # noqa: E402

PREREGISTRATION_ID = "PREREG-COUNTERFEIT-2026-11"
DIGEST = ROOT / "preregistration" / "DIGEST-COUNTERFEIT.json"
OUT = ROOT / "benchmarks" / "counterfeit_laws.json"
FROZEN = (
    "preregistration/PREREG-COUNTERFEIT-2026-11.md", "preregistration/FORECAST-COUNTERFEIT-2026-11.json",
    "scripts/run_counterfeit_benchmark.py", "benchmarks/counterfeit_laws.json", "scripts/run_red_team.py",
    "scripts/build_invented_laws.py", "scripts/run_srbench_baseline.py", "scripts/run_srbench_main_search.py",
    "apedp_dyn/redteam/counterfeit.py", "apedp_dyn/redteam/worlds.py", "apedp_dyn/redteam/plain_selection.py",
    "apedp_dyn/discovery_l2/refined_search.py", "apedp_dyn/discovery_l2/main_search.py",
    "apedp_dyn/discovery_l2/contrast_confirmation.py", "apedp_dyn/discovery_l2/guided_search.py",
    "apedp_dyn/discovery_l2/vouching.py", "apedp_dyn/proposers/fit_guard.py", "apedp_dyn/proposers/portfolio.py",
    "benchmarks/red_team_worlds.jsonl.gz", "benchmarks/vouching_dev_replay_rows.jsonl.gz",       # what the forecast is made of
)
SEED = 20261130
LAW_COUNT = 60
NOISE_LEVELS = (0.0, 0.01, 0.05)
ROWS = 1000
#: bands of distance (shares of the law's spread). Near and middle are accurate by the benchmarks' line
#: (R^2 of 0.999 is a distance of 0.0316); far is wrong, with a gap left between so that no case sits on the line.
BANDS = ((0.001, 0.01), (0.01, 0.0316), (0.05, 0.3))
WITHHELD = 3                     # wrong counterfeits handed over where the law is withheld, where a law has that many
FIRST_ID = 6000
PENALTY = (1.0 - REFEREE_MAX_FPR) / REFEREE_MAX_FPR
METHODS = ("route", "vouched", "best_fit", "validated")
TIME_LIMIT_S = 600.0
KEPT_SHARE = REFEREE_MIN_VRR     # of the laws the route reports right, the share the checks must leave standing
FORECAST = ROOT / "preregistration" / "FORECAST-COUNTERFEIT-2026-11.json"
CAMPAIGN_WORLDS = ROOT / "benchmarks" / "red_team_worlds.jsonl.gz"
REPLAY_ROWS = ROOT / "benchmarks" / "vouching_dev_replay_rows.jsonl.gz"
#: the campaign's survey worlds a noise level of this benchmark is forecast from: (arena, noise above, noise up to)
FORECAST_FROM = {0.0: ("measured", -1.0, 0.0), 0.01: ("measured", 0.003, 0.0101), 0.05: ("noise_level", 0.03, 0.10)}
FORECAST_DRAWS = 20_000
#: what a method's report on one case can be, by the benchmarks' scoring
OUTCOMES = ("the law", "accurate, not the law", "wrong", "nothing")
FORECAST_PRIOR = 0.5             # added to every count before a share is drawn from it
FORECAST_POOL_WEIGHT = 2.0       # the weight, in reports, of what the checks did on all the survey's worlds


def _taken() -> set[str]:
    """Structures already used somewhere: the red team's laws, the memory's development and test laws."""
    used = {grammar.structure(sp.sympify(law["law"])) for law in red_team.laws()}
    used |= {grammar.structure(sp.sympify(law["text"])) for law in development.draw_laws(development.TRAIN + development.VALIDATION)}
    used |= {grammar.structure(sp.sympify(law["text"]))
             for law in json.loads(memory_run.LAWS_FILE.read_text(encoding="utf-8"))["laws"]}
    return used


def build(count: int = LAW_COUNT, seed: int = SEED) -> dict[str, Any]:
    """The benchmark: laws with one counterfeit per band of distance, and the cases made of them."""
    in_benchmark = set(json.loads(grammar.FINGERPRINTS.read_text(encoding="utf-8"))["fingerprints"])
    taken = _taken()
    rng = np.random.default_rng(seed)
    laws: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _ in range(grammar.MAX_DRAWS):
        if len(laws) == count:
            break
        inputs = int(rng.integers(1, 4))
        recipe = counterfeit.draw_recipe(rng, inputs, grammar.CONSTANTS)
        expr = counterfeit.law_of(recipe, inputs)
        key = grammar.structure(expr)
        if key in seen or not grammar.acceptable(expr, inputs, rng):
            continue
        seen.add(key)
        if grammar.fingerprint(str(expr)) in in_benchmark or key in taken:
            continue
        fakes = counterfeit.counterfeits(recipe, inputs, grammar.LOW, grammar.HIGH, seed=seed + len(laws))
        chosen = []
        for low, high in BANDS:
            band = [fake for fake in fakes if low <= fake.distance < high]
            if band:
                chosen.append(band[int(rng.integers(0, len(band)))])
        far = [fake for fake in fakes if BANDS[-1][0] <= fake.distance < BANDS[-1][1]]
        if not far or len(chosen) < 2:
            continue                                    # a law needs a wrong counterfeit and one that is nearly right
        others = [fake for fake in far if fake is not chosen[-1]]
        picked = [int(i) for i in rng.permutation(len(others))[: WITHHELD - 1]]
        wrong = sorted([chosen[-1], *(others[i] for i in picked)], key=lambda fake: (fake.distance, fake.text))
        laws.append({"id": FIRST_ID + len(laws), "inputs": inputs, "law": str(expr),
                     "counterfeits": [asdict(fake) for fake in chosen], "wrong_counterfeits": [asdict(fake) for fake in wrong]})
    if len(laws) != count:
        raise SystemExit(f"only {len(laws)} laws after {grammar.MAX_DRAWS} draws")
    cases = []
    for law in laws:
        beside = [fake["text"] for fake in law["counterfeits"]]
        alone = [fake["text"] for fake in law["wrong_counterfeits"]]
        for noise in NOISE_LEVELS:
            for offered in (True, False):
                texts = [*beside, law["law"]] if offered else alone
                order = rng.permutation(len(texts))
                world = rt.World(law=law["law"], inputs=law["inputs"], candidates=tuple(texts[int(i)] for i in order),
                                 truth_listed=offered, rows=ROWS, noise=noise, seed=int(rng.integers(0, 2 ** 31 - 1)))
                cases.append({"case": f"{law['id']}-{noise:g}-{'offered' if offered else 'withheld'}", "law_id": law["id"],
                              "world": world.as_dict()})
    return {"what": "The counterfeit-law benchmark: every case hands a method counterfeits of a law as candidates: the law "
                    "beside one of each band, or no law and only wrong ones. Scored on right reports minus a heavy cost for "
                    "wrong ones.",
            "seed": seed, "domain": [grammar.LOW, grammar.HIGH], "noise_levels": list(NOISE_LEVELS), "rows": ROWS,
            "penalty_for_a_wrong_law": PENALTY, "laws": laws, "cases": cases}


def cases() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = json.loads(OUT.read_text(encoding="utf-8"))["cases"]
    return out


SMOKE_LAW = "0.8*x0 + 1.9/(x0 + 0.6)"
SMOKE_COUNTERFEITS = ("0.62*x0 + 1.49*x0**(-0.57) + 0.36", "0.7302*x0 + 1.0981/x0 + 0.5134")


def smoke_cases() -> list[dict[str, Any]]:
    """Four cases of a law that is in no benchmark, for the smoke job: no case of the benchmark is run before the run."""
    out = []
    for k, (noise, offered) in enumerate(((0.0, True), (0.0, False), (0.01, True), (0.01, False))):
        texts = (*SMOKE_COUNTERFEITS, *((SMOKE_LAW,) if offered else ()))
        world = rt.World(law=SMOKE_LAW, inputs=1, candidates=texts, truth_listed=offered, rows=ROWS, noise=noise, seed=77 + k)
        out.append({"case": f"smoke-{noise:g}-{'offered' if offered else 'withheld'}", "law_id": -1, "world": world.as_dict()})
    return out


def vouched_row(world: rt.World, route: dict[str, Any]) -> dict[str, Any]:
    """The route's report after the checks of ``vouching``: scored like any method, with why a report was withheld."""
    names = [f"x{i}" for i in range(world.inputs)]
    truth = sp.sympify(world.law, locals={n: sp.Symbol(n) for n in names})
    rows = rt.rows_of(world)
    if not route.get("reported") or not route.get("law"):
        return silent("the route reported nothing")
    order = np.random.default_rng([world.seed, 4]).permutation(world.rows)          # the split the route was given
    cut = round(path_run.DEV_SHARE * world.rows)
    dev, pool = order[:cut], order[cut:]
    done = vouching.vouch(route["law"], list(route.get("equivalent") or []), names, rows.X[dev], rows.y[dev], rows.X[pool])
    return {**red_team._scored(done.law, truth, names, rows), "reason": done.reason, "switched": done.switched,
            "power": done.power, "members": len(done.members), "refused": sum(m.refused for m in done.members)}


def silent(reason: str) -> dict[str, Any]:
    """A method that reported nothing, as ``run_red_team._scored`` records it, with why."""
    return {"reported": False, "law": None, "symbolic": False, "test_r2": None, "accurate": False, "wrong": False,
            "badness": 0.0, "reason": reason, "switched": False}


def evaluate(worlds: list[rt.World], cores: int, time_limit: float = TIME_LIMIT_S) -> list[dict[str, Any]]:
    """Every method on every case. A case that failed for the route failed for all of them.

    Checks that crash on a case vouch for nothing there: the case counts as nothing reported by
    ``vouched``, and the error is kept in the row.
    """
    rows: list[dict[str, Any]] = red_team.evaluate(worlds, cores, time_limit)
    for world, row in zip(worlds, rows, strict=True):
        if row.get("status") == "ok":
            try:
                row["vouched"] = vouched_row(world, row["route"])
            except Exception as exc:  # noqa: BLE001 - checks that crash vouch for nothing; recorded with the error's class
                row["vouched"] = {**silent(f"the checks failed: {type(exc).__name__}: {exc}"[:200]), "failed": True}
    return rows


def tally(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    """One method on a set of cases: what it reported, and its score."""
    ok = [row for row in rows if row.get("status") == "ok"]
    offered = [row for row in rows if row["world"]["truth_listed"]]
    right = sum(bool(row[method]["symbolic"]) for row in ok if row[method]["reported"])
    wrong = sum(bool(row[method]["wrong"]) for row in ok)
    reported = sum(bool(row[method]["reported"]) for row in ok)
    return {"cases": len(rows), "law_offered": len(offered), "errors": sum(row.get("status") == "error" for row in rows),
            "timeouts": sum(row.get("status") == "timeout" for row in rows),
            "reported": reported, "right": right, "wrong": wrong, "accurate_not_the_law": reported - right - wrong,
            "wrong_share_of_reported": wrong / reported if reported else None,
            "law_reported_when_offered": sum(bool(row[method]["symbolic"]) for row in ok if row["world"]["truth_listed"]),
            "reported_when_withheld": sum(bool(row[method]["reported"]) for row in ok if not row["world"]["truth_listed"]),
            "score": (right - PENALTY * wrong) / len(offered) if offered else None}


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Every method overall, per noise level, and with the law offered or withheld; then the expectations as written."""
    out: dict[str, Any] = {}
    for method in METHODS:
        out[f"{method}|all"] = tally(rows, method)
        for noise in NOISE_LEVELS:
            out[f"{method}|{noise:g}"] = tally([row for row in rows if float(row["world"]["noise"]) == noise], method)
        for offered in (True, False):
            sel = [row for row in rows if bool(row["world"]["truth_listed"]) == offered]
            out[f"{method}|{'offered' if offered else 'withheld'}"] = tally(sel, method)
    measured = [row for row in rows if float(row["world"]["noise"]) <= 0.01]
    route_measured = tally(measured, "route")
    route, vouched = out["route|all"], out["vouched|all"]
    ok = [row for row in rows if row.get("status") == "ok"]
    right = [row for row in ok if row["route"]["reported"] and row["route"]["symbolic"]]
    kept = sum(bool(row["vouched"]["reported"] and row["vouched"]["symbolic"]) for row in right)
    worst = max((out[f"vouched|{noise:g}"]["wrong_share_of_reported"] or 0.0) for noise in NOISE_LEVELS)
    plain = [method for method in METHODS if method not in ("route", "vouched")]
    out["expectations"] = {
        "1_wrong_share_of_the_routes_reports_up_to_1_percent_noise": route_measured["wrong_share_of_reported"],
        "1_limit": REFEREE_MAX_FPR,
        "1_holds": (route_measured["wrong_share_of_reported"] or 0.0) <= REFEREE_MAX_FPR,
        "2_scores": {method: out[f"{method}|all"]["score"] for method in METHODS},
        "2_holds": route["score"] is not None and all(route["score"] > out[f"{method}|all"]["score"] for method in plain),
        "3_wrong_share_of_vouched_reports_by_noise": {f"{noise:g}": out[f"vouched|{noise:g}"]["wrong_share_of_reported"]
                                                      for noise in NOISE_LEVELS},
        "3_limit": REFEREE_MAX_FPR,
        "3_holds": worst <= REFEREE_MAX_FPR,
        "4_holds": vouched["score"] is not None and route["score"] is not None and vouched["score"] > route["score"],
        "5_right_laws_of_the_route_kept": [kept, len(right)],
        "5_limit": KEPT_SHARE,
        "5_holds": bool(right) and kept >= KEPT_SHARE * len(right),
    }
    out["why_vouched_withheld"] = {}
    for row in ok:
        if row["route"]["reported"] and not row["vouched"]["reported"]:
            reason = str(row["vouched"].get("reason", ""))[:60]
            out["why_vouched_withheld"][reason] = out["why_vouched_withheld"].get(reason, 0) + 1
    out["checks_failed"] = sum(bool(row["vouched"].get("failed")) for row in ok)
    out["vouched_switched"] = {"runs": sum(bool(row["vouched"].get("switched")) for row in ok),
                               "into_the_law": sum(bool(row["vouched"].get("switched") and row["vouched"]["symbolic"]
                                                        and not row["route"]["symbolic"]) for row in ok)}
    return out


def _outcome(scored: dict[str, Any]) -> str:
    if not scored["reported"]:
        return "nothing"
    if scored["symbolic"]:
        return "the law"
    return "wrong" if scored["wrong"] else "accurate, not the law"


def _counts(labels: list[str]) -> list[int]:
    return [sum(label == outcome for label in labels) for outcome in OUTCOMES]


def simulate(cells: dict[tuple[float, bool], dict[str, Any]], pooled: dict[str, list[int]], rng: np.random.Generator,
             draws: int = FORECAST_DRAWS, per_cell: int = LAW_COUNT) -> dict[str, Any]:
    """The run drawn many times over from counts: every method's totals, and how often each expectation holds.

    ``cells[(noise, offered)]`` holds, for the route and the two plain methods, how many of the
    campaign's worlds of that kind ended in each of ``OUTCOMES``, and under ``after`` what the
    checks made of the route's reports of each kind there. A draw first draws the shares
    themselves (a Dirichlet on the counts with ``FORECAST_PRIOR`` added to each), then
    ``per_cell`` cases a cell at those shares. The checks act on what the route reported in the
    same draw: a report of each kind is passed on, changed or withheld at shares drawn from the
    replay's counts for that cell, with ``pooled`` (the replay's counts over all its worlds) as a
    prior worth ``FORECAST_POOL_WEIGHT`` reports. The expectations are judged on each draw as
    ``summarise`` judges them on the run.
    """
    law, _close, wrong, nothing = range(len(OUTCOMES))
    seen = {method: {noise: np.zeros((draws, len(OUTCOMES))) for noise in NOISE_LEVELS} for method in METHODS}
    kept = np.zeros(draws)
    for (noise, _offered), cell in cells.items():
        drawn = {}
        for method in ("route", "best_fit", "validated"):
            shares = rng.dirichlet(np.asarray(cell[method], dtype=float) + FORECAST_PRIOR, size=draws)
            drawn[method] = rng.multinomial(per_cell, shares)
            seen[method][noise] += drawn[method]
        seen["vouched"][noise][:, nothing] += drawn["route"][:, nothing]
        for before, name in enumerate(OUTCOMES[:nothing]):
            centre = np.asarray(pooled[name], dtype=float) + FORECAST_PRIOR
            alpha = np.asarray(cell["after"][name], dtype=float) + FORECAST_POOL_WEIGHT * centre / centre.sum()
            after = rng.multinomial(drawn["route"][:, before], rng.dirichlet(alpha, size=draws))
            seen["vouched"][noise] += after
            if before == law:
                kept += after[:, law]

    def total(method: str, outcome: int) -> np.ndarray:
        summed: np.ndarray = np.sum([seen[method][noise][:, outcome] for noise in NOISE_LEVELS], axis=0)
        return summed

    def wrong_share(method: str, levels: list[float]) -> np.ndarray:
        reported = np.sum([seen[method][noise][:, :nothing].sum(axis=1) for noise in levels], axis=0)
        bad = np.sum([seen[method][noise][:, wrong] for noise in levels], axis=0)
        share: np.ndarray = np.where(reported > 0, bad / np.maximum(reported, 1), 0.0)
        return share

    offered_cases = per_cell * len(NOISE_LEVELS)
    score = {method: (total(method, law) - PENALTY * total(method, wrong)) / offered_cases for method in METHODS}
    within = {f"{noise:g}": wrong_share("vouched", [noise]) <= REFEREE_MAX_FPR for noise in NOISE_LEVELS}
    holds = {"1": wrong_share("route", [noise for noise in NOISE_LEVELS if noise <= 0.01]) <= REFEREE_MAX_FPR,
             "2": (score["route"] > score["best_fit"]) & (score["route"] > score["validated"]),
             "3": np.all(list(within.values()), axis=0),
             "4": score["vouched"] > score["route"],
             "5": (total("route", law) > 0) & (kept >= KEPT_SHARE * total("route", law))}
    totals: dict[str, Any] = {}
    for method in METHODS:
        totals[method] = {name: {"expected": round(float(values.mean()), 2),
                                 "interval": [round(float(np.quantile(values, 0.1)), 3), round(float(np.quantile(values, 0.9)), 3)],
                                 "level": 0.8}
                          for name, values in (("right", total(method, law)), ("wrong", total(method, wrong)),
                                               ("score", score[method]))}
    return {"totals": totals,
            "chance_that_an_expectation_holds": {**{key: round(float(value.mean()), 3) for key, value in holds.items()},
                                                 "3_by_noise": {key: round(float(value.mean()), 3)
                                                                for key, value in within.items()}}}


def forecast(seed: int = SEED) -> dict[str, Any]:
    """What the red-team campaign's randomly drawn worlds lead one to expect of this run, method by method.

    For each noise level and for the law offered or withheld, the campaign's survey worlds of that
    kind are counted by how each method ended on them, and the replayed reports of that kind by
    what the checks made of them. ``simulate`` then draws the run from those counts many times
    over, for the totals, their 80 % intervals and the chance that each expectation holds.
    """
    import gzip

    with gzip.open(CAMPAIGN_WORLDS, "rt", encoding="utf-8") as handle:
        survey = [row for row in map(json.loads, handle) if row.get("stage") == "survey" and row.get("status") == "ok"
                  and row["world"]["sampling"] == "uniform" and row["world"]["noise_shape"] == "gaussian"
                  and int(row["world"]["rows"]) == ROWS]
    with gzip.open(REPLAY_ROWS, "rt", encoding="utf-8") as handle:
        replayed = [row for row in map(json.loads, handle) if row["stage"] == "survey" and int(row["rows"]) == ROWS]
    shares: dict[str, dict[str, Any]] = {}
    cells: dict[tuple[float, bool], dict[str, Any]] = {}
    for noise, (arena, above, up_to) in FORECAST_FROM.items():
        for offered in (True, False):
            def of_this_kind(world_noise: float, listed: bool, row_arena: str, best_fit_wrong: bool, arena: str = arena,
                             above: float = above, up_to: float = up_to, offered: bool = offered) -> bool:
                # a withheld case here offers wrong candidates only: of the campaign's worlds, those whose best fit was wrong
                return (row_arena == arena and above < world_noise <= up_to and listed == offered
                        and (offered or best_fit_wrong))

            cell = [row for row in survey if of_this_kind(float(row["world"]["noise"]), bool(row["world"]["truth_listed"]),
                                                          row["arena"], bool(row["best_fit"]["wrong"]))]
            again = [row for row in replayed if of_this_kind(float(row["noise"]), bool(row["truth_listed"]), row["arena"],
                                                             bool(row["best_fit_wrong"]))]
            cells[(noise, offered)] = {
                **{method: _counts([_outcome(row[method]) for row in cell]) for method in ("route", "best_fit", "validated")},
                "after": {name: _counts([row["after"] for row in again if row["before"] == name]) for name in OUTCOMES[:-1]}}
            key = f"{noise:g}|{'offered' if offered else 'withheld'}"
            shares[key] = {"survey_worlds": len(cell), "replayed_reports": len(again),
                           "outcomes": list(OUTCOMES), **cells[(noise, offered)]}
    pooled = {name: _counts([row["after"] for row in replayed if row["before"] == name]) for name in OUTCOMES[:-1]}
    drawn = simulate(cells, pooled, np.random.default_rng([seed, 9]))
    return {"what": "Filed before the run. Counts come from the red-team campaign's randomly drawn worlds of the same noise, "
                    "sampling and number of rows, and for the vouched route from the replay of its checks on them. Those "
                    "worlds drew their counterfeits at random distances; this benchmark takes one from each band where the "
                    "law is offered and only wrong ones where it is withheld, for which the campaign's worlds whose best "
                    "fit was wrong stand in. The intervals and chances cover the draw of sixty cases a cell and what the "
                    "counts leave open about the shares, and nothing else: not the difference between the campaign's "
                    "laws and these.",
            "from": {"campaign": str(CAMPAIGN_WORLDS.relative_to(ROOT)), "replay": str(REPLAY_ROWS.relative_to(ROOT))},
            "cases_per_cell": LAW_COUNT, "draws": FORECAST_DRAWS, "counts": shares,
            "checks_on_all_replayed_reports": {"outcomes": list(OUTCOMES), **pooled}, **drawn}


def frozen_digests() -> dict[str, str]:
    return {rel: hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() for rel in FROZEN}


def check_frozen() -> None:
    recorded = json.loads(DIGEST.read_text(encoding="utf-8"))["sha256"]
    now = frozen_digests()
    changed = sorted(rel for rel in FROZEN if recorded.get(rel) != now[rel])
    if changed or set(recorded) != set(FROZEN):
        raise SystemExit(f"frozen files differ from {DIGEST.name}: {changed or sorted(set(recorded) ^ set(FROZEN))}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("command", choices=("build", "smoke", "forecast", "run", "collect"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--check", action="store_true", help="for build: compare the file with the generator")
    args = parser.parse_args(argv)
    if args.command == "build":
        text = json.dumps(build(), indent=1, sort_keys=True) + "\n"
        if args.check:
            if OUT.read_text(encoding="utf-8") != text:
                print(f"{OUT.relative_to(ROOT)} is not what the generator gives")
                return 1
            print(f"{OUT.relative_to(ROOT)}: as generated")
            return 0
        OUT.write_text(text, encoding="utf-8")
        record = json.loads(text)
        print(f"wrote {OUT.relative_to(ROOT)}: {len(record['laws'])} laws, {len(record['cases'])} cases")
        return 0
    if args.command == "forecast":
        filed = forecast()
        FORECAST.write_text(json.dumps(filed, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        for method, total in filed["totals"].items():
            print(f"{method:10s} right {total['right']['expected']:6.1f} {total['right']['interval']}  wrong "
                  f"{total['wrong']['expected']:6.1f} {total['wrong']['interval']}  score {total['score']['expected']:7.2f} "
                  f"{total['score']['interval']}")
        print("chance that an expectation holds:", json.dumps(filed["chance_that_an_expectation_holds"]))
        return 0
    if args.command == "smoke":
        picked = smoke_cases()
        rows = evaluate([rt.World.from_dict(case["world"]) for case in picked], red_team._cores())
        for case, row in zip(picked, rows, strict=True):
            if row["status"] != "ok":
                print(case["case"], row["status"], row.get("error"), row.get("trace", ""))
                return 1
            print(f"{case['case']:28s} route: {row['route']['action']:17s} wrong={row['route']['wrong']!s:5s} "
                  f"vouched: reported={row['vouched']['reported']!s:5s} wrong={row['vouched']['wrong']!s:5s} "
                  f"best_fit wrong={row['best_fit']['wrong']!s}")
        print(json.dumps(summarise(rows)["expectations"], indent=1))
        return 0
    if args.out is None:
        parser.error("--out is required")
    args.out.mkdir(parents=True, exist_ok=True)
    k, n = (int(v) for v in args.shard.split("/"))
    if args.command == "run":
        check_frozen()
        mine = [case for i, case in enumerate(cases()) if i % n == k]
        rows = evaluate([rt.World.from_dict(case["world"]) for case in mine], red_team._cores(), TIME_LIMIT_S)
        path = args.out / f"runs-shard-{k:03d}-of-{n:03d}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for case, row in zip(mine, rows, strict=True):
                handle.write(json.dumps({"case": case["case"], "law_id": case["law_id"], **row}, sort_keys=True, default=str) + "\n")
        return 0
    rows = [json.loads(line) for path in sorted(args.out.glob("runs-shard-*.jsonl")) for line in path.read_text().splitlines()]
    (args.out / "runs.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    summary = summarise(rows)
    record = {"preregistration": PREREGISTRATION_ID, "rows": len(rows), "summary": summary, "frozen": frozen_digests()}
    (args.out / "results.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"collected {len(rows)} rows into {args.out}")
    print(json.dumps({key: value for key, value in summary.items() if key.endswith("|all") or key == "expectations"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
