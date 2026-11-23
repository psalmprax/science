"""The mark that says whether a table lies inside the conditions the route in force was measured on."""
from __future__ import annotations

import gzip
import json
import math
import re
from pathlib import Path

import numpy as np

from apedp_dyn.discovery_l2 import measured_range as mr
from apedp_dyn.discovery_l2 import vouched_route_v2

ROOT = Path(__file__).resolve().parents[1]
LAW = "0.8*x0 + 1.9/(x0 + 0.6)"


def _table(noise: float, rows: int = 700, inputs: int = 1, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(1.0, 5.0, (rows, inputs))
    clean = 0.8 * X[:, 0] + 1.9 / (X[:, 0] + 0.6)
    y = clean + noise * float(np.sqrt(np.mean(clean ** 2))) * rng.standard_normal(rows)
    return [f"x{i}" for i in range(inputs)], X, y


def test_the_measured_conditions_are_the_blind_runs_own() -> None:
    laws = json.loads((ROOT / "benchmarks" / "vouching_v2_blind_laws.json").read_text())
    assert mr.MEASURED_NOISE == max(laws["noise_levels"])
    worlds = [case["world"] for case in laws["handed"]["cases"]]
    assert mr.MEASURED_INPUTS == max(w["inputs"] for w in worlds)
    # the runner's split (run_srbench_main_search.DEV_SHARE): 70 % of the rows to fit on
    share = re.search(r"^DEV_SHARE = ([0-9.]+)$", (ROOT / "scripts" / "run_srbench_main_search.py").read_text(), re.M)
    assert share is not None
    assert {round(float(share.group(1)) * w["rows"]) for w in worlds} == {mr.MEASURED_ROWS}
    assert mr.SOURCE == vouched_route_v2.EVIDENCE


def test_the_noise_estimate_is_the_noise_and_the_mark_follows_it() -> None:
    for noise, state in ((0.0, mr.INSIDE), (0.01, mr.INSIDE), (0.10, mr.OUTSIDE)):
        names, X, y = _table(noise)
        mark = mr.assess(LAW, names, X, y)
        assert mark["state"] == state, (noise, mark)
        if noise:
            assert abs(mark["noise"] / noise - 1.0) < 0.1
    names, X, y = _table(0.05)
    assert mr.assess(LAW, names, X, y)["state"] == mr.EDGE


def test_what_else_puts_a_table_outside() -> None:
    names, X, y = _table(0.01, inputs=4)
    assert mr.assess(LAW, names, X, y)["state"] == mr.OUTSIDE
    names, X, y = _table(0.01, rows=200)
    mark = mr.assess(LAW, names, X, y)
    assert mark["state"] == mr.OUTSIDE and "200 rows" in mark["reasons"][0]
    assert "Gaussian" in mark["unchecked"]


def test_what_cannot_be_marked_says_so() -> None:
    names, X, y = _table(0.01)
    assert mr.assess(None, names, X, y)["state"] == mr.UNKNOWN
    assert mr.assess("x0 +* 2", names, X, y)["state"] == mr.UNKNOWN
    other = mr.assess(LAW, names, X, y, route="main")
    assert other["state"] == mr.UNKNOWN and "refined" in other["reasons"][0]
    share, error = mr.noise_share("0.0*x0", names, X, y) or (0.0, 0.0)
    assert math.isinf(share) and math.isinf(error)


def test_a_wrong_law_reads_as_noisier() -> None:
    names, X, y = _table(0.01)
    right = mr.assess(LAW, names, X, y)["noise"]
    wrong = mr.assess("0.52*x0 + 2.2", names, X, y)["noise"]
    assert wrong > 3 * right


def test_the_claim_states_the_blind_runs_numbers() -> None:
    rows = [json.loads(line) for line in gzip.open(ROOT / "preregistration" / "RESULT-VOUCHING-V2-2026-11" / "searched.jsonl.gz", "rt")]
    ok = [r for r in rows if r["status"] == "ok"]
    reported = [r for r in ok if r["second"]["reported"]]
    recovered = [r for r in reported if r["second"]["symbolic"]]
    assert f"{len(recovered)} of the {len(reported)} laws reported" in vouched_route_v2.CLAIM
    assert f"the other {len(reported) - len(recovered)} accurate" in vouched_route_v2.CLAIM
    assert not any(not r["second"]["accuracy_solution"] for r in reported)
    for noise, text in ((0.0, "at noise 0"), (0.01, "at 1 %"), (0.05, "at 5 %")):
        sel = [r for r in reported if r["noise"] == noise]
        assert f"{sum(r['second']['symbolic'] for r in sel)} of {len(sel)} {text}" in vouched_route_v2.CLAIM
