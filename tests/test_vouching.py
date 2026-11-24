"""What a route can vouch for: the structure test's null, the class rule and the power rule.

Status: audit-only

Everything here is on made-up rows with a known law. Nothing runs a search.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from apedp_dyn.conf import canonical_spec
from apedp_dyn.discovery_l2 import main_search, vouching
from apedp_dyn.discovery_l2.guided_search import neighbours, residual_structure

ROOT = Path(__file__).resolve().parents[1]
LAW = "1.7*x0/(x0 + 2.3) + 0.4*x1"
NAMES = ["x0", "x1"]
FAST = {"flips": 511, "trials": 16}


def _rows(n: int = 500, noise: float = 0.01, seed: int = 0, shape: str = "gaussian"):
    rng = np.random.default_rng(seed)
    X = rng.uniform(1.0, 5.0, (n, 2))
    clean = 1.7 * X[:, 0] / (X[:, 0] + 2.3) + 0.4 * X[:, 1]
    sd = noise * float(np.sqrt(np.mean(clean ** 2)))
    error = rng.standard_normal(n) * sd
    if shape == "proportional":
        error = error * clean / float(np.sqrt(np.mean(clean ** 2)))
    return X, clean + error, clean, sd


def test_the_settings_are_the_canonical_ones_and_the_benchmarks_accuracy_line():
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_srbench_baseline as base

    assert vouching.ACCURACY_R2 == base.R2_ACCURACY
    assert vouching.LEVEL == canonical_spec.DIEBOLD_MARIANO_P_VALUE_THRESHOLD
    assert vouching.POWER == canonical_spec.TEMPORAL_RATE_CALIBRATION_POWER_MIN
    assert vouching.NUMERIC_FLOOR == main_search.NUMERIC_FLOOR
    assert (vouching.FLIPS + 1) * vouching.LEVEL / len(vouching.SCALES) >= 1.0     # the smallest p can fall below the level


def test_at_the_paths_own_scale_the_statistic_is_the_paths_own():
    X, y, clean, _ = _rows()
    residuals = y - clean
    (k, matrix), *_ = vouching.differences(X, (8,))
    assert k == 8
    assert vouching._statistic(matrix, residuals[:, None])[0] == pytest.approx(residual_structure(residuals, neighbours(X, 8)))


def test_a_right_law_is_refused_about_as_often_as_the_level_says():
    refused, smallest = 0, []
    for seed in range(120):
        X, y, clean, _ = _rows(300, 0.02, seed, "proportional" if seed % 2 else "gaussian")
        p, _ = vouching.structure_test(y - clean, vouching.differences(X), flips=511, seed=seed)
        refused += p < vouching.LEVEL
        smallest.append(p)
    assert refused <= 5, refused                         # 1.2 expected at a level of one in a hundred
    assert np.median(smallest) > 0.3                     # and the p-values are not piled up near zero


def test_an_error_a_third_of_the_noise_passes_the_paths_check_and_not_this_one():
    X, y, clean, sd = _rows(700, 0.02, 3)
    bend = (X[:, 0] - 3.0) ** 2 - np.mean((X[:, 0] - 3.0) ** 2)
    wrong = clean + 0.33 * sd * bend / bend.std()
    old = main_search.adequacy(y - wrong, X, y, 8)
    assert old["passed"] and old["residual_structure"] < 0.0            # the frozen check lets it through
    p, scales = vouching.structure_test(y - wrong, vouching.differences(X), flips=1023)
    assert p < vouching.LEVEL
    assert [s["neighbours"] for s in scales] == [8.0, 32.0, 128.0]
    assert all(s["null_mean"] == pytest.approx(-1.0 / s["neighbours"], abs=0.02) for s in scales)
    assert scales[-1]["statistic"] > scales[-1]["critical"]              # the coarse scale is the one that sees it


def test_a_right_law_is_reported_as_it_was():
    X, y, _, _ = _rows(700, 0.01, 1)
    done = vouching.vouch(LAW, [], NAMES, X, y, **FAST)
    assert done.law == LAW and done.reason == "" and not done.switched and done.power is not None and done.power >= vouching.POWER
    assert done.record["fit_error"] < done.record["allowed_error"] and len(done.record["chances"]) == 5


def test_a_law_that_is_slightly_wrong_is_refused_for_its_structure():
    X, y, _, _ = _rows(700, 0.003, 2)
    near = "1.7*x0/(x0 + 2.3) + 0.4*x1 + 0.02*sin(1.3*x0)"            # R^2 against the law about 0.9993: counts as accurate...
    off = "1.7*x0/(x0 + 2.3) + 0.4*x1 + 0.05*sin(1.3*x0)"             # ...and this one about 0.996: wrong
    for text in (near, off):
        done = vouching.vouch(text, [], NAMES, X, y, **FAST)
        assert done.law is None and "structure" in done.reason, text
        assert done.members[0].refused and done.members[0].p_value < vouching.LEVEL


def test_when_the_simplest_of_a_class_leaves_structure_the_member_that_does_not_is_reported():
    X, y, _, _ = _rows(700, 0.005, 4)
    counterfeit = "0.52*x0**0.5 + 0.4*x1 + 0.02"                       # shorter, and not the law
    done = vouching.vouch(counterfeit, [LAW], NAMES, X, y, **FAST)
    assert done.law == LAW and done.switched and "another member" in done.reason
    assert [m.refused for m in done.members] == [True, False]


def test_members_that_pass_and_disagree_are_named_and_nothing_is_reported():
    X, y, _, _ = _rows(120, 0.15, 5)
    other = "0.9 + 0.33*x0 - 0.028*x0**2 + 0.4*x1"                     # close to the law at this noise, and not the law
    done = vouching.vouch(LAW, [other], NAMES, X, y, **FAST)
    assert done.law is None and "differ by more than the accuracy claimed" in done.reason
    assert done.record["class"] == [LAW, other] and done.record["apart"] == 1


def test_where_an_error_that_size_could_not_have_been_seen_nothing_is_vouched_for():
    X, y, _, _ = _rows(150, 0.02, 6)
    done = vouching.vouch(LAW, [], NAMES, X, y, **FAST)
    assert done.law is None and "would have been seen" in done.reason and done.power is not None and done.power < vouching.POWER
    assert done.record["fit_error"] < done.record["allowed_error"]      # the constants could be fitted; the error could not be seen
    many, y_many, _, _ = _rows(1500, 0.02, 6)                           # ten times the rows, and it can
    assert vouching.vouch(LAW, [], NAMES, many, y_many, **FAST).law == LAW
    crowded = "1.7*x0/(x0 + 2.3) + 0.4*x1 + 0.001*x0**2 + 0.001*x1**2 + 0.001*x0*x1 + 0.001*x0**3 + 0.001"
    X, y, _, _ = _rows(60, 0.3, 7)
    costly = vouching.vouch(crowded, [], NAMES, X, y, **FAST)
    assert costly.law is None and "constants cannot be fitted" in costly.reason


def test_an_exact_fit_is_reported_and_an_inexact_rival_beside_it_is_refused():
    X, _, clean, _ = _rows(300, 0.0, 8)
    done = vouching.vouch(LAW, ["0.52*x0**0.5 + 0.4*x1 + 0.02"], NAMES, X, clean, **FAST)
    assert done.law == LAW and done.reason == "" and done.members[0].exact and done.members[1].refused and done.power == 1.0


def test_a_candidate_that_is_not_finite_on_the_rows_is_not_a_candidate():
    X, y, _, _ = _rows(300, 0.01, 9)
    only = vouching.vouch("log(x0 - 50)", [], NAMES, X, y, **FAST)
    assert only.law is None and not only.members[0].usable and "every candidate" in only.reason
    beside = vouching.vouch("log(x0 - 50)", [LAW], NAMES, X, y, **FAST)
    assert beside.law == LAW and beside.switched
    assert vouching.vouch("not a law (", [], NAMES, X, y, **FAST).law is None


def test_members_are_compared_where_inputs_are_known_and_not_only_where_targets_are():
    rng = np.random.default_rng(10)
    X = rng.uniform(1.0, 2.0, (400, 2))                                 # the rows with targets cover half the box
    more = rng.uniform(1.0, 5.0, (400, 2))
    clean = 1.7 * X[:, 0] / (X[:, 0] + 2.3) + 0.4 * X[:, 1]
    y = clean + 0.002 * rng.standard_normal(400) * float(np.sqrt(np.mean(clean ** 2)))
    rival = LAW + " + 0.004*(x0 - 1)**2*(x0 - 2)**2"                    # the law on those rows to within the noise, and not beyond
    inside = vouching.vouch(LAW, [rival], NAMES, X, y, **FAST)
    outside = vouching.vouch(LAW, [rival], NAMES, X, y, more, **FAST)
    assert inside.law == LAW and inside.record["apart"] == 0 and inside.record["refused"] == 0
    assert outside.law is None and outside.record["apart"] == 1
