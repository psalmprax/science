"""The red team: counterfeit laws, worlds, the search over worlds, and the plain methods it is compared with."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import sympy as sp

from apedp_dyn.redteam import attack, counterfeit, plain_selection, worlds

ROOT = Path(__file__).resolve().parents[1]
CONSTANTS = (0.3, 0.45, 0.7, 1.3, 1.7, 2.3, 2.9, 3.7)
RATIO = (counterfeit.Step("ratio", 0, inner=1.5, scale=2.0), counterfeit.Step("linear", 1, join="add", scale=0.5))


def _entry() -> dict:
    fakes = counterfeit.counterfeits(RATIO, 2, 1.0, 5.0)
    return {"law": str(counterfeit.law_of(RATIO, 2)), "inputs": 2,
            "counterfeits": [{"text": f.text, "change": f.change, "distance": f.distance} for f in fakes]}


# ---------------------------------------------------------------- counterfeits
def test_a_recipe_is_the_law_it_describes() -> None:
    x0, x1 = sp.symbols("x0 x1")
    assert sp.simplify(counterfeit.law_of(RATIO, 2) - (2.0 * x0 / (1.5 + x0) + 0.5 * x1)) == 0
    quotient = (counterfeit.Step("square", 0, scale=1.3), counterfeit.Step("decay", 1, inner=0.7, join="divide", scale=2.3))
    assert sp.simplify(counterfeit.law_of(quotient, 2) - 1.3 * x0 ** 2 / (2.3 + sp.exp(-0.7 * x1))) == 0


def test_every_constant_of_a_recipe_is_a_parameter() -> None:
    expr, params, values = counterfeit.parametrised(RATIO, 2)
    assert len(params) == len(values) == 3 and sorted(values) == [0.5, 1.5, 2.0]
    assert sp.simplify(expr.subs(dict(zip(params, values, strict=True))) - counterfeit.law_of(RATIO, 2)) == 0


def test_a_drawn_recipe_uses_every_input() -> None:
    rng = np.random.default_rng(0)
    for inputs in (1, 2, 3):
        for _ in range(20):
            recipe = counterfeit.draw_recipe(rng, inputs, CONSTANTS)
            assert {step.symbol for step in recipe} == set(range(inputs))
            assert recipe[0].join == "start" and all(step.join in counterfeit.JOINS for step in recipe[1:])


def test_counterfeits_are_other_forms_at_their_stated_distance() -> None:
    fakes = counterfeit.counterfeits(RATIO, 2, 1.0, 5.0)
    assert fakes and fakes == sorted(fakes, key=lambda f: (f.distance, f.text))
    law = counterfeit.law_of(RATIO, 2)
    X = np.random.default_rng(5).uniform(1.0, 5.0, (3000, 2))
    target = counterfeit._values(law, 2, X)
    for fake in fakes:
        expr = sp.sympify(fake.text)
        assert 1e-3 <= fake.distance <= 0.3
        assert expr.free_symbols == law.free_symbols
        assert sp.simplify(expr - law) != 0
        assert counterfeit.distance(expr, 2, X, target) == pytest.approx(fake.distance, rel=0.25)
    assert any("tanh" in f.text for f in fakes)                      # the saturating ratio's closest relatives are found
    assert any("left out" in f.change for f in fakes) or all(f.distance < 0.3 for f in fakes)


def test_fitting_a_recipe_to_its_own_law_returns_the_law() -> None:
    X = np.random.default_rng(1).uniform(1.0, 5.0, (300, 2))
    target = counterfeit._values(counterfeit.law_of(RATIO, 2), 2, X)
    moved = (counterfeit.Step("ratio", 0, inner=1.1, scale=1.6), counterfeit.Step("linear", 1, join="add", scale=0.8))
    fitted = counterfeit.fitted_to(moved, 2, X, target)
    assert fitted is not None and counterfeit.distance(fitted, 2, X, target) < 1e-3


# ---------------------------------------------------------------- worlds
def _world(**changes: object) -> worlds.World:
    entry = _entry()
    settings = {"law": entry["law"], "inputs": 2, "candidates": (entry["counterfeits"][0]["text"],), "seed": 3}
    return worlds.World(**{**settings, **changes})          # type: ignore[arg-type]


def test_a_world_gives_the_same_rows_every_time_and_other_rows_for_another_seed() -> None:
    a, b, c = worlds.rows_of(_world()), worlds.rows_of(_world()), worlds.rows_of(_world(seed=4))
    assert np.array_equal(a.X, b.X) and np.array_equal(a.y, b.y) and np.array_equal(a.X_test, b.X_test)
    assert not np.array_equal(a.X, c.X)
    assert len(a.X) == 1000 and len(a.X_test) == worlds.TEST_ROWS
    assert np.array_equal(a.y_test, worlds.law_values(_world(), a.X_test))      # test targets carry no noise


@pytest.mark.parametrize("sampling,spread", [("uniform", 1.0), ("narrow", 0.3), ("clustered", 0.05), ("correlated", 0.1)])
def test_every_sampling_stays_in_the_box_and_test_rows_are_drawn_the_same_way(sampling: str, spread: float) -> None:
    rows = worlds.rows_of(_world(sampling=sampling, spread=spread))
    for X in (rows.X, rows.X_test):
        assert X.min() >= 1.0 and X.max() <= 5.0
    if sampling == "narrow":
        assert np.all(np.ptp(rows.X, axis=0) <= 0.3 * 4.0 + 1e-9)
        assert np.all(np.abs(rows.X.min(axis=0) - rows.X_test.min(axis=0)) < 0.2)     # the same part of the box
    if sampling == "correlated":
        assert np.corrcoef(rows.X.T)[0, 1] > 0.8
    if sampling == "uniform":
        assert np.all(np.ptp(rows.X, axis=0) > 3.9)


@pytest.mark.parametrize("shape", worlds.NOISE_SHAPES)
def test_noise_has_the_stated_size_whatever_its_shape(shape: str) -> None:
    world = _world(noise=0.05, noise_shape=shape, rows=20000)
    rows = worlds.rows_of(world)
    noise = rows.y - rows.clean
    stated = 0.05 * float(np.sqrt(np.mean(rows.clean ** 2)))
    measured = float(np.sqrt(np.mean(noise ** 2)))
    if shape == "outliers":                                  # the outliers come on top: 2 % of rows at ten times the size
        assert measured == pytest.approx(stated * np.sqrt(1 + 0.02 * 100), rel=0.1)
        assert np.mean(np.abs(noise) > 6 * stated) == pytest.approx(0.02, abs=0.006)
    elif shape == "student":
        assert measured == pytest.approx(stated, rel=0.5)    # the variance of a t with 3 degrees of freedom converges slowly
        assert np.mean(np.abs(noise) > 4 * stated) > 20 * 6.3e-5     # far more large errors than a Gaussian has
    else:
        assert measured == pytest.approx(stated, rel=0.05)
    if shape == "proportional":
        big = np.abs(rows.clean) > np.median(np.abs(rows.clean))
        assert np.std(noise[big]) > np.std(noise[~big])


def test_a_world_without_noise_has_none_and_a_world_survives_its_record() -> None:
    world = _world(noise=0.0, noise_shape="outliers")
    rows = worlds.rows_of(world)
    assert np.array_equal(rows.y, rows.clean)
    assert worlds.World.from_dict(json.loads(json.dumps(world.as_dict()))) == world
    with pytest.raises(ValueError):
        worlds.rows_of(_world(sampling="elsewhere"))


# ---------------------------------------------------------------- the search over worlds
def test_a_drawn_world_and_every_change_of_it_stay_inside_the_arena() -> None:
    entry = _entry()
    by_text = {entry["law"]: entry}
    rng = np.random.default_rng(0)
    for arena in attack.ARENAS.values():
        world = attack.draw(arena, [entry], rng)
        for _ in range(60):
            assert arena.noise[0] <= world.noise <= arena.noise[1] or (world.noise == 0.0 and arena.noise[0] == 0.0)
            assert arena.rows[0] <= world.rows <= arena.rows[1]
            assert world.sampling in arena.samplings and world.noise_shape in arena.noise_shapes
            fakes = [text for text in world.candidates if text != world.law]
            assert world.candidates and len(fakes) <= attack.MAX_COUNTERFEITS
            assert world.truth_listed == (world.law in world.candidates)
            assert set(fakes) <= {fake["text"] for fake in entry["counterfeits"]}
            world = attack.change(world, arena, by_text, rng)


def test_the_measured_arena_is_the_conditions_of_the_preregistered_runs() -> None:
    measured = attack.ARENAS["measured"]
    assert measured.noise == (0.0, 0.01) and measured.rows == (1000, 1000)
    assert measured.samplings == ("uniform",) and measured.noise_shapes == ("gaussian",)


def test_the_worst_worlds_are_kept_once_each() -> None:
    a, b = _world(seed=1), _world(seed=2)
    kept = attack.keep_worst([(0.1, a), (0.5, b), (0.3, a), (0.0, _world(seed=3))], 2)
    assert kept == [(0.5, b), (0.3, a)]


def test_a_climb_moves_towards_the_worse_outcome() -> None:
    entry = _entry()
    arena = attack.ARENAS["noise_level"]
    rng = np.random.default_rng(1)
    start = [attack.draw(arena, [entry], rng) for _ in range(4)]

    def badness(world: worlds.World) -> float:        # a stand-in: more noise is worse, and worst without the law
        return world.noise * (1.0 if world.truth_listed else 2.0)

    held = attack.climb(start, badness, arena, {entry["law"]: entry}, rounds=25, seed=2)
    assert held[0][0] >= max(badness(world) for world in start)
    assert held[0][0] > 0.15 and not held[0][1].truth_listed
    assert [score for score, _world_ in held] == sorted((score for score, _world_ in held), reverse=True)


# ---------------------------------------------------------------- plain selection
def _shape(text: str) -> str:
    expr = sp.sympify(text)
    return str(expr.xreplace(dict.fromkeys(expr.atoms(sp.Float), sp.Symbol("C"))))


def test_best_fit_takes_the_law_when_it_is_offered_and_a_counterfeit_when_it_is_not() -> None:
    entry = _entry()
    rows = worlds.rows_of(_world(noise=0.01))
    names = ["x0", "x1"]
    fakes = [fake["text"] for fake in entry["counterfeits"][-2:]]           # the two farthest
    with_law = plain_selection.best_fit([*fakes, entry["law"]], names, rows.X, rows.y)
    assert with_law.law is not None and _shape(with_law.law) == _shape(entry["law"])
    without = plain_selection.best_fit(fakes, names, rows.X, rows.y)        # it cannot say "none of these"
    assert without.law is not None and _shape(without.law) in {_shape(fake) for fake in fakes}
    assert plain_selection.best_fit(["1.5", "x9 + 1", "not a ( formula"], names, rows.X, rows.y).law is None


def test_validation_refuses_a_law_that_fails_on_fresh_rows_and_passes_one_that_does_not() -> None:
    rng = np.random.default_rng(0)
    X = rng.uniform(1.0, 2.0, (300, 1))
    y = 2.0 * X[:, 0] + 0.01 * rng.standard_normal(300)
    fresh_X = rng.uniform(1.0, 2.0, (30, 1))
    same = plain_selection.validated(["2.0*x0"], ["x0"], X, y, fresh_X, 2.0 * fresh_X[:, 0] + 0.01 * rng.standard_normal(30))
    assert same.law is not None and same.p_value is not None and same.p_value > plain_selection.ALPHA
    moved = plain_selection.validated(["2.0*x0"], ["x0"], X, y, fresh_X, 2.0 * fresh_X[:, 0] + 0.5)
    assert moved.law is None and moved.p_value is not None and moved.p_value < plain_selection.ALPHA


# ---------------------------------------------------------------- the campaign script
@pytest.fixture(scope="module")
def campaign():
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_red_team

    return run_red_team


def test_the_route_is_scored_by_the_benchmarks_standard(campaign) -> None:
    world = _world()
    rows = worlds.rows_of(world)
    truth = sp.sympify(world.law)
    names = ["x0", "x1"]
    right = campaign._scored(world.law, truth, names, rows)
    assert right["reported"] and right["symbolic"] and not right["wrong"] and right["badness"] == 0.0
    nothing = campaign._scored(None, truth, names, rows)
    assert not nothing["reported"] and not nothing["wrong"] and nothing["badness"] == 0.0
    far = campaign._scored("0.9*x0 + 0.1*x1", truth, names, rows)
    assert far["reported"] and far["wrong"] and 0.001 < far["badness"] <= 1.0
    near = campaign._scored(_entry()["counterfeits"][0]["text"], truth, names, rows)
    assert near["reported"] and not near["symbolic"] and near["badness"] == pytest.approx(1.0 - near["test_r2"])
    assert near["wrong"] == (near["test_r2"] < 0.999)


def test_one_world_end_to_end_and_the_summary(campaign) -> None:
    entry = _entry()
    world = worlds.World(law=entry["law"], inputs=2, candidates=(entry["law"], entry["counterfeits"][-1]["text"]),
                         truth_listed=True, rows=300, noise=0.01, seed=11)
    rows = campaign.evaluate([world], cores=1)
    assert rows[0]["status"] == "ok" and rows[0]["world"] == world.as_dict()
    route = rows[0]["route"]
    assert route["action"] in ("confirmed", "equivalent_class", "unchallenged", "widen_search")
    assert not route["wrong"]                                    # the law was offered beside a far counterfeit
    assert rows[0]["best_fit"]["reported"] and campaign.badness(rows[0]) == route["badness"]
    tagged = [{**rows[0], "arena": "few_rows", "stage": "survey", "shard": 0}]
    summary = campaign.summarise(tagged)
    tally = summary["survey"]["few_rows"]["route"]
    assert tally["worlds"] == 1 and tally["law_offered"] == 1 and tally["wrong"] == 0
    assert summary["survey"]["measured"]["route"]["worlds"] == 0


def test_the_survey_shares_cover_every_arena(campaign) -> None:
    assert set(campaign.SURVEY_SHARES) == set(attack.ARENAS)
    assert sum(campaign.SURVEY_SHARES.values()) == pytest.approx(1.0)
