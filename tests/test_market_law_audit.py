"""The audit of published laws of asset returns: its numbers, its verdict rule, its simulated worlds.

What is checked here is the machinery, on inputs whose answer is known by construction. How often
the verdicts are right on panels of the real one's shape is measured by
``scripts/run_law_audit_finance.py power`` and recorded with the preregistration.
"""
from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from apedp_dyn.market import law_audit as la
from apedp_dyn.market import law_audit_worlds as lw

ROOT = Path(__file__).resolve().parents[1]


def _runner():
    name = "run_law_audit_finance"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "run_law_audit_finance.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _small_panel(world: lw.World, seed: int = 3) -> la.Panel:
    dates, assets, mask = _runner().shape_of_the_panel(None)
    return lw.simulate(world, dates, assets, mask, np.random.default_rng(seed))


# --- the verdict rule ---------------------------------------------------------------------------

def test_an_interval_wholly_outside_the_published_set_fails_the_law():
    assert la.verdict(5.2, 6.0, (2.0, 5.0)) == la.FAILS
    assert la.verdict(1.0, 1.9, (2.0, 5.0)) == la.FAILS
    assert la.verdict(1.02, 1.08, (1.0, 1.0)) == la.FAILS
    assert la.verdict(0.01, 0.05, (-la.INF, 0.0)) == la.FAILS


def test_an_interval_wholly_inside_survives_and_one_that_straddles_is_not_rejected():
    assert la.verdict(2.4, 3.1, (2.0, 5.0)) == la.SURVIVES
    assert la.verdict(-0.04, -0.01, (-la.INF, 0.0)) == la.SURVIVES
    assert la.verdict(1.8, 2.6, (2.0, 5.0)) == la.NOT_REJECTED
    assert la.verdict(-0.02, 0.01, (-la.INF, 0.0)) == la.NOT_REJECTED


def test_a_point_claim_can_be_failed_but_never_survives():
    # an interval cannot lie inside a single value: agreement with it is "not rejected", with the width
    assert la.verdict(0.97, 1.03, (1.0, 1.0)) == la.NOT_REJECTED
    assert la.verdict(1.0, 1.0, (1.0, 1.0)) == la.NOT_REJECTED


def test_no_interval_is_no_verdict_against_the_law():
    assert la.verdict(float("nan"), float("nan"), (0.2, 0.4)) == la.NOT_REJECTED


def test_every_claim_reads_a_measured_number_and_every_number_has_a_widening_factor():
    assert set(la.NUMBER_OF) == set(la.PUBLISHED)
    assert set(la.NUMBER_OF.values()) <= set(la.NUMBERS)
    assert set(la.STRETCH) == set(la.NUMBER_OF.values())
    assert all(factor >= 1.0 for factor in la.STRETCH.values())      # an interval is never narrowed


def test_the_level_is_the_referees_own_threshold():
    from apedp_dyn.conf.canonical_spec import DIEBOLD_MARIANO_P_VALUE_THRESHOLD

    assert la.LEVEL == DIEBOLD_MARIANO_P_VALUE_THRESHOLD


# --- blocks and sums ----------------------------------------------------------------------------

def test_a_short_last_block_joins_the_one_before():
    index, blocks = la.block_index(200, 90)        # 90 + 90 + 20
    assert blocks == 2 and (index == 1).sum() == 110
    index, blocks = la.block_index(230, 90)        # 90 + 90 + 50: long enough to stand alone
    assert blocks == 3 and (index == 2).sum() == 50
    index, blocks = la.block_index(30, 90)
    assert blocks == 1 and (index == 0).all()


def test_lagged_pairs_never_straddle_two_blocks():
    days = 10
    x = np.arange(1.0, days + 1.0)[:, None]        # one asset: 1, 2, ..., 10
    block, blocks = la.block_index(days, 5)        # days 0-4 and 5-9
    sums, counts = la.lag_sums(x, x, [0, 2], block, blocks)
    assert counts[:, 0].tolist() == [5, 5]
    # lag 2 inside the first block: (1,3), (2,4), (3,5); inside the second: (6,8), (7,9), (8,10)
    assert counts[:, 1].tolist() == [3, 3]
    assert sums[:, 1].tolist() == [1 * 3 + 2 * 4 + 3 * 5, 6 * 8 + 7 * 9 + 8 * 10]
    half, pairs = la.variogram_sums(x, [2], block, blocks)
    assert pairs[:, 0].tolist() == [3, 3] and half[:, 0].tolist() == [6.0, 6.0]


def test_a_pair_with_a_missing_day_is_not_counted():
    x = np.array([[1.0], [np.nan], [3.0], [4.0]])
    block, blocks = la.block_index(4, 4)
    sums, counts = la.lag_sums(x, x, [1], block, blocks)
    assert counts[0, 0] == 1 and sums[0, 0] == 12.0


def test_the_variogram_does_not_depend_on_the_level_of_the_series():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((120, 4))
    block, blocks = la.block_index(120, 40)
    plain = la.variogram_sums(x, [1, 3, 7], block, blocks)[0]
    lifted = la.variogram_sums(x + 50.0, [1, 3, 7], block, blocks)[0]
    assert np.allclose(plain, lifted)
    # ... while the usual lagged product does
    assert not np.allclose(la.lag_sums(x, x, [1], block, blocks)[0], la.lag_sums(x + 50.0, x + 50.0, [1], block, blocks)[0])


def test_ranks_share_ties_and_skip_missing_cells():
    x = np.array([[0.5, np.nan], [0.1, 2.0], [0.5, 1.0], [np.nan, 2.0]])
    ranks = la._ranks(x)
    assert ranks[:, 0].tolist()[:3] == [2.5, 1.0, 2.5] and np.isnan(ranks[3, 0])
    assert np.isnan(ranks[0, 1]) and ranks[1:, 1].tolist() == [2.5, 1.0, 2.5]


def test_the_per_asset_quantile_matches_numpy_and_survives_an_empty_column():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((200, 3))
    x[:50, 1] = np.nan
    x[:, 2] = np.nan
    got = la._quantile(x, 0.95)
    assert np.allclose(got[:2], np.nanquantile(x[:, :2], 0.95, axis=0)) and np.isnan(got[2])


# --- preparation --------------------------------------------------------------------------------

def _panel_from_returns(returns: np.ndarray, volume: np.ndarray | None = None) -> la.Panel:
    days, count = returns.shape
    close = 100.0 * np.exp(np.cumsum(np.where(np.isfinite(returns), returns, 0.0), axis=0))
    close = np.where(np.isfinite(returns), close, np.nan)
    dates = np.arange(np.datetime64("2021-01-01"), np.datetime64("2021-01-01") + np.timedelta64(days, "D"))
    volume = np.full_like(close, 1000.0) if volume is None else volume
    return la.Panel(dates, tuple(f"A{i}" for i in range(count)), close, np.where(np.isfinite(close), volume, np.nan))


def test_preparation_drops_the_first_returns_short_assets_and_returns_across_a_gap():
    rng = np.random.default_rng(2)
    days = la.MIN_DAYS + 60
    returns = 0.02 * rng.standard_normal((days, 3))
    returns[: days - 100, 1] = np.nan              # an asset with too short a history
    returns[150, 2] = np.nan                       # one missing bar in an otherwise full asset
    prepared = la.prepare(_panel_from_returns(returns))
    assert prepared.assets == ("A0", "A2")
    first = np.flatnonzero(np.isfinite(prepared.g[:, 0]))[0]
    assert first == 1 + la.SKIP_FIRST              # day 0 has no return; the next SKIP_FIRST are dropped
    # no return on the day of the gap nor on the day after it: neither has two consecutive bars
    assert np.isnan(prepared.g[150, 1]) and np.isnan(prepared.g[151, 1]) and np.isfinite(prepared.g[152, 1])
    assert prepared.asset_days == int(np.isfinite(prepared.g).sum())


def test_the_tail_is_each_assets_own_largest_returns():
    rng = np.random.default_rng(3)
    returns = 0.02 * rng.standard_normal((1000, 2)) * np.array([[1.0, 7.0]])      # very different scales
    prepared = la.prepare(_panel_from_returns(returns))
    share = np.isfinite(prepared.excess).mean(axis=0) / np.isfinite(prepared.g).mean(axis=0)
    assert np.allclose(share, 1.0 - la.TAIL_QUANTILE, atol=0.002)
    assert np.nanmin(prepared.excess) > 1.0


def test_relative_volume_uses_only_earlier_days():
    rng = np.random.default_rng(4)
    log_volume = rng.standard_normal((120, 2))
    before = la.relative_volume(log_volume)
    changed = log_volume.copy()
    changed[80:] += 3.0                             # nothing before day 80 may notice
    after = la.relative_volume(changed)
    assert np.allclose(before[:80], after[:80], equal_nan=True)
    assert np.allclose(after[80], before[80] + 3.0)  # day 80 itself moves, its reference does not
    assert np.isnan(before[: la.VOLUME_MIN]).all() and np.isfinite(before[la.VOLUME_MIN]).all()


# --- the numbers, on inputs whose answer is known -----------------------------------------------------

@pytest.mark.parametrize("beta", [0.1, 0.3, 0.7, 1.5])
def test_the_decay_exponent_of_an_exact_power_law_is_recovered(beta):
    lag = np.arange(1.0, la.MEMORY_LAGS + 1.0)
    variogram = 1.7 - 0.2 * lag ** -beta           # variance times (1 - autocorrelation)
    assert la.decay_exponent(variogram)[0] == pytest.approx(beta, abs=0.006)


def test_a_variogram_that_does_not_rise_has_no_decay_exponent():
    lag = np.arange(1.0, la.MEMORY_LAGS + 1.0)
    assert np.isnan(la.decay_exponent(np.ones(la.MEMORY_LAGS))[0])
    assert np.isnan(la.decay_exponent(1.0 - 0.1 * np.log(lag))[0])


def test_each_family_of_shapes_fits_its_own_curve_best():
    lag = np.arange(1.0, la.MEMORY_LAGS + 1.0)
    powers, exponentials = la.power_shapes(), la.exponential_shapes()
    power_law = 1.0 - 0.3 * lag ** -0.3
    exponential = 1.0 - 0.3 * 0.5 ** (lag / 6.0)
    assert la.best_shape(power_law, powers)[1][0] == pytest.approx(1.0, abs=1e-9)
    assert la.best_shape(power_law, exponentials)[1][0] < 0.99
    index, share = la.best_shape(exponential, exponentials)
    assert share[0] == pytest.approx(1.0, abs=1e-4) and la.HALFLIFE_GRID[index[0]] == pytest.approx(6.0, rel=0.02)
    assert la.best_shape(exponential, powers)[1][0] < 0.99


def test_the_tail_index_of_exact_pareto_tails_is_recovered():
    rng = np.random.default_rng(5)
    alpha = 2.7
    size = rng.pareto(alpha, (2000, 12)) + 1.0
    returns = 0.01 * size * np.where(rng.random(size.shape) < 0.5, 1.0, -1.0)
    measured = la.measure(la.prepare(_panel_from_returns(returns)), resamples=200)
    cells = measured["tail_cells"]
    assert cells == pytest.approx((1.0 - la.TAIL_QUANTILE) * returns[1 + la.SKIP_FIRST:].size, rel=0.02)
    assert measured["tail"]["estimate"] == pytest.approx(alpha, abs=4.0 * alpha / np.sqrt(cells))
    low, high = measured["tail"]["percentile"]
    assert low < measured["tail"]["estimate"] < high


def test_independent_returns_have_a_variance_ratio_of_one_and_no_dependence():
    rng = np.random.default_rng(6)
    returns = 0.02 * rng.standard_normal((1500, 20))
    volume = np.exp(rng.standard_normal(returns.shape))
    audit = la.audit(_panel_from_returns(returns, volume), resamples=400)
    readings = audit["readings"]
    assert readings["random"]["verdict"] == la.NOT_REJECTED and abs(readings["random"]["estimate"] - 1.0) < 0.05
    assert readings["independent"]["verdict"] == la.NOT_REJECTED
    assert readings["leverage"]["verdict"] == la.NOT_REJECTED and readings["volume"]["verdict"] == la.NOT_REJECTED
    assert readings["tail"]["verdict"] == la.FAILS            # Gaussian returns are not heavy-tailed
    assert readings["tail"]["estimate"] > 5.0


def test_autocorrelated_returns_move_the_variance_ratio():
    rng = np.random.default_rng(7)
    shock = 0.02 * rng.standard_normal((1501, 20))
    for theta, side in ((0.25, 1.0), (-0.25, -1.0)):
        audit = la.audit(_panel_from_returns(shock[1:] + theta * shock[:-1]), resamples=400)
        reading = audit["readings"]["random"]
        assert reading["verdict"] == la.FAILS and side * (reading["estimate"] - 1.0) > 0.1


def test_widening_moves_each_side_away_from_the_estimate_and_never_inwards():
    assert la.stretched(1.0, 0.8, 1.4, 2.0) == pytest.approx((0.6, 1.8))
    assert la.stretched(1.0, 0.8, 1.4, 1.0) == pytest.approx((0.8, 1.4))
    assert la.stretched(1.0, 1.1, 1.4, 2.0) == pytest.approx((1.1, 1.8))      # the low side lies beyond the estimate
    assert all(np.isnan(v) for v in la.stretched(1.0, float("nan"), 1.4, 2.0))


def test_a_reading_carries_the_widened_interval_and_the_factor_used():
    panel = _small_panel(lw.World("all_hold"))
    plain = la.audit(panel, resamples=300)
    wide = la.audit(panel, resamples=300, stretch=dict.fromkeys(la.STRETCH, 3.0))
    for claim in la.PUBLISHED:
        a, b = plain["readings"][claim], wide["readings"][claim]
        assert b["stretch"] == 3.0 and a["estimate"] == b["estimate"]
        assert b["low"] <= a["low"] and b["high"] >= a["high"]
    assert wide["settings"]["stretch"]["tail"] == 3.0


# --- simulated worlds ----------------------------------------------------------------------------

def test_every_world_has_a_valid_volatility_autocorrelation():
    # circulant embedding is exact only if the embedded spectrum is non-negative
    for world in lw.WORLDS:
        rho = lw.volatility_autocorrelation(world, 1826)
        assert rho[0] == 1.0
        spectrum = np.fft.fft(np.concatenate([rho, rho[-2:0:-1]])).real
        assert spectrum.min() > -1e-9, world.name


def test_the_power_law_worlds_decay_exactly_as_a_power_law():
    for beta in (0.1, 0.2, 0.3, 0.4, 0.7):
        rho = lw.volatility_autocorrelation(lw.World("w", beta=beta), 60)
        lag = np.arange(1.0, 60.0)
        assert np.allclose(rho[1:], lw.POWER_LEVEL * lag ** -beta)


def test_simulated_series_have_the_requested_autocorrelation():
    rng = np.random.default_rng(8)
    rho = lw.volatility_autocorrelation(lw.World("w", beta=0.4), 400)
    series = lw.gaussian_series(rho, 3000, rng)
    assert series.var() == pytest.approx(1.0, abs=0.03)
    for lag in (1, 5, 40):
        assert float(np.mean(series[:-lag] * series[lag:])) == pytest.approx(rho[lag], abs=0.03)


def test_surprises_are_white_and_of_unit_variance():
    rng = np.random.default_rng(9)
    rho = lw.volatility_autocorrelation(lw.World("w"), 300)
    series = lw.gaussian_series(rho, 200, rng)
    surprise = np.column_stack([lw.surprises(rho, series[:, j]) for j in range(200)])
    assert surprise.var() == pytest.approx(1.0, abs=0.03)
    for lag in (1, 2, 10):
        assert abs(float(np.mean(surprise[:-lag] * surprise[lag:]))) < 0.02
    # the series is strongly autocorrelated; what is new in it each day is not
    assert float(np.mean(series[:-1] * series[1:])) > 0.5


def test_a_world_is_reproducible_and_trades_only_where_the_asset_is_present():
    dates, assets, mask = _runner().shape_of_the_panel(None)
    world = lw.World("all_hold")
    a = lw.simulate(world, dates, assets, mask, np.random.default_rng(11))
    b = lw.simulate(world, dates, assets, mask, np.random.default_rng(11))
    assert np.array_equal(a.close, b.close, equal_nan=True) and np.array_equal(a.volume, b.volume, equal_nan=True)
    assert np.isnan(a.close[~mask]).all() and np.isfinite(a.close[mask]).all() and (a.close[mask] > 0).all()
    assert np.isnan(a.volume[~mask]).all() and (a.volume[mask] > 0).all()


def test_the_leverage_dial_sets_the_sign_of_the_dependence_and_nothing_else():
    readings = {}
    for name, dial in (("with", 0.9), ("without", 0.0), ("inverse", -0.9)):
        measured = la.measure(la.prepare(_small_panel(lw.World(name, leverage=dial), seed=12)), resamples=200)
        readings[name] = measured
    assert readings["with"]["leverage"]["estimate"] < readings["without"]["leverage"]["estimate"] \
        < readings["inverse"]["leverage"]["estimate"]
    assert readings["with"]["leverage"]["percentile"][1] < 0 < readings["inverse"]["leverage"]["percentile"][0]
    # the same seed gives the same volatility path whatever the dial: the memory number does not move much
    assert abs(readings["with"]["independent"]["estimate"] - readings["inverse"]["independent"]["estimate"]) < 0.02


def test_the_controls_behave_in_a_world_where_the_laws_hold():
    audit = la.audit(_small_panel(lw.World("all_hold"), seed=13), resamples=600)
    readings = audit["readings"]
    assert readings["independent"]["verdict"] == la.FAILS and readings["independent"]["estimate"] > 0.05
    assert readings["volume"]["verdict"] == la.SURVIVES
    assert readings["tail"]["verdict"] != la.FAILS and readings["memory"]["verdict"] != la.FAILS
    assert audit["assets"] == 60 and audit["asset_days"] > 50_000
    assert set(audit["numbers"]) >= set(la.NUMBERS) | {"curves", "blocks", "years", "tail_cells"}
    assert len(audit["numbers"]["curves"]["variogram"]) == la.MEMORY_LAGS


def test_the_controls_turn_round_in_a_world_of_independent_returns():
    world = lw.World("constant_volatility", memory="none", vol_scale=0.0, leverage=0.0, volume_link=0.0)
    readings = la.audit(_small_panel(world, seed=14), resamples=600)["readings"]
    assert readings["independent"]["verdict"] == la.NOT_REJECTED
    assert readings["volume"]["verdict"] == la.NOT_REJECTED
    assert readings["leverage"]["verdict"] == la.NOT_REJECTED


def test_the_form_tests_tell_a_power_law_tail_from_an_exponential_one():
    r = _runner()
    heavy = la.form_tests(*r.split(_small_panel(lw.World("all_hold"), seed=15)), resamples=400)
    assert heavy["tail"]["favours"] == la.PUBLISHED_FORM and heavy["tail"]["estimate"] > 0
    assert set(heavy["memory"]) >= {"estimate", "low", "high", "favours", "beta_fit", "halflife_fit"}
    # Laplace returns: an exponential tail exactly
    rng = np.random.default_rng(16)
    returns = 0.02 * rng.laplace(size=(1500, 30))
    panel = _panel_from_returns(returns)
    light = la.form_tests(panel.between("2021-01-01", "2023-01-31"), panel.between("2023-02-01", "2025-12-31"),
                          resamples=400)
    assert light["tail"]["favours"] == la.RIVAL_FORM and light["tail"]["estimate"] < 0


# --- the runner ----------------------------------------------------------------------------------

def test_the_needed_widening_is_the_distance_to_the_centre_in_half_widths():
    r = _runner()
    values = np.array([[1.0, 0.5, 1.5],      # centre 2 lies two half-widths above
                       [3.0, 2.0, 3.5],      # one half-width below
                       [2.0, 1.9, 2.1],      # at the estimate
                       [1.0, 1.0, 1.0]])     # no width: cannot be reached
    needed = r.needed_stretch(values, 2.0)
    assert needed[:3].tolist() == [2.0, 1.0, 0.0] and np.isnan(needed[3])


def test_the_widening_factor_is_the_largest_need_rounded_up_and_never_below_one():
    r = _runner()
    table = {"w1": {"numbers": {n: {"stretch_needed": 0.7} for n in la.NUMBERS}},
             "w2": {"numbers": {n: {"stretch_needed": 1.41 if n == "tail" else 0.9} for n in la.NUMBERS}}}
    factors = r.stretch_from(table)
    assert factors["tail"] == 1.5 and factors["random"] == 1.0
    assert set(factors) == set(la.STRETCH)


def test_the_calibration_table_measures_coverage_on_rows_it_is_given():
    r = _runner()
    rng = np.random.default_rng(17)
    estimates = rng.standard_normal(2000)
    rows = [{"world": "w", "replicate": i,
             "numbers": {n: [float(e), float(e - 1.0), float(e + 1.0)] for n in la.NUMBERS}}
            for i, e in enumerate(estimates)]
    cell = r.calibration_table(rows, level=0.05)["w"]["numbers"]["tail"]
    # a half-width of one standard deviation covers about 68 % and needs widening to about 1.96
    assert cell["covered_unwidened"] == pytest.approx(0.683, abs=0.03)
    assert cell["stretch_needed"] == pytest.approx(1.96, abs=0.12)


def test_the_committed_widening_factors_are_the_calibrated_ones():
    r = _runner()
    record = json.loads((ROOT / r.POWER_DIR / "calibration.json").read_text(encoding="utf-8"))
    assert dict(la.STRETCH) == record["stretch"] == r.stretch_from(record["table"])
    assert sorted(record["table"]) == sorted(r.CALIBRATION_WORLDS)
    assert all(cell["replicates"] == r.CALIBRATION_REPLICATES for cell in record["table"].values())
    assert record["level"] == la.LEVEL


def test_the_committed_power_summary_was_made_with_the_frozen_settings():
    r = _runner()
    summary = json.loads((ROOT / r.POWER_DIR / "summary.json").read_text(encoding="utf-8"))
    assert summary["settings"] == json.loads(json.dumps(la.audit_settings()))
    assert sorted(summary["summary"]) == sorted(w.name for w in lw.WORLDS)
    assert all(cell["replicates"] == r.POWER_REPLICATES for cell in summary["summary"].values())
    assert summary["worlds"] == {w.name: json.loads(json.dumps(w.__dict__)) for w in lw.WORLDS}


def test_the_committed_summaries_are_what_the_committed_rows_give(tmp_path, capsys):
    # every simulated panel's row is committed beside the two summaries the preregistration quotes
    r = _runner()
    folder = ROOT / r.POWER_DIR
    for packed, stem, command, summary in (("calibration_rows.jsonl.gz", "calibrate", "stretch", "calibration.json"),
                                           ("power_rows.jsonl.gz", "power", "summarise", "summary.json")):
        out = tmp_path / stem
        out.mkdir()
        (out / f"{stem}.jsonl").write_bytes(gzip.decompress((folder / packed).read_bytes()))
        assert r.main([command, "--out", str(out)]) == 0
        assert json.loads((out / summary).read_text(encoding="utf-8")) \
            == json.loads((folder / summary).read_text(encoding="utf-8")), summary
    capsys.readouterr()


def test_the_commands_run_end_to_end_on_a_small_stand_in_panel(tmp_path, capsys):
    r = _runner()
    assert r.main(["calibrate", "--out", str(tmp_path), "--replicates", "2", "--worlds", "all_hold"]) == 0
    assert r.main(["stretch", "--out", str(tmp_path)]) == 0
    record = json.loads((tmp_path / "calibration.json").read_text(encoding="utf-8"))
    assert record["rows"] == 2 and set(record["stretch"]) == set(la.STRETCH)
    assert r.main(["power", "--out", str(tmp_path), "--replicates", "1", "--worlds", "all_hold,no_volume_link",
                   "--shard", "0/1"]) == 0
    assert r.main(["summarise", "--out", str(tmp_path)]) == 0
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert set(summary["summary"]) == {"all_hold", "no_volume_link"}
    cell = summary["summary"]["all_hold"]["readings"]["volume"]
    assert cell[la.FAILS] + cell[la.SURVIVES] + cell[la.NOT_REJECTED] == 1
    assert "stretch:" in capsys.readouterr().out


def test_shards_partition_the_jobs():
    r = _runner()
    jobs = list(range(11))
    parts = [r.shard_of(jobs, f"{k}/3", "power")[0] for k in range(3)]
    assert sorted(job for part in parts for job in part) == jobs
    assert r.shard_of(jobs, None, "power") == (jobs, "power.jsonl")


def test_the_run_refuses_a_panel_that_is_not_the_preregistered_one(tmp_path, monkeypatch):
    r = _runner()
    monkeypatch.setattr(r, "check_frozen", lambda: None)
    (tmp_path / r.PANEL_FILE).write_bytes(gzip.compress(b"observation_date,X_CLOSE,X_VOL\n2021-01-01,1,1\n"))
    with pytest.raises(SystemExit, match="preregistered panel"):
        r.run_once(tmp_path, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def _write_panel_file(path: Path, panel: la.Panel) -> None:
    """A panel in the builder's own layout: one row per calendar day, four columns per asset, empty for no bar."""
    header = ["observation_date"]
    for asset in panel.assets:
        header += [f"{asset}_CLOSE", f"{asset}_VOL", f"{asset}_TAKER_BUY", f"{asset}_OFI"]
    lines = [",".join(header)]
    for day, date in enumerate(panel.dates):
        row = [str(date)]
        for j in range(len(panel.assets)):
            close, volume = panel.close[day, j], panel.volume[day, j]
            row += ["", "", "", ""] if np.isnan(close) else [repr(float(close)), repr(float(volume)),
                                                              repr(float(volume) / 2.0), "0.0"]
        lines.append(",".join(row))
    path.write_bytes(gzip.compress(("\n".join(lines) + "\n").encode("utf-8")))


def test_the_run_completes_on_a_panel_with_the_blemishes_of_a_real_one(tmp_path, monkeypatch):
    r = _runner()
    dates = np.arange(np.datetime64(r.PERIOD[0]), np.datetime64(r.PERIOD[1]) + np.timedelta64(1, "D"))
    names = [f"A{i:02d}" for i in range(34)] + ["DEAD", "GAP", "LATE", "SHORT", "STALE", "币"]
    mask = np.ones((len(dates), len(names)), dtype=bool)
    mask[:700, :8] = False                               # listed later
    mask[1000:, names.index("DEAD")] = False             # stopped trading in 2023
    mask[900:930, names.index("GAP")] = False            # a month without bars
    mask[:1700, names.index("LATE")] = False             # listed in the last months: too short to enter
    mask[100:, names.index("SHORT")] = False             # a hundred bars and gone
    panel = lw.simulate(lw.World("all_hold"), dates, tuple(names), mask, np.random.default_rng(21))
    stale = names.index("STALE")
    panel.close[400:460, stale] = panel.close[399, stale]      # no trades: the price stands still ...
    panel.volume[400:460, stale] = 0.0                         # ... and the volume is zero
    _write_panel_file(tmp_path / r.PANEL_FILE, panel)
    monkeypatch.setattr(r, "check_frozen", lambda: None)
    monkeypatch.setattr(r, "PANEL_SHA256", hashlib.sha256((tmp_path / r.PANEL_FILE).read_bytes()).hexdigest())
    assert r.run_once(tmp_path, tmp_path / "out") == 0
    results = json.loads((tmp_path / "out" / "results.json").read_text(encoding="utf-8"))
    assert results["panel"]["days"] == len(dates) and results["panel"]["assets_in_file"] == len(names)
    assert results["audit"]["assets"] == len(names) - 2            # LATE and SHORT are too short
    readings = results["audit"]["readings"]
    assert set(readings) == set(la.PUBLISHED)
    assert all(reading["verdict"] in (la.FAILS, la.SURVIVES, la.NOT_REJECTED) for reading in readings.values())
    assert all(np.isfinite([reading["estimate"], reading["low"], reading["high"]]).all() for reading in readings.values())
    assert readings["independent"]["verdict"] == la.FAILS and readings["volume"]["verdict"] == la.SURVIVES
    assert set(results["forms"]) == {"tail", "memory"}
    assert set(results["subgroups"]) == {"fit_period", "test_period", "stopped_trading", "still_trading",
                                         "larger_half", "smaller_half"}
    assert results["subgroups"]["stopped_trading"]["assets"] == 1          # DEAD; SHORT is too short to count
    assert results["subgroups"]["still_trading"]["assets"] == len(names) - 3
    assert results["subgroups"]["larger_half"]["assets"] + results["subgroups"]["smaller_half"]["assets"] \
        == results["audit"]["assets"]
    assert len(results["audit"]["numbers"]["tail"]["yearly"]) == 5


def test_the_run_refuses_a_panel_whose_rows_are_not_consecutive_days(tmp_path, monkeypatch):
    r = _runner()
    text = "observation_date,X_CLOSE,X_VOL\n2021-01-01,1.0,1\n2021-01-02,1.1,1\n2021-01-04,1.2,1\n"
    (tmp_path / r.PANEL_FILE).write_bytes(gzip.compress(text.encode()))
    monkeypatch.setattr(r, "check_frozen", lambda: None)
    monkeypatch.setattr(r, "PANEL_SHA256", hashlib.sha256((tmp_path / r.PANEL_FILE).read_bytes()).hexdigest())
    with pytest.raises(SystemExit, match="consecutive calendar days"):
        r.run_once(tmp_path, tmp_path / "out")


def test_the_run_refuses_to_start_if_a_frozen_file_has_changed(tmp_path, monkeypatch):
    r = _runner()
    digest = tmp_path / "DIGEST.json"
    recorded = dict(r.frozen_digests())
    recorded[r.FROZEN[1]] = "0" * 64
    digest.write_text(json.dumps({"sha256": recorded}), encoding="utf-8")
    monkeypatch.setattr(r, "DIGEST", digest)
    with pytest.raises(SystemExit, match="frozen files differ"):
        r.run_once(tmp_path, tmp_path / "out")
    monkeypatch.setattr(r, "DIGEST", tmp_path / "missing.json")
    with pytest.raises(SystemExit, match="not frozen"):
        r.check_frozen()


def test_the_frozen_files_match_the_recorded_digest():
    r = _runner()
    recorded = json.loads(r.DIGEST.read_text(encoding="utf-8"))
    assert recorded["preregistration"] == r.PREREGISTRATION_ID
    assert recorded["sha256"] == r.frozen_digests()
    r.check_frozen()


def test_nothing_in_the_audit_names_the_holdout_panel():
    # the audit reads the development panel only; the holdout file stays unopened for a later claim
    r = _runner()
    for path in ("scripts/run_law_audit_finance.py", "apedp_dyn/market/law_audit.py",
                 "apedp_dyn/market/law_audit_worlds.py", ".github/workflows/law-audit-finance.yml"):
        assert "panel_holdout" not in (ROOT / path).read_text(encoding="utf-8"), path
    assert r.PANEL_FILE == "panel_development.csv.gz" and r.PERIOD[1] < "2026-01-01"


def test_the_panel_loader_reads_closes_and_volumes_by_asset(tmp_path):
    text = ("observation_date,BBB_CLOSE,BBB_VOL,BBB_OFI,AAA_CLOSE,AAA_VOL\n"
            "2021-01-01,1.0,10,0.1,,\n2021-01-02,1.1,11,0.2,5.0,50\n")
    path = tmp_path / "panel_development.csv.gz"
    path.write_bytes(gzip.compress(text.encode()))
    panel = la.load_panel(path)
    assert panel.assets == ("AAA", "BBB") and panel.dates[1] == np.datetime64("2021-01-02")
    assert np.isnan(panel.close[0, 0]) and panel.close[1].tolist() == [5.0, 1.1] and panel.volume[1].tolist() == [50.0, 11.0]
    assert panel.between("2021-01-02", "2021-01-02").close.shape == (1, 2)
    assert panel.columns(np.array([False, True])).assets == ("BBB",)
