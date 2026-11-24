"""The second batch of audited regularities: its numbers, its worlds, its runner.

What is checked here is the machinery, on inputs whose answer is known by construction. How often
the verdicts are right on panels of the real one's shape is measured by
``scripts/run_law_audit_finance_more.py power`` and recorded with the preregistration.
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
from apedp_dyn.market import law_audit_more as lm
from apedp_dyn.market import law_audit_more_worlds as mw
from apedp_dyn.market import law_audit_worlds as lw

ROOT = Path(__file__).resolve().parents[1]
NO_VOLATILITY = {"memory": "none", "vol_scale": 0.0, "leverage": 0.0, "volume_link": 0.0}


def _runner():
    for name in ("run_law_audit_finance", "run_law_audit_finance_more"):
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
    return sys.modules["run_law_audit_finance_more"]


def _flows(world: mw.MoreWorld, seed: int = 3) -> lm.FlowPanel:
    dates, assets, mask = _runner().first.shape_of_the_panel(None)
    return mw.simulate(world, dates, assets, mask, np.random.default_rng(seed))


def _numbers(world: mw.MoreWorld, seed: int = 3) -> dict[str, float]:
    measured = lm.measure(lm.prepare(_flows(world, seed)), resamples=50)
    return {n: measured[n]["estimate"] for n in lm.NUMBERS}


# --- the pieces ------------------------------------------------------------------------------------

def test_a_window_sum_sits_under_its_first_day_and_needs_every_day():
    x = np.arange(1.0, 9.0)[:, None] * np.array([[1.0, 10.0]])
    x[4, 1] = np.nan
    w = lm.window_sum(x, 3)
    assert w[:6, 0].tolist() == [6.0, 9.0, 12.0, 15.0, 18.0, 21.0] and np.isnan(w[6:, 0]).all()
    assert w[:2, 1].tolist() == [60.0, 90.0] and np.isnan(w[2:5, 1]).all() and w[5, 1] == 210.0
    assert np.isnan(lm.window_sum(x[:2], 3)).all()


def test_a_large_move_exceeds_a_multiple_of_the_assets_own_median():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((400, 3)) * np.array([[1.0, 50.0, 0.001]])      # three scales: the share is the same
    x[7, 0], x[9, 1] = 30.0, -2000.0
    moves = lm.large_moves(x, 1)
    size = np.abs(x)
    expected = (size > lm.TAIL_MULTIPLE * np.median(size, axis=0)[None, :]).sum(axis=1)
    assert moves.hits.tolist() == expected.tolist() and moves.hits[7] >= 1 and moves.hits[9] >= 1
    assert moves.count.tolist() == [3.0] * 400
    # for the panel itself half the cells lie below their medians: the correction is zero
    assert moves.below.sum() == pytest.approx(0.5 * moves.count.sum())
    assert 0.0 < moves.slope < 1.0


def test_medians_are_taken_within_the_days_that_share_a_remainder():
    x = np.ones((280, 1))
    x[::7] = 100.0                         # every seventh day is a hundred times larger ...
    x[14, 0] = 100.0 * 5.0                 # ... and one of those is large even among its own
    x[15, 0] = 5.0                         # large among the ordinary days
    together, apart = lm.large_moves(x, 1), lm.large_moves(x, 7)
    assert together.hits.sum() == 41.0     # against one median, every seventh day is "large", and day 15
    assert apart.hits.sum() == 2.0 and apart.hits[14] == 1.0 and apart.hits[15] == 1.0


def test_a_thin_group_or_a_median_of_zero_is_not_judged():
    x = np.ones((30, 2))
    x[:, 1] = 0.0
    x[::2, 1] = 1.0                        # the second asset moves on every other day: its median is not zero
    assert lm.large_moves(x[:10], 1).count.sum() == 0.0            # ten values are too few
    assert lm.large_moves(x, 1).count.tolist() == [2.0] * 30
    x[:, 1] = 0.0
    assert lm.large_moves(x, 1).count.tolist() == [1.0] * 30       # an asset that never moves is left out


def test_the_correction_stands_in_for_medians_recomputed_in_a_resample():
    # two regimes of volatility; a resample that draws the turbulent blocks more often
    rng = np.random.default_rng(5)
    scale = np.repeat(np.where(np.arange(20) % 2 == 0, 1.0, 3.0), 90)[:, None]
    x = rng.standard_t(3, (1800, 40)) * scale
    moves = lm.large_moves(x, 1)
    block, blocks = la.block_index(1800, 90)
    weights = np.where(np.arange(blocks) % 2 == 0, 0.0, 2.0)       # only the turbulent blocks, twice
    sums = [np.bincount(block, weights=part, minlength=blocks) for part in moves.sums]
    fixed = (weights @ sums[0]) / (weights @ sums[2])
    corrected = fixed + moves.slope * ((weights @ sums[1]) / (weights @ sums[2]) - 0.5)
    drawn = x[np.repeat(np.arange(1800).reshape(blocks, 90)[1::2], 2, axis=0).ravel()]
    exact = lm.large_moves(drawn, 1)
    truth = exact.hits.sum() / exact.count.sum()
    assert abs(corrected - truth) < 0.25 * abs(fixed - truth)
    assert fixed > truth                   # fixed medians are too low for the turbulent blocks: too many large moves


def test_the_volatility_estimate_uses_earlier_days_only():
    rng = np.random.default_rng(1)
    g = rng.standard_normal((120, 2))
    before = lm.volatility_estimate(g)
    assert np.isnan(before[: lm.FILTER_MIN]).all() and np.isfinite(before[lm.FILTER_MIN:]).all()
    changed = g.copy()
    changed[80:] *= 25.0
    after = lm.volatility_estimate(changed)
    assert np.array_equal(before[:81], after[:81], equal_nan=True) and (after[82:] > before[82:]).all()
    still = g.copy()
    still[40:60, 0] = 0.0                  # a stretch without a move counts as the floor, not as minus infinity
    assert np.isfinite(lm.volatility_estimate(still)).sum() == np.isfinite(before).sum()
    assert lm.volatility_estimate(still)[60, 0] > 0.9 * lm.FILTER_FLOOR


def test_the_volatility_estimate_follows_the_scale_of_the_returns():
    rng = np.random.default_rng(2)
    g = rng.standard_normal((600, 1))
    low, high = lm.volatility_estimate(g)[-200:], lm.volatility_estimate(4.0 * g)[-200:]
    assert np.allclose(high / low, 4.0, rtol=0.1)      # not exactly: the floor is in the returns' own units
    assert np.allclose(lm.volatility_estimate(g, floor=0.0)[-200:] * 4.0, lm.volatility_estimate(4.0 * g, floor=0.0)[-200:])


def test_independent_normal_returns_leave_the_reference_share_and_nothing_more():
    present = np.ones((1500, 40), dtype=bool)
    present[:500, :10] = False
    reference = lm.normal_reference(present)
    assert 0.007 < reference < 0.02        # above the 0.7 % of a normal: the estimate and the medians are noisy
    assert lm.normal_reference(present) == reference                       # seeded and kept
    assert lm.normal_reference(present[:, :20]) != reference               # a different pattern of cells
    g = np.where(present, np.random.default_rng(9).standard_normal(present.shape), np.nan)
    moves = lm.large_moves(lm.filtered(g), 1)
    assert moves.hits.sum() / moves.count.sum() == pytest.approx(reference, abs=0.002)


def test_ranks_across_assets_are_centred_scaled_and_skip_thin_days():
    x = np.arange(60.0).reshape(2, 30)
    x[1, 25:] = np.nan
    x[1, 0] = x[1, 1]                      # a tie shares its rank
    ranks = lm.cross_ranks(x, minimum=20)
    assert np.allclose(np.nanmean(ranks, axis=1), 0.0) and np.allclose(np.nanstd(ranks, axis=1), 1.0)
    assert np.isnan(ranks[1, 25:]).all() and ranks[1, 0] == ranks[1, 1]
    assert np.isnan(lm.cross_ranks(x[:, :10], minimum=20)).all()


def test_the_second_batch_uses_the_cells_of_the_first():
    flows = _flows(mw.MoreWorld("quiet"))
    prepared = lm.prepare(flows)
    r, keep = lm.raw_returns(flows.panel)
    assert tuple(a for a, k in zip(flows.panel.assets, keep, strict=True) if k) == prepared.first.assets
    assert np.array_equal(np.isfinite(r), np.isfinite(prepared.first.g))
    assert np.allclose(la._standardise(r), prepared.first.g, equal_nan=True)
    assert np.isfinite(prepared.imbalance).sum() == np.isfinite(prepared.first.g).sum()
    assert np.nanmax(np.abs(prepared.imbalance)) <= 1.0


def test_the_imbalance_is_buys_minus_sells_over_volume():
    dates = np.arange(np.datetime64("2021-01-01"), np.datetime64("2022-06-01"))
    close = np.cumprod(1.0 + 0.01 * np.random.default_rng(4).standard_normal((len(dates), 1)), axis=0)
    volume = np.full((len(dates), 1), 100.0)
    volume[50, 0] = 0.0
    taker = np.full((len(dates), 1), 75.0)
    prepared = lm.prepare(lm.FlowPanel(la.Panel(dates, ("A",), close, volume), taker))
    assert np.nanmin(prepared.imbalance) == np.nanmax(prepared.imbalance) == 0.5
    assert np.isnan(prepared.imbalance[50, 0]) and np.isfinite(prepared.first.g[50, 0])


# --- the numbers, on worlds where the answer is known ---------------------------------------------------

def test_the_skew_dial_sets_the_sign_of_the_asymmetry():
    assert _numbers(mw.MoreWorld("losses", skew=0.3))["asymmetry"] > 0.05
    assert _numbers(mw.MoreWorld("gains", skew=-0.3))["asymmetry"] < -0.05


def test_a_stable_law_keeps_its_shape_and_a_heavy_tail_of_finite_variance_loses_it():
    heavy = _numbers(mw.MoreWorld("t3", base=lw.World("c", **NO_VOLATILITY)))
    stable = _numbers(mw.MoreWorld("stable", stable=1.7))
    assert heavy["gaussianity"] > 0.01 and heavy["share_day"] > heavy["share_week"] > heavy["share_month"]
    assert abs(stable["gaussianity"]) < 0.5 * heavy["gaussianity"]


def test_dividing_by_the_volatility_estimate_helps_only_where_volatility_moves():
    moving = _numbers(mw.MoreWorld("moving", base=lw.World("m", tail=float("inf"))))
    constant = _numbers(mw.MoreWorld("constant", base=lw.World("c", **NO_VOLATILITY)))
    assert moving["conditional"] > 0.0 > constant["conditional"]


def test_what_is_left_after_the_division_is_heavy_unless_the_returns_are_independent_and_normal():
    normal = _numbers(mw.MoreWorld("normal", base=lw.World("n", tail=float("inf"), **NO_VOLATILITY)))
    heavy = _numbers(mw.MoreWorld("t3", base=lw.World("c", **NO_VOLATILITY)))
    assert abs(normal["still_heavy"]) < 0.004 < 0.015 < heavy["still_heavy"]


def test_the_cascade_dial_sets_which_volatility_predicts_which():
    up, down = _numbers(mw.MoreWorld("up", cascade=0.4)), _numbers(mw.MoreWorld("down", cascade=-0.4))
    assert up["cascade"] > 0.02 and up["coarse_to_fine"] > up["fine_to_coarse"]
    assert down["cascade"] < -0.02


def test_a_wandering_expected_return_is_momentum_and_a_fading_price_component_is_reversal():
    assert _numbers(mw.MoreWorld("drift", drift=0.3))["momentum"] > 0.05
    reversal = _numbers(mw.MoreWorld("fading", transitory=1.5))
    assert reversal["momentum"] < -0.02 and all(reversal[f"momentum_{k}"] < 0 for k in lm.MOMENTUM_WEEKS)


def test_the_flow_dial_sets_what_order_flow_says_about_the_next_day():
    carried, reversed_ = _numbers(mw.MoreWorld("carried", flow=0.2)), _numbers(mw.MoreWorld("reversed", flow=-0.2))
    assert carried["flow"] > 0.05 and carried["flow_partial"] > 0.05 and reversed_["flow"] < -0.05
    assert carried["flow_same_day"] > 0.2                # the imbalance follows the same day's return


def test_in_a_quiet_world_the_four_dialled_numbers_are_near_zero():
    quiet = _numbers(mw.MoreWorld("quiet"))
    assert abs(quiet["asymmetry"]) < 0.1 and abs(quiet["cascade"]) < 0.05
    assert abs(quiet["momentum"]) < 0.03 and abs(quiet["flow"]) < 0.03


def test_every_claim_has_a_number_a_published_set_and_a_widening_factor():
    assert set(lm.CLAIMS) == set(lm.PUBLISHED) == set(lm.STRETCH) and set(lm.CLAIMS) <= set(lm.NUMBERS)
    assert all(published == (0.0, la.INF) for published in lm.PUBLISHED.values())
    assert all(factor >= 1.0 for factor in lm.STRETCH.values())


def test_a_reading_carries_the_widened_interval_and_its_verdict():
    audit = lm.audit(_flows(mw.MoreWorld("all", skew=0.3, drift=0.3, cascade=0.4, flow=0.2)), resamples=200,
                     stretch=dict.fromkeys(lm.CLAIMS, 2.0))
    assert set(audit["readings"]) == set(lm.CLAIMS)
    for claim, reading in audit["readings"].items():
        low, high = audit["numbers"][claim]["percentile"]
        assert reading["stretch"] == 2.0 and reading["low"] <= low <= high <= reading["high"]
        assert reading["verdict"] == la.verdict(reading["low"], reading["high"], lm.PUBLISHED[claim])
    assert audit["readings"]["flow"]["verdict"] == la.SURVIVES and audit["readings"]["momentum"]["verdict"] == la.SURVIVES
    assert len(audit["numbers"]["momentum"]["yearly"]) == 3
    assert audit["settings"]["level"] == la.LEVEL


def test_an_imbalance_that_never_varies_gives_no_verdict_on_flow():
    flows = _flows(mw.MoreWorld("quiet"))
    flat = lm.FlowPanel(flows.panel, flows.panel.volume / 2.0)
    reading = lm.audit(flat, resamples=50)["readings"]["flow"]
    assert np.isnan(reading["estimate"]) and reading["verdict"] == la.NOT_REJECTED


# --- the worlds --------------------------------------------------------------------------------------

def test_with_every_dial_at_zero_the_panel_is_the_first_batchs():
    r = _runner()
    dates, assets, mask = r.first.shape_of_the_panel(None)
    base = lw.simulate(lw.World("all_hold"), dates, assets, mask, np.random.default_rng(3))
    quiet = _flows(mw.MoreWorld("quiet"))
    assert np.array_equal(quiet.panel.close, base.close, equal_nan=True)
    assert np.array_equal(quiet.panel.volume, base.volume, equal_nan=True)
    assert np.array_equal(np.isfinite(quiet.taker_buy), mask)
    again = _flows(mw.MoreWorld("quiet"))
    assert np.array_equal(quiet.taker_buy, again.taker_buy, equal_nan=True)


def test_the_returns_of_a_panel_are_the_ones_it_was_built_from():
    r = _runner()
    dates, assets, mask = r.first.shape_of_the_panel(None)
    base = lw.simulate(lw.World("all_hold"), dates, assets, mask, np.random.default_rng(3))
    returns = mw.returns_of(base, mask)
    rebuilt = np.where(mask, 100.0 * np.exp(np.cumsum(np.where(mask, returns, 0.0), axis=0)), np.nan)
    assert np.allclose(rebuilt, base.close, equal_nan=True) and (returns[~mask] == 0.0).all()


def test_a_dialled_world_keeps_volumes_and_presence_and_changes_prices():
    quiet, skewed = _flows(mw.MoreWorld("quiet")), _flows(mw.MoreWorld("skew", skew=0.2))
    assert np.array_equal(quiet.panel.volume, skewed.panel.volume, equal_nan=True)
    assert np.array_equal(np.isfinite(quiet.panel.close), np.isfinite(skewed.panel.close))
    assert not np.allclose(quiet.panel.close, skewed.panel.close, equal_nan=True)


def test_the_persistent_series_has_the_requested_half_life():
    series = mw.persistent(10.0, (20000, 4), np.random.default_rng(6))
    assert np.allclose(series.std(axis=0), 1.0, atol=0.06)
    lagged = [float(np.mean(series[:-10, j] * series[10:, j])) for j in range(4)]
    assert np.allclose(lagged, 0.5, atol=0.08)


def test_stable_draws_are_symmetric_heavy_and_keep_their_shape_when_summed():
    draws = mw.stable_draws(1.7, (200_000, 1), np.random.default_rng(7))[:, 0]
    assert abs(np.median(draws)) < 0.02
    single = np.quantile(np.abs(draws), 0.99) / np.median(np.abs(draws))
    summed = np.abs(draws.reshape(-1, 8).sum(axis=1))
    assert np.quantile(summed, 0.99) / np.median(summed) == pytest.approx(single, rel=0.15)
    assert np.median(summed) / np.median(np.abs(draws)) == pytest.approx(8.0 ** (1.0 / 1.7), rel=0.05)


def test_each_claim_is_true_in_the_world_built_for_it_and_false_where_its_dial_is_off():
    worlds = {w.name: w for w in mw.WORLDS}
    assert all(mw.holds(worlds["all_hold"], claim) for claim in lm.CLAIMS)
    assert not any(mw.holds(worlds["all_inverse"], claim) for claim in ("asymmetry", "cascade", "momentum", "flow"))
    assert not any(mw.holds(worlds["quiet"], claim) for claim in ("asymmetry", "cascade", "momentum", "flow"))
    assert not mw.holds(worlds["stable_1.7"], "gaussianity") and mw.holds(worlds["stable_1.7"], "still_heavy")
    assert not any(mw.holds(worlds["normal"], claim) for claim in lm.CLAIMS)
    assert not mw.holds(worlds["constant_volatility"], "conditional") and mw.holds(worlds["constant_volatility"], "gaussianity")
    assert mw.holds(worlds["normal_shocks"], "conditional") and mw.holds(worlds["normal_shocks"], "still_heavy")
    with pytest.raises(ValueError, match="unknown claim"):
        mw.holds(worlds["quiet"], "tail")
    assert len({w.name for w in mw.WORLDS}) == len(mw.WORLDS)
    assert {w.name for w in mw.CALIBRATION_WORLDS} >= {"quiet", "stable_1.7", "normal"}


# --- the runner --------------------------------------------------------------------------------------

def test_the_widening_factor_is_the_largest_need_rounded_up_and_never_below_one():
    r = _runner()
    table = {"a": {"numbers": {c: {"stretch_needed": 1.21} for c in lm.CLAIMS}},
             "b": {"numbers": {c: {"stretch_needed": 0.4} for c in lm.CLAIMS}}}
    table["b"]["numbers"]["flow"] = {"stretch_needed": 1.7}
    factors = r.stretch_from(table)
    assert factors["flow"] == 1.7 and factors["momentum"] == 1.3 and set(factors) == set(lm.CLAIMS)
    table["a"]["numbers"]["cascade"] = {"stretch_needed": 0.2}
    assert r.stretch_from(table)["cascade"] == 1.0


def test_the_commands_run_end_to_end_on_a_small_stand_in_panel(tmp_path, capsys):
    r = _runner()
    assert r.main(["calibrate", "--out", str(tmp_path), "--replicates", "2", "--worlds", "quiet,normal"]) == 0
    assert r.main(["stretch", "--out", str(tmp_path)]) == 0
    record = json.loads((tmp_path / "calibration.json").read_text(encoding="utf-8"))
    assert set(record["stretch"]) == set(lm.CLAIMS) and set(record["table"]) == {"quiet", "normal"}
    assert r.main(["power", "--out", str(tmp_path), "--replicates", "1", "--worlds", "all_hold,normal"]) == 0
    assert r.main(["summarise", "--out", str(tmp_path)]) == 0
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert set(summary["summary"]) == {"all_hold", "normal"} and set(summary["totals"]) == set(lm.CLAIMS)
    assert summary["summary"]["all_hold"]["claims_that_hold"] == list(lm.CLAIMS)
    assert summary["summary"]["normal"]["claims_that_hold"] == []
    capsys.readouterr()


def test_the_committed_widening_factors_are_the_calibrated_ones():
    r = _runner()
    record = json.loads((ROOT / r.POWER_DIR / "calibration.json").read_text(encoding="utf-8"))
    assert dict(lm.STRETCH) == record["stretch"] == r.stretch_from(record["table"])
    assert sorted(record["table"]) == sorted(w.name for w in mw.CALIBRATION_WORLDS)
    assert all(cell["replicates"] == r.CALIBRATION_REPLICATES for cell in record["table"].values())
    assert record["level"] == la.LEVEL


def test_the_committed_power_summary_was_made_with_the_committed_factors_and_worlds():
    r = _runner()
    summary = json.loads((ROOT / r.POWER_DIR / "summary.json").read_text(encoding="utf-8"))
    assert summary["stretch"] == dict(lm.STRETCH)
    assert sorted(summary["summary"]) == sorted(w.name for w in mw.WORLDS)
    assert all(cell["replicates"] == r.POWER_REPLICATES for cell in summary["summary"].values())
    assert summary["worlds"] == {w.name: json.loads(json.dumps(r.world_dials(w))) for w in mw.WORLDS}


def test_the_committed_summaries_are_what_the_committed_rows_give(tmp_path, capsys):
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


def _write_panel_file(path: Path, flows: lm.FlowPanel) -> None:
    """A panel in the builder's own layout: one row per calendar day, four columns per asset, empty for no bar."""
    panel = flows.panel
    header = ["observation_date"]
    for asset in panel.assets:
        header += [f"{asset}_CLOSE", f"{asset}_VOL", f"{asset}_TAKER_BUY", f"{asset}_OFI"]
    lines = [",".join(header)]
    for day, date in enumerate(panel.dates):
        row = [str(date)]
        for j in range(len(panel.assets)):
            close, volume, taker = panel.close[day, j], panel.volume[day, j], flows.taker_buy[day, j]
            ofi = (2.0 * taker - volume) / volume if volume > 0 else 0.0
            row += ["", "", "", ""] if np.isnan(close) else [repr(float(close)), repr(float(volume)),
                                                              repr(float(taker)), repr(float(ofi))]
        lines.append(",".join(row))
    path.write_bytes(gzip.compress(("\n".join(lines) + "\n").encode("utf-8")))


def test_the_flow_panel_loader_reads_taker_buy_volume_by_asset(tmp_path):
    text = ("observation_date,BBB_CLOSE,BBB_VOL,BBB_TAKER_BUY,BBB_OFI,AAA_CLOSE,AAA_VOL,AAA_TAKER_BUY,AAA_OFI\n"
            "2021-01-01,1.0,10,6,0.2,,,,\n2021-01-02,1.1,11,5,-0.09,5.0,50,30,0.2\n")
    path = tmp_path / "panel_development.csv.gz"
    path.write_bytes(gzip.compress(text.encode()))
    flows = lm.load_flow_panel(path)
    assert flows.panel.assets == ("AAA", "BBB") and flows.taker_buy[1].tolist() == [30.0, 5.0]
    assert np.isnan(flows.taker_buy[0, 0]) and flows.taker_buy[0, 1] == 6.0
    assert flows.between("2021-01-02", "2021-01-02").taker_buy.shape == (1, 2)
    assert flows.columns(np.array([False, True])).taker_buy[:, 0].tolist() == [6.0, 5.0]


def test_the_run_completes_on_a_panel_with_the_blemishes_of_a_real_one(tmp_path, monkeypatch):
    r = _runner()
    period = r.first.PERIOD
    dates = np.arange(np.datetime64(period[0]), np.datetime64(period[1]) + np.timedelta64(1, "D"))
    names = [f"A{i:02d}" for i in range(34)] + ["DEAD", "GAP", "LATE", "SHORT", "STALE", "币"]
    mask = np.ones((len(dates), len(names)), dtype=bool)
    mask[:700, :8] = False                               # listed later
    mask[1000:, names.index("DEAD")] = False             # stopped trading in 2023
    mask[900:930, names.index("GAP")] = False            # a month without bars
    mask[:1700, names.index("LATE")] = False             # listed in the last months: too short to enter
    mask[100:, names.index("SHORT")] = False             # a hundred bars and gone
    flows = mw.simulate(mw.MoreWorld("all", skew=0.1, drift=0.06, cascade=0.2, flow=0.05), dates, tuple(names), mask,
                        np.random.default_rng(21))
    stale = names.index("STALE")
    flows.panel.close[400:460, stale] = flows.panel.close[399, stale]     # no trades: the price stands still ...
    flows.panel.volume[400:460, stale] = 0.0                              # ... and the volume is zero
    flows.taker_buy[400:460, stale] = 0.0
    _write_panel_file(tmp_path / r.first.PANEL_FILE, flows)
    monkeypatch.setattr(r, "check_frozen", lambda: None)
    monkeypatch.setattr(r.first, "PANEL_SHA256", hashlib.sha256((tmp_path / r.first.PANEL_FILE).read_bytes()).hexdigest())
    assert r.run_once(tmp_path, tmp_path / "out") == 0
    results = json.loads((tmp_path / "out" / "results.json").read_text(encoding="utf-8"))
    assert results["preregistration"] == r.PREREGISTRATION_ID
    assert results["panel"]["days"] == len(dates) and results["panel"]["assets_in_file"] == len(names)
    assert results["audit"]["assets"] == len(names) - 2            # LATE and SHORT are too short
    readings = results["audit"]["readings"]
    assert set(readings) == set(lm.CLAIMS)
    assert all(reading["verdict"] in (la.FAILS, la.SURVIVES, la.NOT_REJECTED) for reading in readings.values())
    assert all(np.isfinite([reading["estimate"], reading["low"], reading["high"]]).all() for reading in readings.values())
    assert readings["flow"]["verdict"] == la.SURVIVES and readings["still_heavy"]["verdict"] == la.SURVIVES
    assert set(results["subgroups"]) == {"fit_period", "test_period", "stopped_trading", "still_trading",
                                         "larger_half", "smaller_half"}
    assert results["subgroups"]["larger_half"]["assets"] + results["subgroups"]["smaller_half"]["assets"] \
        == results["audit"]["assets"]
    assert len(results["audit"]["numbers"]["momentum"]["yearly"]) == 5


def test_the_run_refuses_a_panel_that_is_not_the_preregistered_one(tmp_path, monkeypatch):
    r = _runner()
    monkeypatch.setattr(r, "check_frozen", lambda: None)
    (tmp_path / r.first.PANEL_FILE).write_bytes(gzip.compress(b"observation_date,X_CLOSE,X_VOL,X_TAKER_BUY\n2021-01-01,1,1,1\n"))
    with pytest.raises(SystemExit, match="preregistered panel"):
        r.run_once(tmp_path, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_the_run_refuses_to_start_if_a_frozen_file_has_changed(tmp_path, monkeypatch):
    r = _runner()
    digest = tmp_path / "DIGEST.json"
    recorded = dict(r.frozen_digests())
    recorded[r.FROZEN[2]] = "0" * 64
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
    # the first batch's frozen files are frozen here too: this batch is built around them
    assert {"apedp_dyn/market/law_audit.py", "apedp_dyn/market/law_audit_worlds.py",
            "scripts/run_law_audit_finance.py"} <= set(r.FROZEN)


def test_nothing_in_the_second_batch_names_the_holdout_panel():
    r = _runner()
    for path in ("scripts/run_law_audit_finance_more.py", "apedp_dyn/market/law_audit_more.py",
                 "apedp_dyn/market/law_audit_more_worlds.py"):
        assert "panel_holdout" not in (ROOT / path).read_text(encoding="utf-8"), path
    assert r.first.PANEL_FILE == "panel_development.csv.gz" and r.first.PERIOD[1] < "2026-01-01"
