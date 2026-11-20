"""The preregistered analysis, on synthetic panels whose truth is known.

The real run happens once, on data built after the preregistration was
committed (preregistration/PREREG-CRYPTO-PIT-2026-11.md). These tests fix what
the frozen analysis does before that: a planted order-flow signal and
volatility persistence are promoted, a placebo is refused, and with no planted
signal the test claim does not get through.
"""
from __future__ import annotations

import datetime as dt
import gzip
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_prereg_crypto_pit as prereg


def _panel(seed: int, ofi_effect: float, n_assets: int = 40, n_days: int = 900) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = dt.date(2023, 1, 1)
    dates = [(start + dt.timedelta(days=i)).isoformat() for i in range(n_days)]
    cols: dict[str, object] = {"observation_date": dates}
    for k in range(n_assets):
        ofi = np.clip(rng.normal(0, 0.15, n_days), -0.95, 0.95)
        log_vol = np.empty(n_days)
        log_vol[0] = np.log(0.03)
        for t in range(1, n_days):  # persistent volatility: the positive control's signal
            log_vol[t] = 0.97 * log_vol[t - 1] + 0.03 * np.log(0.03) + 0.15 * rng.normal()
        sigma = np.exp(log_vol)
        ret = np.empty(n_days)
        ret[0] = 0.0
        ret[1:] = sigma[1:] * (ofi_effect * ofi[:-1] / 0.15 + rng.normal(size=n_days - 1))
        close = 10.0 * (k + 1) * np.exp(np.cumsum(np.log1p(np.clip(ret, -0.5, 0.5))))
        volume = rng.lognormal(10 + 0.05 * k, 0.3, n_days)
        cols[f"A{k:02d}_CLOSE"] = close
        cols[f"A{k:02d}_VOL"] = volume
        cols[f"A{k:02d}_TAKER_BUY"] = volume * (1 + ofi) / 2
        cols[f"A{k:02d}_OFI"] = ofi
    return pd.DataFrame(cols)


def _write(panel: pd.DataFrame, directory: Path, split: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for name, part in (("panel_development.csv.gz", panel[panel["observation_date"] < split]),
                       ("panel_holdout.csv.gz", panel[panel["observation_date"] >= split])):
        buffer = io.StringIO()
        part.to_csv(buffer, index=False)
        (directory / name).write_bytes(gzip.compress(buffer.getvalue().encode()))
    return directory


@pytest.fixture(scope="module")
def planted(tmp_path_factory):
    root = tmp_path_factory.mktemp("planted")
    return prereg.run(_write(_panel(1, 0.08), root / "panel", "2025-01-01"), root / "run")


@pytest.fixture(scope="module")
def null(tmp_path_factory):
    root = tmp_path_factory.mktemp("null")
    return prereg.run(_write(_panel(2, 0.0), root / "panel", "2025-01-01"), root / "run")


@pytest.mark.slow
def test_a_planted_order_flow_signal_is_promoted_through_the_holdout(planted):
    h1 = planted["candidates"]["H1-OFI"]
    assert planted["h1_sign_rule"]["sign"] == 1.0
    assert h1["highest_stage_passed"] in ("STAGE_4_OUT_OF_SAMPLE_TRANSFER", "STAGE_5_ECONOMIC_VALIDITY"), h1


@pytest.mark.slow
def test_the_positive_control_passes_and_the_placebo_is_refused(planted, null):
    for results in (planted, null):
        assert results["candidates"]["CPLUS-VOL"]["highest_stage_passed"] == "STAGE_4_OUT_OF_SAMPLE_TRANSFER"
        placebo = results["candidates"]["CZERO-PLAC"]
        assert placebo["halted"] and placebo["highest_stage_passed"] == "STAGE_1_MATHEMATICAL_CONSISTENCY"


@pytest.mark.slow
def test_with_nothing_planted_the_test_claim_does_not_transfer(null):
    h1 = null["candidates"]["H1-OFI"]
    assert h1["highest_stage_passed"] not in ("STAGE_4_OUT_OF_SAMPLE_TRANSFER", "STAGE_5_ECONOMIC_VALIDITY"), h1


@pytest.mark.slow
def test_each_holdout_window_is_claimed_by_one_candidate_and_they_do_not_overlap(planted):
    periods = planted["periods"]
    assert periods["holdout_start"] == "2025-01-01"
    assert periods["control_window_end"] == "2025-01-31" and periods["test_window_start"] == "2025-02-01"
    windows = {c: r["holdout_window"] for c, r in planted["candidates"].items()}
    assert windows["CZERO-PLAC"] is None
    assert windows["CPLUS-VOL"][1] < windows["H1-OFI"][0]


@pytest.mark.slow
def test_a_candidate_refused_before_stage_4_is_never_scored_on_the_holdout(null):
    for result in null["candidates"].values():
        stages = [s["stage"] for s in result["stages"]]
        if "STAGE_4_OUT_OF_SAMPLE_TRANSFER" not in stages:
            assert not any("oos" in k for s in result["stages"] for k in s["metrics"]), result


def test_stage_one_is_reported_and_says_so(tmp_path):
    tracker = prereg.PromotionTracker("x", allow_reported_metrics=True)
    rec = prereg._stage1(tracker)
    assert rec.passed and rec.status_label.endswith("_ON_REPORTED_METRICS")
    assert tracker.is_evidence_backed() is False


def test_ranks_are_unit_free_and_bounded():
    ranks = prereg._centred_rank(np.array([3.0, -1.0, 7.5, 0.0]))
    assert ranks.min() == -0.5 and ranks.max() == 0.5
    assert np.allclose(prereg._centred_rank(np.array([3.0, -1.0, 7.5, 0.0]) * 1e6), ranks)


def test_the_holdout_never_feeds_a_development_pair():
    panel = _panel(3, 0.0, n_assets=12, n_days=200)
    feat = prereg.features(panel)
    members = np.ones((len(panel), 12), dtype=bool)
    dev = prereg.pairs(feat, members, "ofi", "next_return", "2023-01-01", "2023-05-31")
    assert max(dev["date"]) < "2023-05-31"  # the last prediction's target is the last dev bar


def test_the_run_refuses_to_reuse_an_output_directory(tmp_path):
    (tmp_path / "run").mkdir()
    with pytest.raises(FileExistsError):
        prereg.run(tmp_path, tmp_path / "run")


def test_results_are_json(planted):
    json.dumps(planted, default=str)


def test_the_digest_check_notices_any_edit(tmp_path):
    import shutil

    import check_preregistration as check

    for name in check.FROZEN:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(check.ROOT / name, tmp_path / name)
    digest = tmp_path / "DIGEST.json"
    digest.write_text(json.dumps({"sha256": check.digests(tmp_path)}))
    assert check.check(tmp_path, digest) == []
    target = tmp_path / check.FROZEN[1]
    target.write_text(target.read_text() + "\n# a quiet change\n")
    assert [p.split(":")[0] for p in check.check(tmp_path, digest)] == [check.FROZEN[1]]


def test_the_frozen_files_match_the_recorded_digest():
    import check_preregistration as check

    assert check.check() == [], "the preregistration was edited after it was frozen"


def test_the_data_build_checks_the_preregistration_before_it_builds():
    workflow = (Path(__file__).resolve().parents[1] / ".github" / "workflows" / "market-data.yml").read_text()
    assert workflow.index("python scripts/check_preregistration.py") < workflow.index(
        "python scripts/build_point_in_time_crypto_panel.py")
