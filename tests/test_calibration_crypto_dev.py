"""The frozen calibration (preregistration/PREREG-CRYPTO-CAL-2026-11.md), on synthetic panels only.

The real panel is for the one preregistered run. These tests check that the
script measures what it says: placebos are judged on the frozen pairing and
almost never pass, a strong injected signal always does, a literature factor
that passes stage 2 is refused at stage 3, and the script cannot be pointed at
anything past the development period.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import run_calibration_crypto_dev as cal
from test_prereg_crypto_pit import _panel


def _development_file(directory: Path, panel: pd.DataFrame) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    panel.to_csv(buffer, index=False)
    path = directory / "panel_development.csv.gz"
    path.write_bytes(gzip.compress(buffer.getvalue().encode()))
    return path


@pytest.fixture(scope="module")
def small_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("cal")
    panel = _panel(seed=5, ofi_effect=0.0)
    path = _development_file(root / "data", panel)
    return cal.run(path, root / "out", n_placebos=30, strengths=(0.01, 0.5), n_per_strength=6), root / "out"


def test_placebos_are_judged_and_rarely_pass(small_run):
    results, _ = small_run
    placebo = results["placebo"]
    assert placebo["n"] == 30
    assert placebo["stage2_false_passes"] <= 1
    assert abs(placebo["ic_mean"]) < 0.02
    lo, hi = placebo["stage2_rate_interval_95"]
    assert 0.0 <= lo <= hi <= 1.0


def test_a_strong_injected_signal_always_passes_and_a_faint_one_rarely_does(small_run):
    results, _ = small_run
    by_rho = {row["rho"]: row for row in results["power"]}
    assert by_rho[0.5]["stage2_passes"] == by_rho[0.5]["n"] == 6
    assert by_rho[0.5]["stage3_passes"] == 6
    assert by_rho[0.5]["mean_measured_ic"] > 0.3
    assert by_rho[0.01]["stage2_passes"] <= 2


def test_a_literature_factor_never_passes_stage_three(small_run):
    results, _ = small_run
    names = [row["candidate"] for row in results["literature"]]
    assert names == ["CAL-LIT-reversal", "CAL-LIT-momentum"]
    for row in results["literature"]:
        assert row["stage3"] in (None, False)
        if row["stage2"]:
            assert row["max_literature_r2"] >= 0.85


def test_every_candidate_is_recorded_with_its_ic_and_verdict(small_run):
    _, out = small_run
    rows = [json.loads(line) for line in (out / "candidates.jsonl").read_text().splitlines()]
    assert len(rows) == 30 + 12 + 2
    assert all({"candidate", "ic", "stage2", "stage3", "highest_stage_passed"} <= set(r) for r in rows)
    assert len({r["candidate"] for r in rows}) == len(rows)
    assert (out / "promotion_journal.jsonl").is_file()
    assert json.loads((out / "results.json").read_text())["preregistration"] == cal.PREREGISTRATION_ID


def test_ranks_match_the_frozen_analysis():
    day = np.array(["d1"] * 5 + ["d2"] * 3)
    values = np.array([0.3, -1.0, 2.0, 0.1, 0.0, 5.0, 4.0, 6.0])
    ours = cal.within_day_centred_rank(day, values)
    import run_prereg_crypto_pit as pit

    theirs = np.concatenate([pit._centred_rank(values[:5]), pit._centred_rank(values[5:])])
    assert np.allclose(ours, theirs)


def test_it_refuses_a_panel_that_reaches_the_holdout(tmp_path):
    panel = _panel(seed=1, ofi_effect=0.0, n_days=1200)  # starts 2023-01-01, runs into 2026
    path = _development_file(tmp_path / "data", panel)
    with pytest.raises(SystemExit, match="past the development period"):
        cal.run(path, tmp_path / "out", n_placebos=1, strengths=(0.5,), n_per_strength=1)


def test_it_refuses_to_reuse_an_output_directory(tmp_path):
    path = _development_file(tmp_path / "data", _panel(seed=2, ofi_effect=0.0, n_days=400))
    (tmp_path / "out").mkdir()
    with pytest.raises(FileExistsError):
        cal.run(path, tmp_path / "out", n_placebos=1, strengths=(0.5,), n_per_strength=1)


def test_it_refuses_to_run_once_a_frozen_file_has_changed(tmp_path):
    for name in cal.FROZEN:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes((ROOT / name).read_bytes())
    digest = tmp_path / "DIGEST-CAL.json"
    digest.write_text(json.dumps({"sha256": cal.frozen_digests(tmp_path)}))
    cal.check_frozen(tmp_path, digest)
    (tmp_path / cal.FROZEN[0]).write_bytes((ROOT / cal.FROZEN[0]).read_bytes() + b"\n")
    with pytest.raises(SystemExit, match="changed since it was frozen"):
        cal.check_frozen(tmp_path, digest)


def test_the_frozen_files_match_the_recorded_digest():
    recorded = json.loads(cal.DIGEST.read_text())["sha256"]
    assert recorded == {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in cal.FROZEN
    }
