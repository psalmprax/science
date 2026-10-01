"""CI test verifying NANO-DS10 Discovery Stability & Identifiability Benchmark."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.run_nano_ds10 import run_nano_ds10

ROOT = Path(__file__).resolve().parent.parent


class TestNanoDS10CI:
    """Verifies the recovery vs identification separation on M4 nanoscale crossover."""

    def test_nano_ds10_stability_and_non_identification(self):
        summary, details = run_nano_ds10(n_seeds=10)

        assert summary.fit_seeds == 10
        assert summary.winner_recurrence == {"piecewise_hp_inv": 1.0}
        assert summary.entropy_nats == 0.0
        assert summary.k_hp_cv == pytest.approx(0.045, abs=0.005)
        assert summary.adjudication_status == "multi_hypothesis"
        assert summary.passed_contract is True

        # Check variance split structure
        vsplit = summary.variance_split
        assert "fit_seed" in vsplit
        assert "scoring_seed" in vsplit
        assert "fit_share" in vsplit
        assert 0.0 <= vsplit["fit_share"] <= 1.0

        # Check frozen json contract
        freeze_file = ROOT / "outputs" / "nano_ds10_freeze.json"
        assert freeze_file.exists()
        with open(freeze_file, encoding="utf-8") as f:
            data = json.load(f)
        assert data["benchmark"] == "NANO-DS10"
        assert data["winner_recurrence"] == {"piecewise_hp_inv": 1.0}
        assert data["entropy_nats"] == 0.0
        assert data["k_hp_cv"] == pytest.approx(0.045, abs=0.005)
