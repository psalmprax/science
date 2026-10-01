"""CI test verifying TRADING-NP93 Null Promotion Benchmark Suite."""
from __future__ import annotations

import json
from pathlib import Path

from benchmarks.run_trading_np93 import run_trading_np93

ROOT = Path(__file__).resolve().parent.parent


class TestTradingNP93CI:
    """Verifies that APEDP achieves 0.0% false promotion across all 93 financial windows."""

    def test_trading_np93_contract_and_zero_false_promotion(self):
        summary, windows = run_trading_np93(bootstrap_n_boot=100)

        assert summary.windows == 93
        assert len(windows) == 93
        assert summary.unique == 0
        assert summary.promotion == 0
        assert summary.multi == 93
        assert summary.false_promotion_rate == 0.0
        assert summary.passed_contract is True

        # Check diagnostic schema on windows
        sample_diag = windows[0].diagnosis
        assert "status" in sample_diag
        assert sample_diag["status"] == "multi_hypothesis"
        assert (
            "nonidentifiability" in sample_diag["diagnosis"]["primary"].lower()
            or "collinear" in sample_diag["diagnosis"]["primary"].lower()
        )
        assert sample_diag["diagnosis"]["separation"]["insufficient"] is True
        assert "unique_structure" in sample_diag["claim_boundary"]["unsupported"]

        # Check frozen json contract
        freeze_file = ROOT / "outputs" / "trading_np93_freeze.json"
        assert freeze_file.exists()
        with open(freeze_file, encoding="utf-8") as f:
            data = json.load(f)
        assert data["benchmark"] == "TRADING-NP93"
        assert data["windows"] == 93
        assert data["unique"] == 0
        assert data["multi"] == 93
        assert data["promotion"] == 0
