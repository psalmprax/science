"""Unit test for real market L2 open discovery sweep."""
from __future__ import annotations

import json
from dataclasses import dataclass

import apedp_dyn.sandbox.real_market_l2_sweep as sweep
from apedp_dyn.sandbox.real_market_l2_sweep import (
    WINDOWS,
    execute_window_discovery,
)


class TestRealMarketL2Sweep:
    """Verifies the real market L2 open discovery sweep."""

    def test_single_window_sp500_calm_sweep(self):
        """Window W2 (2019 calm bull market): sole adequate candidate promotes under audit.

        Audit trail (2026-10-19): the grammar now yields exactly ONE adequate
        candidate for this window, ``vol - 1/ma_dev**(3/2)`` (complexity 4,
        pool-of-one: ranking and Pareto set each contain a single entry). The
        earlier pin expected a refusal because a reorder twin of the then-winner
        (``ma_dev**(-2)+sqrt(ma_dev)``) sat within tau; that twin is no longer
        generated, so the sole-survivor contract applies by design: absolute
        holdout fidelity (holdout RMSE 4.65e-3 vs holdout noise sigma 8.0e-3,
        fidelity margin 0.35 <= 1) plus whiteness (rho1 ~ -0.04) are required
        before promotion, and both hold.

        The apparent ~130x train/holdout error gap is a units artifact:
        ``train_loss`` is MSE (3.58e-5 -> RMSE ~ 6.0e-3) while
        ``holdout_error`` is already RMSE (4.65e-3). In RMSE terms the survivor
        generalizes (train 6.0e-3, holdout 4.7e-3), matching the historically
        accepted candidate's train/holdout profile.
        """
        w2 = WINDOWS[1]  # W2_2019
        rec = execute_window_discovery("sp500", w2, bootstrap=False)
        assert rec.window_id == "W2_2019"
        assert rec.asset == "sp500"
        # Promotion under the hardened sole-survivor contract (pool of one).
        assert rec.epistemic_verdict == "unique_support"
        assert rec.epistemic_attribution == "decisive_evidential_support"
        assert "no structurally distinct rival survived scoring" in rec.full_reason
        # Anti-spurious guards: the survivor must actually be adequate,
        # otherwise this test would be endorsing "promoted because nothing fit".
        assert rec.out_of_grammar is False
        assert rec.holdout_error > 0
        assert rec.fidelity_margin is not None and rec.fidelity_margin <= 1.0
        assert rec.whiteness_passed is True
        assert rec.boundary_artifact is False
        assert "ma_dev" in rec.discovered_expression

    def test_single_window_nasdaq_covid_sweep(self):
        """Window W3 (2020 COVID shock on Nasdaq) recovers interaction term with UNIQUE_SUPPORT."""
        w3 = WINDOWS[2]  # W3_COVID
        rec = execute_window_discovery("nasdaq100", w3, bootstrap=False)
        assert rec.window_id == "W3_COVID"
        assert rec.asset == "nasdaq100"
        assert rec.epistemic_verdict == "unique_support"
        assert "vol" in rec.discovered_expression
        assert rec.out_of_grammar is False

    def test_window_discovery_records_search_multiplicity(self):
        """The promoted candidate is an order statistic over the engine's search pool.

        The sweep must record every candidate it scored (shared search id, the
        true evaluated count, and the promoted candidate's rank) so a 1-of-N
        promotion is never presented without its selection bias.
        """
        w2 = WINDOWS[1]  # W2_2019
        rec = execute_window_discovery("sp500", w2, bootstrap=False)

        mult = rec.trial_multiplicity
        assert mult is not None
        assert mult["recorded_trials"] >= 1
        assert mult["effective_trials"] == mult["recorded_trials"]
        assert mult["evaluated_count"] >= mult["recorded_trials"]
        assert mult["search_id"].startswith("l2:sp500:W2_2019:")
        assert "DSR not applicable" in mult["selection_statistic"]

        # The rank is a disclosure, not a promise: the promoted candidate is
        # selected by HOLDOUT adjudication, so it is generally NOT the best
        # in-sample fit. On this window it ranks 217th of 3195 by train loss
        # (the better-fitting candidates failed the holdout), which is exactly
        # the selection bias the disclosure exists to expose.
        assert 0 <= mult["promoted_rank"] < mult["recorded_trials"]
        assert mult["promoted_rank"] > 0, (
            "a rank-0 promotion would mean the winner was also the best "
            "in-sample fit; the disclosure must reflect the real ordering"
        )

    def test_multiterm_market_sweep_calm_refuses_spurious_models(self):
        """MultiTermL2Engine on W2_2019 calm market abstains with INSUFFICIENT_EVIDENCE."""
        w2 = WINDOWS[1]  # W2_2019
        rec = execute_window_discovery("sp500", w2, bootstrap=False, multiterm=True)
        assert rec.window_id == "W2_2019"
        assert rec.epistemic_verdict == "insufficient_evidence"

    def test_multiterm_market_sweep_covid_promotes_parsimonious(self):
        """MultiTermL2Engine on W3_COVID selects parsimonious interaction term under Pareto pruning."""
        w3 = WINDOWS[2]  # W3_COVID
        rec = execute_window_discovery("nasdaq100", w3, bootstrap=False, multiterm=True)
        assert rec.window_id == "W3_COVID"
        assert rec.epistemic_verdict == "unique_support"
        assert "vol" in rec.discovered_expression

    def test_partition_cli_executes_and_persists_assigned_window(self, tmp_path, monkeypatch):
        @dataclass
        class FakeRecord:
            asset: str
            window_id: str

        calls = []

        def fake_execute(asset, window, bootstrap):
            calls.append((asset, window.window_id, bootstrap))
            return FakeRecord(asset, window.window_id)

        monkeypatch.setattr(sweep, "execute_window_discovery", fake_execute)
        records = sweep.run_partition(0, sweep.PARTITION_COUNT, tmp_path)

        assert calls == [("sp500", WINDOWS[0].window_id, True)]
        assert len(records) == 1
        saved = json.loads((tmp_path / "partition_000.json").read_text(encoding="utf-8"))
        assert saved == [{"asset": "sp500", "window_id": WINDOWS[0].window_id}]
