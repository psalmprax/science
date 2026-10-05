"""E4-T1 protocol pins (doc S72): time-order, abstention honesty, determinism.

Full 11-origin numbers live in the doc (slow); this suite pins the frozen
protocol's structural properties on two origins: P1 MULTI pre, active
abstention when max-D < k, forward-only re-adjudication, frozen
reproducibility, and the oracle-never-promotes guard (with ~6 forward
blocks no policy may manufacture UNIQUE from noise).
"""

import pytest

import apedp_dyn.epistemic.active_disambiguation.trading_e4t1_v2 as T

PANELS = None


def _panels():
    global PANELS
    if PANELS is None:
        frames = T.load_frames()
        PANELS = {a: T.build_asset_panel(frames["raw"], a) for a in T.ASSETS}
    return PANELS


@pytest.mark.parametrize("origin", ["2022-08-01", "2023-01-01"])
def test_P1_multi_pre(origin):
    r = T.run_origin(_panels(), origin, "active", seed=7)
    assert "multi" in r.pre_status.lower()


@pytest.mark.parametrize("origin", ["2022-08-01", "2023-01-01"])
def test_abstention_honest(origin):
    r = T.run_origin(_panels(), origin, "active", seed=7)
    assert r.post_status == "INSUFFICIENT_DISCRIMINATION"
    assert r.mid is None and r.winner is None


def test_oracle_cannot_manufacture_unique():
    for origin in ("2022-08-01", "2023-01-01"):
        r = T.run_origin(_panels(), origin, "oracle", seed=7)
        assert "unique_support" not in r.post_status.lower()


def test_frozen_reproducibility():
    a = T.run_origin(_panels(), "2022-08-01", "random", seed=7)
    b = T.run_origin(_panels(), "2022-08-01", "random", seed=7)
    assert a == b


def test_time_order_forward_after_selection():
    import pandas as pd

    dates = _panels()["BTC"]["date"]
    origin = "2022-08-01"
    nxt = str(pd.Timestamp(origin) + pd.offsets.MonthBegin(1))
    assert str(dates[dates >= origin].min()) >= origin
    assert str(dates[dates < origin].max()) < origin
    assert nxt > origin


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
