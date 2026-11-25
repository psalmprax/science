"""The share of wrong laws among those reported: held, missed or not shown."""
from __future__ import annotations

import pytest

from apedp_dyn.conf import discovery_checks
from apedp_dyn.discovery_l2 import report_share


def test_a_share_inside_the_limit_is_held_only_if_its_upper_bound_is_small_too() -> None:
    assert report_share.verdict(0, 189)["verdict"] == report_share.HELD      # the second checks' handed reports
    assert report_share.verdict(2, 61)["verdict"] == report_share.HELD
    for wrong, reported in ((2, 44), (2, 41), (0, 17), (3, 60)):            # inside 5 %, and not enough reports
        got = report_share.verdict(wrong, reported)
        assert got["verdict"] == report_share.NOT_SHOWN
        assert got["share"] <= discovery_checks.REPORTED_WRONG_SHARE_MAX < got["upper_bound"]


def test_a_share_above_the_limit_is_missed_whatever_its_bound() -> None:
    assert report_share.verdict(3, 44)["verdict"] == report_share.MISSED
    assert report_share.verdict(19, 231)["verdict"] == report_share.MISSED   # the route alone, handed counterfeits
    assert report_share.verdict(5, 5)["upper_bound"] == 1.0


def test_no_report_decides_nothing() -> None:
    got = report_share.verdict(0, 0)
    assert (got["verdict"], got["share"], got["upper_bound"]) == (report_share.NOT_SHOWN, None, None)


def test_the_bound_is_the_exact_one_sided_bound() -> None:
    assert report_share.upper_bound(0, 29) == pytest.approx(1 - 0.05 ** (1 / 29))      # no wrong report: closed form
    assert report_share.upper_bound(2, 44) == pytest.approx(0.136259, abs=1e-6)         # as the blind test recorded
    assert report_share.upper_bound(2, 41) == pytest.approx(0.145706, abs=1e-6)
    with pytest.raises(ValueError, match="not a count"):
        report_share.upper_bound(3, 2)


def test_what_held_needs() -> None:
    """The docstring's numbers: none wrong in 29, one in 46, two in 61, three in 76."""
    assert [report_share.reports_needed(wrong) for wrong in range(4)] == [29, 46, 61, 76]
    assert report_share.verdict(0, 28)["verdict"] == report_share.NOT_SHOWN


def test_the_verdict_carries_the_limits_it_used() -> None:
    got = report_share.verdict(1, 46)
    assert got["share_allowed"] == discovery_checks.REPORTED_WRONG_SHARE_MAX
    assert got["upper_bound_allowed"] == discovery_checks.REPORTED_WRONG_SHARE_BOUND_MAX
    assert got["upper_bound_confidence"] == discovery_checks.REPORTED_WRONG_SHARE_BOUND_CONFIDENCE
