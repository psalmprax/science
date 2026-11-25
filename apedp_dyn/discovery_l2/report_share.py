"""The share of wrong laws among those a method reports: when it is shown to be small enough.

Status: experimental

Five plans up to PREREG-REAL-TABLES-2026-11 asked one thing of this share: at most 5 % of the
reports wrong, with at least 20 reports. That passes 2 wrong of 44, whose 95 % upper bound is
13.6 %: the test "held" without showing that the true share is small. Two independent reviews said
so (``preregistration/REVIEW-VOUCHING-V2-2026-11.md``, ``REVIEW-REAL-TABLES-2026-11.md``).

By the owner's decision of 2026-11-25 the limit has its own name and a second condition
(``specifications/discovery_checks_v1.json``, pinned by ``apedp_dyn/conf/discovery_checks.py``):

* **missed** if the share of wrong reports is above ``REPORTED_WRONG_SHARE_MAX`` (5 %);
* **held** if it is not, and its one-sided exact upper bound at
  ``REPORTED_WRONG_SHARE_BOUND_CONFIDENCE`` (95 %) is at most ``REPORTED_WRONG_SHARE_BOUND_MAX`` (10 %);
* **not shown** otherwise: the reports seen do not decide. No report at all is not shown.

What "held" needs: no wrong report in 29, one in 46, two in 61, three in 76.

For plans frozen from that date. Recorded results keep the verdicts their own plans fixed; ``verdict``
can be run on their counts to say what the new rule would have said, and that is all it says.
"""
from __future__ import annotations

from typing import Any

from scipy.stats import beta

from apedp_dyn.conf.discovery_checks import (
    REPORTED_WRONG_SHARE_BOUND_CONFIDENCE,
    REPORTED_WRONG_SHARE_BOUND_MAX,
    REPORTED_WRONG_SHARE_MAX,
)

HELD, MISSED, NOT_SHOWN = "held", "missed", "not shown"


def upper_bound(wrong: int, reported: int, confidence: float = REPORTED_WRONG_SHARE_BOUND_CONFIDENCE) -> float | None:
    """The one-sided exact (Clopper-Pearson) upper bound of the share of wrong reports; None with no report."""
    wrong, reported = int(wrong), int(reported)
    if wrong < 0 or reported < 0 or wrong > reported:
        raise ValueError(f"{wrong} wrong of {reported} reported is not a count")
    if reported == 0:
        return None
    if wrong == reported:
        return 1.0
    return float(beta.ppf(confidence, wrong + 1, reported - wrong))


def verdict(wrong: int, reported: int) -> dict[str, Any]:
    """Whether the share of wrong reports is held, missed or not shown, with the numbers behind it."""
    bound = upper_bound(wrong, reported)
    share = None if not reported else wrong / reported
    if share is None or bound is None:
        outcome = NOT_SHOWN
    elif share > REPORTED_WRONG_SHARE_MAX:
        outcome = MISSED
    elif bound <= REPORTED_WRONG_SHARE_BOUND_MAX:
        outcome = HELD
    else:
        outcome = NOT_SHOWN
    return {"verdict": outcome, "wrong": int(wrong), "reported": int(reported), "share": share,
            "share_allowed": REPORTED_WRONG_SHARE_MAX, "upper_bound": bound,
            "upper_bound_confidence": REPORTED_WRONG_SHARE_BOUND_CONFIDENCE,
            "upper_bound_allowed": REPORTED_WRONG_SHARE_BOUND_MAX}


def reports_needed(wrong: int) -> int:
    """The fewest reports with ``wrong`` wrong among them for which the share is held."""
    reported = max(int(wrong), 1)
    while verdict(wrong, reported)["verdict"] != HELD:
        reported += 1
        if reported > 100_000:
            raise ValueError(f"no number of reports makes {wrong} wrong a held share")
    return reported
