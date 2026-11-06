"""Input-contract pin for RimeAdProposer.fit_and_propose.

The proposer accepts a DataFrame (or a dict of DataFrames), not an arbitrary
list-of-dicts structure. This pins that contract so a future “copy from the
docs” usage path does not silently accept a different input shape.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from apedp_dyn.proposers.rime_ad_proposer import RimeAdProposer


def _synthetic():
    x = np.linspace(0.2, 8.0, 80)
    y = 2.0 * np.floor(x / 1.5) + 0.5 * x + 0.08 * (x ** 2)
    return x, y


@pytest.fixture
def composite_case():
    x, y = _synthetic()
    return pd.DataFrame({"x": x, "target": y})


def test_dataframe_input_produces_composite_candidate(composite_case):
    prop = RimeAdProposer()
    cands = prop.fit_and_propose(composite_case, target_col="target")
    comps = [c for c in cands if c.candidate_id == "tv_smooth_composite"]
    assert comps, "expected the composite candidate on the pinned example"
    r2 = comps[0].metadata["r2"]
    assert abs(r2 - 0.999965) < 1e-4, (r2, comps[0].metadata)


def test_dict_bundle_is_accepted(composite_case):
    """A dict-of-DataFrames bundle is an accepted input shape."""
    prop = RimeAdProposer()
    bundle = {"bundle": composite_case}
    cands_bundle = prop.fit_and_propose(bundle, target_col="target")
    r2_bundle = next(
        (c.metadata["r2"] for c in cands_bundle if c.candidate_id == "tv_smooth_composite"),
        None,
    )
    assert r2_bundle is not None, [c.candidate_id for c in cands_bundle]


def test_list_of_dicts_fails_loudly_instead_of_returning_a_candidate(composite_case):
    """A JSON-style list-of-records is NOT an accepted input shape.

    The contract is `DataFrame | dict[str, DataFrame]`. A list of records reaches
    the `.columns` access in `fit_and_propose` and raises; it does not quietly
    return a candidate fitted on something else, which is the failure mode that
    would matter. The raised type is an AttributeError rather than a TypeError —
    unhelpful, and the reason the accept/reject guard belongs at the top of the
    method; pinning the current behaviour keeps the door open for that fix while
    asserting the property that matters (no silent wrong answer).
    """
    prop = RimeAdProposer()
    as_list = composite_case.to_dict(orient="records")
    with pytest.raises((AttributeError, TypeError, ValueError)):
        prop.fit_and_propose(as_list, target_col="target")
