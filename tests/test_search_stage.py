"""The default route's search stage follows PREREG-PORTFOLIO-SEARCH's rule: four searches where four cores are free."""
from __future__ import annotations

import pickle

from apedp_dyn.discovery_l2 import search_stage
from apedp_dyn.proposers.deep_pass import DeepPassProposer
from apedp_dyn.proposers.fit_guard import FitGuardProposer
from apedp_dyn.proposers.portfolio import PortfolioProposer
from apedp_dyn.proposers.pysr_proposer import PySRProposer


def test_four_cores_give_the_measured_portfolio():
    guarded, label = search_stage.default_search_stage(60.0, cores=4)
    assert isinstance(guarded, FitGuardProposer)  # an unfittable candidate is dropped, not fatal (audit M-24)
    stage = guarded._inner
    assert isinstance(stage, PortfolioProposer) and "four searches" in label
    members = stage._members
    assert [m.func for m in members] == [PySRProposer, DeepPassProposer, PySRProposer, PySRProposer]
    assert members[0].keywords["budget_seconds"] == 60.0
    assert stage._max == search_stage.MAX_PROPOSALS == 10
    assert stage._timeout == 60.0 + search_stage.MEMBER_GRACE_S
    for member in members:
        assert pickle.dumps(member)


def test_fewer_cores_give_the_single_search_the_refined_run_measured():
    for cores in (1, 2, 3):
        stage, label = search_stage.default_search_stage(60.0, cores=cores)
        assert isinstance(stage, FitGuardProposer) and isinstance(stage._inner, PySRProposer) and label == "PySR, 60 s"


def test_the_core_count_is_read_from_the_machine_when_not_given():
    assert search_stage.available_cores() >= 1
    stage, _ = search_stage.default_search_stage(30.0)
    expected = PortfolioProposer if search_stage.available_cores() >= 4 else PySRProposer
    assert isinstance(stage._inner, expected)
