"""The video comparison module's process-wide state (call counters, clip size) must not pass from one test to the next."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "packages" / "jetflow-video") not in sys.path:
    sys.path.insert(0, str(ROOT / "packages" / "jetflow-video"))

from jetflow_video import head_to_head as h

from tests import conftest as suite_hooks


def _calls(model: torch.nn.Module) -> int:
    counter = h.counter_for(model)
    model(torch.zeros(1, 2))
    return int(counter.calls)


def test_a_counter_inherited_from_another_model_counts_nothing_and_forgetting_cures_it() -> None:
    first, second = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    assert _calls(first) == 1
    # What a reused id does: the second model finds the first one's counter under its own id.
    h._COUNTERS[id(second)] = h._COUNTERS[id(first)]
    assert _calls(second) == 0                    # the fault seen on stage: the model ran and no call was counted
    assert suite_hooks.forget_video_call_counters() >= 1
    assert _calls(second) == 1


def test_a_counter_left_behind_by_this_test() -> None:
    assert _calls(torch.nn.Linear(2, 2)) == 1
    assert h._COUNTERS                            # left for the next test to find, if nothing forgot it


def test_every_test_starts_with_no_counters() -> None:
    # Whatever ran before in this process, the test above included, left none behind.
    assert not h._COUNTERS


def test_a_size_set_by_this_test() -> None:
    assert (h.FRAMES, h.SIZE, h.JETFLOW["hidden_dim"]) == (6, 16, 48)
    h.FRAMES, h.SIZE = 8, 32                      # what run_prereg_video_hybrid.set_size does, and never undoes
    h.JETFLOW = {**h.JETFLOW, "hidden_dim": 64}


def test_every_test_starts_at_the_module_s_own_size() -> None:
    # The sizes the module is written with, whatever the test above (or any other in this process) set.
    assert (h.FRAMES, h.SIZE, h.JETFLOW["hidden_dim"]) == (6, 16, 48)


def test_forgetting_does_nothing_when_the_module_is_not_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "jetflow_video.head_to_head")
    assert suite_hooks.forget_video_call_counters() == 0
