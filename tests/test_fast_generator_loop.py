"""A fast hosted model in the generator loop: the timing, the ceiling on calls, the choice of model, the fresh laws.

No model and no key: scripted clients stand in, as in ``tests/test_evolution.py``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_invented_laws as grammar
import run_fast_generator_loop as fast
import run_search_evolution as rse

from apedp_dyn.evolution.seed_generator import SEED_GENERATOR

GOOD = "THINK: fit constants inside each shape.\n```python\n" + rse.IMPROVED_GENERATOR + "```"


class _Clock:
    """A clock the test moves: every answer of the scripted client takes ``answer`` seconds."""

    def __init__(self) -> None:
        self.now, self.slept = 0.0, []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(round(seconds, 3))
        self.now += seconds


def _answering(clock: _Clock, replies: list[str], seconds: float = 2.0, tokens: int = 100):
    def client(prompt: str) -> str:
        clock.now += seconds
        if not replies:
            raise RuntimeError("unreachable")
        client.last = {"usage": {"completion_tokens": tokens, "prompt_tokens": 50}}  # type: ignore[attr-defined]
        return replies.pop(0)

    return client


def test_the_clock_starts_after_the_wait_between_calls() -> None:
    clock = _Clock()
    timed = fast.TimedClient(_answering(clock, ["a", "b", "c"]), max_calls=10, interval=1.5, sleep=clock.sleep, clock=clock)
    assert [timed("p"), timed("p"), timed("p")] == ["a", "b", "c"]
    assert clock.slept == [1.5, 1.5]                         # no wait before the first call; then the service's interval
    assert [r["seconds"] for r in timed.records] == [2.0, 2.0, 2.0]   # the model's own time, without the waits
    cost = timed.time()
    assert cost == {"calls": 3, "answered": 3, "model_seconds": 6.0, "median_seconds_a_call": 2.0,
                    "slowest_call_seconds": 2.0, "completion_tokens": 300, "tokens_a_second": 50.0}


def test_the_ceiling_on_calls_stops_the_client_and_a_failed_call_is_recorded() -> None:
    clock = _Clock()
    timed = fast.TimedClient(_answering(clock, ["a"]), max_calls=2, interval=0.0, sleep=clock.sleep, clock=clock)
    assert timed("p") == "a"
    with pytest.raises(RuntimeError, match="unreachable"):
        timed("p")
    assert timed.records[-1] == {"ok": False, "seconds": 2.0, "error": "RuntimeError"}
    with pytest.raises(fast.CallLimitReachedError):
        timed("p")
    assert timed.calls == 2 and len(timed.records) == 2      # the refused call reached no model


def test_the_models_tried_are_the_known_ones_then_names_that_suggest_speed() -> None:
    listed = ["acme/giant-900b", "acme/tiny-flash", fast.KNOWN[1], "acme/coder-7b", "zeta/embed-fast", fast.KNOWN[0]]
    chosen = fast.candidates(listed, limit=4)
    assert chosen[:2] == [fast.KNOWN[0], fast.KNOWN[1]]       # known first, in their own order, only if the key reaches them
    assert chosen[2:] == ["acme/coder-7b", "acme/tiny-flash"]
    assert "acme/giant-900b" not in fast.candidates(listed, limit=10)
    assert fast.candidates([], limit=3) == []


def test_the_model_picked_is_the_one_that_gave_a_running_version_soonest(tmp_path: Path) -> None:
    clock = _Clock()
    scripts = {"acme/slow-flash": (["THINK: x\n```python\n" + SEED_GENERATOR + "```"], 9.0),
               "acme/quick-flash": ([GOOD], 3.0),
               "acme/chatty-mini": (["I would rather talk about it."], 1.0),
               "acme/broken-coder": ([], 0.5)}

    def make(model: str):
        replies, seconds = scripts[model]
        return _answering(clock, list(replies), seconds=seconds)

    record = fast.speed(tmp_path, 6, make, sorted(scripts), clock=clock)
    rows = {row["model"]: row for row in record["rows"]}
    assert rows["acme/quick-flash"]["ran"] is True and rows["acme/quick-flash"]["seconds"] == 3.0
    assert rows["acme/slow-flash"]["ran"] is True              # the seed again is still a version that runs
    assert rows["acme/chatty-mini"] == {"model": "acme/chatty-mini", "answered": True, "seconds": 1.0,
                                        "completion_tokens": 100, "tokens_a_second": 100.0, "code": False, "ran": False}
    assert rows["acme/broken-coder"]["answered"] is False and "unreachable" in rows["acme/broken-coder"]["error"]
    assert record["picked"] == "acme/quick-flash"              # not the fastest to answer: the fastest whose code ran
    assert json.loads((tmp_path / "speed.json").read_text())["picked"] == "acme/quick-flash"
    assert fast.pick([{"model": "a", "ran": False, "seconds": 1.0}]) is None


def test_the_report_says_where_the_time_went() -> None:
    clock = _Clock()
    timed = fast.TimedClient(_answering(clock, ["a", "b"], seconds=5.0), max_calls=9, interval=0.0, sleep=clock.sleep,
                             clock=clock)
    timed("p")
    timed("p")
    report = fast.with_time({"seconds": 50.0, "versions": 4}, timed)
    assert report["time"]["model_seconds"] == 10.0 and report["time"]["everything_else_seconds"] == 40.0
    assert report["time"]["share_of_the_time_spent_waiting_for_the_model"] == 0.2
    assert report["time"]["seconds_a_version_outside_the_model"] == 10.0
    assert report["ceiling_on_calls"] == 9


def test_fresh_laws_share_no_structure_with_the_laws_of_the_loop() -> None:
    of_the_loop = {grammar.structure(sp.sympify(law["text"])) for law in rse.draw_laws(rse.TRAIN + rse.VALIDATION + rse.TEACH)}
    fresh = fast.fresh_laws(12)
    assert len(fresh) == 12 and len({law["text"] for law in fresh}) == 12
    assert all(law["name"].startswith("fresh_") for law in fresh)
    assert not of_the_loop & {grammar.structure(sp.sympify(law["text"])) for law in fresh}
    assert fresh == fast.fresh_laws(12)                        # drawn from a fixed seed


def test_the_seed_against_itself_on_fresh_laws_gains_and_loses_nothing() -> None:
    record = fast.compare_on_fresh(SEED_GENERATOR, 8, seconds_per_call=1.5)
    assert record["the_champion_is_the_seed"] is True
    assert record["seed"]["ran"] and record["champion"]["ran"]
    assert record["seed"]["laws_proposed"] == record["champion"]["laws_proposed"]
    assert (record["gained_by_the_champion"], record["lost_by_the_champion"], record["could_be_chance_p"]) == (0, 0, 1.0)


def test_the_sign_test() -> None:
    assert fast.sign_test(0, 0) == 1.0
    assert fast.sign_test(5, 0) == pytest.approx(2 / 32)
    assert fast.sign_test(3, 3) == 1.0
    assert fast.sign_test(8, 1) == pytest.approx(2 * (1 + 9) / 512)


def test_without_a_key_nothing_is_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in ("CODING_MODEL_API_KEY", "NVIDIA_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(SystemExit, match="Nothing was run"):
        fast.main(["speed", "--out", str(tmp_path)])
    with pytest.raises(SystemExit, match="gave no champion"):
        fast.main(["fresh", "--out", str(tmp_path)])
