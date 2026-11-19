"""The LLM proposer's guards (with a scripted stand-in for the model) and the PCMCI+ proposer."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from apedp_dyn.proposers import llm_proposer as lp
from apedp_dyn.proposers import pcmci_proposer as pc

RNG = np.random.default_rng(0)
FRAME = pd.DataFrame({"mass": RNG.uniform(1, 5, 200), "speed": RNG.uniform(1, 5, 200)})
FRAME["target"] = 0.5 * FRAME.mass * FRAME.speed ** 2


class Scripted:
    def __init__(self, probe: str, rounds: list[str]):
        self.probe, self.rounds, self.prompts = probe, list(rounds), []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if "No data is given" in prompt:
            return self.probe
        return self.rounds.pop(0) if self.rounds else ""


def test_a_structure_found_from_data_is_fitted_and_not_marked_recited():
    client = Scripted(probe="c0*x0 + c1*x1", rounds=["1. y = c0*x0*x1\n2. c0*x0*x1**2\n3. c0*exp(x0)", "c0*x0*x1^2 + c1"])
    best = lp.LlmEquationProposer(client, rounds=2).fit_and_propose(FRAME)[0]
    assert best.parameters["dev_relative_rmse"] < 1e-6 and abs(best.parameters["c0"] - 0.5) < 1e-6
    assert best.metadata["recited_before_data"] is False
    assert np.allclose(best.compute_score(FRAME), FRAME.target)
    assert best.metadata["selection_pool_size"] == 4


def test_a_law_the_model_states_before_seeing_data_is_marked_as_recall():
    client = Scripted(probe="y = c0*x0*x1**2", rounds=["0.5*x0*x1**2"])
    best = lp.LlmEquationProposer(client, rounds=1, description="kinetic energy").fit_and_propose(FRAME)[0]
    assert best.parameters["dev_relative_rmse"] < 1e-6
    assert best.metadata["recited_before_data"] is True


def test_the_model_never_sees_the_columns_names():
    client = Scripted(probe="", rounds=["c0*x0"])
    lp.LlmEquationProposer(client, rounds=1).fit_and_propose(FRAME)
    assert all("mass" not in p and "speed" not in p for p in client.prompts)
    assert "x0" in client.prompts[-1]


def test_scores_of_earlier_attempts_are_fed_back():
    client = Scripted(probe="", rounds=["c0*x0", "c0*x0*x1**2"])
    lp.LlmEquationProposer(client, rounds=2).fit_and_propose(FRAME)
    assert "c0*x0" in client.prompts[-1] and "relative error" in client.prompts[-1]


@pytest.mark.parametrize("text", [
    "__import__('os').system('true')", "x0.__class__", "open('f')", "x0; x1", "lambda: 1", "x9 + 1", "x0 if x1 else 2",
    "[x0]", "x0 @ x1", '"a"',
])
def test_text_outside_the_whitelist_is_never_evaluated(text):
    with pytest.raises(lp.UnsafeExpressionError):
        lp.parse_skeleton(text, n_vars=2)


def test_unsafe_replies_are_counted_and_skipped():
    client = Scripted(probe="", rounds=["__import__('os').getcwd()\nc0*x0*x1**2"])
    best = lp.LlmEquationProposer(client, rounds=1).fit_and_propose(FRAME)[0]
    assert best.metadata["replies_rejected_as_unsafe_or_unparseable"] == 1


def test_structure_ignores_the_values_of_constants():
    a = lp.structure_key(lp.parse_skeleton("c0*x0*x1**2", 2))
    assert a == lp.structure_key(lp.parse_skeleton("0.5*x0*x1**2", 2)) == lp.structure_key(lp.parse_skeleton("c3*x1**2*x0", 2))
    assert a != lp.structure_key(lp.parse_skeleton("c0*x0*x1", 2))


def _echo_world(seed: int, n: int = 600) -> pd.DataFrame:
    r = np.random.default_rng(seed)
    x1, x2, y = np.zeros(n), np.zeros(n), np.zeros(n)
    for t in range(2, n):
        x1[t] = 0.6 * x1[t - 1] + r.standard_normal()
        x2[t] = 0.8 * x1[t - 1] + 0.5 * r.standard_normal()  # an echo of x1, not a cause of y
        y[t] = 0.7 * y[t - 1] + 0.5 * x1[t - 2] + 0.5 * r.standard_normal()
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": r.standard_normal(n), "target": y})


@pytest.mark.skipif(not pc.is_available(), reason="tigramite is not installed")
def test_pcmci_keeps_the_direct_parents_and_drops_the_echo():
    exact = 0
    for seed in range(5):
        frame = _echo_world(seed)
        assert abs(np.corrcoef(frame.target[1:], frame.x2[:-1])[0, 1]) > 0.4  # the echo is strongly correlated
        proposal = pc.PcmciProposer().fit_and_propose(frame)[0]
        exact += {(link["source"], link["lag"]) for link in proposal.metadata["links"]} == {("target", 1), ("x1", 2)}
    assert exact >= 4
    score = proposal.compute_score(frame)
    assert np.isnan(score[:2]).all() and np.isfinite(score[2:]).all()  # uses the past only
    assert proposal.parameters["dev_r2"] > 0.8


def test_pcmci_says_so_when_tigramite_is_missing(monkeypatch):
    monkeypatch.setattr(pc, "is_available", lambda: False)
    with pytest.raises(pc.PcmciUnavailableError):
        pc.PcmciProposer().fit_and_propose(_echo_world(0))
