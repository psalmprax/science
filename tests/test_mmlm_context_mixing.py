"""MMLM-CM: the experts look only backwards, the bits add up, and the mixture learns."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from apedp_mmlm import neural_expert
from apedp_mmlm.context_mixing import (
    ALL_EXPERTS,
    ContextMixingLM,
    _counts_so_far,
    _matches,
    _word_contexts,
)

TEXT = (b"The referee promotes a law only when no distinct rival survives. "
        b"The search proposes laws; the referee tests them on rows it has not seen. ") * 60


def _model() -> ContextMixingLM:
    return ContextMixingLM(table_bits=16).fit_counts(TEXT).calibrate(TEXT[1000:4000])


def test_counts_so_far_counts_only_earlier_positions():
    zeros, ones = _counts_so_far(np.array([5, 3, 5, 5, 3, 9]), np.array([1, 0, 0, 1, 1, 0]))
    assert zeros.tolist() == [0, 0, 0, 1, 1, 0]
    assert ones.tolist() == [0, 0, 1, 1, 0, 0]


def test_match_expert_predicts_the_byte_that_followed_last_time():
    history = np.frombuffer(b"abcdefghijXYZ", dtype=np.uint8)
    predicted, length = _matches(history, np.frombuffer(b"..abcdefghijQ", dtype=np.uint8), 8)
    assert chr(predicted[-1]) == "X" and length[-1] == 10
    assert predicted[0] == -1 and length[0] == 0


def test_word_contexts_agree_for_the_same_word_prefix_and_previous_word():
    text = b"the cat. The cat sat; the cat"
    w1, w2 = _word_contexts(np.frombuffer(text, dtype=np.uint8))
    first, last = text.index(b"cat") + 2, len(text) - 1  # both just after "ca" of "cat" following "the"
    assert w1[first] == w1[last] and w2[first] == w2[last]
    assert w1[first] != w1[first - 1]


def test_no_expert_and_no_selector_sees_the_future():
    """Changing the bytes after position k must not change anything computed for the bytes before k."""
    model = _model()
    data = TEXT[5000:5600]
    altered = data[:300] + bytes(reversed(data[300:]))
    rng = np.random.default_rng(0)
    neural = rng.uniform(0.05, 0.95, (len(data), 8))
    a, b = model.features(data, neural), model.features(altered, neural)
    for name in a["columns"]:
        assert np.array_equal(a["columns"][name][: 300 * 8], b["columns"][name][: 300 * 8]), name
    for name in a["selectors"]:
        assert np.array_equal(a["selectors"][name][: 300 * 8], b["selectors"][name][: 300 * 8]), name
    model.fit_mixer(a, epochs=1)
    assert np.allclose(model.byte_bits(a)[:300], model.byte_bits(b)[:300])


def test_mixture_learns_and_beats_its_first_expert():
    model = _model()
    fit, held_out = model.features(TEXT[4000:7000]), model.features(TEXT[7000:8300])
    model.fit_mixer(fit, epochs=3)
    bits = model.byte_bits(held_out)
    assert bits.shape == (1300,) and np.all(bits > 0)
    assert bits.mean() < ContextMixingLM.expert_alone(held_out, "o0") < 8.0
    assert set(model.experts) == set(ALL_EXPERTS) - {"nn"}


def test_an_expert_can_be_left_out():
    model = _model()
    fit = model.features(TEXT[4000:6000])
    model.fit_mixer(fit, experts=("o0", "o2", "m8"), epochs=1)
    assert model.experts == ("o0", "o2", "m8")
    assert np.isfinite(model.byte_bits(fit)).all()


def test_neural_bit_probabilities_multiply_back_to_the_byte_probability():
    torch.manual_seed(0)
    net = neural_expert.ByteLSTM(hidden=16, embed=8)
    data = TEXT[:400]
    bits = neural_expert.bit_probabilities(net, data, segments=1)
    assert bits.shape == (400, 8) and np.all((bits > 0) & (bits < 1))
    with torch.no_grad():
        logits, _ = net(torch.frombuffer(bytearray(data), dtype=torch.uint8).long()[None, :])
        direct = torch.log_softmax(logits[0], dim=-1).numpy()
    array = np.frombuffer(data, dtype=np.uint8)
    for t in (1, 57, 399):  # byte t is predicted from bytes before t
        along_path = sum(math.log(bits[t, j] if (array[t] >> (7 - j)) & 1 else 1 - bits[t, j]) for j in range(8))
        assert along_path == pytest.approx(float(direct[t - 1, array[t]]), abs=1e-4)
    assert np.allclose(bits[0], 0.5)  # the first byte is predicted from nothing


def test_neural_expert_does_not_see_the_future():
    torch.manual_seed(1)
    net = neural_expert.ByteLSTM(hidden=16, embed=8)
    data = TEXT[:600]
    altered = data[:300] + bytes(reversed(data[300:]))
    a = neural_expert.bit_probabilities(net, data, segments=2)
    b = neural_expert.bit_probabilities(net, altered, segments=2)
    assert np.allclose(a[:300, 0], b[:300, 0])  # the first bit of a byte depends on earlier bytes only
    assert np.allclose(a[:299], b[:299])


def test_training_runs_for_its_budget_and_round_trips(tmp_path):
    model, info = neural_expert.train(TEXT, seconds=1.0, hidden=16, embed=8, batch=4, steps_per_batch=20)
    assert info["steps"] >= 1 and info["parameters"] == sum(p.numel() for p in model.parameters())
    neural_expert.save(model, tmp_path / "w.pt")
    again = neural_expert.load(tmp_path / "w.pt")
    assert np.allclose(neural_expert.bit_probabilities(model, TEXT[:100], segments=1),
                       neural_expert.bit_probabilities(again, TEXT[:100], segments=1))
