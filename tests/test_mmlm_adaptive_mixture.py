"""Text track A with self-adjusting weights (apedp_mmlm/adaptive_mixture.py)."""
from __future__ import annotations

import numpy as np

from apedp_mmlm.adaptive_mixture import ENGINES, SelfWeightingA0
from apedp_mmlm.engines import MMLMTokenizer
from apedp_mmlm.models import NonNeuralMMLM_A0

TRAIN = ["the referee refuses a wrong law .", "the referee promotes a right law .", "a wrong law is refused .",
         "the search proposes a law .", "the referee refuses the search ."] * 6
TEST = ["the referee refuses a right law .", "the search proposes a wrong law ."]


def _setup():
    tokenizer = MMLMTokenizer(max_vocab_size=64).fit(TRAIN)
    return tokenizer, NonNeuralMMLM_A0(tokenizer).fit(TRAIN)


def test_weights_stay_a_distribution_and_move_with_the_evidence():
    tokenizer, model = _setup()
    mixer = SelfWeightingA0(model)
    start = mixer.state()
    costs = [c for line in TEST for c in mixer.line_bits(tokenizer.encode(line))]
    end = mixer.state()
    assert all(np.isfinite(costs)) and all(c >= 0 for c in costs)
    assert abs(sum(end.values()) - 1.0) < 1e-9 and set(end) == set(ENGINES)
    assert end != start and min(end.values()) >= mixer.share / len(ENGINES) - 1e-12


def test_the_mixture_is_never_far_behind_its_best_engine():
    """Bayesian mixing with a share costs at most a few bits more than the best single engine over a text."""
    tokenizer, model = _setup()
    mixer = SelfWeightingA0(model)
    mixed, single = 0.0, np.zeros(len(ENGINES))
    for line in TEST:
        ids = tokenizer.encode(line)
        for i in range(1, len(ids)):
            single -= np.log2(mixer.engine_probabilities(ids[:i])[:, ids[i]])
        mixed += sum(mixer.line_bits(ids))
    assert mixed <= single.min() + np.log2(len(ENGINES)) + 0.1 * sum(len(tokenizer.encode(line)) for line in TEST)


def test_weights_reset_per_line_when_asked():
    tokenizer, model = _setup()
    mixer = SelfWeightingA0(model, carry=False)
    ids = tokenizer.encode(TEST[0])
    assert mixer.line_bits(ids) == mixer.line_bits(ids)
