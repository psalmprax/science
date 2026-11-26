"""The look-ahead flip walk and the scorer it learns from its own descents (development harness)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import search_gf2_guided_walk as gw


@pytest.fixture(scope="module")
def binary(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return gw.build(tmp_path_factory.mktemp("guided") / "guided")


def test_plain_walk_is_exact_and_descends(binary: Path) -> None:
    start = len(gw.plain.start(5, "schoolbook"))
    row = gw.walk(binary, 5, seed=1, seconds=2)
    assert row["exact"]
    assert gw.KNOWN[5] <= row["best"] < start
    assert "triples" not in row                      # kept whole only below what is known


def test_log_holds_every_candidate_with_the_walks_own_probability(binary: Path, tmp_path: Path) -> None:
    log = tmp_path / "log.txt"
    row = gw.walk(binary, 6, seed=2, seconds=2, k=4, temperature=0.5, weights=gw.GREEDY, log=log)
    assert row["exact"]
    steps = gw.read_log(log)
    assert steps
    for pos, logpi, phi, pick in steps:
        assert 0 <= pos < 64 and 0 <= pick < 4
        assert logpi.shape == (4,) and phi.shape == (4, len(gw.FEATURES))
        assert abs(np.log(np.exp(logpi).sum())) < 1e-4    # a probability over the candidates
    assert any(pos == 0 for pos, *_ in steps)
    last = [s for s in steps if s[0] == 0]
    merges = sum(s[2][s[3], 1] == 1.0 for s in last)
    assert merges >= 0.5 * len(last)                  # the move that made a new best was mostly a merge


def _synthetic(v: np.ndarray, offsets_scale: float, n: int = 4000, k: int = 6, seed: int = 0) -> list[gw.Step]:
    rng = np.random.default_rng(seed)
    steps: list[gw.Step] = []
    for _ in range(n):
        phi = rng.normal(size=(k, len(v)))
        off = rng.normal(size=k) * offsets_scale
        off -= np.log(np.exp(off).sum())
        u = off + phi @ v
        p = np.exp(u - u.max())
        steps.append((0, off, phi, int(rng.choice(k, p=p / p.sum()))))
    return steps


def test_fit_recovers_planted_weights() -> None:
    v = np.array([0.0, 2.0, -1.0, 0.0, 0.5, 0.0])
    got = gw.fit(_synthetic(v, 0.0), l2=1.0)
    assert np.allclose(got, v, atol=0.25)


def test_fit_is_not_fooled_by_the_walks_own_preferences() -> None:
    """Moves chosen by a non-uniform policy alone carry no signal once the policy is the offset."""
    steps = _synthetic(np.zeros(6), offsets_scale=2.0, seed=3)
    got = gw.fit(steps, l2=1.0)
    assert np.abs(got).max() < 0.15
    assert abs(gw.gain_per_step(got, _synthetic(np.zeros(6), 2.0, seed=4))) < 0.01


def test_pairs_keeps_the_six_and_adds_their_products() -> None:
    x = np.arange(12, dtype=float).reshape(2, 6)
    out = gw.pairs(x)
    assert out.shape == (2, 6 + 15)
    assert np.array_equal(out[:, :6], x)


def test_a_corrupted_decomposition_is_refused() -> None:
    triples = gw.plain.start(5, "schoolbook")
    assert gw.plain.verified(triples, 5)
    a, b, c = triples[0]
    assert not gw.plain.verified([(a, b, c ^ 1), *triples[1:]], 5)
