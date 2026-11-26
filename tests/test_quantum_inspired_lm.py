"""Pins for the quantum-inspired byte models and their classical controls (apedp_mmlm/quantum_inspired.py)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from apedp_mmlm import quantum_inspired as qi

ARMS = ("pure-complex-6", "pure-real-6", "mixed-complex-5-2", "mixed-real-5-3", "hmm-12")


@pytest.mark.parametrize("arm", ARMS)
def test_every_model_gives_a_distribution_and_its_step_agrees(arm: str) -> None:
    torch.manual_seed(0)
    model = qi.build(arm, seed=1)
    ops = model.operators()
    state = model.initial(4)
    x = torch.tensor([3, 97, 255, 32])
    for _ in range(5):
        dist = model.distribution(ops, state)
        assert torch.allclose(dist.sum(-1), torch.ones(4, dtype=dist.dtype), atol=1e-5)
        assert bool((dist >= 0).all())
        logp, state = model.step(ops, state, x)
        assert torch.allclose(torch.log(dist[torch.arange(4), x]), logp, atol=1e-4)
        x = (x * 7 + 11) % 256


@pytest.mark.parametrize("arm", ARMS[:4])
def test_the_operators_are_complete(arm: str) -> None:
    model = qi.build(arm, seed=2)
    k = model.operators().reshape(-1, model.dim, model.dim)
    total = torch.einsum("nji,njk->ik", k.conj(), k)
    assert torch.allclose(total, torch.eye(model.dim, dtype=total.dtype), atol=1e-5)


def test_a_pure_state_model_is_a_born_machine_on_a_matrix_product_state() -> None:
    """The product of its next-byte probabilities is |K_xT ... K_x1 psi0|^2."""
    model = qi.build("pure-complex-5", seed=3)
    ops = model.operators()[:, 0]
    seq = torch.tensor([10, 200, 10, 65, 66, 67])
    state = model.initial(1)
    total = 0.0
    with torch.no_grad():
        for x in seq:
            logp, state = model.step(model.operators(), state, x.view(1))
            total += float(logp)
        psi = model.initial(1)[0]
        for x in seq:
            psi = ops[x] @ psi
    assert math.isclose(total, math.log(float((psi.abs() ** 2).sum())), rel_tol=1e-4)


def test_a_mixed_state_model_is_linear_in_rho() -> None:
    """The product of its probabilities is the trace of the unnormalised state: a weighted automaton in rho."""
    model = qi.build("mixed-complex-4-2", seed=4)
    seq = torch.tensor([1, 2, 3, 250, 0])
    with torch.no_grad():
        ops = model.operators()
        state = model.initial(1)
        total = 0.0
        for x in seq:
            logp, state = model.step(ops, state, x.view(1))
            total += float(logp)
        rho = model.initial(1)[0]
        for x in seq:
            rho = sum(ops[x, r] @ rho @ ops[x, r].conj().T for r in range(2))
    assert math.isclose(total, math.log(float(torch.diagonal(rho).real.sum())), rel_tol=1e-4)


def test_training_learns_a_repeating_text() -> None:
    torch.manual_seed(0)
    text = b"abcab" * 4000
    model = qi.build("pure-complex-4", seed=0)
    before = qi.bits_per_byte(model, text[:5000], chunks=4)
    trained = qi.train(model, text, text[:5000], seconds=10, batch=8, segment=32, lr=3e-2, evaluate_every=2, valid_chunks=4)
    assert trained.best_valid_bpb < 0.5 * before
    assert trained.best_valid_bpb < math.log2(3)          # better than the bytes' own frequencies
    assert qi.bits_per_byte(trained.model, text[:5000], chunks=4) == pytest.approx(trained.best_valid_bpb, abs=1e-6)


@pytest.mark.parametrize("order", [0, 1, 2, 3])
def test_kneser_ney_sums_to_one_over_the_next_byte(order: int) -> None:
    rng = np.random.default_rng(order)
    train = bytes(rng.choice(np.frombuffer(b"the cat sat on a mat ", dtype=np.uint8), size=3000).tolist())
    history = b"at sa"[-order:] if order else b""
    total = sum(float(qi.kneser_ney_probabilities(train, b"zz" + history + bytes([w]), order)[-1]) for w in range(256))
    assert total == pytest.approx(1.0, abs=1e-9)


def test_the_expectations_are_read_as_written() -> None:
    import run_prereg_mmlm_qi as run

    bpb = dict.fromkeys(run.ARMS, 3.0)
    bpb.update({"pure-complex-16": 2.5, "pure-complex-32": 2.4, "pure-complex-64": 2.3, "hmm-256": 2.6, "hmm-1024": 2.5,
                "pure-real-23": 2.55, "pure-real-45": 2.45, "pure-real-90": 2.33,
                "mixed-complex-16-2": 2.45, "mixed-complex-32-2": 2.39, "pure-complex-16@1": 2.51, "hmm-256@1": 2.6,
                "pure-complex-23": 2.5, "pure-complex-45": 2.4, "pure-complex-16@2": 2.49, "pure-complex-32@1": 2.4,
                "pure-complex-32@2": 2.4, "hmm-256@2": 2.6, "hmm-1024@1": 2.5, "hmm-1024@2": 2.5})
    kn = {f"Kneser-Ney, {k} bytes of history": v for k, v in enumerate((4.6, 3.4, 2.7, 2.2, 1.85))}
    hybrid = dict.fromkeys(run.ARMS, 1.84)
    for arm, value in {"pure-complex-16": 1.82, "pure-complex-32": 1.80, "hmm-256": 1.83, "hmm-1024": 1.83}.items():
        hybrid.update({arm: value, f"{arm}@1": value, f"{arm}@2": value})
    selection = dict(bpb)
    e = run.expectations(bpb, kn, hybrid, counting_alone=1.85, selection=selection)
    assert e["best_quantum_inspired_arm"] == "pure-complex-64"
    assert e["E1 learns more than three-byte statistics"]["held"]
    assert e["E2 beats a hidden Markov model with D^2 states by 0.05"]["margin_by_D_on_three_seed_means"] == {16: 0.1, 32: 0.1}
    assert e["E2 beats a hidden Markov model with D^2 states by 0.05"]["held"]
    assert e["E3 complex beats real of the same size by 0.02"]["margin_by_D"] == {16: 0.05, 32: 0.05, 64: 0.03}
    assert e["E3 complex beats real of the same size by 0.02"]["held"]
    e4 = e["E4 a mixed state beats a pure one of the same size by 0.02"]
    assert not e4["held"] and e4["margin_by_D"] == {16: 0.05, 32: 0.01}       # against pure 23 and 45
    assert e["E5 better with a larger state"]["held"]
    assert not e["E6 beats the strongest Kneser-Ney"]["held"]
    assert e["E2 beats a hidden Markov model with D^2 states by 0.05"]["larger_seed_spread_by_D"][16] == pytest.approx(0.02)
    del bpb["pure-real-90"]                                                  # an arm that failed: only E3 is not read
    partial = run.expectations(bpb, kn, hybrid, counting_alone=1.85, selection=selection)
    assert partial["E3 complex beats real of the same size by 0.02"]["held"] is None
    assert partial["E2 beats a hidden Markov model with D^2 states by 0.05"]["held"] is not None
    assert partial["E1 learns more than three-byte statistics"]["held"] is None        # needs all 15 arms
    del bpb["hmm-256@2"]                                                     # a missing seed: E2 is not read
    assert run.expectations(bpb, kn, hybrid, counting_alone=1.85, selection=selection)[
        "E2 beats a hidden Markov model with D^2 states by 0.05"]["held"] is None
    e7 = e["E7 mixed with counting it adds what an automaton of its size does not"]
    assert e7["counting_alone_minus_hybrid_by_D"] == {16: 0.03, 32: 0.05}
    assert e7["hmm_hybrid_minus_hybrid_by_D"] == {16: 0.01, 32: 0.03}
    assert not e7["held"]                                                    # 0.01 at D = 16


def test_the_hybrid_mixture_never_loses_to_either_part_on_the_bytes_it_was_chosen_on() -> None:
    rng = np.random.default_rng(1)
    a, b = rng.uniform(0.01, 1, 5000), rng.uniform(0.01, 1, 5000)
    w = qi.best_mixture_weight(a, b)
    assert qi.mixture(a, b, w) <= min(qi.mixture(a, b, 1.0), qi.mixture(a, b, 0.0)) + 1e-12
    model = qi.build("pure-complex-4", seed=0)
    text = b"hello world " * 300
    probs = qi.byte_probabilities(model, text, chunks=4)
    assert len(probs) == len(text) // 4 * 4
    assert qi.bits_per_byte(model, text, chunks=4) == pytest.approx(float(-np.log2(probs).mean()))
