"""Quantum-inspired byte language models, trained, and the classical models they must beat (MMLM-QI).

Status: experimental

The owner asked on 26 November for a test of the proposal MMLM-QI-001 (a language model built on
quantum states, Kraus operators and tensor trains instead of a neural network). Three facts shape it:

* A Kraus model with a state of size D predicts byte x with probability Tr(K_x rho K_x^dagger) and
  updates rho to K_x rho K_x^dagger over that probability. The completeness condition
  sum_x K_x^dagger K_x = I makes the probabilities sum to one for every state, so training needs
  only the operator of the byte that came: O(D^2) per byte for a pure state, O(r D^3) for a mixed one
  with r operators per byte. The operators are the row blocks of Q from a QR decomposition of a free
  matrix, so completeness holds exactly at every step (the proposal's construction), and gradients go
  through the QR.
* With a pure state (one operator per byte) this is the same model as a uniform matrix product state
  in left-canonical form used as a Born machine, read left to right; so the proposal's candidates B
  (tensor train) and C (open system) meet here, C being the mixed-state generalisation of B.
* rho evolves linearly, so the model is a weighted automaton with D^2 states (Hermitian matrices form
  a real space of that dimension; D(D+1)/2 for real symmetric ones). A test of "quantum" structure is
  only fair against a classical automaton of that size: here a hidden Markov model with N = D^2 states
  (a nonnegative weighted automaton), trained the same way for the same time.

Everything is by gradient on the bytes (Adam, truncated back-propagation through time with the state
carried between segments), for a wall-clock budget, keeping the parameters that score best on held-out
text. ``kneser_ney_bits`` is the interpolated Kneser-Ney byte model, a counting baseline.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

V = 256
FLOOR = 1e-30                                   #: a byte's probability never below this (no log of zero, no NaN state)
Ops = Tensor | tuple[Tensor, Tensor]            #: Kraus operators, or a hidden Markov model's transition and emission


class KrausLM(nn.Module):
    """Byte model on a state of size ``dim``: pure (``rank`` 1) or mixed, complex or real."""

    def __init__(self, dim: int, rank: int = 1, complex_valued: bool = True, seed: int = 0) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        shape = (V * rank * dim, dim)
        raw = torch.randn(shape, generator=g)
        start = torch.randn(dim, generator=g)
        if complex_valued:
            raw = torch.complex(raw, torch.randn(shape, generator=g))
            start = torch.complex(start, torch.randn(dim, generator=g))
        self.dim, self.rank, self.complex_valued = dim, rank, complex_valued
        self.raw = nn.Parameter(raw)
        self.start = nn.Parameter(start)

    def operators(self) -> Tensor:
        """(V, rank, dim, dim); the row blocks of an isometry, so sum of K^dagger K is the identity."""
        q, _ = torch.linalg.qr(self.raw)
        ops: Tensor = q.reshape(V, self.rank, self.dim, self.dim)
        return ops

    def initial(self, batch: int) -> Tensor:
        psi = self.start / torch.linalg.vector_norm(self.start)
        if self.rank == 1:
            pure: Tensor = psi.expand(batch, self.dim).clone()
            return pure
        rho = torch.outer(psi, psi.conj())
        mixed: Tensor = rho.expand(batch, self.dim, self.dim).clone()
        return mixed

    def step(self, ops: Ops, state: Tensor, x: Tensor) -> tuple[Tensor, Tensor]:
        """log p(x | state) and the state after x, for a batch of states and bytes."""
        if not isinstance(ops, Tensor):
            raise TypeError("Kraus operators expected")
        k = ops[x]                                                    # (B, rank, D, D)
        if self.rank == 1:
            phi = torch.einsum("bij,bj->bi", k[:, 0], state)
            p = (phi.abs() ** 2).sum(-1).clamp_min(FLOOR)
            return torch.log(p), phi / torch.sqrt(p)[:, None]
        sigma = torch.einsum("brij,bjk,brlk->bil", k, state, k.conj())
        p = torch.diagonal(sigma, dim1=-2, dim2=-1).real.sum(-1).clamp_min(FLOOR)
        return torch.log(p), sigma / p[:, None, None]

    def distribution(self, ops: Ops, state: Tensor) -> Tensor:
        """p(. | state) over all 256 bytes (for checks and generation; training never needs it)."""
        if not isinstance(ops, Tensor):
            raise TypeError("Kraus operators expected")
        if self.rank == 1:
            phi = torch.einsum("vij,bj->bvi", ops[:, 0], state)
            return (phi.abs() ** 2).sum(-1)
        sigma = torch.einsum("vrij,bjk,vrlk->bvil", ops, state, ops.conj())
        return torch.diagonal(sigma, dim1=-2, dim2=-1).real.sum(-1)

    def parameter_count(self) -> int:
        return sum(p.numel() * (2 if p.is_complex() else 1) for p in self.parameters())


class HiddenMarkovLM(nn.Module):
    """The classical control: a hidden Markov model with ``states`` states, by its filter."""

    def __init__(self, states: int, seed: int = 0) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.states = states
        self.transition = nn.Parameter(0.01 * torch.randn(states, states, generator=g))
        self.emission = nn.Parameter(0.01 * torch.randn(states, V, generator=g))
        self.start = nn.Parameter(torch.zeros(states))

    def operators(self) -> tuple[Tensor, Tensor]:
        return torch.softmax(self.transition, -1), torch.softmax(self.emission, -1)

    def initial(self, batch: int) -> Tensor:
        return torch.softmax(self.start, -1).expand(batch, self.states).clone()

    def step(self, ops: Ops, state: Tensor, x: Tensor) -> tuple[Tensor, Tensor]:
        if not isinstance(ops, tuple):
            raise TypeError("a transition and an emission matrix expected")
        transition, emission = ops
        joint = state * emission[:, x].T                              # (B, N)
        p = joint.sum(-1).clamp_min(FLOOR)
        return torch.log(p), (joint / p[:, None]) @ transition

    def distribution(self, ops: Ops, state: Tensor) -> Tensor:
        if not isinstance(ops, tuple):
            raise TypeError("a transition and an emission matrix expected")
        out: Tensor = state @ ops[1]
        return out

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


Model = KrausLM | HiddenMarkovLM


def build(arm: str, seed: int = 0) -> Model:
    """``pure-complex-D``, ``pure-real-D``, ``mixed-complex-D-r``, ``mixed-real-D-r`` or ``hmm-N``."""
    parts = arm.split("-")
    if parts[0] == "hmm":
        return HiddenMarkovLM(int(parts[1]), seed)
    kind, field = parts[0], parts[1]
    dim = int(parts[2])
    rank = 1 if kind == "pure" else int(parts[3])
    if kind not in ("pure", "mixed") or field not in ("complex", "real"):
        raise ValueError(arm)
    return KrausLM(dim, rank, field == "complex", seed)


def _streams(data: bytes, count: int) -> Tensor:
    """``count`` contiguous, equal pieces of the bytes, one per row."""
    n = len(data) // count
    return torch.from_numpy(np.frombuffer(data[: n * count], dtype=np.uint8).astype(np.int64).reshape(count, n))


@torch.no_grad()
def byte_probabilities(model: Model, data: bytes, chunks: int = 32) -> np.ndarray:
    """p(byte | the bytes before it in its piece) for the first ``chunks * (len(data) // chunks)`` bytes, in
    the order of ``data``; the data is cut into ``chunks`` contiguous pieces, each from the initial state."""
    streams = _streams(data, chunks)
    ops = model.operators()
    state = model.initial(chunks)
    out = torch.empty(streams.shape, dtype=torch.float64)
    for t in range(streams.shape[1]):
        logp, state = model.step(ops, state, streams[:, t])
        out[:, t] = logp.double()
    probs: np.ndarray = torch.exp(out).reshape(-1).numpy()
    return probs


def bits_per_byte(model: Model, data: bytes, chunks: int = 32) -> float:
    """Mean bits per byte over ``chunks`` contiguous pieces, each scored from the model's initial state."""
    return float(-np.log2(byte_probabilities(model, data, chunks)).mean())


MIX_WEIGHTS = tuple(round(0.05 * k, 2) for k in range(21))


def mixture(p_model: np.ndarray, p_counts: np.ndarray, weight: float) -> float:
    """Bits per byte of weight * the model + (1 - weight) * the counting model, byte by byte."""
    return float(-np.log2(weight * p_model + (1.0 - weight) * p_counts).mean())


def best_mixture_weight(p_model: np.ndarray, p_counts: np.ndarray) -> float:
    """The weight on the model, from ``MIX_WEIGHTS``, that costs least on the bytes given."""
    return min(MIX_WEIGHTS, key=lambda w: mixture(p_model, p_counts, w))


@dataclass
class Trained:
    model: Model
    best_valid_bpb: float
    bytes_seen: int
    seconds: float
    steps: int
    curve: list[tuple[float, int, float]]


def train(model: Model, train_bytes: bytes, valid_bytes: bytes, seconds: float, batch: int = 32,
          segment: int = 128, lr: float = 3e-3, evaluate_every: float = 300.0, valid_chunks: int = 32,
          log: Callable[[str], Any] | None = None) -> Trained:
    """Adam on the bytes for ``seconds``; the parameters kept are those best on ``valid_bytes``."""
    streams = _streams(train_bytes, batch)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    state = model.initial(batch)
    best, best_params = math.inf, {k: v.detach().clone() for k, v in model.state_dict().items()}
    curve: list[tuple[float, int, float]] = []
    t0, last_eval, pos, steps, seen = time.monotonic(), time.monotonic(), 0, 0, 0
    while True:
        now = time.monotonic()
        done = now - t0 >= seconds
        if done or now - last_eval >= evaluate_every:
            bpb = bits_per_byte(model, valid_bytes, valid_chunks)
            curve.append((round(now - t0, 1), seen, round(bpb, 4)))
            if log:
                log(f"t={now - t0:.0f}s bytes={seen} valid={bpb:.4f}")
            if bpb < best:
                best, best_params = bpb, {k: v.detach().clone() for k, v in model.state_dict().items()}
            last_eval = time.monotonic()
            if done:
                break
        if pos + segment > streams.shape[1]:
            pos, state = 0, model.initial(batch)
        ops = model.operators()
        state = state.detach()
        loss = torch.zeros(())
        for t in range(pos, pos + segment):
            logp, state = model.step(ops, state, streams[:, t])
            loss = loss - logp.mean()
        optimiser.zero_grad()
        (loss / segment).backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        pos, steps, seen = pos + segment, steps + 1, seen + segment * batch
    model.load_state_dict(best_params)
    return Trained(model, best, seen, round(time.monotonic() - t0, 1), steps, curve)


def kneser_ney_bits(train_bytes: bytes, test_bytes: bytes, order: int) -> float:
    """Interpolated Kneser-Ney over bytes (``order`` bytes of history, 0 to 4), bits per byte on the test."""
    return float(-np.log2(kneser_ney_probabilities(train_bytes, test_bytes, order)).mean())


def kneser_ney_probabilities(train_bytes: bytes, test_bytes: bytes, order: int) -> np.ndarray:
    """p(each test byte | the bytes before it, up to ``order`` of them) under interpolated Kneser-Ney."""
    if not 0 <= order <= 4:
        raise ValueError("order 0 to 4")
    tr: np.ndarray = np.frombuffer(train_bytes, dtype=np.uint8).astype(np.int64)
    te: np.ndarray = np.frombuffer(test_bytes, dtype=np.uint8).astype(np.int64)

    def grams(a: np.ndarray, m: int) -> np.ndarray:
        key: np.ndarray = np.zeros(len(a) - m + 1, dtype=np.int64)
        for j in range(m):
            key = key * V + a[j:len(a) - m + 1 + j]
        return key

    # adjusted counts of m-grams: raw for the highest order, continuation counts below it
    tables: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for m in range(1, order + 2):
        if m == order + 1:
            keys, counts = np.unique(grams(tr, m), return_counts=True)
        else:
            longer = np.unique(grams(tr, m + 1))
            keys, counts = np.unique(longer % (V ** m), return_counts=True)
        tables[m] = (keys, counts.astype(np.float64))
    # test positions with a full history; the first ``order`` bytes are scored at the orders they have
    prob = np.full(len(te), 1.0 / V)
    for m in range(1, order + 2):
        keys, counts = tables[m]
        n1, n2 = float((counts == 1).sum()), float((counts == 2).sum())
        d = n1 / (n1 + 2 * n2) if n1 + 2 * n2 > 0 else 0.5
        hist_keys, hist_index = np.unique(keys // V, return_inverse=True)
        hist_total = np.bincount(hist_index, weights=counts)
        hist_types: np.ndarray = np.bincount(hist_index).astype(np.float64)
        idx = np.arange(m - 1, len(te))
        g: np.ndarray = np.zeros(len(idx), dtype=np.int64)
        for j in range(m):
            g = g * V + te[idx - (m - 1) + j]
        h: np.ndarray = g // V
        hi = np.clip(np.searchsorted(hist_keys, h), 0, len(hist_keys) - 1)
        seen_h = hist_keys[hi] == h
        gi = np.clip(np.searchsorted(keys, g), 0, len(keys) - 1)
        c = np.where(keys[gi] == g, counts[gi], 0.0)
        total = np.where(seen_h, hist_total[hi], 1.0)
        lower = prob[idx]
        mixed = np.maximum(c - d, 0.0) / total + d * np.where(seen_h, hist_types[hi], 0.0) / total * lower
        prob[idx] = np.where(seen_h, mixed, lower)
    return prob
