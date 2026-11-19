"""MMLM-CM: a byte-level language model built from counting experts, one small network and a learned mixer.

Status: experimental

The original tracks combine their engines with fixed weights and, measured on
real text, cost more bits than their own automaton alone
(``benchmarks/mmlm_text_dev_study.json``). This model keeps the idea -- several
small predictors, each one inspectable -- and changes how they are built and
combined:

- **One measure.** It predicts bytes, so its score is bits per byte on any text,
  comparable with compressors and with neural models, with no tokenizer and no
  unknown-word bucket. A byte is predicted as eight binary decisions (most
  significant bit first), so every expert only has to say "how likely is the
  next bit to be 1".
- **Experts are counts, calibrated on held-out text.** Each counting expert keeps
  two numbers per context: how often the next bit was 0 and 1 in the training
  text. What a pair of counts is worth (a context seen once says much less than
  its raw frequency suggests) is not assumed: it is read off a second text the
  counts never saw (``calibrate``).
- **The mixture is learned, and it depends on the situation.** The experts'
  log-odds are added with weights fitted by gradient descent on the log loss.
  Separate weight sets are kept for the bit position, for the longest context
  that was ever seen, for the length of the current match and for the kind of
  byte just read, so the model learns when to trust which expert.
- **Every expert can be left out** (``experts=`` in ``fit_mixer``), so what each
  one contributes in bits per byte is a measured number.

Experts:

- ``o0`` .. ``o6``, ``o8``: the next bit given the previous k bytes and the bits
  of the current byte so far, counted over the training text.
- ``d2`` .. ``d6``, ``dw1``, ``dw2``: the same contexts counted over the text read
  so far, so a name or a term introduced by the document in hand is learned
  while reading it.
- ``w1``, ``w2``: the next bit given the letters of the word typed so far (and
  the separator before it), and given those letters and the previous word.
- ``m8``, ``m4``: the long-range memory. Finds the most recent earlier place
  (training text and everything read so far) where the last 8 (or 4) bytes
  occurred and predicts the byte that followed there; how much that is worth is
  calibrated per verified match length.
- ``nn``: the one neural expert (:mod:`apedp_mmlm.neural_expert`), passed in as
  bit probabilities.

Everything is vectorised over the stream: a model is fitted on 10 MB in about a
minute on one core.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

K1 = np.uint64(0x9E3779B97F4A7C15)
K2 = np.uint64(0xC2B2AE3D27D4EB4F)
K3 = np.uint64(0x165667B19E3779F9)
Q = 32  # count buckets per side
MATCH_CAP = 32
_QLUT = np.array([n if n <= 7 else min(Q - 1, 8 + int(2.5 * math.log2(n / 8))) for n in range(4096)], dtype=np.int64)
_MATCH_BUCKET = np.array([0] * 4 + [1] * 2 + [2] * 2 + [3] * 4 + [4] * 4 + [5] * 8 + [6] * 8 + [7], dtype=np.int64)
COUNT_EXPERTS = ("o0", "o1", "o2", "o3", "o4", "o5", "o6", "o8", "w1", "w2")
ORDERS = (1, 2, 3, 4, 5, 6, 8)
# the same contexts counted over the text read so far instead of the training text
RECENT_EXPERTS = {"d2": "o2", "d3": "o3", "d4": "o4", "d5": "o5", "d6": "o6", "dw1": "w1", "dw2": "w2"}
MATCH_EXPERTS = ("m8", "m4")
ALL_EXPERTS = (*COUNT_EXPERTS, *RECENT_EXPERTS, *MATCH_EXPERTS, "nn")
SELECTORS = {"order": 8 * 8, "match": 8 * 17, "class": 8 * 8, "recent": 8 * 6}


def _context_hash(x: np.ndarray, order: int) -> np.ndarray:
    """A 64-bit hash of the ``order`` bytes before each position."""
    n = len(x)
    padded = np.concatenate([np.zeros(order, dtype=np.uint64), x.astype(np.uint64)])
    g: np.ndarray = np.zeros(n, dtype=np.uint64)
    for d in range(1, order + 1):
        g = g * K1 + padded[order - d : order - d + n] + np.uint64(1)
    return g


def _word_contexts(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hashes of (word so far, separator before it) and of (previous word, word so far)."""
    n = len(x)
    upper = (x >= 65) & (x <= 90)
    letter = upper | ((x >= 97) & (x <= 122)) | (x >= 128)
    low = np.where(upper, x + 32, x).astype(np.uint64)
    index: np.ndarray = np.arange(n, dtype=np.int64)
    last_non = np.maximum.accumulate(np.where(~letter, index, -1))
    # run[i]: how many letters end exactly before position i
    run: np.ndarray = np.zeros(n, dtype=np.int64)
    run[1:] = index[:-1] - last_non[:-1]
    padded = np.concatenate([np.zeros(16, dtype=np.uint64), low])
    word: np.ndarray = np.zeros(n, dtype=np.uint64)
    for d in range(1, 17):
        take = run >= d
        word = np.where(take, word * K1 + padded[16 - d : 16 - d + n] + np.uint64(1), word)
    start = index - run
    sep = np.where(start > 0, x[np.maximum(start - 1, 0)], 0).astype(np.uint64)
    # a word ends at e when x[e] is not a letter and run[e] > 0; its hash is word[e]
    ended = (~letter) & (run > 0)
    last_end = np.maximum.accumulate(np.where(ended, index, -1))
    look = np.maximum(start - 1, 0)
    previous = np.where((start > 0) & (last_end[look] >= 0), word[np.maximum(last_end[look], 0)], 0).astype(np.uint64)
    return word ^ ((sep + np.uint64(1)) * K2), (previous * K3) ^ word


def _matches(history: np.ndarray, x: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    """For each byte of ``x``: the byte that followed the most recent earlier occurrence of the
    last ``order`` bytes (-1 if none) and the verified length of that match (0..MATCH_CAP)."""
    stream = np.concatenate([history, x])
    g = _context_hash(stream, order)
    by_hash = np.argsort(g, kind="stable")
    previous: np.ndarray = np.full(len(stream), -1, dtype=np.int64)
    same = g[by_hash[1:]] == g[by_hash[:-1]]
    previous[by_hash[1:][same]] = by_hash[:-1][same]
    here: np.ndarray = np.arange(len(history), len(stream), dtype=np.int64)
    there = previous[here]
    found = (there >= 0) & (here >= order)
    length: np.ndarray = np.zeros(len(x), dtype=np.int64)
    alive = found.copy()
    for back in range(1, MATCH_CAP + 1):
        a: np.ndarray = here - back
        b: np.ndarray = there - back
        alive &= b >= 0
        alive[alive] = stream[a[alive]] == stream[b[alive]]
        length += alive
    predicted = np.where(found, stream[np.maximum(there, 0)].astype(np.int64), -1)
    return predicted, length


def _counts_so_far(key: np.ndarray, bit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each position: how many earlier positions had the same key with bit 0, and with bit 1."""
    n = len(key)
    by_key = np.argsort(key, kind="stable")
    k, b = key[by_key], bit[by_key].astype(np.int64)
    index: np.ndarray = np.arange(n, dtype=np.int64)
    first = np.concatenate([[True], k[1:] != k[:-1]])
    start = np.maximum.accumulate(np.where(first, index, 0))
    ones = np.cumsum(b) - b
    ones_before = ones - ones[start]
    zeros_before = (index - start) - ones_before
    n0: np.ndarray = np.empty(n, dtype=np.int64)
    n1: np.ndarray = np.empty(n, dtype=np.int64)
    n0[by_key], n1[by_key] = zeros_before, ones_before
    return n0, n1


def _stretch(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    out: np.ndarray = np.log(p / (1 - p)).astype(np.float32)
    return out


def _byte_class(x: np.ndarray) -> np.ndarray:
    c: np.ndarray = np.full(len(x), 7, dtype=np.int64)
    c[(x >= 97) & (x <= 122)] = 0
    c[(x >= 65) & (x <= 90)] = 1
    c[(x >= 48) & (x <= 57)] = 2
    c[x == 32] = 3
    c[x == 10] = 4
    c[np.isin(x, np.frombuffer(b".,;:!?'\"()-", dtype=np.uint8))] = 5
    c[x >= 128] = 6
    return c


class ContextMixingLM:
    """Counting experts, calibrated and mixed. ``fit_counts`` -> ``calibrate`` -> ``fit_mixer`` -> ``byte_bits``."""

    def __init__(self, table_bits: int = 24) -> None:
        self.table_bits = table_bits
        self.counts: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.history: np.ndarray = np.zeros(0, dtype=np.uint8)
        self.state_maps: dict[str, np.ndarray] = {}
        self.match_maps: dict[str, np.ndarray] = {}
        self.experts: tuple[str, ...] = ()
        self.mixer: dict[str, torch.Tensor] = {}

    # -- contexts ---------------------------------------------------------------------------
    def _slots(self, name: str, context: np.ndarray, node: np.ndarray, x: np.ndarray) -> np.ndarray:
        if name == "o0":
            return node.astype(np.int64)
        out: np.ndarray
        if name == "o1":
            prev: np.ndarray = np.concatenate([[0], x[:-1]]).astype(np.int64)
            out = (prev << 8) | node
            return out
        if name == "o2":
            p1: np.ndarray = np.concatenate([[0], x[:-1]]).astype(np.int64)
            p2: np.ndarray = np.concatenate([[0, 0], x[:-2]]).astype(np.int64)
            out = (p2 << 16) | (p1 << 8) | node
            return out
        out = (((context + node.astype(np.uint64) * K2) * K3) >> np.uint64(64 - self.table_bits)).astype(np.int64)
        return out

    def _size(self, name: str) -> int:
        return {"o0": 1 << 8, "o1": 1 << 16, "o2": 1 << 24}.get(name, 1 << self.table_bits)

    @staticmethod
    def _contexts(x: np.ndarray) -> dict[str, np.ndarray]:
        w1, w2 = _word_contexts(x)
        empty: np.ndarray = np.zeros(0, dtype=np.uint64)
        return {"o0": empty, "o1": empty, "o2": empty, **{f"o{k}": _context_hash(x, k) for k in (3, 4, 5, 6, 8)}, "w1": w1, "w2": w2}

    # -- fitting ----------------------------------------------------------------------------
    def fit_counts(self, train: bytes) -> ContextMixingLM:
        x: np.ndarray = np.frombuffer(train, dtype=np.uint8)
        self.history = x.copy()
        contexts = self._contexts(x)
        for name in COUNT_EXPERTS:
            size = self._size(name)
            n0: np.ndarray = np.zeros(size, dtype=np.uint32)
            n1: np.ndarray = np.zeros(size, dtype=np.uint32)
            for j in range(8):
                node = ((1 << j) | (x >> (8 - j))).astype(np.int64) if j else np.ones(len(x), dtype=np.int64)
                bit = ((x >> (7 - j)) & 1).astype(bool)
                slot = self._slots(name, contexts[name], node, x)
                n1 += np.bincount(slot[bit], minlength=size).astype(np.uint32)
                n0 += np.bincount(slot[~bit], minlength=size).astype(np.uint32)
            self.counts[name] = (n0, n1)
        return self

    def _raw(self, data: bytes) -> dict[str, Any]:
        """Per bit of ``data``: the true bit, each counting expert's count state, each match expert's state."""
        x: np.ndarray = np.frombuffer(data, dtype=np.uint8)
        n = len(x)
        contexts = self._contexts(x)
        bits: np.ndarray = np.empty((n, 8), dtype=np.int64)
        states: dict[str, np.ndarray] = {name: np.empty((n, 8), dtype=np.int64) for name in COUNT_EXPERTS}
        seen: dict[str, np.ndarray] = {name: np.empty((n, 8), dtype=bool) for name in COUNT_EXPERTS}
        nodes: np.ndarray = np.empty((n, 8), dtype=np.int64)
        recent: dict[str, np.ndarray] = {name: np.empty((n, 8), dtype=np.int64) for name in RECENT_EXPERTS}
        recent_seen: dict[str, np.ndarray] = {name: np.empty((n, 8), dtype=bool) for name in RECENT_EXPERTS}
        for j in range(8):
            node = ((1 << j) | (x >> (8 - j))).astype(np.int64) if j else np.ones(n, dtype=np.int64)
            nodes[:, j] = node
            bits[:, j] = (x >> (7 - j)) & 1
            for name in COUNT_EXPERTS:
                n0, n1 = self.counts[name]
                slot = self._slots(name, contexts[name], node, x)
                a, b = n0[slot].astype(np.int64), n1[slot].astype(np.int64)
                states[name][:, j] = _QLUT[np.minimum(a, 4095)] * Q + _QLUT[np.minimum(b, 4095)]
                seen[name][:, j] = (a + b) > 0
            for name, source in RECENT_EXPERTS.items():
                if source == "o2":
                    key = self._slots("o2", contexts["o2"], node, x)
                else:
                    key = ((contexts[source] + node.astype(np.uint64) * K2) * K3).view(np.int64)
                a, b = _counts_so_far(key, bits[:, j])
                recent[name][:, j] = _QLUT[np.minimum(a, 4095)] * Q + _QLUT[np.minimum(b, 4095)]
                recent_seen[name][:, j] = (a + b) > 0
        match_states: dict[str, np.ndarray] = {}
        for name in MATCH_EXPERTS:
            predicted, length = _matches(self.history, x, int(name[1:]))
            bucket = _MATCH_BUCKET[length]
            state: np.ndarray = np.zeros((n, 8), dtype=np.int64)
            for j in range(8):
                agrees = (predicted >= 0) & ((predicted >> (8 - j)) == (x.astype(np.int64) >> (8 - j)))
                expected = (predicted >> (7 - j)) & 1
                state[:, j] = np.where(agrees, 1 + bucket * 2 + expected, 0) + 17 * j
            match_states[name] = state
        longest: np.ndarray = np.zeros((n, 8), dtype=np.int64)
        for rank, k in enumerate(ORDERS, start=1):
            longest = np.where(seen[f"o{k}"], rank, longest)
        fresh: np.ndarray = np.zeros((n, 8), dtype=np.int64)
        for rank, name in enumerate(("d2", "d3", "d4", "d5", "d6"), start=1):
            fresh = np.where(recent_seen[name], rank, fresh)
        previous = np.concatenate([[10], x[:-1]]).astype(np.uint8)
        selectors = {
            "order": np.arange(8)[None, :] * 8 + longest,
            "match": np.arange(8)[None, :] * 17 + (match_states[MATCH_EXPERTS[0]] % 17),
            "class": np.arange(8)[None, :] * 8 + _byte_class(previous)[:, None],
            "recent": np.arange(8)[None, :] * 6 + fresh,
        }
        return {"bits": bits, "states": {**states, **recent}, "match_states": match_states, "selectors": selectors, "n": n}

    def calibrate(self, data: bytes, strength: float = 2.0) -> ContextMixingLM:
        """Read what each count state and each match state is worth off text the counts never saw."""
        raw = self._raw(data)
        bit = raw["bits"].ravel().astype(bool)
        grid = np.arange(Q * Q)
        a: np.ndarray = grid // Q
        b: np.ndarray = grid % Q
        for name in (*COUNT_EXPERTS, *RECENT_EXPERTS):
            s = raw["states"][name].ravel()
            c1: np.ndarray = np.bincount(s[bit], minlength=Q * Q).astype(np.float64)
            c0: np.ndarray = np.bincount(s[~bit], minlength=Q * Q).astype(np.float64)
            prior = (b + 0.4) / (a + b + 0.8)  # bucket indices stand in for the counts in unseen states
            self.state_maps[name] = (c1 + strength * prior) / (c0 + c1 + strength)
        for name in MATCH_EXPERTS:
            s = raw["match_states"][name].ravel()
            c1 = np.bincount(s[bit], minlength=17 * 8).astype(np.float64)
            c0 = np.bincount(s[~bit], minlength=17 * 8).astype(np.float64)
            self.match_maps[name] = (c1 + 0.5 * strength) / (c0 + c1 + strength)
        return self

    def features(self, data: bytes, neural: np.ndarray | None = None) -> dict[str, Any]:
        """The mixer's inputs for every bit of ``data``: one log-odds per expert, plus the selectors."""
        raw = self._raw(data)
        columns: dict[str, np.ndarray] = {}
        for name in (*COUNT_EXPERTS, *RECENT_EXPERTS):
            columns[name] = _stretch(self.state_maps[name][raw["states"][name].ravel()])
        for name in MATCH_EXPERTS:
            columns[name] = _stretch(self.match_maps[name][raw["match_states"][name].ravel()])
        if neural is not None:
            columns["nn"] = _stretch(np.asarray(neural, dtype=np.float64).ravel())
        return {"columns": columns, "bits": raw["bits"].ravel(), "n": raw["n"],
                "selectors": {k: v.ravel() for k, v in raw["selectors"].items()}}

    # -- mixer ------------------------------------------------------------------------------
    @staticmethod
    def _tensors(feats: dict[str, Any], experts: Sequence[str]) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(np.stack([feats["columns"][name] for name in experts], axis=1))
        selectors = [torch.from_numpy(feats["selectors"][k]) for k in SELECTORS]
        position = torch.arange(8).repeat(feats["n"])
        return x, selectors, position, torch.from_numpy(feats["bits"].astype(np.float32))

    def _logit(self, x: torch.Tensor, selectors: list[torch.Tensor], position: torch.Tensor) -> torch.Tensor:
        first = torch.stack([(self.mixer[f"w{k}"][s] * x).sum(1) for k, s in enumerate(selectors)], dim=1)
        return (self.mixer["v"][position] * first).sum(1)

    def fit_mixer(self, feats: dict[str, Any], experts: Sequence[str] | None = None, *, epochs: int = 8,
                  batch: int = 32768, lr: float = 0.01, seed: int = 0) -> ContextMixingLM:
        self.experts = tuple(experts) if experts is not None else tuple(n for n in ALL_EXPERTS if n in feats["columns"])
        x, selectors, position, y = self._tensors(feats, self.experts)
        width = len(self.experts)
        torch.manual_seed(seed)
        self.mixer = {f"w{k}": torch.full((size, width), 1.0 / max(1, width) ** 0.5, requires_grad=True)
                      for k, size in enumerate(SELECTORS.values())}
        self.mixer["v"] = torch.full((8, len(SELECTORS)), 1.0 / len(SELECTORS), requires_grad=True)
        optimiser = torch.optim.Adam(list(self.mixer.values()), lr=lr)
        steps = epochs * math.ceil(len(y) / batch)
        schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=lr, total_steps=steps)
        generator = torch.Generator().manual_seed(seed)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        for _ in range(epochs):
            order = torch.randperm(len(y), generator=generator)
            for begin in range(0, len(y), batch):
                rows = order[begin : begin + batch]
                loss = loss_fn(self._logit(x[rows], [s[rows] for s in selectors], position[rows]), y[rows])
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                schedule.step()
        self.mixer = {k: v.detach() for k, v in self.mixer.items()}
        return self

    @torch.no_grad()
    def byte_bits(self, feats: dict[str, Any]) -> np.ndarray:
        """Code length of every byte, in bits."""
        x, selectors, position, y = self._tensors(feats, self.experts)
        logit = self._logit(x, selectors, position).double()
        nats = torch.nn.functional.binary_cross_entropy_with_logits(logit, y.double(), reduction="none")
        out: np.ndarray = (nats.view(-1, 8).sum(1) / math.log(2)).numpy()
        return out

    @staticmethod
    def expert_alone(feats: dict[str, Any], name: str) -> float:
        """Bits per byte if this expert's calibrated prediction were used on its own."""
        z = feats["columns"][name].astype(np.float64)
        y = feats["bits"]
        nats = np.logaddexp(0.0, np.where(y == 1, -z, z))
        return float(nats.sum() / math.log(2) / feats["n"])
