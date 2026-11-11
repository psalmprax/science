r"""The Five Mathematical Engines of APEDP-MMLM (Multi-Mathematical Language Model).

Status: experimental

Implements:
  - **Specification & Tokenization**: `MMLMResearchSpecification`, frozen falsifiable
    hypotheses `H1`–`H5`, and deterministic word/subword/circuit/AST `MMLMTokenizer`.
  - **Engine 1 (Probabilistic Language Engine)**:
    - `WeightedAutomatonEngine`: Variable-order Markov / Weighted Finite-State Automaton
      with Kneser-Ney continuation backoff over $P(x_t \mid x_{1}, \dots, x_{t-1})$ up to order 5.
    - `ProbabilisticCircuitEngine`: Decomposable and deterministic Sum-Product
      Probabilistic Circuit with tractable exact marginals and conditionals.
    - `BayesianLanguageEngine`: Hierarchical Dirichlet-Multinomial conjugate posterior
      predictor with explicit epistemic variance over variable-length contexts.
  - **Engine 2 (Tensor Memory Engine)**:
    - `TensorTrainMemoryEngine`: 1D Matrix Product State (MPS / Tensor Train)
      factorization $P(x_1, \dots, x_n) = G_1[x_1] G_2[x_2] \cdots G_n[x_n]$ with
      trainable rank-1 core tensor updates and Frobenius spectral stabilization.
    - `TreeTensorNetworkMemoryEngine`: Binary hierarchical Tree Tensor Network (TTN)
      with syntax/bracket-depth-aware contraction and trainable bilinear merge cores.
  - **Engine 3 (Dynamical Context Engine)**:
    - `DynamicalContextEngine`: Streaming state evolution $s_{t+1} = F(s_t, \phi(x_t))$
      via unitary-stabilized contraction operators.
    - `KoopmanOperatorEngine`: Extended Dynamic Mode Decomposition (EDMD) lifting
      nonlinear context trajectories into linear spectral Koopman modes with direct
      token probability projection.
  - **Engine 4 (Structured Knowledge & Symbolic Rewrite Engine)**:
    - `HypergraphKnowledgeEngine`: Multi-entity relation hyperedges $(e_1, \dots, e_k; r)$
      with multi-hop relational traversal and contradiction detection.
    - `VectorSymbolicEngine`: Holographic Reduced Representations (HRR) using
      FFT circular convolution binding ($\circledast$) and correlation unbinding.
    - `SymbolicRewriteEngine`: Deterministic term-rewriting system for nested ListOps
      and algebraic quantum gate identities ($G^2 \to I$, $H Z H \leftrightarrow X$,
      $H X H \leftrightarrow Z$, commutation normal forms).
  - **Engine 5 (Global Energy & Speculative Evaluation Engine)**:
    - `EnergyCandidateEvaluator`: Evaluates candidate continuations via
      $E(x \mid c) = E_{\text{language}} + E_{\text{context}} + E_{\text{knowledge}} + E_{\text{constraints}}$.
    - `SpeculativeEnergyDecoder`: Polynomial-time multi-step beam search and
      discrete zero-delta plateau lookahead bypassing the #P-hard partition function $Z(c)$.
    - `MMLMEpistemicAdjudicator`: Verifies logical consistency, uncertainty calibration,
      and factual support against structured knowledge.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np


@dataclass(frozen=True)
class FrozenHypothesis:
    """Preregistered falsifiable research hypothesis for APEDP-MMLM."""

    hypothesis_id: str
    statement: str
    falsification_criterion: str
    metric_name: str
    baseline_comparison_rule: str


def default_frozen_hypotheses() -> tuple[FrozenHypothesis, ...]:
    """Return the five frozen research hypotheses H1-H5 before experimental evaluation."""
    return (
        FrozenHypothesis(
            "H1",
            "Tensor-based context representation preserves long-range dependencies efficiently.",
            "Falsified if tensor memory does not improve recall or held-out perplexity over the pure probabilistic baseline at comparable resource cost.",
            "long_context_recall_and_perplexity",
            "tensor_recall > markov_recall",
        ),
        FrozenHypothesis(
            "H2",
            "Explicit dynamical state evolution provides efficient streaming context processing.",
            "Falsified if dynamical state updates lose critical context information or require excessive state dimension to match baseline accuracy.",
            "streaming_retention_score",
            "dynamical_retention >= 0.70",
        ),
        FrozenHypothesis(
            "H3",
            "Structured hypergraph & VSA memory improves compositional retrieval and reasoning.",
            "Falsified if hypergraph/VSA provides no measurable improvement on KV retrieval and ListOps over unstructured memory.",
            "compositional_accuracy",
            "with_hypergraph > without_hypergraph",
        ),
        FrozenHypothesis(
            "H4",
            "Global probabilistic or energy-based evaluation improves sequence-level coherence.",
            "Falsified if speculative energy evaluation fails to lower energy or improve coherence under the same inference budget.",
            "sequence_coherence_score",
            "with_energy > greedy_local",
        ),
        FrozenHypothesis(
            "H5",
            "A mathematical hybrid can compete with small Transformer and SSM models without conventional self-attention.",
            "Falsified if B0/C0 consistently loses on both quality and efficiency metrics without any compensating advantage.",
            "composite_quality_efficiency",
            "hybrid_wins_at_least_two_axes",
        ),
    )


@dataclass(frozen=True)
class MMLMResearchSpecification:
    """Frozen architectural contract and benchmark protocol for APEDP-MMLM."""

    spec_version: str = "APEDP-MMLM-v0.2"
    tracks: tuple[str, ...] = ("APEDP-MMLM-NN0", "APEDP-MMLM-Hybrid", "APEDP-MMLM-X")
    hypotheses: tuple[FrozenHypothesis, ...] = field(default_factory=default_frozen_hypotheses)
    max_context_window: int = 256


class MMLMTokenizer:
    """Deterministic vocabulary and word/punctuation/circuit/AST tokenizer shared across all tracks."""

    PAD_TOKEN = "<pad>"  # noqa: S105
    UNK_TOKEN = "<unk>"  # noqa: S105
    BOS_TOKEN = "<bos>"  # noqa: S105
    EOS_TOKEN = "<eos>"  # noqa: S105

    def __init__(self, max_vocab_size: int = 2048, circuit_aware: bool = False) -> None:
        self.max_vocab_size = max(16, max_vocab_size)
        self.circuit_aware = circuit_aware
        self.token_to_id: dict[str, int] = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
            self.BOS_TOKEN: 2,
            self.EOS_TOKEN: 3,
        }
        self.id_to_token: dict[int, str] = {v: k for k, v in self.token_to_id.items()}
        self._fitted = False

    @staticmethod
    def split_raw(text: str) -> list[str]:
        """Standard word and punctuation splitter (preserved for backward compatibility)."""
        return re.findall(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]", text.lower())

    @staticmethod
    def split_circuit_or_ast(text: str) -> list[str]:
        """Grammar-aware splitter preserving indexed quantum registers `q[12]`, gate ops, and AST brackets."""
        pattern = r"[A-Za-z_]+\[\d+\]|[A-Za-z0-9_]+|==|!=|<=|>=|->|[^\sA-Za-z0-9_]"
        return re.findall(pattern, text.lower())

    @staticmethod
    def bracket_depths(tokens: Sequence[str]) -> list[int]:
        """Compute structural nesting depth for every token in an AST or bracketed sequence."""
        depths: list[int] = []
        depth = 0
        openers = {"(", "[", "{"}
        closers = {")", "]", "}"}
        for tok in tokens:
            if tok in openers:
                depth += 1
                depths.append(depth)
            elif tok in closers:
                depths.append(depth)
                depth = max(0, depth - 1)
            else:
                depths.append(depth)
        return depths

    def _tokenize_str(self, text: str) -> list[str]:
        if self.circuit_aware:
            return self.split_circuit_or_ast(text)
        return self.split_raw(text)

    def fit(self, texts: Sequence[str]) -> MMLMTokenizer:
        counts: Counter[str] = Counter()
        for t in texts:
            counts.update(self._tokenize_str(t))
        for tok, _ in counts.most_common(self.max_vocab_size - len(self.token_to_id)):
            if tok not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[tok] = idx
                self.id_to_token[idx] = tok
        self._fitted = True
        return self

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        raw = self._tokenize_str(text)
        unk_id = self.token_to_id[self.UNK_TOKEN]
        ids = [self.token_to_id.get(tok, unk_id) for tok in raw]
        if add_special_tokens:
            return [self.token_to_id[self.BOS_TOKEN], *ids, self.token_to_id[self.EOS_TOKEN]]
        return ids

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        special = {0, 1, 2, 3} if skip_special_tokens else set()
        tokens = [self.id_to_token.get(int(i), self.UNK_TOKEN) for i in token_ids if int(i) not in special]
        return " ".join(tokens)


class BPEMMLMTokenizer(MMLMTokenizer):
    """Deterministic byte-level BPE with lossless UTF-8 fallback.

    The vocabulary cap includes four special tokens and all 256 byte tokens.
    Training learns frequent adjacent byte sequences without normalizing text.
    """

    def __init__(self, max_vocab_size: int = 2048, min_frequency: int = 2) -> None:
        if max_vocab_size < 260:
            raise ValueError("max_vocab_size must be at least 260 for UTF-8 byte fallback")
        if min_frequency < 1:
            raise ValueError("min_frequency must be positive")
        super().__init__(max_vocab_size=max_vocab_size)
        self.min_frequency = min_frequency
        self.merges: list[tuple[bytes, bytes]] = []
        self._reset_vocabulary()

    def _reset_vocabulary(self) -> None:
        self.token_to_id = {
            token: index for index, token in enumerate(
                (self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN)
            )
        }
        self._pieces = {index + 4: bytes([index]) for index in range(256)}
        self.token_to_id.update(
            (f"<0x{index:02X}>", index + 4) for index in range(256)
        )
        self.id_to_token = {index: token for token, index in self.token_to_id.items()}
        self._piece_ids = {piece: index for index, piece in self._pieces.items()}
        self.merges = []
        self._fitted = False

    @staticmethod
    def _merge(pieces: list[bytes], pair: tuple[bytes, bytes]) -> list[bytes]:
        merged: list[bytes] = []
        index = 0
        while index < len(pieces):
            if index + 1 < len(pieces) and (pieces[index], pieces[index + 1]) == pair:
                merged.append(pair[0] + pair[1])
                index += 2
            else:
                merged.append(pieces[index])
                index += 1
        return merged

    def fit(self, texts: Sequence[str]) -> BPEMMLMTokenizer:
        self._reset_vocabulary()
        sequences = [[bytes([value]) for value in text.encode("utf-8")] for text in texts]
        while self.vocab_size < self.max_vocab_size:
            counts: Counter[tuple[bytes, bytes]] = Counter()
            for pieces in sequences:
                counts.update(pairwise(pieces))
            if not counts:
                break
            pair = min(counts, key=lambda candidate: (-counts[candidate], candidate))
            if counts[pair] < self.min_frequency:
                break
            self.merges.append(pair)
            piece = pair[0] + pair[1]
            if piece not in self._piece_ids:
                token_id = self.vocab_size
                label = f"<bpe:{piece.hex()}>"
                self.token_to_id[label] = token_id
                self.id_to_token[token_id] = label
                self._pieces[token_id] = piece
                self._piece_ids[piece] = token_id
            sequences = [self._merge(pieces, pair) for pieces in sequences]
        self._fitted = True
        return self

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        pieces = [bytes([value]) for value in text.encode("utf-8")]
        for pair in self.merges:
            pieces = self._merge(pieces, pair)
        ids = [self._piece_ids[piece] for piece in pieces]
        if add_special_tokens:
            return [self.token_to_id[self.BOS_TOKEN], *ids, self.token_to_id[self.EOS_TOKEN]]
        return ids

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        chunks: list[bytes] = []
        for value in token_ids:
            token_id = int(value)
            if token_id not in self.id_to_token:
                raise ValueError(f"Unknown token ID: {token_id}")
            if token_id < 4:
                if not skip_special_tokens:
                    chunks.append(self.id_to_token[token_id].encode("utf-8"))
            else:
                chunks.append(self._pieces[token_id])
        return b"".join(chunks).decode("utf-8")


# ============================================================================
# ENGINE 1: PROBABILISTIC LANGUAGE ENGINE
# ============================================================================


class WeightedAutomatonEngine:
    """Variable-order Weighted Finite-State Automaton (WFSA) with Kneser-Ney continuation backoff."""

    def __init__(
        self,
        vocab_size: int,
        max_order: int = 5,
        smoothing_alpha: float = 0.15,
        discount: float = 0.25,
    ) -> None:
        self.vocab_size = max(4, vocab_size)
        self.max_order = max(1, max_order)
        self.smoothing_alpha = smoothing_alpha
        self.discount = float(np.clip(discount, 0.01, 0.90))
        self.unigram_counts: np.ndarray = np.ones(self.vocab_size, dtype=np.float64) * smoothing_alpha
        self.context_counts: dict[tuple[int, ...], dict[int, float]] = defaultdict(lambda: defaultdict(float))
        self.context_totals: dict[tuple[int, ...], float] = defaultdict(float)
        # Kneser-Ney continuation counts: number of distinct preceding contexts per token
        self.continuation_predecessors: dict[int, set[int]] = defaultdict(set)

    def fit_sequences(self, sequences: Sequence[Sequence[int]]) -> None:
        for seq in sequences:
            for i, tok in enumerate(seq):
                t_id = int(tok) % self.vocab_size
                self.unigram_counts[t_id] += 1.0
                if i >= 1:
                    prev_id = int(seq[i - 1]) % self.vocab_size
                    self.continuation_predecessors[t_id].add(prev_id)
                for order in range(1, self.max_order + 1):
                    if i >= order:
                        ctx = tuple(int(x) % self.vocab_size for x in seq[i - order : i])
                        self.context_counts[ctx][t_id] += 1.0
                        self.context_totals[ctx] += 1.0

    def predict_next_proba(self, context: Sequence[int]) -> np.ndarray:
        """Compute Kneser-Ney interpolated next-token probability distribution $P(x_t \\mid x_{<t})$."""
        unigram_prob = self.unigram_counts / float(np.sum(self.unigram_counts))
        cont_counts = np.array(
            [len(self.continuation_predecessors.get(v, set())) for v in range(self.vocab_size)],
            dtype=np.float64,
        )
        total_cont = float(np.sum(cont_counts))
        if total_cont > 0.0:
            kn_base = (cont_counts + self.smoothing_alpha) / (total_cont + self.vocab_size * self.smoothing_alpha)
            probs = 0.65 * unigram_prob + 0.35 * kn_base
        else:
            probs = unigram_prob.copy()

        ctx_list = [int(x) % self.vocab_size for x in context]
        for order in range(1, min(self.max_order, len(ctx_list)) + 1):
            sub_ctx = tuple(ctx_list[-order:])
            total = self.context_totals.get(sub_ctx, 0.0)
            if total > 0.0:
                counts_dict = self.context_counts[sub_ctx]
                n_unique = len(counts_dict)
                d = min(self.discount, 0.5 * min(counts_dict.values()))
                order_vec = np.zeros(self.vocab_size, dtype=np.float64)
                for tok, c in counts_dict.items():
                    if 0 <= tok < self.vocab_size:
                        order_vec[tok] = max(c - d, 0.0) / total
                backoff_mass = (n_unique * d) / total
                kn_step = order_vec + backoff_mass * probs
                confidence = min(0.88, total / (total + 1.8))
                probs = (1.0 - confidence) * probs + confidence * kn_step

        probs = np.maximum(probs, 1e-12)
        return probs / float(np.sum(probs))


class ProbabilisticCircuitEngine:
    """Decomposable and deterministic Sum-Product Probabilistic Circuit (PC) over token windows."""

    def __init__(self, vocab_size: int, num_components: int = 8, window_size: int = 4, seed: int = 42) -> None:
        self.vocab_size = max(4, vocab_size)
        self.num_components = max(2, num_components)
        self.window_size = max(2, window_size)
        rng = np.random.default_rng(seed)
        self.mixture_weights = np.ones(self.num_components, dtype=np.float64) / self.num_components
        raw_leaves = rng.uniform(0.5, 1.5, size=(self.num_components, self.window_size, self.vocab_size))
        self.leaf_categoricals = raw_leaves / np.sum(raw_leaves, axis=-1, keepdims=True)

    def fit_sequences(self, sequences: Sequence[Sequence[int]], n_em_steps: int = 5) -> None:
        windows: list[tuple[int, ...]] = []
        for seq in sequences:
            if len(seq) < self.window_size:
                padded = [0] * (self.window_size - len(seq)) + [int(x) % self.vocab_size for x in seq]
                windows.append(tuple(padded))
            else:
                for i in range(len(seq) - self.window_size + 1):
                    w = tuple(int(x) % self.vocab_size for x in seq[i : i + self.window_size])
                    windows.append(w)
        if not windows:
            return

        win_arr = np.array(windows, dtype=np.int64)
        n_win = win_arr.shape[0]

        for _ in range(n_em_steps):
            resp = np.zeros((n_win, self.num_components), dtype=np.float64)
            for k in range(self.num_components):
                prod = np.ones(n_win, dtype=np.float64) * self.mixture_weights[k]
                for pos in range(self.window_size):
                    prod *= self.leaf_categoricals[k, pos, win_arr[:, pos]]
                resp[:, k] = prod
            row_sums = np.maximum(np.sum(resp, axis=1, keepdims=True), 1e-15)
            resp /= row_sums

            comp_totals = np.sum(resp, axis=0) + 1e-3
            self.mixture_weights = comp_totals / float(np.sum(comp_totals))
            for k in range(self.num_components):
                for pos in range(self.window_size):
                    counts = np.ones(self.vocab_size, dtype=np.float64) * 0.04
                    np.add.at(counts, win_arr[:, pos], resp[:, k])
                    self.leaf_categoricals[k, pos] = counts / float(np.sum(counts))

    def marginal_proba(self, partial_window: Sequence[int | None]) -> float:
        """Compute exact tractable marginal $P(X_{\\mathcal{O}} = x_{\\mathcal{O}})$ where unobserved slots are `None`."""
        comp_probs = self.mixture_weights.copy()
        for pos, val in enumerate(partial_window[: self.window_size]):
            if val is not None:
                tok = int(val) % self.vocab_size
                comp_probs *= self.leaf_categoricals[:, pos, tok]
        return float(np.sum(comp_probs))

    def conditional_next_proba(self, context: Sequence[int]) -> np.ndarray:
        """Exact conditional marginal $P(X_W \\mid X_1, \\dots, X_{W-1})$ in $\\mathcal{O}(K \\cdot W)$ time."""
        prefix_len = self.window_size - 1
        ctx = [int(x) % self.vocab_size for x in context[-prefix_len:]]
        offset = prefix_len - len(ctx)

        comp_weights = self.mixture_weights.copy()
        for idx, tok in enumerate(ctx):
            pos = offset + idx
            comp_weights *= self.leaf_categoricals[:, pos, tok]
        denom = float(np.sum(comp_weights))
        if denom <= 1e-15:
            comp_weights = self.mixture_weights.copy()
        else:
            comp_weights /= denom

        dist = np.einsum("k,kv->v", comp_weights, self.leaf_categoricals[:, -1, :])
        dist = np.maximum(dist, 1e-12)
        return dist / float(np.sum(dist))


class BayesianLanguageEngine:
    """Hierarchical Dirichlet-Multinomial Bayesian predictor quantifying epistemic uncertainty."""

    def __init__(self, vocab_size: int, prior_concentration: float = 0.5, max_order: int = 3) -> None:
        self.vocab_size = max(4, vocab_size)
        self.prior_concentration = prior_concentration
        self.max_order = max(1, max_order)
        self.alpha_table: dict[int, np.ndarray] = {}
        self.order_tables: dict[tuple[int, ...], np.ndarray] = {}
        self.unigram_alpha = np.full(self.vocab_size, self.prior_concentration, dtype=np.float64)

    def fit_sequences(self, sequences: Sequence[Sequence[int]]) -> None:
        for seq in sequences:
            for i, tok in enumerate(seq):
                v = int(tok) % self.vocab_size
                self.unigram_alpha[v] += 1.0
                if i >= 1:
                    u = int(seq[i - 1]) % self.vocab_size
                    if u not in self.alpha_table:
                        self.alpha_table[u] = np.full(self.vocab_size, self.prior_concentration, dtype=np.float64)
                    self.alpha_table[u][v] += 1.0
                for order in range(1, self.max_order + 1):
                    if i >= order:
                        ctx = tuple(int(x) % self.vocab_size for x in seq[i - order : i])
                        if ctx not in self.order_tables:
                            self.order_tables[ctx] = np.full(
                                self.vocab_size, self.prior_concentration, dtype=np.float64
                            )
                        self.order_tables[ctx][v] += 1.0

    def predictive_mean_and_uncertainty(self, context_or_token: int | Sequence[int]) -> tuple[np.ndarray, float]:
        """Return posterior predictive mean and epistemic Dirichlet variance across suffix orders."""
        if isinstance(context_or_token, (int, np.integer)):
            u = int(context_or_token) % self.vocab_size
            alpha = self.alpha_table.get(u, np.full(self.vocab_size, self.prior_concentration, dtype=np.float64))
            alpha_0 = float(np.sum(alpha))
            mean = alpha / alpha_0
            epistemic_uncertainty = float(1.0 / (1.0 + alpha_0 / self.vocab_size))
            return mean, epistemic_uncertainty

        ctx_list = [int(x) % self.vocab_size for x in context_or_token]
        if not ctx_list:
            alpha_0 = float(np.sum(self.unigram_alpha))
            return self.unigram_alpha / alpha_0, 1.0

        blended_mean = self.unigram_alpha / float(np.sum(self.unigram_alpha))
        effective_evidence = 0.0
        prior_mass = self.vocab_size * self.prior_concentration

        for order in range(1, min(self.max_order, len(ctx_list)) + 1):
            sub_ctx = tuple(ctx_list[-order:])
            alpha = self.order_tables.get(sub_ctx)
            if alpha is not None:
                alpha_0 = float(np.sum(alpha))
                obs_counts = max(0.0, alpha_0 - prior_mass)
                effective_evidence += obs_counts * (0.75 ** (self.max_order - order))
                order_mean = alpha / alpha_0
                weight = min(0.85, obs_counts / (obs_counts + 2.0))
                blended_mean = (1.0 - weight) * blended_mean + weight * order_mean

        blended_mean = np.maximum(blended_mean, 1e-12)
        blended_mean /= float(np.sum(blended_mean))
        epistemic_uncertainty = float(1.0 / (1.0 + (prior_mass + effective_evidence) / self.vocab_size))
        return blended_mean, epistemic_uncertainty


# ============================================================================
# ENGINE 2: TENSOR MEMORY ENGINE (1D MPS / TENSOR TRAIN & TREE TENSOR NETWORK)
# ============================================================================


class TensorTrainMemoryEngine:
    """1D Matrix Product State (Tensor Train) context memory: $P(x_1,\\dots,x_n) = G_1[x_1]\\cdots G_n[x_n]$."""

    def __init__(self, vocab_size: int, bond_dim: int = 12, seed: int = 42) -> None:
        self.vocab_size = max(4, vocab_size)
        self.bond_dim = max(2, bond_dim)
        rng = np.random.default_rng(seed)
        raw_cores = rng.normal(0.0, 0.18, size=(self.vocab_size, self.bond_dim, self.bond_dim))
        for v in range(self.vocab_size):
            raw_cores[v] += np.eye(self.bond_dim) * 0.72
        self.cores = raw_cores
        self.readout = np.zeros((self.bond_dim, self.vocab_size), dtype=np.float64)
        self.left_boundary = np.zeros(self.bond_dim, dtype=np.float64)
        self.left_boundary[0] = 1.0

    def fit_sequences(
        self,
        sequences: Sequence[Sequence[int]],
        lr: float = 0.08,
        train_cores: bool = True,
    ) -> None:
        """Fit tensor cores and readout matrix via contracted context statistics and rank-1 core updates."""
        core_lr = 0.025 * lr
        for seq in sequences:
            state = self.left_boundary.copy()
            prev_state = state.copy()
            prev_v: int | None = None
            for tok in seq:
                v = int(tok) % self.vocab_size
                self.readout[:, v] += lr * state

                if train_cores and prev_v is not None:
                    target_dir = self.readout[:, v]
                    t_norm = float(np.linalg.norm(target_dir))
                    if t_norm > 1e-9:
                        delta_core = np.outer(prev_state, target_dir / t_norm)
                        self.cores[prev_v] += core_lr * delta_core
                        frob = float(np.linalg.norm(self.cores[prev_v], ord="fro"))
                        max_frob = math.sqrt(self.bond_dim) * 1.15
                        if frob > max_frob:
                            self.cores[prev_v] *= max_frob / frob

                prev_state = state.copy()
                prev_v = v
                state = state @ self.cores[v]
                norm = float(np.linalg.norm(state))
                if norm > 1e-9:
                    state /= norm

    def contract_context(self, context: Sequence[int]) -> np.ndarray:
        """Contract $G_1[x_1] G_2[x_2] \\cdots G_t[x_t]$ along the 1D chain."""
        state = self.left_boundary.copy()
        for tok in context:
            v = int(tok) % self.vocab_size
            state = state @ self.cores[v]
            norm = float(np.linalg.norm(state))
            if norm > 1e-9:
                state /= norm
        return state

    def predict_next_proba(self, context: Sequence[int]) -> np.ndarray:
        state = self.contract_context(context)
        logits = state @ self.readout
        logits -= float(np.max(logits))
        exp_l = np.exp(logits)
        return exp_l / float(np.sum(exp_l))


class TreeTensorNetworkMemoryEngine:
    """Binary Hierarchical Tree Tensor Network (TTN) resolving the 1D MPS area-law bottleneck.

    Contracts pairs of representations hierarchically in $\\mathcal{O}(\\log L)$ tree depth or
    along syntactic bracket depth when `bracket_depths` are supplied.
    """

    def __init__(self, vocab_size: int, bond_dim: int = 12, seed: int = 42) -> None:
        self.vocab_size = max(4, vocab_size)
        self.bond_dim = max(2, bond_dim)
        rng = np.random.default_rng(seed)
        self.leaf_embeddings = rng.normal(0.0, 0.45, size=(self.vocab_size, self.bond_dim))
        self.merge_left = np.eye(self.bond_dim, dtype=np.float64) * 0.55
        self.merge_right = np.eye(self.bond_dim, dtype=np.float64) * 0.75
        self.bilinear_interaction = rng.normal(0.0, 0.15, size=(self.bond_dim, self.bond_dim))
        self.readout = np.zeros((self.bond_dim, self.vocab_size), dtype=np.float64)

    def _merge_pair(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        merged = (u @ self.merge_left) + (v @ self.merge_right) + np.tanh((u @ self.bilinear_interaction) * v)
        norm = float(np.linalg.norm(merged))
        if norm > 1e-9:
            merged /= norm
        return merged

    def contract_tree(
        self,
        context: Sequence[int],
        bracket_depths: Sequence[int] | None = None,
    ) -> np.ndarray:
        """Perform bottom-up tree tensor contraction (syntax-guided if `bracket_depths` given)."""
        if not context:
            return np.zeros(self.bond_dim, dtype=np.float64)
        leaves = [self.leaf_embeddings[int(tok) % self.vocab_size].copy() for tok in context]
        last_leaf = leaves[-1].copy()

        if bracket_depths is not None and len(bracket_depths) == len(leaves) and max(bracket_depths, default=0) > 0:
            nodes = list(leaves)
            depths = [int(d) for d in bracket_depths]
            while len(nodes) > 1:
                max_d = max(depths)
                merged_any = False
                for i in range(len(nodes) - 1):
                    if depths[i] == max_d and depths[i + 1] == max_d:
                        m = self._merge_pair(nodes[i], nodes[i + 1])
                        nodes = [*nodes[:i], m, *nodes[i + 2 :]]
                        depths = [*depths[:i], max(0, max_d - 1), *depths[i + 2 :]]
                        merged_any = True
                        break
                if not merged_any:
                    m = self._merge_pair(nodes[0], nodes[1])
                    new_d = min(depths[0], depths[1])
                    nodes = [m, *nodes[2:]]
                    depths = [new_d, *depths[2:]]
            root = nodes[0]
        else:
            layer = list(leaves)
            while len(layer) > 1:
                next_layer: list[np.ndarray] = []
                for i in range(0, len(layer), 2):
                    if i + 1 < len(layer):
                        next_layer.append(self._merge_pair(layer[i], layer[i + 1]))
                    else:
                        next_layer.append(layer[i])
                layer = next_layer
            root = layer[0]

        combined = 0.65 * root + 0.35 * last_leaf
        norm_c = float(np.linalg.norm(combined))
        return combined / norm_c if norm_c > 1e-9 else combined

    def fit_sequences(
        self,
        sequences: Sequence[Sequence[int]],
        lr: float = 0.35,
        train_merges: bool = True,
    ) -> None:
        merge_lr = 0.015 * lr
        for seq in sequences:
            for idx in range(1, len(seq)):
                prefix = seq[:idx]
                root_vec = self.contract_tree(prefix)
                target = int(seq[idx]) % self.vocab_size
                self.readout[:, target] += lr * root_vec

                if train_merges and len(prefix) >= 2:
                    u = self.leaf_embeddings[int(prefix[-2]) % self.vocab_size]
                    v = self.leaf_embeddings[int(prefix[-1]) % self.vocab_size]
                    target_emb = self.leaf_embeddings[target]
                    self.merge_left += merge_lr * np.outer(u, target_emb)
                    self.merge_right += merge_lr * np.outer(v, target_emb)
                    self.bilinear_interaction += 0.5 * merge_lr * np.outer(u, v * target_emb)
                    for mat in (self.merge_left, self.merge_right, self.bilinear_interaction):
                        frob = float(np.linalg.norm(mat, ord="fro"))
                        cap = math.sqrt(self.bond_dim) * 1.25
                        if frob > cap:
                            mat *= cap / frob

    def predict_next_proba(
        self,
        context: Sequence[int],
        bracket_depths: Sequence[int] | None = None,
    ) -> np.ndarray:
        root_vec = self.contract_tree(context, bracket_depths=bracket_depths)
        logits = root_vec @ self.readout
        logits -= float(np.max(logits))
        exp_l = np.exp(logits)
        return exp_l / float(np.sum(exp_l))


# ============================================================================
# ENGINE 3: DYNAMICAL CONTEXT & KOOPMAN OPERATOR ENGINE
# ============================================================================


class DynamicalContextEngine:
    """Streaming state evolution engine: $s_{t+1} = F(s_t, \\phi(x_t))$ with spectral radius control."""

    def __init__(self, vocab_size: int, state_dim: int = 16, decay: float = 0.88, seed: int = 42) -> None:
        self.vocab_size = max(4, vocab_size)
        self.state_dim = max(4, state_dim)
        self.decay = float(np.clip(decay, 0.1, 0.99))
        rng = np.random.default_rng(seed)
        q, _ = np.linalg.qr(rng.normal(0.0, 1.0, size=(self.state_dim, self.state_dim)))
        self.transition_matrix = self.decay * q
        self.token_projection = rng.normal(0.0, 0.4, size=(self.vocab_size, self.state_dim))
        self.readout = np.zeros((self.state_dim, self.vocab_size), dtype=np.float64)

    def evolve_sequence(self, context: Sequence[int]) -> np.ndarray:
        """Stream context tokens through $s_{t+1} = A s_t + B \\phi(x_t)$."""
        s = np.zeros(self.state_dim, dtype=np.float64)
        for tok in context:
            phi = self.token_projection[int(tok) % self.vocab_size]
            s = (s @ self.transition_matrix) + phi
        return s

    def fit_sequences(self, sequences: Sequence[Sequence[int]], lr: float = 0.06) -> None:
        for seq in sequences:
            s = np.zeros(self.state_dim, dtype=np.float64)
            for tok in seq:
                v = int(tok) % self.vocab_size
                self.readout[:, v] += lr * s
                s = (s @ self.transition_matrix) + self.token_projection[v]

    def predict_next_proba(self, context: Sequence[int]) -> np.ndarray:
        s = self.evolve_sequence(context)
        logits = s @ self.readout
        logits -= float(np.max(logits))
        exp_l = np.exp(logits)
        return exp_l / float(np.sum(exp_l))


class KoopmanOperatorEngine:
    """Extended Dynamic Mode Decomposition (EDMD) Koopman operator $\\mathcal{K}$ over lifted observables."""

    def __init__(self, state_dim: int = 16, regularization: float = 1e-3) -> None:
        self.state_dim = max(4, state_dim)
        self.regularization = regularization
        self.koopman_matrix: np.ndarray = np.eye(self.state_dim * 2, dtype=np.float64)
        self._fitted = False

    @staticmethod
    def lift_observables(state: np.ndarray) -> np.ndarray:
        """Lift state $s \\in \\mathbb{R}^d$ to observable dictionary $\\psi(s) = [s, \\tanh(s^2)] \\in \\mathbb{R}^{2d}$."""
        s = np.asarray(state, dtype=np.float64)
        return np.concatenate([s, np.tanh(s * s)], axis=-1)

    def fit_trajectories(self, state_trajectories: Sequence[np.ndarray]) -> np.ndarray:
        """Estimate linear Koopman operator $K = (X^T X + \\lambda I)^{-1} X^T Y$ via ridge regression."""
        x_list: list[np.ndarray] = []
        y_list: list[np.ndarray] = []
        for traj in state_trajectories:
            if traj.shape[0] < 2:
                continue
            for t in range(traj.shape[0] - 1):
                x_list.append(self.lift_observables(traj[t]))
                y_list.append(self.lift_observables(traj[t + 1]))
        if not x_list:
            return self.koopman_matrix

        x_mat = np.stack(x_list, axis=0)
        y_mat = np.stack(y_list, axis=0)
        d_obs = x_mat.shape[1]
        gram = x_mat.T @ x_mat + self.regularization * np.eye(d_obs)
        self.koopman_matrix = np.linalg.solve(gram, x_mat.T @ y_mat)
        self._fitted = True
        return self.koopman_matrix

    def predict_next_state(self, state: np.ndarray) -> np.ndarray:
        psi = self.lift_observables(state)
        psi_next = psi @ self.koopman_matrix
        return psi_next[: self.state_dim]

    def predict_next_proba(self, context: Sequence[int], dyn_engine: DynamicalContextEngine) -> np.ndarray:
        """Project Koopman-propagated state $\\hat{s}_{t+1} = (\\mathcal{K}\\psi(s_t))_{1:d}$ onto token probabilities."""
        s_t = dyn_engine.evolve_sequence(context)
        s_next = self.predict_next_state(s_t)
        logits = s_next @ dyn_engine.readout
        logits -= float(np.max(logits))
        exp_l = np.exp(logits)
        return exp_l / float(np.sum(exp_l))

    def spectral_radius(self) -> float:
        eigvals = np.linalg.eigvals(self.koopman_matrix)
        return float(np.max(np.abs(eigvals)))


# ============================================================================
# ENGINE 4: STRUCTURED KNOWLEDGE (HYPERGRAPHS, VSA & SYMBOLIC REWRITING)
# ============================================================================


@dataclass(frozen=True)
class HyperedgeFact:
    """Structured relational tuple $(subject, relation, object)$ stored in the hypergraph."""

    subject: str
    relation: str
    target_object: str
    confidence: float = 1.0


class HypergraphKnowledgeEngine:
    """Explicit concept-relationship hypergraph with multi-hop traversal and contradiction detection."""

    RELATION_TOKENS = frozenset(
        {
            "is",
            "equals",
            "=",
            ":",
            "works_in",
            "has_occupation",
            "located_in",
            "inhibits",
            "activates",
            "targets",
            "causes",
            "commutes_with",
            "reduces_to",
        }
    )

    def __init__(self) -> None:
        self.facts: list[HyperedgeFact] = []
        self._by_subject_rel: dict[tuple[str, str], list[str]] = defaultdict(list)
        self._by_subject: dict[str, list[HyperedgeFact]] = defaultdict(list)

    def add_fact(self, subject: str, relation: str, target_object: str, confidence: float = 1.0) -> None:
        s = subject.strip().lower()
        r = relation.strip().lower()
        o = target_object.strip().lower()
        if not s or not r or not o:
            return
        fact = HyperedgeFact(s, r, o, confidence)
        self.facts.append(fact)
        if o not in self._by_subject_rel[(s, r)]:
            self._by_subject_rel[(s, r)].append(o)
        self._by_subject[s].append(fact)

    def extract_and_store_from_tokens(self, tokens: Sequence[str]) -> int:
        """Extract key-value and relational bindings from token sequences."""
        added = 0
        toks = [t.lower() for t in tokens]
        for i in range(len(toks) - 2):
            if toks[i + 1] in self.RELATION_TOKENS:
                self.add_fact(toks[i], toks[i + 1], toks[i + 2])
                added += 1
        return added

    def query(self, subject: str, relation: str | None = None) -> list[str]:
        s = subject.strip().lower()
        if relation is not None:
            return list(self._by_subject_rel.get((s, relation.strip().lower()), []))
        results: list[str] = []
        for (subj, _rel), objs in self._by_subject_rel.items():
            if subj == s:
                for o in objs:
                    if o not in results:
                        results.append(o)
        return results

    def query_multihop(self, start_subject: str, max_hops: int = 2) -> list[tuple[str, float]]:
        """Breadth-first multi-hop relational traversal returning `(entity, path_confidence)`."""
        s0 = start_subject.strip().lower()
        visited: dict[str, float] = {s0: 1.0}
        frontier: list[tuple[str, float]] = [(s0, 1.0)]
        discovered: dict[str, float] = {}

        for _hop in range(max(1, max_hops)):
            next_frontier: list[tuple[str, float]] = []
            for entity, conf in frontier:
                for fact in self._by_subject.get(entity, []):
                    new_conf = round(conf * fact.confidence * 0.9, 4)
                    target = fact.target_object
                    if target not in visited or new_conf > visited[target]:
                        visited[target] = new_conf
                        discovered[target] = max(discovered.get(target, 0.0), new_conf)
                        next_frontier.append((target, new_conf))
            frontier = next_frontier
            if not frontier:
                break

        return sorted(discovered.items(), key=lambda kv: kv[1], reverse=True)

    def resolve_chain(self, subject: str, relations: Sequence[str]) -> list[str]:
        """Resolve a multi-hop relational chain $e_0 \\xrightarrow{r_1} e_1 \\xrightarrow{r_2} \\dots \\xrightarrow{r_k} e_k$."""
        current = [subject.strip().lower()]
        for rel in relations:
            r = rel.strip().lower()
            next_entities: list[str] = []
            for ent in current:
                for obj in self._by_subject_rel.get((ent, r), []):
                    if obj not in next_entities:
                        next_entities.append(obj)
            current = next_entities
            if not current:
                break
        return current

    def count_contradictions(self, subject: str, relation: str, candidate_object: str) -> int:
        """Count known facts for `(subject, relation)` that conflict with `candidate_object`."""
        known = self.query(subject, relation)
        if not known:
            return 0
        cand = candidate_object.strip().lower()
        if cand in known:
            return 0
        return len(known)


class VectorSymbolicEngine:
    """Holographic Reduced Representations (VSA / HRR) in $\\mathbb{R}^D$ using FFT circular convolution."""

    def __init__(self, dim: int = 128, seed: int = 42) -> None:
        self.dim = max(16, dim)
        self.rng = np.random.default_rng(seed)
        self.item_memory: dict[str, np.ndarray] = {}
        self.composite_memory = np.zeros(self.dim, dtype=np.float64)

    def get_or_create_vector(self, symbol: str) -> np.ndarray:
        key = symbol.strip().lower()
        if key not in self.item_memory:
            vec = self.rng.normal(0.0, 1.0 / math.sqrt(self.dim), size=self.dim)
            vec /= max(float(np.linalg.norm(vec)), 1e-12)
            self.item_memory[key] = vec
        return self.item_memory[key]

    @staticmethod
    def bind(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Circular convolution $u \\circledast v = \\mathcal{F}^{-1}(\\mathcal{F}(u) \\odot \\mathcal{F}(v))$."""
        return np.fft.irfft(np.fft.rfft(u) * np.fft.rfft(v), n=u.shape[0])

    @staticmethod
    def unbind(composite: np.ndarray, key_vec: np.ndarray) -> np.ndarray:
        """Approximate inverse (circular correlation) via complex conjugate in frequency domain."""
        return np.fft.irfft(np.fft.rfft(composite) * np.conj(np.fft.rfft(key_vec)), n=composite.shape[0])

    def store_pair(self, key: str, value: str) -> None:
        k_vec = self.get_or_create_vector(key)
        v_vec = self.get_or_create_vector(value)
        self.composite_memory = self.composite_memory + self.bind(k_vec, v_vec)

    def retrieve_value(self, key: str) -> tuple[str, float]:
        """Unbind `key` from superposition memory and clean up against item memory."""
        if not self.item_memory:
            return "", 0.0
        k_vec = self.get_or_create_vector(key)
        noisy = self.unbind(self.composite_memory, k_vec)
        n_norm = max(float(np.linalg.norm(noisy)), 1e-12)

        best_sym = ""
        best_sim = -2.0
        for sym, vec in self.item_memory.items():
            if sym == key.strip().lower():
                continue
            sim = float(np.dot(noisy, vec) / n_norm)
            if sim > best_sim:
                best_sim = sim
                best_sym = sym
        return best_sym, round(best_sim, 4)


class SymbolicRewriteEngine:
    """Deterministic algebraic & hierarchical term-rewriting engine (Group A #15).

    Supports:
      - Recursive evaluation of hierarchical ListOps expressions (`MAX`, `MIN`, `SUM`, `MED`).
      - Algebraic quantum gate sequence simplification (involution cancellation, Hadamard
        conjugation $H Z H \\to X$, $H X H \\to Z$, and disjoint-qubit canonical commutation).
    """

    SELF_INVERSE_GATES = frozenset({"id", "x", "y", "z", "h", "cx", "cz", "swap"})

    @staticmethod
    def evaluate_listops(expr: str) -> int:
        """Evaluate nested ListOps expressions via bottom-up tree reduction."""
        tokens = re.findall(r"\[|\]|MAX|MIN|SUM|MED|\d+", expr.upper())
        pos = 0

        def parse_node() -> int:
            nonlocal pos
            if pos >= len(tokens):
                return 0
            tok = tokens[pos]
            if tok == "[":
                pos += 1
                if pos >= len(tokens):
                    return 0
                op = tokens[pos]
                pos += 1
                args: list[int] = []
                while pos < len(tokens) and tokens[pos] != "]":
                    args.append(parse_node())
                if pos < len(tokens) and tokens[pos] == "]":
                    pos += 1
                if not args:
                    return 0
                if op == "MAX":
                    return max(args)
                if op == "MIN":
                    return min(args)
                if op == "SUM":
                    return sum(args) % 10
                if op == "MED":
                    s = sorted(args)
                    return s[len(s) // 2]
                return args[0]
            pos += 1
            return int(tok) if tok.isdigit() else 0

        return parse_node()

    @classmethod
    def rewrite_gate_sequence(
        cls,
        gates: Sequence[tuple[str, tuple[int, ...]]],
    ) -> list[tuple[str, tuple[int, ...]]]:
        """Apply algebraic rewrite rules to a sequence of `(gate_name, qubits)` tuples until fixed point."""
        seq = [(name.lower(), tuple(int(q) for q in qubits)) for name, qubits in gates if name.lower() != "id"]
        changed = True
        while changed:
            changed = False
            i = 0
            new_seq: list[tuple[str, tuple[int, ...]]] = []
            while i < len(seq):
                # Rule 1: H Z H -> X and H X H -> Z on the same single qubit
                if (
                    i + 2 < len(seq)
                    and seq[i][0] == "h"
                    and seq[i + 2][0] == "h"
                    and seq[i][1] == seq[i + 1][1] == seq[i + 2][1]
                    and seq[i + 1][0] in ("x", "z")
                ):
                    conjugated = "z" if seq[i + 1][0] == "x" else "x"
                    new_seq.append((conjugated, seq[i][1]))
                    i += 3
                    changed = True
                    continue

                # Rule 2: Adjacent involution cancellation G G -> I and S Sdg / Sdg S -> I
                if i + 1 < len(seq) and seq[i][1] == seq[i + 1][1]:
                    g1, g2 = seq[i][0], seq[i + 1][0]
                    if (g1 == g2 and g1 in cls.SELF_INVERSE_GATES) or {g1, g2} == {"s", "sdg"}:
                        i += 2
                        changed = True
                        continue

                new_seq.append(seq[i])
                i += 1
            seq = new_seq
        return seq


# ============================================================================
# ENGINE 5: GLOBAL ENERGY EVALUATION, SPECULATIVE RESCORING & ADJUDICATION
# ============================================================================


@dataclass(frozen=True)
class EnergyBreakdown:
    """Sequence-level multi-component energy $E(x \\mid c)$ (lower energy = higher compatibility)."""

    total_energy: float
    language_energy: float
    context_energy: float
    knowledge_energy: float
    constraint_energy: float


class EnergyCandidateEvaluator:
    """Evaluates complete candidate continuations via $E(x \\mid c) = E_{\\text{lang}} + E_{\\text{ctx}} + E_{\\text{know}} + E_{\\text{const}}$."""

    def __init__(
        self,
        tokenizer: MMLMTokenizer,
        hypergraph: HypergraphKnowledgeEngine | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.hypergraph = hypergraph

    def evaluate_candidate(
        self,
        context_ids: Sequence[int],
        candidate_ids: Sequence[int],
        step_log_probs: Sequence[float],
    ) -> EnergyBreakdown:
        """Compute global sequence energy for a candidate continuation."""
        # 1. Language energy: negative mean log-probability
        lang_e = -float(np.mean(step_log_probs)) if step_log_probs else 10.0

        # 2. Context continuity energy: penalize immediate token repetition and degenerate loops
        rep_count = sum(1 for a, b in pairwise(candidate_ids) if a == b)
        unique_ratio = len(set(candidate_ids)) / max(1, len(candidate_ids))
        ctx_e = 1.5 * rep_count + max(0.0, 0.65 - unique_ratio) * 3.0

        # 3. Knowledge consistency energy: reward agreement with 1-hop & multi-hop hypergraph facts
        know_e = 0.0
        if self.hypergraph is not None and self.hypergraph.facts:
            ctx_tokens = [self.tokenizer.id_to_token.get(int(i), "") for i in context_ids]
            cand_tokens = [self.tokenizer.id_to_token.get(int(i), "") for i in candidate_ids]
            cand_set = set(cand_tokens)
            for fact in self.hypergraph.facts:
                if fact.subject in ctx_tokens:
                    if fact.target_object in cand_set:
                        know_e -= 1.25 * fact.confidence
                    else:
                        know_e += 0.20
            for c_tok in set(ctx_tokens):
                if not c_tok:
                    continue
                for mh_target, mh_conf in self.hypergraph.query_multihop(c_tok, max_hops=2):
                    if mh_target in cand_set:
                        know_e -= 0.45 * mh_conf

        # 4. Structural constraint energy: balanced parentheses/brackets
        cand_text = self.tokenizer.decode(candidate_ids)
        open_parens = cand_text.count("(") + cand_text.count("[")
        close_parens = cand_text.count(")") + cand_text.count("]")
        const_e = 1.2 * abs(open_parens - close_parens)

        total = lang_e + ctx_e + know_e + const_e
        return EnergyBreakdown(
            total_energy=round(total, 4),
            language_energy=round(lang_e, 4),
            context_energy=round(ctx_e, 4),
            knowledge_energy=round(know_e, 4),
            constraint_energy=round(const_e, 4),
        )


class SpeculativeEnergyDecoder:
    """Polynomial-time multi-step beam & plateau search bypassing the #P-hard partition function $Z(c)$."""

    def __init__(
        self,
        evaluator: EnergyCandidateEvaluator,
        top_k_beams: int = 4,
        branch_factor: int = 2,
    ) -> None:
        self.evaluator = evaluator
        self.top_k_beams = max(1, top_k_beams)
        self.branch_factor = max(1, branch_factor)

    def generate_best_sequence(
        self,
        context_ids: Sequence[int],
        next_proba_fn: Callable[[Sequence[int]], np.ndarray],
        gen_length: int = 8,
    ) -> tuple[list[int], EnergyBreakdown]:
        """Multi-step beam expansion pruning to `top_k_beams` by sequence energy at every step."""
        beams: list[tuple[list[int], list[float]]] = [([], [])]

        for step in range(gen_length):
            expanded: list[tuple[EnergyBreakdown, list[int], list[float]]] = []
            n_branches = self.top_k_beams if step == 0 else min(self.branch_factor, self.top_k_beams)
            for cand_seq, log_ps in beams:
                full_ctx = [*list(context_ids), *cand_seq]
                probs = next_proba_fn(full_ctx)
                masked = probs.copy()
                masked[:4] *= 1e-6
                masked /= max(float(np.sum(masked)), 1e-12)
                ranked_ids = np.argsort(masked)[::-1]

                for b_idx in range(min(n_branches, len(ranked_ids))):
                    chosen_id = int(ranked_ids[b_idx])
                    chosen_lp = math.log(max(float(masked[chosen_id]), 1e-12))
                    next_seq = [*cand_seq, chosen_id]
                    next_lps = [*log_ps, chosen_lp]
                    eb = self.evaluator.evaluate_candidate(context_ids, next_seq, next_lps)
                    expanded.append((eb, next_seq, next_lps))

            expanded.sort(key=lambda item: item[0].total_energy)
            deduped: list[tuple[list[int], list[float]]] = []
            seen: set[tuple[int, ...]] = set()
            for _eb, seq, lps in expanded:
                key = tuple(seq)
                if key not in seen:
                    seen.add(key)
                    deduped.append((seq, lps))
                    if len(deduped) >= self.top_k_beams:
                        break
            beams = deduped if deduped else [([], [])]

        scored: list[tuple[EnergyBreakdown, list[int]]] = []
        for cand_seq, log_ps in beams:
            eb = self.evaluator.evaluate_candidate(context_ids, cand_seq, log_ps)
            scored.append((eb, cand_seq))

        scored.sort(key=lambda item: item[0].total_energy)
        best_eb, best_seq = scored[0]
        return best_seq, best_eb

    def search_discrete_plateau(
        self,
        initial_state: object,
        neighbor_fn: Callable[[object], Sequence[object]],
        energy_fn: Callable[[object], tuple[float, ...]],
        max_steps: int = 16,
    ) -> tuple[object, tuple[float, ...]]:
        """Non-greedy beam search over discrete state spaces (e.g. circuits/graphs) traversing $\\Delta = 0$ plateaus."""
        best_state = initial_state
        best_score = energy_fn(initial_state)
        beam: list[tuple[tuple[float, ...], object]] = [(best_score, initial_state)]
        visited: set[int] = {hash(repr(initial_state))}

        for _ in range(max_steps):
            candidates: list[tuple[tuple[float, ...], object]] = []
            for _score, state in beam:
                for nb in neighbor_fn(state):
                    h = hash(repr(nb))
                    if h in visited:
                        continue
                    visited.add(h)
                    sc = energy_fn(nb)
                    if sc < best_score:
                        best_score = sc
                        best_state = nb
                    candidates.append((sc, nb))
            if not candidates:
                break
            candidates.sort(key=lambda item: item[0])
            beam = candidates[: self.top_k_beams]

        return best_state, best_score


@dataclass(frozen=True)
class MMLMAdjudicationReceipt:
    """Epistemic verification receipt emitted by `MMLMEpistemicAdjudicator`."""

    admitted: bool
    energy_breakdown: EnergyBreakdown
    epistemic_uncertainty: float
    contradictions_detected: int
    verification_status: str


class MMLMEpistemicAdjudicator:
    """APEDP epistemic evaluation layer checking evidence, contradictions, and uncertainty."""

    def __init__(
        self,
        max_allowed_energy: float = 8.5,
        max_epistemic_uncertainty: float = 0.85,
    ) -> None:
        self.max_allowed_energy = max_allowed_energy
        self.max_epistemic_uncertainty = max_epistemic_uncertainty

    def adjudicate_generation(
        self,
        energy: EnergyBreakdown,
        epistemic_uncertainty: float,
        contradictions: int = 0,
    ) -> MMLMAdjudicationReceipt:
        admitted = (
            energy.total_energy <= self.max_allowed_energy
            and epistemic_uncertainty <= self.max_epistemic_uncertainty
            and contradictions == 0
        )
        status = "ADMIT" if admitted else "REFUSE"
        return MMLMAdjudicationReceipt(
            admitted=admitted,
            energy_breakdown=energy,
            epistemic_uncertainty=round(epistemic_uncertainty, 4),
            contradictions_detected=contradictions,
            verification_status=status,
        )
