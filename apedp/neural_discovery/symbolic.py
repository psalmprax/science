# Databricks notebook source
"""Differentiable Symbolic Regression for APEDP.

Implements end-to-end differentiable symbolic regression using:
- Neural network proposes expression trees (Transformer decoder)
- JAX differentiable symbolic execution
- RL (REINFORCE) + supervised training (behavioral cloning from bic_forward_select)
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from jax import grad, jit, random

from apedp.discovery import bic_forward_select
from apedp.neural_discovery.architectures import (
    MLPResNet,
    TransformerDecoder,
    init_model,
)
from apedp.neural_discovery.config import NeuralDiscoveryConfig

Array = jnp.ndarray
PRNGKey = jax.Array
Variables = dict[str, Array]


# Expression grammar in prefix notation
# Operators: ADD, SUB, MUL, DIV, POW, SIN, COS, EXP, LOG, SQRT, NEG
# Terminals: VAR_0, VAR_1, ..., CONST

OPERATORS = {
    'ADD': 2, 'SUB': 2, 'MUL': 2, 'DIV': 2, 'POW': 2,
    'SIN': 1, 'COS': 1, 'EXP': 1, 'LOG': 1, 'SQRT': 1, 'NEG': 1,
}

TERMINALS_BASE = ['CONST']
VAR_PREFIX = 'VAR_'

# Build vocab
VOCAB = ['PAD', 'EOS', 'BOS'] + list(OPERATORS.keys()) + TERMINALS_BASE
VOCAB_TO_ID = {token: i for i, token in enumerate(VOCAB)}
ID_TO_VOCAB = {i: token for i, token in enumerate(VOCAB)}

PAD_ID = VOCAB_TO_ID['PAD']
EOS_ID = VOCAB_TO_ID['EOS']
BOS_ID = VOCAB_TO_ID['BOS']


def build_vocab(n_variables: int) -> tuple[dict[str, int], dict[int, str]]:
    """Build vocabulary for n variables."""
    terminals = TERMINALS_BASE + [f'{VAR_PREFIX}{i}' for i in range(n_variables)]
    vocab = ['PAD', 'EOS', 'BOS'] + list(OPERATORS.keys()) + terminals
    return {token: i for i, token in enumerate(vocab)}, {i: token for i, token in enumerate(vocab)}


class ExpressionTokenizer:
    """Tokenize symbolic expressions to/from prefix notation sequences."""

    def __init__(self, n_variables: int, max_len: int = 128):
        self.vocab_to_id, self.id_to_vocab = build_vocab(n_variables)
        self.n_variables = n_variables
        self.max_len = max_len

    def tokenize(self, expr_str: str) -> list[int]:
        """Convert expression string to token IDs (prefix notation)."""
        # Simple parser for basic expressions
        # In practice, use sympy to parse and convert to prefix
        tokens = self._parse_to_prefix(expr_str)
        ids = [BOS_ID] + [self.vocab_to_id.get(t, PAD_ID) for t in tokens] + [EOS_ID]
        # Pad/truncate
        if len(ids) < self.max_len:
            ids += [PAD_ID] * (self.max_len - len(ids))
        else:
            ids = ids[:self.max_len - 1] + [EOS_ID]
        return ids

    def detokenize(self, ids: list[int]) -> str:
        """Convert token IDs back to expression string."""
        tokens = [self.id_to_vocab.get(i, 'PAD') for i in ids]
        # Remove special tokens
        tokens = [t for t in tokens if t not in ('PAD', 'BOS', 'EOS')]
        return self._prefix_to_infix(tokens)

    def _parse_to_prefix(self, expr: str) -> list[str]:
        """Parse infix expression to prefix tokens (simplified)."""
        # This is a placeholder; real implementation uses sympy
        # For now, return a simple hardcoded example
        return expr.replace(' ', '').replace('+', ' ADD ').replace('-', ' SUB ').split()

    def _prefix_to_infix(self, tokens: list[str]) -> str:
        """Convert prefix tokens to infix string."""
        # Simplified reconstruction
        stack = []
        for token in reversed(tokens):
            if token in OPERATORS:
                arity = OPERATORS[token]
                if arity == 2:
                    a = stack.pop()
                    b = stack.pop()
                    stack.append(f'({a} {token} {b})')
                elif arity == 1:
                    a = stack.pop()
                    stack.append(f'{token}({a})')
            else:
                stack.append(token)
        return stack[0] if stack else ''


class JAXSymbolicExecutor:
    """Execute symbolic expressions in JAX with autodiff support.

    Supports differentiable operators: +, -, *, /, sin, cos, exp, log, sqrt, pow
    """

    def __init__(self, n_variables: int):
        self.n_variables = n_variables
        self.operators = {
            'ADD': lambda a, b: a + b,
            'SUB': lambda a, b: a - b,
            'MUL': lambda a, b: a * b,
            'DIV': lambda a, b: a / (b + 1e-8),
            'POW': lambda a, b: jnp.abs(a) ** b,
            'SIN': lambda a: jnp.sin(a),
            'COS': lambda a: jnp.cos(a),
            'EXP': lambda a: jnp.exp(jnp.clip(a, -10, 10)),
            'LOG': lambda a: jnp.log(jnp.abs(a) + 1e-8),
            'SQRT': lambda a: jnp.sqrt(jnp.abs(a) + 1e-8),
            'NEG': lambda a: -a,
        }

    def execute(self, tokens: list[str], variables: dict[str, Array],
                constants: Array | None = None) -> Array:
        """Execute prefix expression.

        Args:
            tokens: Prefix token list
            variables: Dict of variable arrays {var_name: array}
            constants: Array of constant values for CONST terminals

        Returns:
            Evaluated expression array
        """
        var_arrays = [variables.get(f'{VAR_PREFIX}{i}', jnp.zeros(()))
                      for i in range(self.n_variables)]
        const_idx = [0]

        def eval_recursive(token_iter):
            token = next(token_iter)
            if token in self.operators:
                arity = OPERATORS[token]
                if arity == 2:
                    a = eval_recursive(token_iter)
                    b = eval_recursive(token_iter)
                    return self.operators[token](a, b)
                else:
                    a = eval_recursive(token_iter)
                    return self.operators[token](a)
            elif token == 'CONST':
                if constants is not None and const_idx[0] < len(constants):
                    val = constants[const_idx[0]]
                    const_idx[0] += 1
                    return val
                return 1.0
            elif token.startswith(VAR_PREFIX):
                var_idx = int(token[len(VAR_PREFIX):])
                if var_idx < self.n_variables:
                    return var_arrays[var_idx]
                return 1.0
            else:
                return 1.0

        return eval_recursive(iter(tokens))

    def execute_batch(self, token_sequences: list[list[str]],
                      variables: dict[str, Array],
                      constants: Array | None = None) -> Array:
        """Execute multiple expressions (batched)."""
        results = []
        for tokens in token_sequences:
            try:
                result = self.execute(tokens, variables, constants)
                results.append(result)
            except (RuntimeError, TypeError, ValueError, ArithmeticError,
                    KeyError, IndexError):
                # Invalid generated expressions yield NaN, and the caller sees
                # that as an unusable candidate rather than a silent pass.
                results.append(jnp.full_like(next(iter(variables.values())), jnp.nan))
        return jnp.stack(results)


class NeuralSymbolicRegressor(nn.Module):
    """Neural Symbolic Regressor: Transformer decoder generates expressions.

    Architecture:
    - Encoder: processes input data (x, y) -> latent
    - Decoder: autoregressive Transformer generates prefix expression tokens
    - Executor: JAX symbolic executor evaluates expressions
    - Loss: MSE + complexity penalty
    """

    config: NeuralDiscoveryConfig
    n_variables: int
    vocab_size: int

    def setup(self):
        self.encoder = MLPResNet(
            hidden_dims=self.config.hidden_dims,
            num_layers=2,
            dropout_rate=self.config.dropout_rate,
            activation=self.config.activation,
            output_dim=self.config.transformer_dim,
        )
        self.decoder = TransformerDecoder(
            vocab_size=self.vocab_size,
            dim=self.config.transformer_dim,
            num_heads=self.config.transformer_heads,
            num_layers=self.config.transformer_layers,
            max_seq_len=self.config.max_seq_len,
            dropout_rate=self.config.dropout_rate,
        )
        self.executor = JAXSymbolicExecutor(self.n_variables)

    def encode(self, x: Array, y: Array) -> Array:
        """Encode (x, y) pairs to latent representation."""
        # x: (batch, n_vars), y: (batch,)
        # Concatenate and process
        xy = jnp.concatenate([x, y.reshape(-1, 1)], axis=1)
        latent = self.encoder(xy)
        return latent

    def decode(self, latent: Array, train: bool = False, rng: PRNGKey | None = None) -> tuple[Array, Array]:
        """Autoregressive decoding of expression tokens.

        Returns:
            logits: (batch, seq_len, vocab_size)
            tokens: (batch, seq_len) sampled tokens
        """
        batch_size = latent.shape[0]
        # Start with BOS token
        tokens = jnp.full((batch_size, 1), BOS_ID, dtype=jnp.int32)

        logits_list = []

        for pos in range(self.config.max_seq_len - 1):
            # Decoder forward - pass latent for conditioning
            logits = self.decoder(tokens, latent, train=train)
            next_logits = logits[:, -1, :]  # (batch, vocab)
            logits_list.append(next_logits)

            if train:
                # Teacher forcing would go here
                pass
            else:
                # Sample next token
                if rng is not None:
                    rng, sample_rng = random.split(rng)
                else:
                    sample_rng = self.make_rng('sample')
                next_token = random.categorical(sample_rng, next_logits, axis=-1)
                tokens = jnp.concatenate([tokens, next_token[:, None]], axis=1)

        logits = jnp.stack(logits_list, axis=1) if logits_list else jnp.zeros((batch_size, 0, self.vocab_size))
        return logits, tokens

    def forward(self, x: Array, y: Array, train: bool = False) -> dict[str, Array]:
        """Full forward pass."""
        latent = self.encode(x, y)
        logits, tokens = self.decode(latent, train=train)

        return {
            'latent': latent,
            'logits': logits,
            'tokens': tokens,
        }

    @nn.compact
    def __call__(self, x: Array, y: Array, train: bool = False) -> dict[str, Array]:
        """Default call method - delegates to forward."""
        return self.forward(x, y, train=train)

    def loss_fn(self, params: Variables, x: Array, y: Array,
                key: PRNGKey, baseline: Array | None = None) -> tuple[float, dict]:
        """RL loss with REINFORCE + baseline."""
        variables = {'params': params}
        batch_size = x.shape[0]

        # Sample expressions
        latent = self.apply(variables, x, y, method=self.encode)
        logits, tokens = self.apply(variables, latent, method=self.decode, train=True)

        # Evaluate expressions
        rewards = []
        valid_masks = []

        for b in range(batch_size):
            token_seq = tokens[b]
            # Convert to token strings
            token_strs = [ID_TO_VOCAB.get(int(t), 'PAD') for t in token_seq]
            # Execute
            var_dict = {f'{VAR_PREFIX}{i}': x[b, i] for i in range(self.n_variables)}
            try:
                pred = self.executor.execute(token_strs, var_dict)
                mse = jnp.mean((pred - y[b]) ** 2)
                complexity = len([t for t in token_strs if t != 'PAD' and t != 'EOS'])
                reward = -mse - self.config.symbolic_complexity_weight * complexity
                rewards.append(reward)
                valid_masks.append(1.0)
            except (RuntimeError, TypeError, ValueError, ArithmeticError,
                    KeyError, IndexError):
                rewards.append(-100.0)  # Large negative for invalid
                valid_masks.append(0.0)

        rewards = jnp.array(rewards)
        valid_masks = jnp.array(valid_masks)

        # REINFORCE: -E[reward * log_prob]
        log_probs = jnp.sum(
            jax.nn.log_softmax(logits, axis=-1) *
            jax.nn.one_hot(tokens[:, 1:], self.vocab_size),
            axis=-1
        ).sum(axis=1)  # Sum over sequence

        if baseline is not None:
            rewards = rewards - baseline

        loss = -jnp.mean(rewards * log_probs * valid_masks)

        return float(loss), {
            'mean_reward': float(jnp.mean(rewards * valid_masks)),
            'valid_rate': float(jnp.mean(valid_masks)),
        }

    def supervised_loss(self, params: Variables, x: Array, y: Array,
                        target_tokens: Array) -> tuple[float, dict]:
        """Supervised loss: behavioral cloning from bic_forward_select."""
        variables = {'params': params}
        latent = self.apply(variables, x, y, method=self.encode)
        logits, _ = self.apply(variables, latent, method=self.decode, train=True)

        # Cross-entropy loss
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits.reshape(-1, self.vocab_size),
            target_tokens.reshape(-1)
        ).mean()

        return float(loss), {'ce_loss': float(loss)}


def discover_symbolic(x_data: dict[str, Array],
                      y_data: Array,
                      config: NeuralDiscoveryConfig | None = None,
                      n_iterations: int = 100) -> dict[str, Any]:
    """High-level API for differentiable symbolic regression.

    Args:
        x_data: Dict of variable arrays
        y_data: Target array
        config: NeuralDiscoveryConfig
        n_iterations: Training iterations

    Returns:
        Dict with discovered expressions, rewards, etc.
    """
    if config is None:
        config = NeuralDiscoveryConfig(architecture="symbolic_transformer")

    n_variables = len(x_data)
    x = jnp.column_stack([x_data[k] for k in sorted(x_data.keys())])
    y = y_data

    # Initialize model
    model = NeuralSymbolicRegressor(config, n_variables, config.vocab_size)
    key = random.PRNGKey(config.seed)
    params = init_model(model, x.shape, key, config)

    # Supervised pre-training from bic_forward_select
    print("Generating supervised targets from bic_forward_select...")
    bic_result = bic_forward_select(x_data, np.asarray(y_data))
    target_expr = bic_result.get('terms', [])

    # Tokenize targets (simplified)
    tokenizer = ExpressionTokenizer(n_variables, config.max_seq_len)
    target_tokens = []
    for term in target_expr[:5]:  # Top 5 terms
        tok = tokenizer.tokenize(term)
        target_tokens.append(tok)
    target_tokens = jnp.array(target_tokens) if target_tokens else jnp.zeros((1, config.max_seq_len))

    # Training loop
    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(params)

    @jit
    def train_step(params, opt_state, x, y, key):
        loss, info = model.apply({'params': params}, x, y, method=model.loss_fn, key=key)
        grads = grad(lambda p: model.apply({'params': p}, x, y, method=model.loss_fn, key=key)[0])(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, info

    print("Training neural symbolic regressor...")
    for i in range(n_iterations):
        key, subkey = random.split(key)
        params, opt_state, loss, info = train_step(params, opt_state, x, y, subkey)

        if i % 20 == 0:
            print(f"  Iter {i}: loss={loss:.4f}, reward={info['mean_reward']:.4f}, valid={info['valid_rate']:.2f}")

    # Generate final expressions
    print("Generating final expressions...")
    final_output = model.apply({'params': params}, x[:5], y[:5], method=model.forward, train=False)

    return {
        'params': params,
        'expressions': [tokenizer.detokenize(t.tolist()) for t in final_output['tokens']],
        'rewards': info['mean_reward'],
        'valid_rate': info['valid_rate'],
    }


# MLPResNet import for encoder

__all__ = [
    "ID_TO_VOCAB",
    "OPERATORS",
    "VOCAB_TO_ID",
    "ExpressionTokenizer",
    "JAXSymbolicExecutor",
    "NeuralSymbolicRegressor",
    "discover_symbolic",
]

# COMMAND ----------

# Production parser override: keep the public tokenizer API but parse through SymPy.
import sympy as _sp


class ExpressionTokenizer(ExpressionTokenizer):
    def _parse_to_prefix(self, expr: str) -> list[str]:
        parsed = _sp.sympify(expr, evaluate=False)

        def visit(node: _sp.Expr) -> list[str]:
            if node.is_Number:
                return ["CONST"]
            if node.is_Symbol:
                name = str(node)
                if name.startswith("x") and name[1:].isdigit():
                    name = f"{VAR_PREFIX}{name[1:]}"
                return [name]
            if node.func in (_sp.Add, _sp.Mul):
                operator = "ADD" if node.func is _sp.Add else "MUL"
                values = list(node.args)
                result = visit(values[0])
                for value in values[1:]:
                    result = [operator, *result, *visit(value)]
                return result
            if node.func is _sp.Pow:
                if node.args[1] == _sp.Rational(1, 2):
                    return ["SQRT", *visit(node.args[0])]
                return ["POW", *visit(node.args[0]), *visit(node.args[1])]
            unary = {_sp.sin: "SIN", _sp.cos: "COS", _sp.exp: "EXP", _sp.log: "LOG", _sp.sqrt: "SQRT"}
            if node.func in unary:
                return [unary[node.func], *visit(node.args[0])]
            raise ValueError(f"unsupported symbolic node: {node.func}")

        return visit(parsed)
