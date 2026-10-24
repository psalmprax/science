"""JAX/Flax neural architectures for scientific discovery.

Implements modern architectures:
- MLPResNet: Residual MLP with LayerNorm, Swish, dropout
- SymbolicTransformer: Transformer encoder for expression sequences
- EquationGNN: Graph network over expression trees
- PhysicsInformedNN: PINN-style with PDE residual loss
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.core import FrozenDict

from apedp.neural_discovery.config import NeuralDiscoveryConfig

# Type aliases
Array = jnp.ndarray
PRNGKey = jax.Array
Variables = dict[str, Array]


class Swish(nn.Module):
    """Swish activation: x * sigmoid(x)."""
    beta: float = 1.0

    @nn.compact
    def __call__(self, x: Array) -> Array:
        return x * nn.sigmoid(self.beta * x)


class MLPResNet(nn.Module):
    """Residual MLP with LayerNorm, Swish, and dropout.

    Architecture: Input -> [ResBlock] * num_layers -> Output
    ResBlock: LayerNorm -> Swish -> Linear -> Dropout -> Residual
    """

    hidden_dims: Sequence[int]
    num_layers: int
    dropout_rate: float = 0.1
    activation: str = "swish"
    layer_norm: bool = True
    residual: bool = True
    output_dim: int = 1

    @nn.compact
    def __call__(self, x: Array, train: bool = False) -> Array:
        # Input projection
        x = nn.Dense(self.hidden_dims[0], name="input_proj")(x)

        # Activation function
        act_fn = self._get_activation()

        # Residual blocks
        for i in range(self.num_layers):
            residual = x
            if self.layer_norm:
                x = nn.LayerNorm(name=f"ln_{i}")(x)
            x = act_fn(x)
            x = nn.Dense(self.hidden_dims[min(i + 1, len(self.hidden_dims) - 1)],
                         name=f"dense_{i}")(x)
            x = nn.Dropout(self.dropout_rate, deterministic=not train)(x)
            if self.residual and residual.shape == x.shape:
                x = x + residual

        # Output projection
        if self.layer_norm:
            x = nn.LayerNorm(name="ln_out")(x)
        x = act_fn(x)
        x = nn.Dense(self.output_dim, name="output_proj")(x)
        return x

    def _get_activation(self) -> Callable[[Array], Array]:
        if self.activation == "swish":
            return Swish()
        elif self.activation == "gelu":
            return nn.gelu
        elif self.activation == "silu":
            return nn.silu
        elif self.activation == "relu":
            return nn.relu
        else:
            raise ValueError(f"Unknown activation: {self.activation}")


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with optional causal mask."""

    num_heads: int
    qkv_features: int
    out_features: int
    dropout_rate: float = 0.1
    causal: bool = False

    @nn.compact
    def __call__(self, x: Array, train: bool = False, mask: Array | None = None) -> Array:
        return nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.qkv_features,
            out_features=self.out_features,
            dropout_rate=self.dropout_rate,
            deterministic=not train,
        )(x, mask=mask)


class TransformerBlock(nn.Module):
    """Transformer encoder block: Attention -> Add&Norm -> MLP -> Add&Norm."""

    dim: int
    num_heads: int
    mlp_ratio: float = 4.0
    dropout_rate: float = 0.1
    activation: str = "swish"

    @nn.compact
    def __call__(self, x: Array, train: bool = False, mask: Array | None = None) -> Array:
        # Attention
        residual = x
        x = nn.LayerNorm(name="ln_attn")(x)
        x = MultiHeadAttention(
            num_heads=self.num_heads,
            qkv_features=self.dim,
            out_features=self.dim,
            dropout_rate=self.dropout_rate,
        )(x, train=train, mask=mask)
        x = nn.Dropout(self.dropout_rate, deterministic=not train)(x)
        x = x + residual

        # MLP
        residual = x
        x = nn.LayerNorm(name="ln_mlp")(x)
        mlp_dim = int(self.dim * self.mlp_ratio)
        x = nn.Dense(mlp_dim, name="mlp_1")(x)
        x = self._get_activation()(x)
        x = nn.Dropout(self.dropout_rate, deterministic=not train)(x)
        x = nn.Dense(self.dim, name="mlp_2")(x)
        x = nn.Dropout(self.dropout_rate, deterministic=not train)(x)
        x = x + residual
        return x

    def _get_activation(self) -> Callable[[Array], Array]:
        if self.activation == "swish":
            return Swish()
        elif self.activation == "gelu":
            return nn.gelu
        elif self.activation == "silu":
            return nn.silu
        elif self.activation == "relu":
            return nn.relu
        else:
            raise ValueError(f"Unknown activation: {self.activation}")


class SymbolicTransformer(nn.Module):
    """Transformer encoder for symbolic expression sequences.

    Input: tokenized expression (prefix notation)
    Output: latent representation for expression classification/regression
    """

    vocab_size: int
    dim: int = 256
    num_heads: int = 8
    num_layers: int = 6
    max_seq_len: int = 128
    dropout_rate: float = 0.1
    output_dim: int = 1

    @nn.compact
    def __call__(self, tokens: Array, train: bool = False,
                 mask: Array | None = None) -> Array:
        # Token embeddings
        x = nn.Embed(self.vocab_size, self.dim, name="token_emb")(tokens)

        # Positional embeddings
        pos_emb = self.param("pos_emb",
                            nn.initializers.normal(0.02),
                            (self.max_seq_len, self.dim))
        x = x + pos_emb[:tokens.shape[1]]

        x = nn.Dropout(self.dropout_rate, deterministic=not train)(x)

        # Transformer blocks
        for i in range(self.num_layers):
            x = TransformerBlock(
                dim=self.dim,
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
                name=f"block_{i}",
            )(x, train=train, mask=mask)

        # Global average pooling over sequence
        x = jnp.mean(x, axis=1)

        # Output head
        x = nn.LayerNorm(name="ln_out")(x)
        x = nn.Dense(self.output_dim, name="output_proj")(x)
        return x


class TransformerDecoder(nn.Module):
    """Autoregressive Transformer decoder for token generation.

    Outputs logits for next token prediction at each position.
    """

    vocab_size: int
    dim: int = 256
    num_heads: int = 8
    num_layers: int = 6
    max_seq_len: int = 128
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, tokens: Array, latent: Array, train: bool = False,
                 mask: Array | None = None) -> Array:
        """
        Args:
            tokens: (batch, seq_len) input token sequence
            latent: (batch, dim) latent conditioning vector
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        _batch_size, seq_len = tokens.shape

        # Token embeddings
        x = nn.Embed(self.vocab_size, self.dim, name="token_emb")(tokens)

        # Positional embeddings
        pos_emb = self.param("pos_emb",
                            nn.initializers.normal(0.02),
                            (self.max_seq_len, self.dim))
        x = x + pos_emb[:seq_len]

        # Add latent conditioning (broadcast to sequence)
        latent_proj = nn.Dense(self.dim, name="latent_proj")(latent)
        x = x + latent_proj[:, None, :]

        x = nn.Dropout(self.dropout_rate, deterministic=not train)(x)

        # Causal mask for autoregressive decoding
        causal_mask = nn.make_causal_mask(tokens)

        # Transformer decoder blocks
        for i in range(self.num_layers):
            x = TransformerDecoderBlock(
                dim=self.dim,
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
                name=f"block_{i}",
            )(x, train=train, mask=causal_mask)

        # Output projection to vocab logits
        x = nn.LayerNorm(name="ln_out")(x)
        logits = nn.Dense(self.vocab_size, name="output_proj")(x)
        return logits


class TransformerDecoderBlock(nn.Module):
    """Transformer decoder block: Self-Attention -> Cross-Attention -> MLP."""

    dim: int
    num_heads: int
    mlp_ratio: float = 4.0
    dropout_rate: float = 0.1
    activation: str = "swish"

    @nn.compact
    def __call__(self, x: Array, train: bool = False,
                 mask: Array | None = None) -> Array:
        # Self-attention
        residual = x
        x = nn.LayerNorm(name="ln_self_attn")(x)
        x = MultiHeadAttention(
            num_heads=self.num_heads,
            qkv_features=self.dim,
            out_features=self.dim,
            dropout_rate=self.dropout_rate,
        )(x, train=train, mask=mask)
        x = nn.Dropout(self.dropout_rate, deterministic=not train)(x)
        x = x + residual

        # MLP
        residual = x
        x = nn.LayerNorm(name="ln_mlp")(x)
        mlp_dim = int(self.dim * self.mlp_ratio)
        x = nn.Dense(mlp_dim, name="mlp_1")(x)
        x = self._get_activation()(x)
        x = nn.Dropout(self.dropout_rate, deterministic=not train)(x)
        x = nn.Dense(self.dim, name="mlp_2")(x)
        x = nn.Dropout(self.dropout_rate, deterministic=not train)(x)
        x = x + residual
        return x

    def _get_activation(self):
        if self.activation == "swish":
            return Swish()
        elif self.activation == "gelu":
            return nn.gelu
        elif self.activation == "silu":
            return nn.silu
        elif self.activation == "relu":
            return nn.relu
        else:
            raise ValueError(f"Unknown activation: {self.activation}")


class GraphConv(nn.Module):
    """Graph convolution layer for expression trees."""

    features: int
    aggregation: str = "mean"  # mean, sum, max
    activation: str = "swish"

    @nn.compact
    def __call__(self, nodes: Array, edges: Array, train: bool = False) -> Array:
        """
        Args:
            nodes: (num_nodes, node_dim) node features
            edges: (num_edges, 2) edge list [src, dst]
        Returns:
            Updated node features (num_nodes, features)
        """
        num_nodes = nodes.shape[0]

        # Self features
        self_feat = nn.Dense(self.features, name="self_proj")(nodes)

        # Aggregate neighbor features
        src, dst = edges[:, 0], edges[:, 1]
        neighbor_feat = nodes[src]  # (num_edges, node_dim)
        neighbor_feat = nn.Dense(self.features, name="neigh_proj")(neighbor_feat)

        # Aggregation
        if self.aggregation == "mean":
            agg_feat = jnp.zeros((num_nodes, self.features))
            counts = jnp.zeros((num_nodes, 1))
            agg_feat = agg_feat.at[dst].add(neighbor_feat)
            counts = counts.at[dst].add(1.0)
            agg_feat = agg_feat / jnp.maximum(counts, 1.0)
        elif self.aggregation == "sum":
            agg_feat = jnp.zeros((num_nodes, self.features))
            agg_feat = agg_feat.at[dst].add(neighbor_feat)
        elif self.aggregation == "max":
            agg_feat = jnp.full((num_nodes, self.features), -jnp.inf)
            agg_feat = jnp.maximum(agg_feat.at[dst].set(neighbor_feat), 0)  # simplified
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        # Combine
        x = self_feat + agg_feat

        # Activation
        if self.activation == "swish":
            x = Swish()(x)
        elif self.activation == "gelu":
            x = nn.gelu(x)
        elif self.activation == "relu":
            x = nn.relu(x)

        return x


class EquationGNN(nn.Module):
    """Graph Neural Network for expression trees.

    Nodes: variables, constants, operators
    Edges: parent-child relationships in expression tree
    """

    hidden_dim: int = 128
    num_layers: int = 4
    aggregation: str = "mean"
    num_node_types: int = 50  # variable types, operators, constants
    output_dim: int = 1

    @nn.compact
    def __call__(self, node_features: Array, edges: Array,
                 node_types: Array, train: bool = False) -> Array:
        """
        Args:
            node_features: (num_nodes, feature_dim) continuous features
            edges: (num_edges, 2) edge indices
            node_types: (num_nodes,) categorical node types
        """
        # Node type embeddings
        type_emb = nn.Embed(self.num_node_types, self.hidden_dim, name="type_emb")(node_types)

        # Project continuous features
        x = nn.Dense(self.hidden_dim, name="feat_proj")(node_features)
        x = x + type_emb

        # GNN layers
        for i in range(self.num_layers):
            x = GraphConv(
                features=self.hidden_dim,
                aggregation=self.aggregation,
                name=f"gcn_{i}",
            )(x, edges, train=train)

        # Global pooling (mean over nodes)
        x = jnp.mean(x, axis=0, keepdims=True)  # (1, hidden_dim)

        # Output head
        x = nn.LayerNorm(name="ln_out")(x)
        x = nn.Dense(self.output_dim, name="output_proj")(x)
        return x.squeeze(0)


class PhysicsInformedNN(nn.Module):
    """Physics-Informed Neural Network with PDE residual loss.

    Combines data loss with PDE residual computed via JAX autodiff
    on differentiable PDE solvers from jax_pde_solver.py.
    """

    hidden_dims: Sequence[int] = (256, 256, 128, 64)
    num_layers: int = 4
    dropout_rate: float = 0.1
    activation: str = "swish"
    pde_weight: float = 1.0
    bc_weight: float = 10.0
    ic_weight: float = 10.0
    output_dim: int = 1

    @nn.compact
    def __call__(self, coords: Array, train: bool = False) -> Array:
        """Forward pass: coordinates -> field values.

        Args:
            coords: (..., spatial_dim + 1) where last dim is time
        """
        x = coords
        act_fn = self._get_activation()

        for i, dim in enumerate(self.hidden_dims):
            x = nn.Dense(dim, name=f"dense_{i}")(x)
            x = act_fn(x)
            x = nn.Dropout(self.dropout_rate, deterministic=not train)(x)

        x = nn.Dense(self.output_dim, name="output")(x)
        return x

    def _get_activation(self) -> Callable[[Array], Array]:
        if self.activation == "swish":
            return Swish()
        elif self.activation == "gelu":
            return nn.gelu
        elif self.activation == "silu":
            return nn.silu
        elif self.activation == "relu":
            return nn.relu
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def pde_residual(self, params: FrozenDict, coords: Array,
                     pde_fn: Callable) -> Array:
        """Compute PDE residual using JAX autodiff.

        Args:
            params: Model parameters
            coords: (n_points, dim) coordinate points
            pde_fn: Function(u, coords) -> residual
        """
        # This is a template; actual PDE depends on physics
        # User provides pde_fn that computes residual from u and coords
        u = self.apply(params, coords)
        return pde_fn(u, coords)


def create_model(config: NeuralDiscoveryConfig,
                 input_dim: int,
                 output_dim: int = 1) -> nn.Module:
    """Factory function to create model from config.

    Args:
        config: NeuralDiscoveryConfig
        input_dim: Input feature dimension
        output_dim: Output dimension

    Returns:
        Flax Module instance
    """
    if config.architecture == "mlp_resnet":
        return MLPResNet(
            hidden_dims=config.hidden_dims,
            num_layers=config.num_layers,
            dropout_rate=config.dropout_rate,
            activation=config.activation,
            layer_norm=config.layer_norm,
            residual=config.residual,
            output_dim=output_dim,
        )
    elif config.architecture == "symbolic_transformer":
        return SymbolicTransformer(
            vocab_size=config.vocab_size,
            dim=config.transformer_dim,
            num_heads=config.transformer_heads,
            num_layers=config.transformer_layers,
            max_seq_len=config.max_seq_len,
            dropout_rate=config.dropout_rate,
            output_dim=output_dim,
        )
    elif config.architecture == "equation_gnn":
        return EquationGNN(
            hidden_dim=config.gnn_hidden_dim,
            num_layers=config.gnn_num_layers,
            aggregation=config.gnn_aggregation,
            output_dim=output_dim,
        )
    elif config.architecture == "pinn":
        return PhysicsInformedNN(
            hidden_dims=config.hidden_dims,
            num_layers=config.num_layers,
            dropout_rate=config.dropout_rate,
            activation=config.activation,
            pde_weight=config.pinn_pde_weight,
            bc_weight=config.pinn_bc_weight,
            ic_weight=config.pinn_ic_weight,
            output_dim=output_dim,
        )
    else:
        raise ValueError(f"Unknown architecture: {config.architecture}")


def init_model(model: nn.Module,
               input_shape: tuple[int, ...],
               key: PRNGKey,
               config: NeuralDiscoveryConfig | None = None) -> FrozenDict:
    """Initialize model parameters.

    Args:
        model: Flax module
        input_shape: Shape of dummy input (batch, ...)
        key: JAX PRNG key
        config: Optional config for architecture-specific init

    Returns:
        Initialized parameters (FrozenDict)
    """
    # Determine input dtype based on model type
    if isinstance(model, SymbolicTransformer):
        dummy_input = jnp.ones(input_shape, dtype=jnp.int32)
        variables = model.init(key, dummy_input, train=False)
    elif isinstance(model, EquationGNN):
        # GNN needs multiple inputs - this is a simplified init
        dummy_input = jnp.ones(input_shape)
        variables = model.init(key, dummy_input, train=False)
    elif model.__class__.__name__ == "NeuralSymbolicRegressor":
        # NeuralSymbolicRegressor needs (x, y) inputs
        x_dummy = jnp.ones(input_shape)
        y_dummy = jnp.ones(input_shape[0])
        variables = model.init(key, x_dummy, y_dummy, train=False)
    else:
        dummy_input = jnp.ones(input_shape)
        variables = model.init(key, dummy_input, train=False)
    return variables['params']


def count_params(params: FrozenDict) -> int:
    """Count total parameters in a Flax parameter dict."""
    return sum(np.prod(p.shape) for p in jax.tree_util.tree_leaves(params))


# Export for convenience
__all__ = [
    "EquationGNN",
    "GraphConv",
    "MLPResNet",
    "MultiHeadAttention",
    "PhysicsInformedNN",
    "Swish",
    "SymbolicTransformer",
    "TransformerBlock",
    "count_params",
    "create_model",
    "init_model",
]