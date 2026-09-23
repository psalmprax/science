"""Configuration dataclasses for SOTA Neural Discovery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ArchitectureType = Literal["mlp_resnet", "symbolic_transformer", "equation_gnn", "pinn"]
UncertaintyMethod = Literal["ensemble", "bayesian_svi", "conformal", "dropout"]
AcquisitionType = Literal["bald", "expected_improvement", "max_variance", "max_entropy", "diversity"]


@dataclass(frozen=True)
class NeuralDiscoveryConfig:
    """Global configuration for neural discovery.

    All fields are frozen for reproducibility; create modified configs via
    ``replace()`` or ``NeuralDiscoveryConfig(**changes)``.
    """

    # Architecture
    architecture: ArchitectureType = "mlp_resnet"
    hidden_dims: tuple[int, ...] = (256, 256, 128, 64)
    num_layers: int = 4
    dropout_rate: float = 0.1
    activation: str = "swish"  # swish, gelu, relu, silu
    layer_norm: bool = True
    residual: bool = True

    # Symbolic Transformer specific
    transformer_dim: int = 256
    transformer_heads: int = 8
    transformer_layers: int = 6
    max_seq_len: int = 128
    vocab_size: int = 100

    # Equation GNN specific
    gnn_hidden_dim: int = 128
    gnn_num_layers: int = 4
    gnn_aggregation: str = "mean"  # mean, sum, max

    # PINN specific
    pinn_pde_weight: float = 1.0
    pinn_bc_weight: float = 10.0
    pinn_ic_weight: float = 10.0

    # Uncertainty
    uncertainty_method: UncertaintyMethod = "ensemble"
    ensemble_size: int = 5
    svi_samples: int = 100
    svi_kl_weight: float = 1.0
    conformal_alpha: float = 0.1  # 90% coverage
    dropout_mc_samples: int = 30

    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 500
    early_stopping_patience: int = 50
    lr_schedule: str = "cosine"  # cosine, constant, step
    warmup_epochs: int = 10
    gradient_clip: float = 1.0
    mixed_precision: bool = True

    # Cross-validation
    cv_folds: int = 5
    cv_seeds: int = 5
    cv_stratify: bool = False

    # Hyperparameter optimization
    hyperopt_trials: int = 50
    hyperopt_timeout: int = 3600  # seconds
    hyperopt_pruner: str = "median"  # median, hyperband, none

    # Active learning
    acquisition_type: AcquisitionType = "bald"
    acquisition_batch_size: int = 10
    acquisition_diversity_weight: float = 0.3

    # Symbolic regression
    symbolic_max_expr_len: int = 64
    symbolic_complexity_weight: float = 0.01
    symbolic_rl_epochs: int = 100
    symbolic_supervised_epochs: int = 50
    symbolic_temperature: float = 1.0

    # Hardware / reproducibility
    seed: int = 42
    device: str = "cpu"  # cpu, gpu, tpu
    compile: bool = True  # jax.jit

    # Logging
    log_interval: int = 10
    log_dir: str | None = None
    use_wandb: bool = False
    wandb_project: str = "apedp-neural-discovery"

    def __post_init__(self):
        # Validate
        if self.ensemble_size < 2:
            raise ValueError("ensemble_size must be >= 2")
        if not 0 < self.conformal_alpha < 1:
            raise ValueError("conformal_alpha must be in (0, 1)")
        if self.cv_folds < 2:
            raise ValueError("cv_folds must be >= 2")
        if self.max_epochs < 1:
            raise ValueError("max_epochs must be >= 1")

    def with_updates(self, **kwargs) -> NeuralDiscoveryConfig:
        """Return a new config with updated fields."""
        return NeuralDiscoveryConfig(**{**self.__dict__, **kwargs})


@dataclass(frozen=True)
class TrainingConfig:
    """Training-specific configuration (subset of NeuralDiscoveryConfig)."""

    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 500
    early_stopping_patience: int = 50
    lr_schedule: str = "cosine"
    warmup_epochs: int = 10
    gradient_clip: float = 1.0
    mixed_precision: bool = True
    seed: int = 42
    device: str = "cpu"
    compile: bool = True
    log_interval: int = 10
    log_dir: str | None = None

    @classmethod
    def from_discovery_config(cls, config: NeuralDiscoveryConfig) -> TrainingConfig:
        return cls(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            batch_size=config.batch_size,
            max_epochs=config.max_epochs,
            early_stopping_patience=config.early_stopping_patience,
            lr_schedule=config.lr_schedule,
            warmup_epochs=config.warmup_epochs,
            gradient_clip=config.gradient_clip,
            mixed_precision=config.mixed_precision,
            seed=config.seed,
            device=config.device,
            compile=config.compile,
            log_interval=config.log_interval,
            log_dir=config.log_dir,
        )


def get_default_config() -> NeuralDiscoveryConfig:
    """Get the default neural discovery configuration."""
    return NeuralDiscoveryConfig()


# Preset configurations for common use cases
PRESET_CONFIGS = {
    "fast": NeuralDiscoveryConfig(
        architecture="mlp_resnet",
        hidden_dims=(128, 64),
        num_layers=2,
        ensemble_size=3,
        max_epochs=100,
        cv_folds=3,
        cv_seeds=1,
    ),
    "accurate": NeuralDiscoveryConfig(
        architecture="symbolic_transformer",
        hidden_dims=(512, 512, 256, 128),
        num_layers=4,
        ensemble_size=10,
        max_epochs=1000,
        cv_folds=5,
        cv_seeds=5,
        hyperopt_trials=100,
    ),
    "symbolic": NeuralDiscoveryConfig(
        architecture="symbolic_transformer",
        transformer_dim=512,
        transformer_heads=8,
        transformer_layers=8,
        max_epochs=500,
        symbolic_rl_epochs=200,
        symbolic_supervised_epochs=100,
    ),
    "physics": NeuralDiscoveryConfig(
        architecture="pinn",
        pinn_pde_weight=1.0,
        pinn_bc_weight=10.0,
        pinn_ic_weight=10.0,
        max_epochs=1000,
    ),
    "production": NeuralDiscoveryConfig(
        architecture="mlp_resnet",
        hidden_dims=(256, 256, 128, 64),
        ensemble_size=5,
        max_epochs=500,
        cv_folds=5,
        cv_seeds=5,
        hyperopt_trials=50,
        uncertainty_method="ensemble",
    ),
}


def get_preset(name: str) -> NeuralDiscoveryConfig:
    """Get a preset configuration by name."""
    if name not in PRESET_CONFIGS:
        raise ValueError(f"Unknown preset: {name}. Available: {list(PRESET_CONFIGS.keys())}")
    return PRESET_CONFIGS[name]