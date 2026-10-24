"""SOTA Neural Discovery for APEDP.

JAX-based neural architectures with uncertainty quantification,
differentiable symbolic regression, and active learning.
"""

from __future__ import annotations

# Active learning
from apedp.neural_discovery.active_learning import (
    BALD,
    AcquisitionFunction,
    ActiveLearningLoop,
    DiversityAcquisition,
    ExpectedImprovement,
    MaxEntropy,
    MaxVariance,
)

# Architectures
from apedp.neural_discovery.architectures import (
    EquationGNN,
    MLPResNet,
    PhysicsInformedNN,
    SymbolicTransformer,
    create_model,
)

# Core configuration
from apedp.neural_discovery.config import (
    AcquisitionType,
    ArchitectureType,
    NeuralDiscoveryConfig,
    UncertaintyMethod,
    get_default_config,
)

# Integration bridge
from apedp.neural_discovery.integration import (
    ModelDiscoveryIntegration,
    NeuralDiscoveryEngine,
    distill_via_surrogate,
    neural_relevance,
)

# Differentiable symbolic regression
from apedp.neural_discovery.symbolic import (
    ExpressionTokenizer,
    JAXSymbolicExecutor,
    NeuralSymbolicRegressor,
    discover_symbolic,
)

# Training & hyperopt
from apedp.neural_discovery.training import (
    HyperparameterOptimizer,
    ModelCheckpoint,
    NeuralDiscoveryTrainer,
    TrainingConfig,
)

# Uncertainty quantification
from apedp.neural_discovery.uncertainty import (
    BayesianNN,
    CalibrationMetrics,
    ConformalPredictor,
    DeepEnsemble,
    UncertaintyWrapper,
    predict_with_uncertainty,
)

# Validation & benchmarks
from apedp.neural_discovery.validation import (
    CalibrationSuite,
    ComparisonBaseline,
    OODBenchmarkSuite,
    SRBenchEvaluator,
    run_full_benchmark,
)

__all__ = [
    "BALD",
    # Active learning
    "AcquisitionFunction",
    "AcquisitionType",
    "ActiveLearningLoop",
    "ArchitectureType",
    # Uncertainty
    "BayesianNN",
    "CalibrationMetrics",
    "CalibrationSuite",
    "ComparisonBaseline",
    "ConformalPredictor",
    "DeepEnsemble",
    "DiversityAcquisition",
    "EquationGNN",
    "ExpectedImprovement",
    "ExpressionTokenizer",
    "HyperparameterOptimizer",
    "JAXSymbolicExecutor",
    # Architectures
    "MLPResNet",
    "MaxEntropy",
    "MaxVariance",
    "ModelCheckpoint",
    "ModelDiscoveryIntegration",
    # Config
    "NeuralDiscoveryConfig",
    # Integration
    "NeuralDiscoveryEngine",
    # Training
    "NeuralDiscoveryTrainer",
    # Symbolic
    "NeuralSymbolicRegressor",
    # Validation
    "OODBenchmarkSuite",
    "PhysicsInformedNN",
    "SRBenchEvaluator",
    "SymbolicTransformer",
    "TrainingConfig",
    "UncertaintyMethod",
    "UncertaintyWrapper",
    "create_model",
    "discover_symbolic",
    "distill_via_surrogate",
    "get_default_config",
    "neural_relevance",
    "predict_with_uncertainty",
    "run_full_benchmark",
]

__version__ = "1.0.0"