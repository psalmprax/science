"""APEDP-MMLM: Multi-Mathematical Language Model Research Package & Quantum Circuit Discovery.

Status: experimental

Provides the three-track Multi-Mathematical Language Model research architecture
and the Quantum-Reach MMLM Circuit Discovery subsystem:
  - Track A (`NonNeuralMMLM_A0`): Pure non-neural mathematical language modeling
    using weighted finite-state automata, trainable 1D Tensor Trains / Tree Tensor
    Networks, explicit dynamical state transitions, Koopman spectral projection,
    hypergraph / VSA associative memory, symbolic rewriting, and energy-guided decoding.
  - Track B (`HybridMMLM_B0`): Ablation-governed mathematical hybrid introducing
    optional continuous semantic encoders and syntax-aware Tree Tensor Network
    hierarchy only where quantitative ablations prove net benefit.
  - Track C (`AttentionFreeMMLM_C0`): Attention-free Transformer replacement
    combining selective state-space dynamics, hierarchical tensor memory, and
    speculative energy rescoring.
  - Quantum Discovery (`apedp_mmlm.quantum_discovery`): Local Complementation +
    symplectic CX graph-state synthesis, general stabilizer tableau synthesis,
    commutation-aware Clifford rewriting, DAG front-layer lookahead routing,
    fault-tolerant `invsqrt` / `dqi` / `dfthcsos` proposers, and sparse cubic HUBO
    + Koopman QAOA equation discovery.
"""
from apedp_mmlm.engines import (
    BayesianLanguageEngine,
    DynamicalContextEngine,
    EnergyBreakdown,
    EnergyCandidateEvaluator,
    FrozenHypothesis,
    HypergraphKnowledgeEngine,
    KoopmanOperatorEngine,
    MMLMAdjudicationReceipt,
    MMLMEpistemicAdjudicator,
    MMLMResearchSpecification,
    MMLMTokenizer,
    ProbabilisticCircuitEngine,
    SpeculativeEnergyDecoder,
    SymbolicRewriteEngine,
    TensorTrainMemoryEngine,
    TreeTensorNetworkMemoryEngine,
    VectorSymbolicEngine,
    WeightedAutomatonEngine,
    default_frozen_hypotheses,
)
from apedp_mmlm.models import (
    AblationComparisonReport,
    AttentionFreeMMLM_C0,
    BenchmarkMetrics,
    HybridAblationConfig,
    HybridMMLM_B0,
    Milestone1BenchmarkSuite,
    Milestone1SummaryReport,
    NonNeuralMMLM_A0,
    RecurrentBaseline,
    SSMBaseline,
    TransformerBaseline,
)
from apedp_mmlm.quantum_discovery import (
    DfthcsosBankingProposal,
    DQIScheduleProposal,
    HigherOrderSurrogate,
    InvSqrtPrecisionProposal,
    MMLMDfthcsosOptimizer,
    MMLMDQIScheduleOptimizer,
    MMLMInvSqrtProposer,
    MMLMQEDProposer,
    build_sparse_hubo_surrogate,
    mmlm_clifford_search,
    mmlm_resource_score,
    place_with_dag_lookahead,
    propose_mmlm_clifford_rewrites,
    synthesize_any_target_mmlm,
    synthesize_graph_state_mmlm,
    synthesize_stabilizer_target_mmlm,
)

__all__ = [
    "AblationComparisonReport",
    "AttentionFreeMMLM_C0",
    "BayesianLanguageEngine",
    "BenchmarkMetrics",
    "DQIScheduleProposal",
    "DfthcsosBankingProposal",
    "DynamicalContextEngine",
    "EnergyBreakdown",
    "EnergyCandidateEvaluator",
    "FrozenHypothesis",
    "HigherOrderSurrogate",
    "HybridAblationConfig",
    "HybridMMLM_B0",
    "HypergraphKnowledgeEngine",
    "InvSqrtPrecisionProposal",
    "KoopmanOperatorEngine",
    "MMLMAdjudicationReceipt",
    "MMLMDQIScheduleOptimizer",
    "MMLMDfthcsosOptimizer",
    "MMLMEpistemicAdjudicator",
    "MMLMInvSqrtProposer",
    "MMLMQEDProposer",
    "MMLMResearchSpecification",
    "MMLMTokenizer",
    "Milestone1BenchmarkSuite",
    "Milestone1SummaryReport",
    "NonNeuralMMLM_A0",
    "ProbabilisticCircuitEngine",
    "RecurrentBaseline",
    "SSMBaseline",
    "SpeculativeEnergyDecoder",
    "SymbolicRewriteEngine",
    "TensorTrainMemoryEngine",
    "TransformerBaseline",
    "TreeTensorNetworkMemoryEngine",
    "VectorSymbolicEngine",
    "WeightedAutomatonEngine",
    "build_sparse_hubo_surrogate",
    "default_frozen_hypotheses",
    "mmlm_clifford_search",
    "mmlm_resource_score",
    "place_with_dag_lookahead",
    "propose_mmlm_clifford_rewrites",
    "synthesize_any_target_mmlm",
    "synthesize_graph_state_mmlm",
    "synthesize_stabilizer_target_mmlm",
]
