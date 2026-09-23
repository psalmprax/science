# Databricks notebook source
"""APEDP: Autonomous Physics & Engineering Discovery Platform.

An end-to-end scientific discovery, mathematical verification, epistemic tracking,
dimensional assurance, metrology analysis, and autonomous anomaly repair platform.
"""

from apedp.adversarial_benchmarks import (
    AdversarialScienceBenchmarkSuite,
    AdversarialTrapResult,
    AdversarialTrapType,
)
from apedp.applied_defects import (
    AppliedChemistryDefectAuditor,
    AppliedMaterialScienceDefectAuditor,
    AppliedNanotechnologyDefectAuditor,
    AppliedPhysicsDefectAuditor,
    UniversalAppliedDefectEngine,
)
from apedp.benchmarks import (
    BenchmarkCaseResult,
    BenchmarkCategory,
    BenchmarkHarness,
    BenchmarkSummary,
    NoveltyTier,
    PathologicalCase,
    PathologicalEquationBenchmark,
)
from apedp.calibration import posterior_predictive_calibration
from apedp.composer import Conjecture, HypothesisComposer
from apedp.core import (
    PHYSICAL_CONSTANTS,
    BuckinghamPiEngine,
    ClosedLoopAssuranceSystem,
    ConservationInvariantRegistry,
    DimensionalAssuranceEngine,
    FundamentalConstantsRegistry,
    NumberLike,
    PhysicalConstant,
    QuantityLike,
    VectorizedDimensionalEngine,
    ureg,
)
from apedp.decay_models import fit_shifted_power_decay
from apedp.delay import evaluate_delay_ivp, evaluate_delay_system
from apedp.discovery import (
    BayesianRefinementEngine,
    EvolutionaryDiscoveryEngine,
    ModelEvidence,
    MultiVariableSRE,
    detect_composition_family,
    detect_multiplicative_split,
    recombine_power_law,
    recursive_solve,
)
from apedp.epistemic import (
    AdversarialDeliberationTournament,
    ClaimEvidenceProfile,
    DataPartitionVault,
    EpistemicGraph,
    EpistemicModelTier,
    EvidenceMaturityLevel,
    EvidenceVector,
    FalsificationContract,
    LifecycleManager,
    ModelEpistemicStatus,
    ModelEvidenceRecord,
    ModelInterventionClass,
    NoveltyAuditResult,
    NoveltyStatus,
    PriorArtAuditor,
    RefinementOperator,
    RequirementsTraceabilityMatrix,
    ScientificMaturityLevel,
    ScientificProvenanceChain,
    TestableModelPackage,
    UncertaintyDecomposition,
)
from apedp.eval_extended import (
    evaluate_einsum,
    evaluate_implicit,
    evaluate_implicit_system,
    evaluate_indexed,
    evaluate_ode_events,
    evaluate_ode_ivp,
    evaluate_ode_system,
    evaluate_ode_system_events,
    evaluate_pde_residual,
    evaluate_quadrature,
    parse_indexed_to_sympy,
    simulate_sde_euler_maruyama,
    solve_eigen,
    solve_heat_1d_explicit,
    symbolic_contract,
)
from apedp.falsification import (
    UNREPAIRABLE_FAILURE_CLASSES,
    BlindAdversarialAuditor,
    BlindAuditResult,
    ContradictoryConstraintEngine,
    Counterexample,
    FailureClass,
    FalsificationAxis,
    FalsificationReport,
    IndependentAdversarialFalsifier,
    InfeasibilityCertificate,
    unrepairable_counterexamples,
)
from apedp.family_proposal import (
    FamilyCandidate,
    discover_with_proposed_family,
    propose_univariate_families,
    select_univariate_family,
)
from apedp.gated_assurance import (
    AdjudicationReport,
    AdjudicationVerdict,
    BlindChallengeHarness,
    BlindDiagnosticBenchmark,
    DiagnosticReasoningBenchmark,
    EpistemicActionStance,
    FailureSeverity,
    FalsificationBenchResult,
    GatedScientificAssuranceEngine,
    KnownDiagnosticBenchmark,
    MechanisticAdjudicationEngine,
    OperationalGateAuditor,
    OperationalReadinessReport,
    ParetoFrontierTracker,
    ReplicationIndependenceVector,
    ScientificBenchmarkContract,
    ScientificFeasibilityConfusionMatrix,
    ScientificImprovementMetric,
    SpecificationFeasibility,
    create_butler_volmer_benchmark_contract,
    create_casimir_benchmark_contract,
    create_gibbs_thomson_benchmark_contract,
    create_hall_petch_benchmark_contract,
    create_prandtl_glauert_benchmark_contract,
    map_feasibility_to_stance,
)
from apedp.ingestion import (
    AIFeynmanBenchmarkSuite,
    FeynmanBenchmarkCase,
    NACA0012TransonicBenchmark,
    NanocrystallineMicrocompressionBenchmark,
)
from apedp.integral_equations import (
    solve_nystrom,
    solve_nystrom_fredholm,
    solve_nystrom_volterra,
)
from apedp.meshing import (
    delaunay_triangulate,
    delaunay_triangulate_3d,
    refine_mesh_2d,
    smooth_mesh_2d,
    uniform_mesh_2d,
)
from apedp.metrology import (
    AnalyticalUncertaintyPropagator,
    CalibrationTraceabilityChain,
    MetrologySensorStream,
    UncertaintyReport,
)
from apedp.model_discovery import ModelCandidateResult, ModelDiscoveryEngine, ModelDiscoveryResult
from apedp.pde_solvers import (
    solve_advection_1d,
    solve_heat_2d,
    solve_heat_3d,
    solve_heat_nd_feynman_kac,
    solve_laplace_2d,
    solve_laplace_3d,
    solve_laplace_nd_walk_on_spheres,
    solve_poisson_2d,
    solve_poisson_fem,
    solve_wave_1d,
)
from apedp.pde_templates import TEMPLATES as PDE_TEMPLATES
from apedp.pde_templates import recognize_pde
from apedp.persistence import (
    ExperimentLog,
    ModelStore,
    deserialize_model,
    from_json,
    serialize_model,
    to_json,
)

# COMMAND ----------
from apedp.pipeline import DiscoveryPipeline, run_pipeline
from apedp.qec import (
    ConcatenatedThresholdRecursion,
    KnillLaflammeAuditor,
    QECEquationDefectAuditor,
    QECMetrologyPropagator,
    QuantumCodeValidator,
    SurfaceCodeThresholdEngine,
)
from apedp.repair import (
    AgnosticClosedLoopAssuranceSystem,
    PhysicalAnomalyRepairEngine,
    RegularizationAuditor,
)
from apedp.safety import LimitTimeout, limit_with_timeout, safe_parse_expr, safe_parse_pde
from apedp.scenarios import APEDPScenarioSuite
from apedp.symbolic_regression import RegressionConfig, SymbolicRegressor
from apedp.time_series import build_lagged_supervised_dataset
from apedp.turnover_models import TurnoverCandidate, select_log_scale_turnover
from apedp.visualization import (
    plot_calibration_curve,
    plot_dimensional_analysis,
    plot_epistemic_graph,
    plot_regression_fitness,
    plot_scenario_results,
    plot_uncertainty_propagation,
)

__version__ = "1.0.0"
