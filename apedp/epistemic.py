import hashlib
import hmac
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, ClassVar

import networkx as nx
import numpy as np
import sympy as sp

logger = logging.getLogger(__name__)


class ModelInterventionClass(str, Enum):
    """Classification of the operational intervention that produced the model."""
    MATHEMATICAL_REGULARIZATION = "MATHEMATICAL_REGULARIZATION"  # Removals of coordinate/pole singularities without asserting new physics
    NUMERICAL_REGULARIZATION = "NUMERICAL_REGULARIZATION"        # Conditioning, overflow/underflow, catastrophic cancellation, stiffness
    PHENOMENOLOGICAL_EXTENSION = "PHENOMENOLOGICAL_EXTENSION"    # Bounded functional continuation to span beyond an invalid regime
    MECHANISTIC_REPLACEMENT = "MECHANISTIC_REPLACEMENT"          # Replacement of invalid approximation with physically motivated mechanism


class EvidenceMaturityLevel(str, Enum):
    """Progressive levels of empirical and analytical confirmation."""
    THEORETICAL_SPECIFICATION = "THEORETICAL_SPECIFICATION"  # Candidate formulation prior to dedicated testing
    ANALYTICALLY_VERIFIED = "ANALYTICALLY_VERIFIED"          # Dimensional homogeneity, invariants, and symbolic limits verified
    NUMERICALLY_STRESS_TESTED = "NUMERICALLY_STRESS_TESTED"  # Boundary grids, stability, and non-divergence verified
    FALSIFICATION_SURVIVED = "FALSIFICATION_SURVIVED"        # Survived active adversarial counterexample probes
    EMPIRICALLY_SUPPORTED = "EMPIRICALLY_SUPPORTED"          # Supported by out-of-sample experimental data with uncertainty bounds
    INDEPENDENTLY_CONFIRMED = "INDEPENDENTLY_CONFIRMED"      # Confirmed via independent experimental replication & preregistered predictions


class RefinementOperator(str, Enum):
    """Formal operators for transforming models in the refinement loop: M_{t+1} = R(M_t, E_t, A_t, C_t)."""
    REGULARIZE = "R_regularize"          # Smooth singularities/poles
    MECHANISTIC = "R_mechanistic"        # Shift to alternative physical mechanism family
    SELECT = "R_select"                  # Bayesian/Occam tournament selection among competing hypotheses
    GROUND = "R_ground"                  # Map floating mathematical parameters to physical dimensional scales
    SIMPLIFY = "R_simplify"              # Prune redundant terms or complexity bloat
    EXPAND = "R_expand"                  # Add auxiliary degrees of freedom or transitional envelopes
    FALSIFY = "R_falsify"                # Adversarial boundary stress-testing to expose failure modes


# Backward-compatible alias
EpistemicModelTier = ModelInterventionClass


@dataclass
class ClaimEvidenceProfile:
    """Rigorous epistemic profile documenting what a candidate model guarantees, assumes, regularizes, and leaves unproven."""
    model_name: str
    mathematical_form: str
    epistemic_tier: ModelInterventionClass
    mathematical_guarantees: list[str] = field(default_factory=list)
    underlying_assumptions: list[str] = field(default_factory=list)
    regularized_defects: list[str] = field(default_factory=list)
    unproven_physical_claims: list[str] = field(default_factory=list)
    required_empirical_falsification: str = ""
    falsification_target: str = ""
    failure_interpretation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "mathematical_form": self.mathematical_form,
            "epistemic_tier": self.epistemic_tier.value,
            "mathematical_guarantees": self.mathematical_guarantees,
            "underlying_assumptions": self.underlying_assumptions,
            "regularized_defects": self.regularized_defects,
            "unproven_physical_claims": self.unproven_physical_claims,
            "required_empirical_falsification": self.required_empirical_falsification,
            "falsification_target": self.falsification_target,
            "failure_interpretation": self.failure_interpretation,
        }

@dataclass(frozen=True)
class EvidenceVector:
    """Multidimensional evidence vector decomposing analytical, numerical, adversarial, and empirical channels."""
    analytical_score: float = 0.0          # [0.0, 1.0] formal symmetries, limits, dimensional homogeneity
    numerical_score: float = 0.0           # [0.0, 1.0] grid stability, conditioning, non-divergence
    adversarial_score: float = 0.0         # [0.0, 1.0] survived red-team falsification probes
    empirical_score: float = 0.0           # [0.0, 1.0] out-of-sample data fidelity and goodness-of-fit
    independent_replication_score: float = 0.0 # [0.0, 1.0] replicated by external preregistered tests
    derived_maturity_level: EvidenceMaturityLevel = EvidenceMaturityLevel.THEORETICAL_SPECIFICATION

    def to_dict(self) -> dict[str, Any]:
        return {
            "analytical_score": self.analytical_score,
            "numerical_score": self.numerical_score,
            "adversarial_score": self.adversarial_score,
            "empirical_score": self.empirical_score,
            "independent_replication_score": self.independent_replication_score,
            "derived_maturity_level": self.derived_maturity_level.value,
        }


@dataclass(frozen=True)
class UncertaintyDecomposition:
    """Rigorous uncertainty budget distinguishing measurement, parameter, model, and structural sources."""
    measurement_uncertainty: float = 0.0    # GUM ISO 98-3 aleatoric sensor uncertainty
    parameter_uncertainty: float = 0.0      # Posterior parameter variance
    model_discrepancy: float = 0.0          # Systematic bias between model and true process
    structural_uncertainty: float = 0.0     # Uncertainty from chosen functional form
    extrapolation_uncertainty: float = 0.0  # Variance inflation outside validated domain
    epistemic_total: float = 0.0            # Combined reducible uncertainty
    aleatoric_total: float = 0.0            # Irreducible stochastic noise
    combined_expanded_uncertainty: float = 0.0 # Total combined uncertainty (k=2, 95% confidence)

    def to_dict(self) -> dict[str, float]:
        return {
            "measurement_uncertainty": self.measurement_uncertainty,
            "parameter_uncertainty": self.parameter_uncertainty,
            "model_discrepancy": self.model_discrepancy,
            "structural_uncertainty": self.structural_uncertainty,
            "extrapolation_uncertainty": self.extrapolation_uncertainty,
            "epistemic_total": self.epistemic_total,
            "aleatoric_total": self.aleatoric_total,
            "combined_expanded_uncertainty": self.combined_expanded_uncertainty,
        }


@dataclass(frozen=True)
class FalsificationContract:
    """Quantitative computational contract specifying testable prediction intervals and decision thresholds."""
    falsification_id: str
    target_observable: str
    expected_interval: tuple[float, float]
    domain_interval: tuple[float, float]
    decision_threshold: float
    null_hypothesis: str
    alternative_hypothesis: str
    required_sample_size: int
    statistical_test: str
    preregistered: bool
    failure_interpretation: str
    target_unit: str = "dimensionless"

    def to_dict(self) -> dict[str, Any]:
        return {
            "falsification_id": self.falsification_id,
            "target_observable": self.target_observable,
            "expected_interval": self.expected_interval,
            "domain_interval": self.domain_interval,
            "decision_threshold": self.decision_threshold,
            "null_hypothesis": self.null_hypothesis,
            "alternative_hypothesis": self.alternative_hypothesis,
            "required_sample_size": self.required_sample_size,
            "statistical_test": self.statistical_test,
            "preregistered": self.preregistered,
            "failure_interpretation": self.failure_interpretation,
            "target_unit": self.target_unit,
        }


@dataclass(frozen=True)
class ScientificProvenanceChain:
    """Cryptographically verifiable execution provenance for institutional repeatability."""
    code_commit: str = "HEAD"
    dataset_hashes: tuple[str, ...] = ()
    environment_hash: str = ""
    random_seed: int | None = 42
    execution_id: str = ""
    timestamp: str = ""
    agent_id: str = "APEDP_SDI_ENGINE"
    parent_record_hash: str | None = None
    record_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "code_commit": self.code_commit,
            "dataset_hashes": list(self.dataset_hashes),
            "environment_hash": self.environment_hash,
            "random_seed": self.random_seed,
            "execution_id": self.execution_id,
            "timestamp": self.timestamp,
            "agent_id": self.agent_id,
            "parent_record_hash": self.parent_record_hash,
            "record_hash": self.record_hash,
        }


@dataclass(frozen=True)
class DataPartitionVault:
    """Enforces strict anti-self-confirmation boundaries between development, selection, and falsification data."""
    development_data_hash: str
    selection_data_hash: str
    falsification_data_hash: str
    locked_external_test_hash: str
    is_falsification_locked: bool = True

    def verify_isolation(self, dataset_hash: str, phase: str) -> bool:
        """Ensures that falsification and external test data are never accessed during development or selection."""
        if phase in ("development", "selection") and dataset_hash in (self.falsification_data_hash, self.locked_external_test_hash):
            raise PermissionError(f"ANTI-SELF-CONFIRMATION VIOLATION: {phase} cannot access locked falsification data ({dataset_hash[:8]}).")
        return True


class NoveltyStatus(str, Enum):
    """Rigorous epistemic novelty classification relative to searched reference corpus."""
    KNOWN_MODEL = "KNOWN_MODEL"                              # Exact symbolic or linear/scale isomorphism to cataloged literature model
    RELATED_STRUCTURE = "RELATED_STRUCTURE"                  # Parametric variation or common structural family (e.g. rational saturation)
    APPARENTLY_NOVEL_WITHIN_CORPUS = "APPARENTLY_NOVEL_WITHIN_CORPUS"  # No algebraic isomorphism or family match in corpus


@dataclass(frozen=True)
class NoveltyAuditResult:
    """Outcome of novelty & prior-art audit with explicit epistemic disclaimers."""
    status: NoveltyStatus
    closest_reference_id: str | None
    reference_formula: str | None
    isomorphism_type: str | None
    searched_corpus_size: int
    epistemic_disclaimer: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "closest_reference_id": self.closest_reference_id,
            "reference_formula": self.reference_formula,
            "isomorphism_type": self.isomorphism_type,
            "searched_corpus_size": self.searched_corpus_size,
            "epistemic_disclaimer": self.epistemic_disclaimer,
        }


class PriorArtAuditor:
    """Audits candidate expressions against prior art databases, structural families, and symbolic isomorphisms."""

    CORPUS_VERSION = "2026.09-v2"

    STANDARD_PHYSICS_CORPUS: ClassVar[dict[str, str]] = {
        # Single-variable structural families (legacy, kept for compat).
        "LANGMUIR_MICHAELIS_MENTEN": "a * x / (1 + b * x)",
        "HILL_EQUATION": "a * x**2 / (1 + b * x**2)",
        "ARRHENIUS_BOLTZMANN": "a * exp(-b / x)",
        "POWER_LAW_FREUNDLICH": "a * x**b",
        "EXPONENTIAL_DECAY": "a * exp(-b * x)",
        "LOGISTIC_SIGMOID": "a / (1 + exp(-b * (x - c)))",
        "LINEAR_OHMIC_HOOKE": "a * x",
        "HALL_PETCH": "a + b / sqrt(x)",
        "NEWTON_GRAVITATION_COULOMB": "a / x**2",
        "CASIMIR_FORCE": "a / x**4",
        # Multi-variable textbook laws (rename-insensitive match covers
        # I*R vs V=I*R, m*v**2/2 vs kinetic energy, etc.).
        "OHMIC_PRODUCT": "x * y",
        "KINETIC_ENERGY": "m * v**2 / 2",
        "EINSTEIN_MASS_ENERGY": "m * c**2",
        "NEWTON_GRAVITY": "G * m1 * m2 / r**2",
        "COULOMB_LAW": "q1 * q2 / (4 * pi * eps * r**2)",
        "STEFAN_BOLTZMANN": "sigma * T**4",
        "IDEAL_GAS": "P * V / (R * T)",
        "LORENTZ_GAMMA": "1 / sqrt(1 - (v / c)**2)",
        "PENDULUM_PERIOD": "2 * pi * sqrt(L / g)",
        "WAVE_SPEED": "f * lam",
        "FIRST_ORDER_DECAY": "N0 * exp(-k * t)",
        "BEER_LAMBERT": "eps * l * c",
        "KEPLER_THIRD": "a**3 / T**2",
        "BERNOULLI": "P + rho * v**2 / 2",
        "CARNOT": "1 - Tc / Th",
        "PLANCK_RADIATION": "a * x**3 / (exp(b * x) - 1)",
    }

    @staticmethod
    def corpus_hash(corpus: dict[str, str] | None = None) -> str:
        """SHA-256 over the sorted corpus, so audits cite what was searched."""
        import hashlib
        import json

        body = json.dumps(
            {"version": PriorArtAuditor.CORPUS_VERSION, "corpus": dict(sorted((corpus or PriorArtAuditor.STANDARD_PHYSICS_CORPUS).items()))},
            sort_keys=True,
        )
        return hashlib.sha256(body.encode()).hexdigest()[:16]

    @staticmethod
    def _canonical_rename(expr: sp.Expr) -> sp.Expr:
        """Map free symbols to positional _v0.._vn so I*R matches x*y."""
        syms = sorted(expr.free_symbols, key=str)
        mapping = {s: sp.Symbol(f"_v{i}") for i, s in enumerate(syms)}
        return expr.xreplace(mapping)

    @staticmethod
    def _parse_physics(expr_str: str) -> sp.Expr:
        """Parse with every identifier forced to a Symbol (physics `I` is current, not sqrt(-1))."""
        import re

        from apedp.safety import _SAFE_FUNCS, safe_parse_expr

        idents = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expr_str))
        extra = {
            ident: sp.Symbol(ident)
            for ident in idents
            if ident not in _SAFE_FUNCS and ident not in ("pi", "E", "EulerGamma")
        }
        return safe_parse_expr(expr_str, extra_symbols=extra)

    @classmethod
    def check_rename_isomorphism(cls, expr1: str | sp.Expr, expr2: str | sp.Expr) -> tuple[bool, str]:
        """Structural match under variable renaming (plus scalar multiple)."""
        try:
            s1 = cls._parse_physics(expr1) if isinstance(expr1, str) else expr1
            s2 = cls._parse_physics(expr2) if isinstance(expr2, str) else expr2
            if len(s1.free_symbols) != len(s2.free_symbols):
                return False, "ARITY_MISMATCH"
            c1, c2 = cls._canonical_rename(s1), cls._canonical_rename(s2)
            return cls.check_symbolic_equivalence(c1, c2)
        except (sp.SympifyError, TypeError, ValueError, ZeroDivisionError, AttributeError, ArithmeticError) as e:
            return False, f"COMPARISON_ERROR: {e}"

    @staticmethod
    def pi_signature(expr: sp.Expr) -> tuple[tuple[str, ...], int, tuple[str, ...]]:
        """Coarse structural signature: (sorted func names, total poly degree, sorted symbols)."""
        funcs = sorted({type(a).__name__ for a in expr.atoms(sp.Function)})
        try:
            poly = sp.Poly(expr.as_numer_denom()[0] * expr.as_numer_denom()[1], *expr.free_symbols)
            degree = int(poly.total_degree()) if poly is not None else -1
        except (ValueError, TypeError, ArithmeticError, NotImplementedError, sp.PolynomialError):
            degree = -1
        return (tuple(funcs), degree, tuple(sorted(str(s) for s in expr.free_symbols)))

    @staticmethod
    def check_symbolic_equivalence(expr1: str | sp.Expr, expr2: str | sp.Expr) -> tuple[bool, str]:
        """Checks if two expressions are mathematically identical despite cosmetic differences."""
        try:
            from apedp.safety import safe_parse_expr

            s1 = safe_parse_expr(expr1) if isinstance(expr1, str) else expr1
            s2 = safe_parse_expr(expr2) if isinstance(expr2, str) else expr2
            diff = sp.simplify(s1 - s2)
            if diff == 0:
                return True, f"IDENTICAL_PRIOR_ART: Expression is isomorphic to reference model (difference: {diff})."
            ratio = sp.simplify(s1 / s2)
            if ratio.is_number and ratio != 0:
                return True, f"SCALED_PRIOR_ART: Expression is scalar-multiple isomorphic (factor {ratio})."
            return False, "DISTINCT: Expression represents an algebraically distinct functional form."
        except (sp.SympifyError, TypeError, ValueError, ZeroDivisionError, AttributeError, ArithmeticError) as e:
            return False, f"COMPARISON_ERROR: {e}"

    @classmethod
    def audit_novelty(
        cls,
        candidate_expr: str | sp.Expr,
        variable_name: str = "x",
        custom_corpus: dict[str, str] | None = None,
    ) -> NoveltyAuditResult:
        """Audits candidate against reference corpus, preventing false claims of novelty.
        
        Classifies strictly into KNOWN_MODEL, RELATED_STRUCTURE, or APPARENTLY_NOVEL_WITHIN_CORPUS.
        Always outputs an epistemic disclaimer that novelty is relative to the searched corpus.
        """
        corpus = dict(cls.STANDARD_PHYSICS_CORPUS)
        if custom_corpus:
            corpus.update(custom_corpus)

        x_sym = sp.Symbol(variable_name)
        from apedp.safety import safe_parse_expr

        if isinstance(candidate_expr, str):
            try:
                s_cand = cls._parse_physics(candidate_expr)
            except (TypeError, ValueError):
                s_cand = safe_parse_expr(candidate_expr, extra_symbols={variable_name: x_sym})
        else:
            s_cand = candidate_expr

        # 1. Direct symbolic equivalence against corpus models
        for ref_id, ref_str in corpus.items():
            equiv, _ = cls.check_symbolic_equivalence(s_cand, ref_str)
            if equiv:
                return NoveltyAuditResult(
                    status=NoveltyStatus.KNOWN_MODEL,
                    closest_reference_id=ref_id,
                    reference_formula=ref_str,
                    isomorphism_type="EXACT_OR_SCALED_ISOMORPHISM",
                    searched_corpus_size=len(corpus),
                    epistemic_disclaimer=(
                        f"Model is algebraically isomorphic to known reference '{ref_id}' ({ref_str}). "
                        "Categorized as re-discovered prior art; novelty claim is refused."
                    ),
                )

        # 1b. Rename-insensitive structural match (I*R vs x*y, m*v**2/2 vs kinetic).
        for ref_id, ref_str in corpus.items():
            try:
                equiv, _ = cls.check_rename_isomorphism(s_cand, ref_str)
            except (sp.SympifyError, TypeError, ValueError, ZeroDivisionError, AttributeError, ArithmeticError):
                continue
            if equiv:
                return NoveltyAuditResult(
                    status=NoveltyStatus.KNOWN_MODEL,
                    closest_reference_id=ref_id,
                    reference_formula=ref_str,
                    isomorphism_type="RENAME_ISOMORPHISM",
                    searched_corpus_size=len(corpus),
                    epistemic_disclaimer=(
                        f"Model matches reference '{ref_id}' ({ref_str}) up to variable renaming/scale. "
                        "Categorized as re-discovered prior art; novelty claim is refused."
                    ),
                )

        # 2. Rational functional family analysis (Langmuir / Michaelis-Menten: P(x)/Q(x))
        try:
            num, den = s_cand.as_numer_denom()
            if num.has(x_sym) or den.has(x_sym):
                p_num = sp.Poly(num, x_sym)
                p_den = sp.Poly(den, x_sym)
                deg_num = p_num.degree()
                deg_den = p_den.degree()

                # Case: deg_num == 1 and deg_den == 1: Langmuir / Michaelis-Menten family: c1*x / (c2 + c3*x)
                if deg_num == 1 and deg_den == 1:
                    is_zero_intercept = abs(float(num.subs(x_sym, 0))) < 1e-9
                    if is_zero_intercept:
                        return NoveltyAuditResult(
                            status=NoveltyStatus.KNOWN_MODEL,
                            closest_reference_id="LANGMUIR_MICHAELIS_MENTEN",
                            reference_formula="a * x / (1 + b * x)",
                            isomorphism_type="RATIONAL_SATURATION_ISOMORPHISM",
                            searched_corpus_size=len(corpus),
                            epistemic_disclaimer=(
                                "Model has functional form P1(x)/Q1(x) with P1(0)=0, algebraically isomorphic "
                                "to the classical Langmuir adsorption isotherm / Michaelis-Menten kinetic law. "
                                "Categorized as re-discovered prior art; novelty claim is refused."
                            ),
                        )
                    else:
                        return NoveltyAuditResult(
                            status=NoveltyStatus.RELATED_STRUCTURE,
                            closest_reference_id="LANGMUIR_MICHAELIS_MENTEN",
                            reference_formula="a * x / (1 + b * x)",
                            isomorphism_type="OFFSET_RATIONAL_SATURATION_FAMILY",
                            searched_corpus_size=len(corpus),
                            epistemic_disclaimer=(
                                "Model is a non-zero-intercept variation of the rational saturation family P1(x)/Q1(x). "
                                "Categorized as related structural variation rather than a novel physical law."
                            ),
                        )

                # Case: deg_num == n and deg_den == n with n >= 2: Hill cooperative binding family
                if deg_num == deg_den and deg_num >= 2 and abs(float(num.subs(x_sym, 0))) < 1e-9:
                    return NoveltyAuditResult(
                        status=NoveltyStatus.RELATED_STRUCTURE,
                        closest_reference_id="HILL_EQUATION",
                        reference_formula=f"a * x**{deg_num} / (1 + b * x**{deg_num})",
                        isomorphism_type="HIGHER_ORDER_COOPERATIVE_RATIONAL",
                        searched_corpus_size=len(corpus),
                        epistemic_disclaimer=(
                            f"Model is of degree ({deg_num},{deg_den}), belonging to the generalized Hill cooperative "
                            "saturation family. Categorized as related structural variation."
                        ),
                    )
        except (ArithmeticError, AttributeError, TypeError, ValueError, NotImplementedError, sp.PolynomialError):
            # The structural / isomorphism matcher is best-effort: a candidate whose
            # rational structure SymPy cannot characterize falls through to the
            # corpus-relative verdict. Log it, because an "apparently novel" result
            # must never be produced silently by a defect in the matcher itself.
            logger.warning(
                "Structural novelty matching failed for %s; falling back to corpus-relative novelty",
                s_cand,
                exc_info=True,
            )

        # 3. If no isomorphism or structural archetype match is found:
        return NoveltyAuditResult(
            status=NoveltyStatus.APPARENTLY_NOVEL_WITHIN_CORPUS,
            closest_reference_id=None,
            reference_formula=None,
            isomorphism_type=None,
            searched_corpus_size=len(corpus),
            epistemic_disclaimer=(
                f"Candidate expression is apparently novel within the searched reference corpus ({len(corpus)} cataloged models, "
                f"version {cls.CORPUS_VERSION} hash {cls.corpus_hash(corpus)}). "
                "NOTICE: This designation certifies novelty only relative to the checked corpus and does NOT constitute "
                "proof of absolute novelty or discovery of a new physical law."
            ),
        )


@dataclass(frozen=True)
class ModelEvidenceRecord:
    """Persistent, immutable scientific model evidence record (Assumption Ledger).
    
    Tracks the full lineage, mathematical guarantees, assumptions, failure modes,
    and assurance evidence partitions for an evolving scientific model.
    """
    model_id: str
    model_version: str
    parent_model_id: str | None
    model_lineage: tuple[str, ...]

    # Classification & Mechanism
    intervention_class: ModelInterventionClass
    evidence_maturity: EvidenceMaturityLevel
    domain: str
    mechanism_family: str  # e.g. "continuum", "kinetic", "dislocation_sliding", "steric_repulsion"

    # Mathematical Representation
    equation_str: str
    variables: tuple[str, ...]
    parameters: dict[str, Any]
    derivation_basis: str

    # Guarantees & Invariants
    mathematical_guarantees: tuple[str, ...]
    numerical_guarantees: tuple[str, ...]
    physical_constraints: tuple[str, ...]

    # Assumptions & Boundaries
    assumptions: tuple[str, ...]
    known_failure_modes: tuple[str, ...]
    validated_domain: dict[str, tuple[float, float]]
    extrapolation_limits: dict[str, tuple[float, float]]

    # Falsification Protocol
    falsification_target: str
    falsification_tests: tuple[str, ...]
    failure_interpretation: str

    # Multi-Valued Intervention & Nuanced Evidence
    secondary_intervention_classes: tuple[ModelInterventionClass, ...] = ()
    evidence_vector: EvidenceVector | None = None
    uncertainty_decomposition: UncertaintyDecomposition | None = None
    falsification_contract: FalsificationContract | None = None
    provenance_chain: ScientificProvenanceChain | None = None

    # Assurance Evidence Partitioning
    software_assurance: dict[str, Any] = field(default_factory=dict)
    mathematical_assurance: dict[str, Any] = field(default_factory=dict)
    scientific_assurance: dict[str, Any] = field(default_factory=dict)

    # Asymptotics, Uncertainty, and Epistemic Status
    symbolic_asymptotics: dict[str, str] = field(default_factory=dict)
    uncertainty_budget: dict[str, float] = field(default_factory=dict)
    epistemic_status: str = "CANDIDATE"
    provenance: dict[str, Any] = field(default_factory=dict)
    record_hash: str = ""

    def compute_record_hash(self) -> str:
        """Computes deterministic cryptographic SHA-256 hash over canonical representation.

        The ``record_hash`` field is excluded from the hashed payload: a digest
        cannot cover its own value, and including it would make the result depend
        on whatever string happened to be stored rather than on the record's
        content. Compare against it with :meth:`verify_record_hash`.
        """
        payload = self.to_dict()
        payload.pop("record_hash", None)
        canonical = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode('utf-8')).hexdigest()

    def verify_record_hash(self) -> bool:
        """True when ``record_hash`` matches a digest recomputed from this record.

        This is the only meaningful reproducibility check: a stored hash whose
        length happens to be 64 proves nothing about the content it claims to seal.
        """
        if not self.record_hash:
            return False
        return hmac.compare_digest(self.record_hash, self.compute_record_hash())

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "parent_model_id": self.parent_model_id,
            "model_lineage": list(self.model_lineage),
            "intervention_class": self.intervention_class.value,
            "secondary_intervention_classes": [c.value for c in self.secondary_intervention_classes],
            "evidence_maturity": self.evidence_maturity.value,
            "evidence_vector": self.evidence_vector.to_dict() if self.evidence_vector else None,
            "domain": self.domain,
            "mechanism_family": self.mechanism_family,
            "equation_str": self.equation_str,
            "variables": list(self.variables),
            "parameters": self.parameters,
            "derivation_basis": self.derivation_basis,
            "mathematical_guarantees": list(self.mathematical_guarantees),
            "numerical_guarantees": list(self.numerical_guarantees),
            "physical_constraints": list(self.physical_constraints),
            "assumptions": list(self.assumptions),
            "known_failure_modes": list(self.known_failure_modes),
            "validated_domain": self.validated_domain,
            "extrapolation_limits": self.extrapolation_limits,
            "falsification_target": self.falsification_target,
            "falsification_tests": list(self.falsification_tests),
            "failure_interpretation": self.failure_interpretation,
            "falsification_contract": self.falsification_contract.to_dict() if self.falsification_contract else None,
            "uncertainty_decomposition": self.uncertainty_decomposition.to_dict() if self.uncertainty_decomposition else None,
            "provenance_chain": self.provenance_chain.to_dict() if self.provenance_chain else None,
            "software_assurance": self.software_assurance,
            "mathematical_assurance": self.mathematical_assurance,
            "scientific_assurance": self.scientific_assurance,
            "symbolic_asymptotics": self.symbolic_asymptotics,
            "uncertainty_budget": self.uncertainty_budget,
            "epistemic_status": self.epistemic_status,
            "provenance": self.provenance,
            "record_hash": self.record_hash,
        }


class EpistemicGraph:
    """Formal Epistemic Knowledge Graph tracking 11 entity types and 9 lineage edge semantics."""

    ENTITY_TYPES: ClassVar[set[str]] = {
        'Observation', 'Measurand', 'State', 'Equation', 'Model',
        'Hypothesis', 'Prediction', 'Evidence', 'Claim', 'Decision', 'Experiment'
    }

    EDGE_TYPES: ClassVar[set[str]] = {
        'DERIVED_FROM', 'INFERRED_BY', 'ASSUMED_BY', 'MEASURED_BY',
        'SIMULATED_BY', 'APPROXIMATED_FROM', 'CONTRADICTED_BY',
        'CONSTRAINED_BY', 'VALIDATED_BY',
        'SUPPORTS', 'CONTRADICTS', 'PREDICTS', 'SPECIAL_CASE_OF',
        'TESTED_BY', 'INCONCLUSIVE_ON', 'DEPENDS_ON', 'INVALIDATED_BY',
    }

    def __init__(self):
        self.graph = nx.DiGraph()

    def add_node(self, node_id: str, node_type: str, metadata: dict[str, Any] | None = None) -> None:
        if node_type not in self.ENTITY_TYPES:
            raise ValueError(f"Invalid Epistemic Type '{node_type}'. Must be one of {self.ENTITY_TYPES}")
        self.graph.add_node(node_id, entity_type=node_type, metadata=metadata or {})

    def add_edge(self, source_id: str, target_id: str, relation: str, metadata: dict[str, Any] | None = None) -> None:
        if relation not in self.EDGE_TYPES:
            raise ValueError(f"Invalid Lineage Edge relation '{relation}'. Must be one of {self.EDGE_TYPES}")
        if not self.graph.has_node(source_id) or not self.graph.has_node(target_id):
            raise ValueError(f"Both source '{source_id}' and target '{target_id}' must exist prior to edge creation.")
        self.graph.add_edge(source_id, target_id, relation=relation, metadata=metadata or {})

    def add_lineage_edge(self, source_id: str, target_id: str, edge_type: str, metadata: dict[str, Any] | None = None) -> None:
        """Alias for add_edge to support legacy caller signatures."""
        self.add_edge(source_id, target_id, relation=edge_type, metadata=metadata)

    def get_model_evidence_summary(self, model_id: str) -> dict[str, Any]:
        """Summarizes all upstream and downstream evidence edges connected to model_id."""
        if not self.graph.has_node(model_id):
            raise ValueError(f"Model node '{model_id}' does not exist in the epistemic graph.")

        supporting = []
        contradicting = []
        predictions = []
        tests = []
        derived_from = []
        special_cases = []

        # Incoming edges to model
        for u, _, data in self.graph.in_edges(model_id, data=True):
            rel = data.get("relation")
            if rel in ("SUPPORTS", "VALIDATED_BY"):
                supporting.append(u)
            elif rel in ("CONTRADICTS", "CONTRADICTED_BY", "INVALIDATED_BY"):
                contradicting.append(u)
            elif rel in ("DERIVED_FROM", "APPROXIMATED_FROM"):
                derived_from.append(u)

        # Outgoing edges from model
        for _, v, data in self.graph.out_edges(model_id, data=True):
            rel = data.get("relation")
            if rel == "PREDICTS":
                predictions.append(v)
            elif rel == "TESTED_BY":
                tests.append(v)
            elif rel == "SPECIAL_CASE_OF":
                special_cases.append(v)
            elif rel in ("DERIVED_FROM", "APPROXIMATED_FROM"):
                derived_from.append(v)
            elif rel in ("SUPPORTS", "VALIDATED_BY"):
                supporting.append(v)
            elif rel in ("CONTRADICTS", "CONTRADICTED_BY", "INVALIDATED_BY"):
                contradicting.append(v)

        return {
            "model_id": model_id,
            "supporting_evidence_count": len(supporting),
            "contradicting_evidence_count": len(contradicting),
            "predictions_count": len(predictions),
            "tests_count": len(tests),
            "supporting_nodes": supporting,
            "contradicting_nodes": contradicting,
            "prediction_nodes": predictions,
            "test_nodes": tests,
            "derived_from_nodes": derived_from,
            "special_case_nodes": special_cases,
        }

    def get_root_assumptions(self, claim_id: str) -> list[str]:
        """Traces upstream DAG ancestors to isolate all unverified foundational axioms (nodes linked via ASSUMED_BY)."""
        if not self.graph.has_node(claim_id):
            raise ValueError(f"Node '{claim_id}' does not exist in the epistemic graph.")
        ancestors = nx.ancestors(self.graph, claim_id)
        roots: set[str] = set()
        for anc in ancestors:
            for _, _, data in self.graph.out_edges(anc, data=True):
                if data.get('relation') == 'ASSUMED_BY':
                    roots.add(anc)
        return sorted(roots)

    def get_lineage(self, node_id: str) -> list[str]:
        """Returns all upstream ancestor nodes that contributed to node_id."""
        if not self.graph.has_node(node_id):
            raise ValueError(f"Node '{node_id}' does not exist.")
        return sorted(nx.ancestors(self.graph, node_id))

    def find_contradictions(self) -> list[tuple[str, str, dict[str, Any]]]:
        """Identifies all pairs of nodes linked by CONTRADICTED_BY relationships."""
        contradictions = []
        for u, v, data in self.graph.edges(data=True):
            if data.get('relation') == 'CONTRADICTED_BY':
                contradictions.append((u, v, data))
        return contradictions

    def validate_graph_acyclicity(self) -> bool:
        """Verifies that the graph contains no illegal circular causal derivations."""
        return bool(nx.is_directed_acyclic_graph(self.graph))

    def export_graph(self) -> dict[str, Any]:
        """Serializes the graph into a JSON-compatible dictionary."""
        return {
            'nodes': [
                {'id': n, 'type': self.graph.nodes[n]['entity_type'], 'metadata': self.graph.nodes[n].get('metadata', {})}
                for n in self.graph.nodes
            ],
            'edges': [
                {'source': u, 'target': v, 'relation': data['relation'], 'metadata': data.get('metadata', {})}
                for u, v, data in self.graph.edges(data=True)
            ]
        }


class LifecycleManager:
    """Strict finite state machine governing scientific artifact lifecycle to prevent 'zombie science'."""

    STATES: ClassVar[list[str]] = [
        'Draft', 'Candidate', 'Contract_Validated', 'Validated', 'Challenged', 'Revalidated',
        'Domain_Restricted', 'Operational', 'Superseded', 'Refuted', 'Retired'
    ]

    ALLOWED_TRANSITIONS: ClassVar[dict[str, list[str]]] = {
        'Draft': ['Candidate', 'Retired'],
        'Candidate': ['Contract_Validated', 'Validated', 'Challenged', 'Refuted', 'Retired'],
        'Contract_Validated': ['Operational', 'Validated', 'Challenged', 'Domain_Restricted', 'Superseded', 'Retired'],
        'Validated': ['Operational', 'Challenged', 'Domain_Restricted', 'Superseded', 'Retired'],
        'Challenged': ['Revalidated', 'Contract_Validated', 'Domain_Restricted', 'Superseded', 'Refuted', 'Retired'],
        'Revalidated': ['Operational', 'Contract_Validated', 'Validated', 'Challenged', 'Superseded', 'Retired'],
        'Domain_Restricted': ['Operational', 'Contract_Validated', 'Validated', 'Challenged', 'Superseded', 'Retired'],
        'Operational': ['Challenged', 'Superseded', 'Domain_Restricted', 'Retired'],
        'Superseded': ['Retired'],
        'Refuted': ['Retired'],
        'Retired': []
    }

    def __init__(self):
        self.model_states: dict[str, str] = {}
        self.audit_log: list[dict[str, Any]] = []

    def get_state(self, model_name: str) -> str:
        return self.model_states.get(model_name, 'Draft')

    def register_model(self, model_name: str, initial_state: str = 'Draft') -> None:
        if initial_state not in self.STATES:
            raise ValueError(f"Invalid state '{initial_state}'")
        self.model_states[model_name] = initial_state
        self.audit_log.append({
            'model_name': model_name,
            'from_state': None,
            'to_state': initial_state,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'checks_passed': True,
            'evidence_id': 'INIT'
        })

    def promote(
        self,
        model_name: str,
        target_state: str,
        checks_passed: bool,
        evidence_id: str | None = None
    ) -> None:
        if not checks_passed:
            raise ValueError(f"Cannot promote model '{model_name}' to '{target_state}': verification checks failed.")

        current_state = self.get_state(model_name)
        allowed = self.ALLOWED_TRANSITIONS.get(current_state, [])
        if target_state not in allowed:
            raise ValueError(
                f"Illegal state transition for '{model_name}': cannot transition from '{current_state}' to '{target_state}'. Allowed: {allowed}"
            )

        self.model_states[model_name] = target_state
        self.audit_log.append({
            'model_name': model_name,
            'from_state': current_state,
            'to_state': target_state,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'checks_passed': checks_passed,
            'evidence_id': evidence_id or 'UNSPECIFIED'
        })


class RequirementsTraceabilityMatrix:
    """Authoritative traceability chain: Goal -> Requirement -> Design -> Implementation -> Test -> Evidence."""

    def __init__(self):
        self.requirements: dict[str, dict[str, Any]] = {}

    def register_requirement(
        self,
        req_id: str,
        goal: str,
        owner: str,
        design_adr: str,
        impl_artifact: str,
        acceptance_test: str,
        validation_evidence: str
    ) -> None:
        self.requirements[req_id] = {
            'req_id': req_id,
            'goal': goal,
            'owner': owner,
            'design_adr': design_adr,
            'impl_artifact': impl_artifact,
            'acceptance_test': acceptance_test,
            'validation_evidence': validation_evidence,
            'status': 'VERIFIED' if validation_evidence else 'PENDING'
        }

    def verify_completeness(self) -> tuple[bool, list[str]]:
        if not self.requirements:
            return False, ["No requirements registered in traceability matrix."]
        missing = []
        placeholders = {"unspecified", "todo", "tbd", "n/a", "none", "pending"}
        for req_id, data in self.requirements.items():
            for key in ['goal', 'owner', 'design_adr', 'impl_artifact', 'acceptance_test', 'validation_evidence']:
                val = data.get(key)
                if not val or (isinstance(val, str) and len(val.strip()) < 10):
                    missing.append(f"{req_id} missing {key} (needs >=10 non-blank chars)")
                elif isinstance(val, str) and val.strip().lower() in placeholders:
                    missing.append(f"{req_id} has placeholder {key}={val!r}")
        return len(missing) == 0, missing


class AdversarialDeliberationTournament:
    """Multi-agent adversarial tournament: Generator -> Challenger -> Counterexample -> Regularization -> Promotion.

    .. warning::
        This is lifecycle bookkeeping, NOT mathematical proof. The
        ``boundary_flaw_detected`` flag must come from an actual check
        (e.g. ``AgnosticClosedLoopAssuranceSystem``); the tournament only
        records the state transitions. Returned dicts carry
        ``mathematical_proof: False`` to prevent misreading.
    """

    def __init__(self, epistemic_graph: EpistemicGraph, lifecycle: LifecycleManager):
        self.graph = epistemic_graph
        self.lifecycle = lifecycle

    def run_tournament(
        self,
        hypothesis_id: str,
        formula_str: str,
        adversarial_probe_name: str,
        boundary_flaw_detected: bool,
        repaired_formula_str: str
    ) -> dict[str, Any]:
        """
        Executes deliberation lifecycle:
        1. Ingest Hypothesis
        2. Challenger stress-tests boundary
        3. If flaw detected, mark CONTRADICTED_BY and synthesize regularized Model
        4. Validate repaired model and promote to Validated
        """
        # 1. Register Hypothesis
        self.graph.add_node(hypothesis_id, 'Hypothesis', {'formula': formula_str})
        self.lifecycle.register_model(hypothesis_id, 'Draft')
        self.lifecycle.promote(hypothesis_id, 'Candidate', checks_passed=True, evidence_id="TOURNAMENT-ENTRY")

        if boundary_flaw_detected:
            # 2. Register Counterexample Evidence
            counter_id = f"Evidence_Flaw_{adversarial_probe_name}"
            self.graph.add_node(counter_id, 'Evidence', {'probe': adversarial_probe_name, 'result': 'Singularity/Divergence'})
            self.graph.add_edge(counter_id, hypothesis_id, 'CONTRADICTED_BY')

            # Retire the contradicted hypothesis to prevent "zombie science"
            self.lifecycle.promote(hypothesis_id, 'Retired', checks_passed=True, evidence_id=f"REFUTED_BY_{counter_id}")

            # 3. Register Repaired Model
            repaired_id = f"Model_{hypothesis_id}_Regularized"
            self.graph.add_node(repaired_id, 'Model', {'formula': repaired_formula_str})
            self.graph.add_edge(repaired_id, hypothesis_id, 'APPROXIMATED_FROM')
            self.graph.add_edge(counter_id, repaired_id, 'CONSTRAINED_BY')

            # 4. Lifecycle transition
            self.lifecycle.register_model(repaired_id, 'Candidate')
            self.lifecycle.promote(repaired_id, 'Validated', checks_passed=True, evidence_id="REGULARIZATION-PASS")

            return {
                'outcome': 'REPAIRED_AND_VALIDATED',
                'mathematical_proof': False,
                'hypothesis': hypothesis_id,
                'hypothesis_state': self.lifecycle.get_state(hypothesis_id),
                'counterexample': counter_id,
                'repaired_model': repaired_id,
                'final_state': self.lifecycle.get_state(repaired_id)
            }
        else:
            self.lifecycle.promote(hypothesis_id, 'Validated', checks_passed=True, evidence_id="TOURNAMENT-CLEAN-PASS")
            return {
                'outcome': 'DIRECT_VALIDATION',
                'mathematical_proof': False,
                'hypothesis': hypothesis_id,
                'final_state': self.lifecycle.get_state(hypothesis_id)
            }


class ModelEpistemicStatus(str, Enum):
    """Rigorous scientific epistemic ladder distinguishing hypothesis maturity.
    
    Equation != Hypothesis != Test != Evidence != Established Model
    """
    GENERATED = "GENERATED"                        # Bare equation syntax/symbolic candidate
    MATHEMATICALLY_VALID = "MATHEMATICALLY_VALID"  # Consistent units, bounded, well-posed
    TESTABLE_HYPOTHESIS = "TESTABLE_HYPOTHESIS"    # Equipped with explicit test contract & holdout
    EMPIRICALLY_SUPPORTED = "EMPIRICALLY_SUPPORTED"# Survived out-of-sample empirical test
    PROVISIONALLY_ACCEPTED = "PROVISIONALLY_ACCEPTED" # Multi-regime consensus across independent labs
    CONTRADICTED = "CONTRADICTED"                  # Decisively refuted by evidence
    UNDERDETERMINED = "UNDERDETERMINED"            # Indistinguishable among multiple candidates
    OUT_OF_DOMAIN = "OUT_OF_DOMAIN"                # Outside claimed applicability envelope
    UNTESTABLE = "UNTESTABLE"                      # Lacks observable predictions or required resolution


class DiscoveryMode(str, Enum):
    """The three fundamental scientific discovery modes supported by APEDP."""
    REPAIR = "REPAIR"          # Known baseline M_0 -> regularized/bounded continuation M_1
    COMPOSE = "COMPOSE"        # Multiple partial laws M_1 + M_2 -> hybrid composite model
    SYNTHESIZE = "SYNTHESIZE"  # Ab initio dimensional/invariance search from variables & constraints


class VerificationGate(str, Enum):
    """Decoupled verification gates where passing one never implies passing the next."""
    DIMENSIONAL = "DIMENSIONAL"
    MATHEMATICAL = "MATHEMATICAL"
    PHYSICAL_ADMISSIBILITY = "PHYSICAL_ADMISSIBILITY"
    EMPIRICAL = "EMPIRICAL"
    NOVELTY_AUDIT = "NOVELTY_AUDIT"
    GENERALIZATION = "GENERALIZATION"


class ScientificMaturityLevel(str, Enum):
    """The 5 progressive maturity levels of scientific model generation & assurance."""
    LEVEL_1_EQUATION_REPAIR = "LEVEL_1_EQUATION_REPAIR"              # M_0 -> M_1 regularization
    LEVEL_2_MATHEMATICAL_ASSURANCE = "LEVEL_2_MATHEMATICAL_ASSURANCE"# Dimensions, poles, asymptotics, invariants
    LEVEL_3_EPISTEMIC_ASSURANCE = "LEVEL_3_EPISTEMIC_ASSURANCE"      # 5-way feasibility & action stance
    LEVEL_4_TESTABLE_MODEL_SYNTHESIS = "LEVEL_4_TESTABLE_MODEL_SYNTHESIS" # T = (M, P, D, C, L, H, F, E, G)
    LEVEL_5_NOVEL_SCIENTIFIC_DISCOVERY = "LEVEL_5_NOVEL_SCIENTIFIC_DISCOVERY" # Unseen experiment + independent confirmation


@dataclass(frozen=True)
class TestableModelPackage:
    """Complete scientific hypothesis package binding an equation to its executable validation protocol.
    
    Formalizes the canonical tuple:
    T = (M, P, D, C, L, H, F, E, G)
    where:
      M = Model equation and parameters
      P = Structured generation provenance (mode, parent models, Pi-groups, search space)
      D = Domain applicability envelope and extrapolation boundaries
      C = Physical constraints, conservation invariants, monotonicity
      L = Limiting behavior and recovered classical asymptotic limits
      H = Observable predictions and quantitative hypotheses
      F = Explicit quantitative falsification criteria
      E = Evidence requirements (data partition locks, target resolution)
      G = Competing models and maximal-discrimination active experiment point (x*)
    """
    __test__: ClassVar[bool] = False

    model_id: str
    equation_str: str
    parameters: dict[str, Any]
    domain: dict[str, tuple[float, float]]
    constraints: tuple[str, ...]
    limiting_behavior: dict[str, str]
    recovered_limits: dict[str, str]
    predictions: tuple[str, ...]
    required_data: dict[str, Any]
    holdout_test: dict[str, Any]
    falsification_criteria: tuple[str, ...]
    competing_models: tuple[str, ...] = ()
    maximal_discrimination_point: dict[str, float] = field(default_factory=dict)
    mechanistic_status: str = "phenomenological_continuation"
    epistemic_status: ModelEpistemicStatus = ModelEpistemicStatus.TESTABLE_HYPOTHESIS
    generation_provenance: dict[str, Any] = field(default_factory=dict)
    novelty_audit: NoveltyAuditResult | None = None
    maturity_level: ScientificMaturityLevel = ScientificMaturityLevel.LEVEL_4_TESTABLE_MODEL_SYNTHESIS

    @staticmethod
    def compute_uncertainty_weighted_discrimination(
        eval_f1: Callable[[np.ndarray], np.ndarray],
        eval_f2: Callable[[np.ndarray], np.ndarray],
        x_grid: np.ndarray,
        uncertainty_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> tuple[float, float, float]:
        """Calculates optimal active-experiment coordinate x* maximizing discrepancy relative to uncertainty.
        
        x* = argmax_x |f1(x) - f2(x)| / sigma(x)
        Returns: (x_star, snr_discrepancy, raw_discrepancy)
        """
        f1_vals = eval_f1(x_grid)
        f2_vals = eval_f2(x_grid)
        raw_diff = np.abs(f1_vals - f2_vals)
        if uncertainty_fn is not None:
            sigma = np.maximum(uncertainty_fn(x_grid), 1e-12)
            snr_diff = raw_diff / sigma
        else:
            snr_diff = raw_diff
        idx = int(np.argmax(snr_diff))
        x_star = float(x_grid.flatten()[idx])
        return x_star, float(snr_diff.flatten()[idx]), float(raw_diff.flatten()[idx])

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "equation_str": self.equation_str,
            "parameters": self.parameters,
            "domain": self.domain,
            "constraints": list(self.constraints),
            "limiting_behavior": self.limiting_behavior,
            "recovered_limits": self.recovered_limits,
            "predictions": list(self.predictions),
            "required_data": self.required_data,
            "holdout_test": self.holdout_test,
            "falsification_criteria": list(self.falsification_criteria),
            "competing_models": list(self.competing_models),
            "maximal_discrimination_point": self.maximal_discrimination_point,
            "mechanistic_status": self.mechanistic_status,
            "epistemic_status": self.epistemic_status.value,
            "generation_provenance": self.generation_provenance,
            "novelty_audit": self.novelty_audit.to_dict() if self.novelty_audit else None,
            "maturity_level": self.maturity_level.value,
        }

