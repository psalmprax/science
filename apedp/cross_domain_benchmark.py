# Databricks notebook source
"""Pre-registered cross-domain equation-discovery evaluation protocol."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar


class EvidenceTrack(str, Enum):
    UNINFORMED = "uninformed"
    WEAK_PRIOR = "weak_prior"
    LITERATURE_INFORMED = "literature_informed"


@dataclass(frozen=True)
class InformationFirewall:
    """Declaration of information available to, and withheld from, a run."""

    known: tuple[str, ...]
    hidden: tuple[str, ...]
    allowed_priors: tuple[str, ...]
    forbidden: tuple[str, ...]

    def __post_init__(self) -> None:
        overlap = set(self.known) & set(self.hidden)
        if overlap:
            raise ValueError(f"firewall fields cannot be both known and hidden: {sorted(overlap)}")
        forbidden_overlap = set(self.allowed_priors) & set(self.forbidden)
        if forbidden_overlap:
            raise ValueError(f"firewall priors cannot be both allowed and forbidden: {sorted(forbidden_overlap)}")

    def to_dict(self) -> dict[str, list[str]]:
        return {"known": list(self.known), "hidden": list(self.hidden), "allowed_priors": list(self.allowed_priors), "forbidden": list(self.forbidden)}


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    domain: str
    target: str
    description: str
    source: str
    expected_refusal: bool = False


@dataclass(frozen=True)
class BenchmarkOutcome:
    case_id: str
    track: EvidenceTrack
    seed: int
    dataset_hash: str
    discovered: bool
    admitted: bool
    refused: bool
    baseline_improved: bool | None
    stable_across_seeds: bool | None
    expression: str | None = None
    refusal_reason: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    status: DiscoveryStatus | None = None

    def __post_init__(self) -> None:
        if self.admitted and self.refused:
            raise ValueError("an outcome cannot be both admitted and refused")
        if self.dataset_hash == "":
            raise ValueError("dataset_hash is required")
        derived = self.status
        if derived is None:
            domain_limited = bool(self.evidence.get("domain_limited", False))
            if self.admitted and domain_limited:
                derived = DiscoveryStatus.DOMAIN_LIMITED
            elif self.admitted and self.stable_across_seeds is not None:
                derived = DiscoveryStatus.ADMIT
            elif self.refused and self.refusal_reason:
                derived = DiscoveryStatus.REFUSE
            else:
                derived = DiscoveryStatus.INCONCLUSIVE
            object.__setattr__(self, "status", derived)

    def to_dict(self) -> dict[str, Any]:
        return {"case_id": self.case_id, "track": self.track.value, "seed": self.seed, "dataset_hash": self.dataset_hash, "discovered": self.discovered, "admitted": self.admitted, "refused": self.refused, "status": self.status.value if self.status is not None else None, "baseline_improved": self.baseline_improved, "stable_across_seeds": self.stable_across_seeds, "expression": self.expression, "refusal_reason": self.refusal_reason, "evidence": self.evidence}


@dataclass(frozen=True)
class BenchmarkSummary:
    total: int
    discovery_rate: float | None
    false_admission_rate: float | None
    refusal_rate: float | None
    inconclusive_rate: float | None
    baseline_improvement_rate: float | None
    seed_stability_rate: float | None
    by_track: dict[str, dict[str, float | None]]

    def to_dict(self) -> dict[str, Any]:
        return {"total": self.total, "discovery_rate": self.discovery_rate, "false_admission_rate": self.false_admission_rate, "refusal_rate": self.refusal_rate, "inconclusive_rate": self.inconclusive_rate, "baseline_improvement_rate": self.baseline_improvement_rate, "seed_stability_rate": self.seed_stability_rate, "by_track": self.by_track}


DEFAULT_CASES: tuple[BenchmarkCase, ...] = (
    BenchmarkCase("mechanics_kinetic_energy", "mechanics", "energy", "kinetic energy relation", "synthetic_known_law"),
    BenchmarkCase("thermodynamics_arrhenius", "thermodynamics", "rate", "temperature-dependent reaction rate", "synthetic_known_law"),
    BenchmarkCase("electromagnetism_lorentz", "electromagnetism", "force", "magnetic force relation", "synthetic_known_law"),
    BenchmarkCase("fluid_dynamics_heat_transfer", "fluid_dynamics", "heat_flux", "dimensionless heat-transfer relation", "external_benchmark_adapter"),
    BenchmarkCase("materials_hall_petch", "materials", "strength", "grain-size strength relation with hidden turnover", "external_benchmark_adapter"),
    BenchmarkCase("adversarial_overfit", "cross_domain", "response", "seductive high-fit candidate with poor locked holdout", "adversarial_benchmark", True),
)


def dataset_hash(rows: Iterable[dict[str, Any]]) -> str:
    payload = json.dumps(list(rows), sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def _rate(values: list[bool | None]) -> float | None:
    measured = [value for value in values if value is not None]
    return sum(measured) / len(measured) if measured else None


def summarize(outcomes: Iterable[BenchmarkOutcome], cases: Iterable[BenchmarkCase] = DEFAULT_CASES) -> BenchmarkSummary:
    items = list(outcomes)
    case_map = {case.case_id: case for case in cases}
    expected_refusal = lambda item: case_map.get(item.case_id, BenchmarkCase("", "", "", "", "")).expected_refusal
    by_track: dict[str, dict[str, float | None]] = {}
    for track in EvidenceTrack:
        subset = [item for item in items if item.track == track]
        by_track[track.value] = {"n": float(len(subset)), "discovery_rate": _rate([item.discovered for item in subset]), "false_admission_rate": _rate([item.admitted for item in subset if expected_refusal(item)]), "refusal_rate": _rate([item.refused for item in subset]), "domain_limited_rate": _rate([item.status == DiscoveryStatus.DOMAIN_LIMITED for item in subset]), "inconclusive_rate": _rate([item.status == DiscoveryStatus.INCONCLUSIVE for item in subset])}
    return BenchmarkSummary(len(items), _rate([item.discovered for item in items]), _rate([item.admitted for item in items if expected_refusal(item)]), _rate([item.refused for item in items]), _rate([item.status == DiscoveryStatus.INCONCLUSIVE for item in items]), _rate([item.baseline_improved for item in items]), _rate([item.stable_across_seeds for item in items]), by_track)


def run_protocol(cases: Iterable[BenchmarkCase], evaluator: Callable[[BenchmarkCase, EvidenceTrack, int], BenchmarkOutcome], *, tracks: Iterable[EvidenceTrack] = tuple(EvidenceTrack), seeds: Iterable[int] = (7, 42, 123)) -> list[BenchmarkOutcome]:
    outcomes: list[BenchmarkOutcome] = []
    for case in cases:
        for track in tracks:
            for seed in seeds:
                outcome = evaluator(case, track, seed)
                if outcome.case_id != case.case_id or outcome.track != track or outcome.seed != seed:
                    raise ValueError("evaluator returned an outcome for the wrong protocol cell")
                outcomes.append(outcome)
    return outcomes


__all__ = ["DEFAULT_CASES", "BenchmarkCase", "BenchmarkOutcome", "BenchmarkSummary", "DiscoveryStatus", "EvidenceTrack", "InformationFirewall", "dataset_hash", "run_protocol", "summarize"]

# COMMAND ----------

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class AdapterDataset:
    """Locked data supplied by a benchmark adapter."""

    x_train: dict[str, np.ndarray]
    y_train: np.ndarray
    x_holdout: dict[str, np.ndarray]
    y_holdout: np.ndarray
    dataset_hash: str
    provenance: dict[str, Any]
    firewall: InformationFirewall


class BenchmarkAdapter(Protocol):
    def load(self, case: BenchmarkCase, seed: int) -> AdapterDataset:
        ...

    def evaluate(self, case: BenchmarkCase, track: EvidenceTrack, seed: int, dataset: AdapterDataset) -> BenchmarkOutcome:
        ...


def outcome_from_measurement(case: BenchmarkCase, track: EvidenceTrack, seed: int, dataset: AdapterDataset, *, expression: str | None, holdout_r_squared: float | None, baseline_improved: bool | None, admitted: bool, stable_across_seeds: bool | None, refusal_reason: str | None = None, extra_evidence: dict[str, Any] | None = None) -> BenchmarkOutcome:
    """Convert one adapter measurement into the canonical protocol record."""
    return BenchmarkOutcome(
        case_id=case.case_id,
        track=track,
        seed=seed,
        dataset_hash=dataset.dataset_hash,
        discovered=expression is not None,
        admitted=admitted,
        refused=not admitted,
        baseline_improved=baseline_improved,
        stable_across_seeds=stable_across_seeds,
        expression=expression,
        refusal_reason=refusal_reason,
        evidence={"holdout_r_squared": holdout_r_squared, "provenance": dataset.provenance, "information_firewall": dataset.firewall.to_dict(), **(extra_evidence or {})},
    )


class SyntheticDiscoveryAdapter:
    """Adapter for deterministic known-law cases used in fast CI."""

    _LAW_IDS: ClassVar[dict[str, str]] = {"mechanics_kinetic_energy": "KINETIC_ENERGY", "thermodynamics_arrhenius": "STEFAN_BOLTZMANN", "electromagnetism_lorentz": "OHMIC_PRODUCT"}
    _LITERATURE_FORMS: ClassVar[dict[str, str]] = {"KINETIC_ENERGY": "m*v**2/2", "STEFAN_BOLTZMANN": "sigma*T**4", "OHMIC_PRODUCT": "x*y"}

    def load(self, case: BenchmarkCase, seed: int) -> AdapterDataset:
        from apedp.discovery_benchmark import make_synthetic

        law_id = self._LAW_IDS.get(case.case_id)
        if law_id is None:
            raise KeyError(f"no synthetic generator registered for {case.case_id!r}")
        x_train, y_train, x_holdout, y_holdout, variables, priors = make_synthetic(law_id, seed=seed, n=60, noise=0.0)
        rows = []
        for index in range(len(y_train)):
            rows.append({name: float(x_train[name][index]) for name in variables} | {case.target: float(y_train[index])})
        for index in range(len(y_holdout)):
            rows.append({name: float(x_holdout[name][index]) for name in variables} | {case.target: float(y_holdout[index])})
        firewall = InformationFirewall(
            known=("variables", "units", "measurement_domain"),
            hidden=("generating_equation", "holdout_targets", "literature_parameters"),
            allowed_priors=("dimensional_analysis",),
            forbidden=("target_equation", "hidden_observations", "literature_specific_constants"),
        )
        return AdapterDataset(
            x_train=x_train,
            y_train=np.asarray(y_train, dtype=float),
            x_holdout=x_holdout,
            y_holdout=np.asarray(y_holdout, dtype=float),
            dataset_hash=dataset_hash(rows),
            provenance={"source": case.source, "case_id": case.case_id, "variables": variables, "weak_priors": priors, "literature_form": self._LITERATURE_FORMS[law_id]},
            firewall=firewall,
        )

    def _track_configuration(self, track: EvidenceTrack, dataset: AdapterDataset) -> dict[str, Any]:
        if track == EvidenceTrack.UNINFORMED:
            return {"guided": False, "template_priors": [], "prior_disclosure": "none"}
        if track == EvidenceTrack.WEAK_PRIOR:
            return {"guided": True, "template_priors": dataset.provenance["weak_priors"], "prior_disclosure": "generic_alternatives"}
        return {"guided": True, "template_priors": [dataset.provenance["literature_form"]], "prior_disclosure": "declared_reference_form"}

    def evaluate(self, case, track, seed, dataset):
        import sympy as sp

        from apedp.core import ClosedLoopAssuranceSystem, DimensionalAssuranceEngine
        from apedp.discovery import MultiVariableSRE
        from apedp.discovery_admission import admit_candidate, numeric_conditioning_report
        from apedp.safety import safe_parse_expr

        configuration = self._track_configuration(track, dataset)
        engine = DimensionalAssuranceEngine()
        for variable in dataset.x_train:
            engine.register_variable(variable, "dimensionless")
        search = MultiVariableSRE(ClosedLoopAssuranceSystem(assurance_engine=engine), independent_vars=list(dataset.x_train), target_unit_str="dimensionless", guided=configuration["guided"], template_priors=configuration["template_priors"], seed=seed)
        expression_obj, _, history = search.run_discovery(dataset.x_train, dataset.y_train, generations=10, pop_size=30, seed=seed)
        expression = str(expression_obj) if expression_obj is not None else None
        holdout_r_squared = None
        admitted = False
        reason = "no candidate"
        admission = None
        if expression is not None:
            expr = safe_parse_expr(expression)
            symbols = sorted(expr.free_symbols, key=str)
            prediction = np.asarray(sp.lambdify(symbols, expr, "numpy")(*[dataset.x_holdout[str(symbol)] for symbol in symbols]), dtype=float)
            residual = prediction - dataset.y_holdout
            total = np.sum((dataset.y_holdout - np.mean(dataset.y_holdout)) ** 2)
            holdout_r_squared = float(1.0 - np.sum(residual ** 2) / total) if total else float("nan")
            admission = admit_candidate(expr, x_holdout=dataset.x_holdout, y_holdout=dataset.y_holdout, min_holdout_r_squared=0.1, dimensionally_consistent=True, repair_status="ACCEPTED", conditioning=numeric_conditioning_report(expr, dataset.x_holdout))
            admitted = admission.status.value == "PERSISTED"
            reason = "; ".join(admission.reasons) if admission.reasons else None
        return outcome_from_measurement(case, track, seed, dataset, expression=expression, holdout_r_squared=holdout_r_squared, baseline_improved=None, admitted=admitted, stable_across_seeds=None, refusal_reason=reason, extra_evidence={"track_configuration": configuration, "history_head": history[:2], "admission": admission.to_dict() if admission else None})


def run_adapter_protocol(adapter: BenchmarkAdapter, cases: Iterable[BenchmarkCase], *, tracks: Iterable[EvidenceTrack] = tuple(EvidenceTrack), seeds: Iterable[int] = (7, 42, 123)) -> list[BenchmarkOutcome]:
    """Run the declared protocol using an adapter-owned locked dataset."""
    outcomes = []
    for case in cases:
        for track in tracks:
            for seed in seeds:
                dataset = adapter.load(case, seed)
                outcomes.append(adapter.evaluate(case, track, seed, dataset))
    return outcomes


__all__ += ["AdapterDataset", "BenchmarkAdapter", "SyntheticDiscoveryAdapter", "outcome_from_measurement", "run_adapter_protocol"]


@dataclass(frozen=True)
class SensitivityVariant:
    """One documented alternative analysis of an unchanged primary dataset."""

    variant_id: str
    rationale: str
    rule: str
    outcome: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.variant_id or not self.rationale or not self.rule:
            raise ValueError("sensitivity variants require an id, rationale, and rule")

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "rationale": self.rationale,
            "rule": self.rule,
            "outcome": self.outcome,
        }


@dataclass(frozen=True)
class SensitivityAnalysis:
    """Primary result plus separately reported sensitivity variants."""

    primary: dict[str, Any]
    variants: tuple[SensitivityVariant, ...]
    robustness: str

    def __post_init__(self) -> None:
        if self.robustness not in {"ROBUST", "SENSITIVE", "UNRESOLVED"}:
            raise ValueError("robustness must be ROBUST, SENSITIVE, or UNRESOLVED")

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary": self.primary,
            "variants": [variant.to_dict() for variant in self.variants],
            "robustness": self.robustness,
        }


def run_sensitivity_analysis(
    primary: dict[str, Any],
    variants: Iterable[SensitivityVariant],
    *,
    structural_key: str = "structure",
    quantitative_key: str = "holdout_error",
    tolerance: float = 0.1,
) -> SensitivityAnalysis:
    """Compare documented variants without overwriting the primary result.

    Structural agreement across variants is robust when all variants report the
    same non-empty structure. Quantitative behavior is sensitive when the
    holdout error changes by more than ``tolerance`` relative to the primary.
    Missing comparable measurements remain unresolved rather than being treated
    as evidence of robustness.
    """
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    variant_list = tuple(variants)
    if not variant_list:
        return SensitivityAnalysis(primary, (), "UNRESOLVED")
    structures = [primary.get(structural_key)] + [variant.outcome.get(structural_key) for variant in variant_list]
    structural_values = [value for value in structures if value not in (None, "")]
    structure_robust = bool(structural_values) and len(set(map(str, structural_values))) == 1
    primary_error = primary.get(quantitative_key)
    comparable_errors = [variant.outcome.get(quantitative_key) for variant in variant_list]
    # Narrow once, explicitly: a non-numeric primary error means the comparison
    # has no reference point, so every variant counts as "missing".
    numeric_primary = primary_error if isinstance(primary_error, (int, float)) else None
    quantitative_values = [value for value in comparable_errors if isinstance(value, (int, float))]
    quantitative_stable = (
        numeric_primary is not None and bool(quantitative_values) and all(
            abs(float(value) - float(numeric_primary))
            <= tolerance * max(abs(float(numeric_primary)), 1.0e-12)
            for value in quantitative_values
        )
    )
    has_missing = len(quantitative_values) != len(comparable_errors) or numeric_primary is None
    if structure_robust and quantitative_stable and not has_missing:
        robustness = "ROBUST"
    elif structure_robust and quantitative_values:
        robustness = "SENSITIVE"
    else:
        robustness = "UNRESOLVED"
    return SensitivityAnalysis(primary, variant_list, robustness)


__all__ += ["SensitivityAnalysis", "SensitivityVariant", "run_sensitivity_analysis"]

# COMMAND ----------

class DiscoveryStatus(str, Enum):
    ADMIT = "ADMIT"
    DOMAIN_LIMITED = "DOMAIN_LIMITED"
    CONDITIONAL = "CONDITIONAL"
    PREDICTIVE_ONLY = "PREDICTIVE_ONLY"
    REFUSE = "REFUSE"
    INCONCLUSIVE = "INCONCLUSIVE"


def _classify_outcome_status(self) -> None:
    if self.admitted and self.refused:
        raise ValueError("an outcome cannot be both admitted and refused")
    if self.dataset_hash == "":
        raise ValueError("dataset_hash is required")
    if self.status is not None:
        return
    evidence = self.evidence
    if self.refused and self.refusal_reason:
        status = DiscoveryStatus.REFUSE
    elif not self.admitted:
        status = DiscoveryStatus.INCONCLUSIVE
    elif evidence.get("domain_limited"):
        status = DiscoveryStatus.DOMAIN_LIMITED
    elif evidence.get("conditional"):
        status = DiscoveryStatus.CONDITIONAL
    elif evidence.get("predictive_only"):
        status = DiscoveryStatus.PREDICTIVE_ONLY
    elif self.stable_across_seeds is not None:
        status = DiscoveryStatus.ADMIT
    else:
        status = DiscoveryStatus.INCONCLUSIVE
    object.__setattr__(self, "status", status)


# Intentional monkey-patch (notebook composition): the frozen dataclass is
# given its status classifier after definition.
BenchmarkOutcome.__post_init__ = _classify_outcome_status  # type: ignore[method-assign]
__all__.extend(["CONDITIONAL", "PREDICTIVE_ONLY"])
