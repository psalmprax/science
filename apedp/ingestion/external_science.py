# Databricks notebook source
"""External-science dataset adapters with provenance and grouped holdouts."""
from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from apedp.cross_domain_benchmark import AdapterDataset, InformationFirewall, dataset_hash
from apedp.source_audit import SourceMetadata


@dataclass(frozen=True)
class NISTWebBookCase:
    """Declaration for one NIST WebBook property export."""
    case_id: str
    compound: str
    property_name: str
    source_url: str = "https://webbook.nist.gov/chemistry/"
    property_unit: str = ""
    variable_units: dict[str, str] | None = None
    literature_equation: str | None = None


class NISTWebBookAdapter:
    """Load a caller-supplied NIST export without exposing the target equation."""

    def load_csv(self, path: str | Path, case: NISTWebBookCase, *, feature_columns: tuple[str, ...], target_column: str, group_column: str, holdout_groups: tuple[str, ...], source_metadata: SourceMetadata | None = None) -> AdapterDataset:
        """Create locked train/holdout arrays using whole-group holdout."""
        if not holdout_groups:
            raise ValueError("holdout_groups must contain at least one group")
        with open(path, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        required = set(feature_columns) | {target_column, group_column}
        if not rows or not required.issubset(rows[0]):
            raise ValueError(f"CSV must contain columns: {sorted(required)}")
        parsed: list[dict[str, Any]] = []
        for row in rows:
            try:
                parsed.append({"group": str(row[group_column]), "features": {name: float(row[name]) for name in feature_columns}, "target": float(row[target_column])})
            except (TypeError, ValueError) as exc:
                raise ValueError(f"non-numeric NIST row: {row}") from exc
        holdout_set = set(holdout_groups)
        train_rows = [row for row in parsed if row["group"] not in holdout_set]
        holdout_rows = [row for row in parsed if row["group"] in holdout_set]
        if not train_rows or not holdout_rows:
            raise ValueError("group split produced an empty train or holdout partition")
        groups_present = {row["group"] for row in parsed}
        missing_groups = sorted(holdout_set - groups_present)
        if missing_groups:
            raise ValueError(f"holdout groups are absent from CSV: {missing_groups}")
        canonical_rows = [{**row["features"], case.property_name: row["target"], group_column: row["group"]} for row in parsed]
        source = source_metadata or SourceMetadata(source_id=f"nist-webbook:{case.case_id}", source_kind="database", value_kind="measured", citation=case.source_url, dataset_id=case.case_id, selection_criteria=f"compound={case.compound}; property={case.property_name}", preprocessing="numeric CSV export; whole-group holdout", unit_conventions={**(case.variable_units or {}), case.property_name: case.property_unit}, notes="Literature equation withheld from discovery track.")
        provenance = {"source": source.to_dict(), "case_id": case.case_id, "compound": case.compound, "property": case.property_name, "feature_columns": list(feature_columns), "group_column": group_column, "holdout_groups": list(holdout_groups), "literature_equation_withheld": case.literature_equation is not None, "file_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
        firewall = InformationFirewall(known=("feature_values", "target_values_train", "units", "compound_groups"), hidden=("target_equation", "literature_parameters", "holdout_targets"), allowed_priors=("dimensional_analysis", "declared_property_semantics"), forbidden=("literature_equation", "holdout_group_targets", "post-discovery_tuning"))
        train_x = {name: np.asarray([row["features"][name] for row in train_rows], dtype=float) for name in feature_columns}
        holdout_x = {name: np.asarray([row["features"][name] for row in holdout_rows], dtype=float) for name in feature_columns}
        return AdapterDataset(x_train=train_x, y_train=np.asarray([row["target"] for row in train_rows], dtype=float), x_holdout=holdout_x, y_holdout=np.asarray([row["target"] for row in holdout_rows], dtype=float), dataset_hash=dataset_hash(canonical_rows), provenance={**provenance, "train_rows": len(train_rows), "holdout_rows": len(holdout_rows), "groups": sorted(groups_present)}, firewall=firewall)


def apply_noise_ladder(dataset: AdapterDataset, noise_levels: tuple[float, ...] = (0.0, 0.001, 0.005, 0.01, 0.02, 0.05), *, seed: int = 42) -> list[AdapterDataset]:
    """Create reproducible noisy-training variants while preserving clean holdout truth."""
    if any(level < 0 for level in noise_levels):
        raise ValueError("noise levels must be non-negative")
    rng = np.random.default_rng(seed)
    scale = float(np.std(dataset.y_train))
    variants: list[AdapterDataset] = []
    for level in noise_levels:
        noisy_target = np.asarray(dataset.y_train, dtype=float) + rng.normal(0.0, level * scale, size=dataset.y_train.shape)
        variants.append(AdapterDataset(x_train={name: values.copy() for name, values in dataset.x_train.items()}, y_train=noisy_target, x_holdout={name: values.copy() for name, values in dataset.x_holdout.items()}, y_holdout=dataset.y_holdout.copy(), dataset_hash=dataset.dataset_hash, provenance={**dataset.provenance, "noise_level": float(level), "noise_seed": seed, "noise_target_scale": scale, "clean_holdout_preserved": True}, firewall=dataset.firewall))
    return variants


def build_nist_manifest(dataset: AdapterDataset, case: NISTWebBookCase, *, seeds: tuple[int, ...], grammar: dict[str, Any], operator_limits: dict[str, Any]) -> dict[str, Any]:
    """Build a reproducible experiment manifest before discovery is run."""
    return {
        "protocol_status": "INFRASTRUCTURE_ONLY",
        "case_id": case.case_id,
        "source_url": case.source_url,
        "source_sha256": dataset.provenance["file_sha256"],
        "dataset_hash": dataset.dataset_hash,
        "compound": case.compound,
        "property": case.property_name,
        "property_unit": case.property_unit,
        "variable_units": case.variable_units or {},
        "groups": dataset.provenance["groups"],
        "holdout_groups": dataset.provenance["holdout_groups"],
        "train_rows": dataset.provenance["train_rows"],
        "holdout_rows": dataset.provenance["holdout_rows"],
        "seeds": list(seeds),
        "grammar": grammar,
        "operator_limits": operator_limits,
        "literature_equation_withheld": True,
    }


def run_nist_blind_discovery(dataset: AdapterDataset, case: NISTWebBookCase, *, seed: int = 42, generations: int = 30, pop_size: int = 100, template_priors: list[str] | None = None) -> dict[str, Any]:
    """Discover and score one frozen candidate without literature comparison."""
    import sympy as sp

    from apedp.core import ClosedLoopAssuranceSystem, DimensionalAssuranceEngine
    from apedp.discovery import BayesianRefinementEngine, MultiVariableSRE
    from apedp.discovery_admission import admit_candidate, numeric_conditioning_report
    from apedp.safety import safe_parse_expr

    units = case.variable_units or {}
    engine = DimensionalAssuranceEngine()
    for name in dataset.x_train:
        engine.register_variable(name, units.get(name, "dimensionless"))
    system = ClosedLoopAssuranceSystem(assurance_engine=engine)
    search = MultiVariableSRE(system, independent_vars=list(dataset.x_train), target_unit_str=case.property_unit or "dimensionless", guided=not template_priors, template_priors=template_priors or [], seed=seed)
    expression_obj, score, history = search.run_discovery(dataset.x_train, dataset.y_train, generations=generations, pop_size=pop_size, seed=seed)
    if expression_obj is None:
        return {"status": "REFUSED", "reason": "no candidate", "dataset_hash": dataset.dataset_hash, "literature_comparison": "NOT_RUN"}
    expression = safe_parse_expr(str(expression_obj))
    symbols = sorted(expression.free_symbols, key=str)
    predictor = sp.lambdify(symbols, expression, "numpy")
    train_prediction = np.asarray(predictor(*[dataset.x_train[str(symbol)] for symbol in symbols]), dtype=float)
    holdout_prediction = np.asarray(predictor(*[dataset.x_holdout[str(symbol)] for symbol in symbols]), dtype=float)
    train_r2 = BayesianRefinementEngine.compute_r_squared(dataset.y_train, train_prediction)
    holdout_r2 = BayesianRefinementEngine.compute_r_squared(dataset.y_holdout, holdout_prediction)
    dimensions = engine.check_expression_consistency(expression)
    expected = engine.registry(case.property_unit or "dimensionless")
    dimensional = dimensions.dimensionality == expected.dimensionality
    admission = admit_candidate(expression, x_holdout=dataset.x_holdout, y_holdout=dataset.y_holdout, train_r_squared=train_r2, min_holdout_r_squared=0.1, max_r2_drop=0.2, dimensionally_consistent=dimensional, repair_status="ACCEPTED", conditioning=numeric_conditioning_report(expression, dataset.x_holdout))
    return {"status": admission.status.value, "expression": str(expression), "score": float(score), "train_r_squared": train_r2, "holdout_r_squared": holdout_r2, "dimensional_validation": {"consistent": dimensional, "computed": str(dimensions), "expected": str(expected)}, "admission": admission.to_dict(), "history": history, "dataset_hash": dataset.dataset_hash, "literature_comparison": "NOT_RUN"}


__all__ = ["NISTWebBookAdapter", "NISTWebBookCase", "apply_noise_ladder", "build_nist_manifest", "run_nist_blind_discovery"]