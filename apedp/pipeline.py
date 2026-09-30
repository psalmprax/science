# Databricks notebook source
"""APEDP Pipeline — end-to-end discovery pipeline: ingest → discover → validate → repair → export."""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp

from apedp.core import ClosedLoopAssuranceSystem, DimensionalAssuranceEngine
from apedp.discovery import BayesianRefinementEngine, check_data_adequacy
from apedp.discovery import MultiVariableSRE as _BaseMultiVariableSRE
from apedp.discovery_admission import admit_candidate, numeric_conditioning_report
from apedp.epistemic import DataPartitionVault
from apedp.metrology import AnalyticalUncertaintyPropagator
from apedp.persistence import ExperimentLog, ModelStore
from apedp.repair import AgnosticClosedLoopAssuranceSystem
from apedp.rust_bridge import RustKernelBridge
from apedp.safety import safe_parse_expr
from apedp.source_audit import SourceMetadata, audit_dataset

logger = logging.getLogger(__name__)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

#: Fraction of unparseable rows above which CSV ingest refuses the file.
#:
#: Chosen, not derived -- and stated as such because the value is load-bearing:
#: a file whose rows are more than a fifth unparseable is far likelier to be the
#: wrong file (bad delimiter, header mis-detected, units written into a numeric
#: column) than to be a clean table with sporadic blanks, and fitting a
#: regression to the valid remnant of a mis-parsed file produces a confident
#: model of a data-loading bug. Overridable per pipeline via
#: ``DiscoveryPipeline(max_bad_row_fraction=...)``.
DEFAULT_MAX_BAD_ROW_FRACTION = 0.2

#: Expression-evaluation backends for :meth:`DiscoveryPipeline._predict_expression`.
#:
#: ``rust``  -- prefer the native ``apedp_kernel``, fall back to SymPy on error.
#: ``python`` -- pin the SymPy fallback, never touching the native kernel.
#: ``auto``   -- same as ``rust``; the default for ordinary runs.
VALID_BACKENDS = frozenset({"auto", "python", "rust"})


class DiscoveryPipeline:
    """End-to-end scientific discovery pipeline.

    Flow: Ingest Data → Dimensional Check → Symbolic Regression →
    Anomaly Repair → Persist Results.

    Limitation: CSV carries no unit metadata, so ingested variables are
    registered as dimensionless. Supply units explicitly via
    ``variable_units`` when dimensional validation matters.
    """

    def __init__(
        self,
        output_dir: str = "pipeline_results",
        max_bad_row_fraction: float = DEFAULT_MAX_BAD_ROW_FRACTION,
        strict_units: bool = False,
        backend: str = "auto",
    ):
        if backend not in VALID_BACKENDS:
            raise ValueError(
                f"backend must be one of {sorted(VALID_BACKENDS)}, got {backend!r}")
        self.backend = backend
        # "python" pins the SymPy fallback; "rust"/"auto" prefer the native
        # kernel and only fall back when it is unavailable or errors.
        self._bridge = RustKernelBridge(force_fallback=(backend == "python"))
        self.max_bad_row_fraction = max_bad_row_fraction
        self.strict_units = strict_units
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.engine = DimensionalAssuranceEngine()
        self.platform = ClosedLoopAssuranceSystem(self.engine)
        self.repair_engine = AgnosticClosedLoopAssuranceSystem()
        self.store = ModelStore(str(self.output_dir / "models"))
        self.experiment = ExperimentLog(str(self.output_dir / "experiments.jsonl"))
        self.propagator = AnalyticalUncertaintyPropagator()
        self._results: list[dict] = []

    def ingest_csv(self, filepath: str, target_column: str) -> dict:
        """Load data from a CSV file.

        Non-numeric cells are NOT silently coerced to 0.0 (the old behavior
        corrupted fits). Rows with unparseable values are skipped and counted;
        if no valid rows remain, or the bad-row fraction exceeds
        ``self.max_bad_row_fraction`` (default
        :data:`DEFAULT_MAX_BAD_ROW_FRACTION`), a ``ValueError`` is raised.
        """
        with open(filepath) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise ValueError(f"No data rows in {filepath}")
        if target_column not in rows[0]:
            raise ValueError(f"Target column '{target_column}' not in {filepath}")

        columns = list(rows[0])
        variables = [c for c in columns if c != target_column]
        data: dict[str, list[float]] = {v: [] for v in variables}
        target_values: list[float] = []
        skipped = 0

        for row in rows:
            parsed: dict[str, float] = {}
            valid = True
            for v in variables:
                try:
                    parsed[v] = float(row[v])
                except (TypeError, ValueError):
                    valid = False
                    break
            try:
                parsed[target_column] = float(row[target_column])
            except (TypeError, ValueError):
                valid = False
            if not valid:
                skipped += 1
                continue
            for v in variables:
                data[v].append(parsed[v])
            target_values.append(parsed[target_column])

        if not target_values:
            raise ValueError(f"No valid numeric rows in {filepath} ({skipped} skipped)")
        bad_frac = skipped / (skipped + len(target_values))
        if bad_frac > self.max_bad_row_fraction:
            raise ValueError(
                f"Too many non-numeric rows in {filepath}: {skipped} skipped "
                f"({bad_frac:.1%} > {self.max_bad_row_fraction:.0%}); raise "
                f"max_bad_row_fraction if the data is known to be dirty"
            )
        if skipped:
            logger.warning("Skipped %d non-numeric rows in %s", skipped, filepath)

        data_arr = {k: np.asarray(v, dtype=float) for k, v in data.items()}
        data_arr[target_column] = np.asarray(target_values, dtype=float)

        logger.info("Ingested %d rows, %d variables from %s", len(target_values), len(variables), filepath)
        return {"variables": variables, "data": data_arr, "target": target_column}

    def discover(
        self,
        variables: list[str],
        data: dict[str, np.ndarray],
        target_values: np.ndarray,
        output_unit: str = "dimensionless",
        variable_units: dict[str, str] | None = None,
        generations: int = 30,
        pop_size: int = 100,
        min_r_squared: float = 0.1,
    ) -> dict:
        """Run symbolic regression with real BIC/AIC/R² selection.

        Returns penalized MSE plus ``r_squared``/``bic``/``aic`` and a
        ``degenerate`` flag. Degenerate winners are reported, not hidden.
        """
        for var in variables:
            unit_str = (variable_units or {}).get(var, "dimensionless")
            if var not in (variable_units or {}):
                logger.warning(
                    "Variable '%s' has no unit metadata; registering as dimensionless. "
                    "Supply variable_units={'%s': '<unit>'} for dimensional validation.",
                    var,
                    var,
                )
            try:
                self.engine.register_variable(var, unit_str)
            except (ValueError, TypeError, AttributeError) as e:
                logger.warning("Variable registration failed for '%s': %s", var, e)
        sre = _BaseMultiVariableSRE(
            assurance_system=self.platform,
            independent_vars=variables,
            target_unit_str=output_unit,
        )
        adequacy = check_data_adequacy(
            {k: data[k] for k in variables}, target_values
        )
        if adequacy["verdict"] == "INSUFFICIENT":
            return {
                "best_expression": "None",
                "best_score": float("inf"),
                "generations_run": 0,
                "variables": variables,
                "r_squared": float("nan"),
                "bic": float("inf"),
                "aic": float("inf"),
                "degenerate": True,
                "degenerate_reason": "; ".join(adequacy["reasons"]),
                "search_stagnated": False,
                "stagnation": {},
                "data_adequacy": adequacy,
            }
        best_expr, best_score, history = sre.run_discovery(
            x_data={k: data[k] for k in variables},
            y_data=target_values,
            generations=generations,
            pop_size=pop_size,
            seed=42,
        )
        r_squared = float("nan")
        bic = float("inf")
        aic = float("inf")
        degenerate = True
        degenerate_reason = "no candidate"
        if best_expr is not None:
            try:
                y_pred = self._predict_expression(best_expr, {k: data[k] for k in variables})
                scored = BayesianRefinementEngine.score_candidate(best_expr, target_values, y_pred)
                r_squared, bic, aic = scored["r_squared"], scored["bic"], scored["aic"]
                degenerate, degenerate_reason = BayesianRefinementEngine.is_degenerate(
                    target_values, y_pred, best_expr, min_r_squared=min_r_squared
                )
            except (ValueError, TypeError, ArithmeticError) as e:
                degenerate_reason = f"scoring failed: {e}"

        # A stagnated search means the returned model reflects the seeded
        # population only, so callers must not read it as a search result.
        stagnation = dict(sre.stagnation_report)
        if stagnation.get("stagnated"):
            logger.warning(
                "Search stagnated for target unit '%s': no generation improved on the "
                "seeded population (%d generation(s) without improvement).",
                output_unit,
                stagnation.get("generations_without_improvement", 0),
            )

        return {
            "best_expression": str(best_expr) if best_expr is not None else "None",
            "best_score": float(best_score),
            "generations_run": len(history),
            "variables": variables,
            "r_squared": float(r_squared),
            "bic": float(bic),
            "aic": float(aic),
            "degenerate": bool(degenerate),
            "degenerate_reason": str(degenerate_reason),
            "search_stagnated": bool(stagnation.get("stagnated")),
            "stagnation": stagnation,
            "data_adequacy": adequacy,
        }

    def _predict_expression(self, expr: sp.Expr, x_data: dict[str, np.ndarray]) -> np.ndarray:
        var_names = sorted(x_data.keys())
        syms = [sp.Symbol(k) for k in var_names]
        subs = {sp.Symbol(k): v for k, v in self.platform.constants_values.items()}
        substituted = expr.subs(subs)
        args = [np.asarray(x_data[k], dtype=float) for k in var_names]
        if not args:
            # No independent variables: evaluate as scalar constant
            f = sp.lambdify(syms, substituted, modules=["numpy"])
            y_pred = f()
            if not isinstance(y_pred, np.ndarray):
                return np.array([float(y_pred)], dtype=float)
            return np.asarray(y_pred, dtype=float)
        if self.backend != "python":
            # Route through the Rust bridge (native kernel, SymPy fallback) so
            # `backend` actually selects an evaluator. The direct lambdify
            # path stays as a guard for expressions the string round-trip
            # cannot represent.
            try:
                return np.asarray(
                    self._bridge.evaluate_expression(str(substituted), var_names, args),
                    dtype=float,
                )
            except Exception as e:  # noqa: BLE001 - bridge must not break discovery
                logger.warning(
                    "Backend evaluation failed for '%s' (%s); using direct lambdify.",
                    expr, e,
                )
        f = sp.lambdify(syms, substituted, modules=["numpy"])
        y_pred = f(*args)
        if not isinstance(y_pred, np.ndarray):
            y_pred = np.full_like(args[0], float(y_pred), dtype=float)
        return np.asarray(y_pred, dtype=float)

    def validate_dimensions(self, expression: sp.Expr, expected_unit_str: str) -> dict:
        """Check dimensional consistency of a discovered expression."""
        try:
            computed = self.engine.check_expression_consistency(expression)
            expected = self.engine.registry(expected_unit_str)
            consistent = computed.dimensionality == expected.dimensionality
            return {
                "dimensionally_consistent": consistent,
                "computed": str(computed),
                "expected": str(expected),
            }
        except (TypeError, ValueError, NotImplementedError, ArithmeticError) as e:
            return {"dimensionally_consistent": False, "error": str(e)}

    def audit_repair(
        self, expression: sp.Expr, variable_units: dict[str, str] | None = None,
        strict_units: bool | None = None,
    ) -> dict:
        """Stress-test an expression for boundary anomalies and regularize."""
        result = self.repair_engine.autonomous_stress_test_and_repair(
            str(expression), variable_units=variable_units,
            strict_units=self.strict_units if strict_units is None else strict_units,
        )
        return {
            "status": result.get("status"),
            "anomalies": [f"{s}: {k}" for s, k in result.get("anomalies", [])],
            "repaired_expression": str(result.get("repaired_expression", expression)),
            "inferred_unit": str(result.get("inferred_unit", "unknown")),
            "eps_symbols": list(result.get("eps_symbols", [])),
            "distortion": dict(result.get("distortion", {}) or {}),
            "validity_domain": str(result.get("validity_domain", "")),
            "error": result.get("error"),
        }

    def run(
        self,
        csv_path: str,
        target_column: str,
        output_unit: str = "dimensionless",
        variable_units: dict[str, str] | None = None,
        min_r_squared: float = 0.1,
        strict_units: bool | None = None,
        param_uncertainties: dict[str, float] | None = None,
        source_metadata: SourceMetadata | dict[str, Any] | None = None,
    ) -> dict:
        """Execute the full pipeline: ingest → discover → validate → repair → persist.

        Refuses to persist degenerate/failed models: when discovery yields
        no candidate, a non-finite score, or R² below ``min_r_squared``,
        the run returns ``status="REFUSED"`` with no ``model_id`` and no
        ``0`` fallback expression. Repair ``REJECTED`` status is propagated
        the same way instead of being silently recorded.

        When ``param_uncertainties`` is supplied, the persisted result also
        carries a ``metrology`` report (GUM first-order, or Monte Carlo when
        ``has_singularity_risk`` fires) evaluated at the data means.
        """
        logger.info("Starting discovery pipeline for %s (target: %s)", csv_path, target_column)

        ingested = self.ingest_csv(csv_path, target_column)
        variables = ingested["variables"]
        data = ingested["data"]
        target_values = data[target_column]

        source_audit = None
        if source_metadata is not None:
            source = (SourceMetadata(**source_metadata)
                      if isinstance(source_metadata, dict) else source_metadata)
            rows = [
                {column: float(data[column][index]) for column in [*variables, target_column]}
                for index in range(len(target_values))
            ]
            source_audit = audit_dataset(
                rows,
                source,
                required_columns=[*variables, target_column],
            )
            if source_audit.errors:
                return {
                    "status": "REFUSED",
                    "reason": "source audit failed: " + "; ".join(source_audit.errors),
                    "source_audit": source_audit.to_dict(),
                    "variables": variables,
                }

        discovery = self.discover(
            variables, data, target_values, output_unit, variable_units, min_r_squared=min_r_squared
        )
        logger.info("Discovered: %s (score=%.4g)", discovery["best_expression"], discovery["best_score"])

        # Surface a search that never improved on its seeded population: the
        # returned model is then an artefact of initialisation, not of search. The
        # refusal itself is issued below, once the cheaper refusals have reported.
        stagnation = discovery.get("stagnation", {})
        if stagnation.get("stagnated"):
            logger.warning(
                "Discovery added nothing over its seeded population for target '%s' "
                "(normalized score %.3g, %d generation(s) without improvement); "
                "treat the model as initialisation-only.",
                target_column,
                stagnation.get("final_normalized_score", float("nan")),
                stagnation.get("generations_without_improvement", 0),
            )

        if discovery["best_expression"] == "None" or not np.isfinite(discovery["best_score"]):
            result = {
                "status": "REFUSED",
                "reason": "no dimensionally consistent candidate",
                "best_score": discovery["best_score"],
                "variables": variables,
                "search_stagnated": bool(stagnation.get("stagnated")),
                "stagnation": stagnation,
            }
            self.experiment.log(
                experiment_type="discovery_pipeline",
                parameters={"csv_path": csv_path, "target_column": target_column, "output_unit": output_unit},
                results={"status": "REFUSED", "reason": result["reason"]},
                status="refused",
            )
            return result
        if discovery.get("degenerate", False):
            result = {
                "status": "REFUSED",
                "reason": f"degenerate candidate: {discovery.get('degenerate_reason')}",
                "expression": discovery["best_expression"],
                "best_score": discovery["best_score"],
                "r_squared": discovery.get("r_squared"),
                "variables": variables,
                "search_stagnated": bool(stagnation.get("stagnated")),
                "stagnation": stagnation,
            }
            self.experiment.log(
                experiment_type="discovery_pipeline",
                parameters={"csv_path": csv_path, "target_column": target_column, "output_unit": output_unit},
                results={"status": "REFUSED", "reason": result["reason"]},
                status="refused",
            )
            return result

        if stagnation.get("stagnated"):
            # A search that never improved on its own seeded population did not
            # discover anything: the candidate is the initialisation. Persisting it
            # would file an artefact of the seed as a result of search, which is the
            # one thing a discovery pipeline must not do.
            result = {
                "status": "REFUSED",
                "reason": (
                    "search stagnated: the best candidate equals its seeded initialisation "
                    f"({stagnation.get('generations_without_improvement', 0)} generation(s) "
                    "without improvement), so it is not a search result"
                ),
                "expression": discovery["best_expression"],
                "best_score": discovery["best_score"],
                "r_squared": discovery.get("r_squared"),
                "variables": variables,
                "search_stagnated": True,
                "stagnation": stagnation,
            }
            self.experiment.log(
                experiment_type="discovery_pipeline",
                parameters={"csv_path": csv_path, "target_column": target_column, "output_unit": output_unit},
                results={"status": "REFUSED", "reason": result["reason"]},
                status="refused",
            )
            return result

        try:
            expr = safe_parse_expr(discovery["best_expression"])
        except (ValueError, TypeError) as e:
            logger.warning("Could not parse discovered expression: %s", e)
            result = {
                "status": "REFUSED",
                "reason": f"unparseable expression: {e}",
                "search_stagnated": bool(stagnation.get("stagnated")),
                "stagnation": stagnation,
            }
            self.experiment.log(
                experiment_type="discovery_pipeline",
                parameters={"csv_path": csv_path, "target_column": target_column, "output_unit": output_unit},
                results={"status": "REFUSED", "reason": result["reason"]},
                status="refused",
            )
            return result
        # Validation must not depend on ``discover`` having performed a
        # registration side effect; callers may replace discovery or resume a
        # candidate from persistence.
        for variable in variables:
            unit_str = (variable_units or {}).get(variable, "dimensionless")
            try:
                self.engine.register_variable(variable, unit_str)
            except (ValueError, TypeError, AttributeError) as exc:
                result = {
                    "status": "REFUSED",
                    "reason": f"variable registration failed for '{variable}': {exc}",
                    "expression": str(expr),
                    "variables": variables,
                    "search_stagnated": bool(stagnation.get("stagnated")),
                    "stagnation": stagnation,
                }
                self.experiment.log(
                    experiment_type="discovery_pipeline",
                    parameters={"csv_path": csv_path, "target_column": target_column, "output_unit": output_unit},
                    results={"status": "REFUSED", "reason": result["reason"]},
                    status="refused",
                )
                return result
        validation = self.validate_dimensions(expr, output_unit)
        if not validation.get("dimensionally_consistent", False):
            result = {
                "status": "REFUSED",
                "reason": "dimensional validation failed",
                "expression": str(expr),
                "dimensional_validation": validation,
                "variables": variables,
                "search_stagnated": bool(stagnation.get("stagnated")),
                "stagnation": stagnation,
            }
            self.experiment.log(
                experiment_type="discovery_pipeline",
                parameters={"csv_path": csv_path, "target_column": target_column, "output_unit": output_unit},
                results={"status": "REFUSED", "reason": result["reason"]},
                status="refused",
            )
            return result
        repair_result = self.audit_repair(expr, variable_units, strict_units=strict_units)
        # ``UNDETERMINED`` is a kill signal too: the boundary audit could not
        # answer its own question, so the repaired model is unverified. Filing it
        # as a success would be the same mistake as persisting a stagnated search.
        if repair_result.get("status") in ("REJECTED", "UNDETERMINED"):
            if repair_result.get("status") == "REJECTED":
                reason = f"repair rejected: {repair_result.get('error')}"
            else:
                reason = f"repair undetermined: {repair_result.get('validity_domain')}"
            result = {
                "status": "REFUSED",
                "reason": reason,
                "expression": str(expr),
                "dimensional_validation": validation,
                "anomaly_repair": repair_result,
                "search_stagnated": bool(stagnation.get("stagnated")),
                "stagnation": stagnation,
            }
            self.experiment.log(
                experiment_type="discovery_pipeline",
                parameters={"csv_path": csv_path, "target_column": target_column, "output_unit": output_unit},
                results={"status": "REFUSED", "reason": result["reason"]},
                status="refused",
            )
            return result

        model_id = self.store.save_model(
            model_name=f"pipeline_{target_column}",
            expression=expr,
            variables=variables,
            r_squared=float(discovery.get("r_squared", float("nan"))),
            complexity=len(discovery["best_expression"]),
            metadata={
                "validation": validation,
                "repair": repair_result,
                "score": discovery["best_score"],
                "evidence_level": "in_sample_exploratory",
                "scientific_claim": "consistency_screening_only",
                "bic": discovery.get("bic"),
                "aic": discovery.get("aic"),
                "stagnation": stagnation,
                "source_audit": source_audit.to_dict() if source_audit else None,
            },
        )

        metrology: dict | None = None
        if param_uncertainties:
            try:
                from apedp.metrology import has_singularity_risk

                nominal = {k: float(np.mean(data[k])) for k in variables}
                if has_singularity_risk(expr):
                    mc = self.propagator.propagate_uncertainty_monte_carlo(
                        expr, param_values=nominal,
                        param_uncertainties=dict(param_uncertainties),
                    )
                    metrology = {"method": "monte_carlo", **{k: v for k, v in mc.items() if True}}
                else:
                    rep = self.propagator.propagate_uncertainty(
                        expr, param_values=nominal,
                        param_uncertainties=dict(param_uncertainties),
                    )
                    metrology = {
                        "method": "gum",
                        "evaluated_value": rep.evaluated_value,
                        "combined_uncertainty": rep.combined_uncertainty,
                        "expanded_uncertainty_95": rep.expanded_uncertainty_95,
                        "relative_uncertainty": rep.relative_uncertainty,
                    }
            except (ValueError, TypeError, ArithmeticError) as e:
                logger.warning("Metrology passthrough failed (%s)", e)
                metrology = {"method": "failed", "error": str(e)}

        self.experiment.log(
            experiment_type="discovery_pipeline",
            parameters={"csv_path": csv_path, "target_column": target_column, "output_unit": output_unit},
            results={
                "best_score": discovery["best_score"],
                "model_id": model_id,
                "r_squared": discovery.get("r_squared"),
                "status": "PERSISTED",
                "search_stagnated": bool(stagnation.get("stagnated")),
            },
        )

        result = {
            "status": "PERSISTED",
            "model_id": model_id,
            "expression": str(expr),
            "best_score": discovery["best_score"],
            "score_semantics": "penalized MSE (lower is better); not R²",
            "evidence_level": "in_sample_exploratory",
            "scientific_claim": "consistency_screening_only",
            "r_squared": discovery.get("r_squared"),
            "bic": discovery.get("bic"),
            "aic": discovery.get("aic"),
            "variables": variables,
            "dimensional_validation": validation,
            "anomaly_repair": repair_result,
            "metrology": metrology,
            "search_stagnated": bool(stagnation.get("stagnated")),
            "stagnation": stagnation,
            "source_audit": source_audit.to_dict() if source_audit else None,
        }
        self._results.append(result)

        # Write result atomically before returning (durability)
        out_path = self.output_dir / "pipeline_result.json"
        tmp_path = out_path.with_name(f".{out_path.name}.tmp")
        with open(tmp_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        os.replace(tmp_path, out_path)
        logger.info("Pipeline complete. Result saved to %s", out_path)
        return result

    def run_prospective(
        self,
        train_csv: str,
        holdout_csv: str,
        target_column: str,
        output_unit: str = "dimensionless",
        variable_units: dict[str, str] | None = None,
        min_r_squared: float = 0.1,
        max_r2_drop: float = 0.1,
        strict_units: bool | None = None,
        coefficient_magnitude_limit: float | None = None,
        cancellation_ratio_limit: float = 1.0e8,
    ) -> dict:
        """Discover on train, freeze, then evaluate once on a locked holdout.

        The holdout file is hashed before discovery and never passed to
        ``discover()``: ``DataPartitionVault`` raises if the development phase
        touches it. A train-`PERSISTED` model that scores below
        ``min_r_squared`` on holdout, or drops more than ``max_r2_drop``
        vs train, is `REFUSED` and removed from the store — a fit that does
        not generalize is not filed as a discovery. Callers may additionally
        declare coefficient-scale and cancellation limits for the benchmark;
        these are measured on the locked holdout before admission.
        """
        train_hash = _sha256_file(train_csv)
        holdout_hash = _sha256_file(holdout_csv)
        vault = DataPartitionVault(
            development_data_hash=train_hash,
            selection_data_hash=train_hash,
            falsification_data_hash=holdout_hash,
            locked_external_test_hash=holdout_hash,
        )
        vault.verify_isolation(train_hash, "development")

        train_result = self.run(
            train_csv, target_column, output_unit,
            variable_units=variable_units, min_r_squared=min_r_squared,
            strict_units=strict_units,
        )
        base: dict = {
            "train_status": train_result.get("status"),
            "train_hash": train_hash,
            "holdout_hash": holdout_hash,
        }
        if train_result.get("status") != "PERSISTED":
            self.experiment.log(
                experiment_type="prospective_validation",
                parameters={"train_csv": train_csv, "holdout_csv": holdout_csv},
                results={**base, "status": "REFUSED", "reason": "train refused"},
                status="refused",
            )
            return {**base, "status": "REFUSED", "reason": f"train {train_result.get('status')}: {train_result.get('reason')}"}

        expr = safe_parse_expr(str(train_result["expression"]))
        # Re-hash holdout immediately before evaluation to detect tampering
        holdout_hash_now = _sha256_file(holdout_csv)
        if holdout_hash_now != holdout_hash:
            self.store.delete_model(train_result["model_id"])
            reason = "holdout file modified after initial hash (tampering detected)"
            self.experiment.log("prospective_validation", {"train_csv": train_csv, "holdout_csv": holdout_csv}, {**base, "status": "REFUSED", "reason": reason}, "refused")
            return {**base, "status": "REFUSED", "reason": reason}
        holdout = self.ingest_csv(holdout_csv, target_column)
        holdout_vars = list(holdout["variables"])
        if set(holdout_vars) != set(train_result["variables"]):
            self.store.delete_model(train_result["model_id"])
            reason = f"holdout variables {holdout_vars} != train {train_result['variables']}"
            self.experiment.log("prospective_validation", {"train_csv": train_csv}, {**base, "status": "REFUSED", "reason": reason}, "refused")
            return {**base, "status": "REFUSED", "reason": reason}
        y_true = holdout["data"][target_column]
        holdout_x = {k: holdout["data"][k] for k in holdout_vars}
        y_pred = self._predict_expression(expr, holdout_x)
        holdout_r2 = BayesianRefinementEngine.compute_r_squared(y_true, y_pred)
        train_r2 = float(train_result.get("r_squared", float("nan")))
        drop = (train_r2 - holdout_r2) if np.isfinite(train_r2) and np.isfinite(holdout_r2) else float("inf")
        admission = admit_candidate(
            expr,
            x_holdout=holdout_x,
            y_holdout=y_true,
            train_r_squared=train_r2,
            min_holdout_r_squared=min_r_squared,
            max_r2_drop=max_r2_drop,
            dimensionally_consistent=bool(train_result.get("dimensional_validation", {}).get("dimensionally_consistent")),
            repair_status=train_result.get("anomaly_repair", {}).get("status"),
            conditioning=numeric_conditioning_report(
                expr,
                holdout_x,
                coefficient_magnitude_limit=coefficient_magnitude_limit,
                cancellation_ratio_limit=cancellation_ratio_limit,
            ),
        )
        if admission.status.value != "PERSISTED":
            self.store.delete_model(train_result["model_id"])
            reason = "; ".join(admission.reasons)
            refusal = {**base, "status": "REFUSED", "reason": reason, "holdout_r_squared": holdout_r2, "train_r_squared": train_r2, "admission": admission.to_dict()}
            self.experiment.log("prospective_validation", {"train_csv": train_csv}, refusal, "refused")
            return refusal
        result = {
            **base, "status": "PERSISTED", "model_id": train_result["model_id"],
            "expression": train_result["expression"], "train_r_squared": train_r2,
            "holdout_r_squared": holdout_r2, "r2_drop": drop,
            "evidence_level": "locked_holdout_prospective",
            "scientific_claim": "prospective_generalization_on_one_holdout",
            "admission": admission.to_dict(),
        }
        self.experiment.log("prospective_validation", {"train_csv": train_csv, "holdout_csv": holdout_csv}, result, "success")
        return result


def run_pipeline(
    csv_path: str,
    target_column: str,
    output_unit: str = "dimensionless",
    variable_units: dict[str, str] | None = None,
    min_r_squared: float = 0.1,
    strict_units: bool = False,
    output_dir: str = "pipeline_results",
    param_uncertainties: dict[str, float] | None = None,
    source_metadata: SourceMetadata | dict[str, Any] | None = None,
) -> dict:
    """Convenience function to run a full discovery pipeline."""
    pipeline = DiscoveryPipeline(output_dir=output_dir, strict_units=strict_units)
    return pipeline.run(
        csv_path, target_column, output_unit,
        variable_units=variable_units, min_r_squared=min_r_squared,
        strict_units=strict_units, param_uncertainties=param_uncertainties,
        source_metadata=source_metadata,
    )


# COMMAND ----------

# Use generic family proposals automatically for compatible pipeline discovery.
from apedp.family_guided_discovery import FamilyGuidedMultiVariableSRE

# Intentional notebook-cell swap: the base engine is used under its alias above
# this cell; downstream code should use the family-guided engine by default.
MultiVariableSRE = FamilyGuidedMultiVariableSRE
