"""APEDP Pipeline — end-to-end discovery pipeline: ingest → discover → validate → repair → export."""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import numpy as np
import sympy as sp

from apedp.core import ClosedLoopAssuranceSystem, DimensionalAssuranceEngine
from apedp.discovery import BayesianRefinementEngine, MultiVariableSRE
from apedp.metrology import AnalyticalUncertaintyPropagator
from apedp.persistence import ExperimentLog, ModelStore
from apedp.repair import AgnosticClosedLoopAssuranceSystem
from apedp.safety import safe_parse_expr

logger = logging.getLogger(__name__)

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
    ):
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
        sre = MultiVariableSRE(
            assurance_system=self.platform,
            independent_vars=variables,
            target_unit_str=output_unit,
        )
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
        }

    def _predict_expression(self, expr: sp.Expr, x_data: dict[str, np.ndarray]) -> np.ndarray:
        var_names = sorted(x_data.keys())
        syms = [sp.Symbol(k) for k in var_names]
        subs = {sp.Symbol(k): v for k, v in self.platform.constants_values.items()}
        f = sp.lambdify(syms, expr.subs(subs), modules=["numpy"])
        args = [np.asarray(x_data[k], dtype=float) for k in var_names]
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
    ) -> dict:
        """Execute the full pipeline: ingest → discover → validate → repair → persist.

        Refuses to persist degenerate/failed models: when discovery yields
        no candidate, a non-finite score, or R² below ``min_r_squared``,
        the run returns ``status="REFUSED"`` with no ``model_id`` and no
        ``0`` fallback expression. Repair ``REJECTED`` status is propagated
        the same way instead of being silently recorded.
        """
        logger.info("Starting discovery pipeline for %s (target: %s)", csv_path, target_column)

        ingested = self.ingest_csv(csv_path, target_column)
        variables = ingested["variables"]
        data = ingested["data"]
        target_values = data[target_column]

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
        validation = self.validate_dimensions(expr, output_unit)
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
                "bic": discovery.get("bic"),
                "aic": discovery.get("aic"),
                "stagnation": stagnation,
            },
        )

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
            "r_squared": discovery.get("r_squared"),
            "bic": discovery.get("bic"),
            "aic": discovery.get("aic"),
            "variables": variables,
            "dimensional_validation": validation,
            "anomaly_repair": repair_result,
            "search_stagnated": bool(stagnation.get("stagnated")),
            "stagnation": stagnation,
        }
        self._results.append(result)

        out_path = self.output_dir / "pipeline_result.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        logger.info("Pipeline complete. Result saved to %s", out_path)
        return result


def run_pipeline(
    csv_path: str,
    target_column: str,
    output_unit: str = "dimensionless",
    variable_units: dict[str, str] | None = None,
    min_r_squared: float = 0.1,
    strict_units: bool = False,
    output_dir: str = "pipeline_results",
) -> dict:
    """Convenience function to run a full discovery pipeline."""
    pipeline = DiscoveryPipeline(output_dir=output_dir, strict_units=strict_units)
    return pipeline.run(
        csv_path, target_column, output_unit,
        variable_units=variable_units, min_r_squared=min_r_squared,
        strict_units=strict_units,
    )
