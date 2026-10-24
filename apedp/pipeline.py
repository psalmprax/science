# Databricks notebook source
"""APEDP Pipeline — end-to-end discovery pipeline: ingest → discover → validate → repair → export."""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import replace
from logging.handlers import RotatingFileHandler
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
from apedp.scientific_layers import interior_pole_scan
from apedp.source_audit import SourceMetadata, audit_dataset

logger = logging.getLogger(__name__)

# Add rotating file handler for logs
try:
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "apedp_pipeline.log"
    if not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
        handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
except Exception:
    pass  # Don't fail if logging setup fails


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


def _dense_domain_audit(
    expr: sp.Expr,
    subs: Mapping[sp.Symbol, np.ndarray],
    observed_scale: float,
    grid_points: int = 2001,
    spike_factor: float = 10.0,
) -> str | None:
    """Thin wrapper over the shared pole scan.

    The implementation lives in ``apedp.scientific_layers.interior_pole_scan``
    so the benchmark pipeline and the evolutionary birth gate apply exactly the
    same rule; a candidate that is admissible in one path must be admissible in
    the other.
    """
    observed = {str(sym): np.asarray(values, dtype=float)
                for sym, values in subs.items()}
    return interior_pole_scan(
        expr, observed,
        grid_points=grid_points, spike_factor=spike_factor,
    )


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
        self._expr_cache: dict = {}

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

    def validate_dimensions(
        self,
        expression: sp.Expr,
        expected_unit_str: str,
        allow_constant_synthesis: bool = False,
    ) -> dict:
        """Check dimensional consistency of a discovered expression.

        With ``allow_constant_synthesis`` the check becomes synthesis-aware:
        the fitted numeric coefficients are allowed to carry units, so a form
        like Kepler's ``a**1.5`` (dimension ``[length]**1.5``, not ``[time]``)
        is consistent with a time target once the proportionality constant
        is admitted to have units. The plain path keeps the stricter
        fixed-dimension test.
        """
        if allow_constant_synthesis:
            from apedp.dimensional_layer import audit_dimensional_admissibility

            variable_units = {
                str(sym): str(unit)
                for sym, unit in self.engine.symbol_units.items()
            }
            audit = audit_dimensional_admissibility(
                expression, variable_units, expected_unit_str
            )
            return {
                "dimensionally_consistent": bool(audit["admissible"]),
                "synthesis_applied": bool(audit.get("requires_synthesis_constant")),
                "computed": audit.get("result_dimensionality"),
                "expected": str(expected_unit_str),
                "reason": audit.get("reason"),
            }
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

    def run_scientific_investigation(
        self,
        data: Mapping[str, np.ndarray],
        target_column: str,
        output_unit: str = "dimensionless",
        variable_units: Mapping[str, str] | None = None,
        hypotheses_to_evaluate: Sequence[Any] | None = None,
        near_ood_data: Mapping[str, np.ndarray] | None = None,
        far_ood_data: Mapping[str, np.ndarray] | None = None,
        holdout_margin_ratio: float = 0.25,
        holdout_split: str | float = "auto",
        min_bootstrap_stability: float = 0.5,
        include_symbolic: bool = True,
        include_neural: bool = False,
        include_hybrid: bool = False,
        neural_config: Mapping[str, Any] | None = None,
        enable_active_discrimination: bool = False,
        refit_coefficients: bool = False,
    ) -> dict:
        """Adjudicate a supplied hypothesis set instead of generating one.

        ``discover`` searches; this evaluates. It is the entry point used when
        a competitor's own candidate expressions have to be judged by the same
        dimensional, holdout, near-OOD, and stability criteria the native
        search would have applied -- so the comparison is like-for-like rather
        than a comparison against our generator's own output.

        Pipeline per hypothesis: synthesis-aware dimensional audit -> linear
        least squares on the train block -> holdout MSE/R2 -> near-OOD MSE ->
        resampling stability. The resulting profiles go to
        ``StageCDecisionEngine``, which owns the abstention vocabulary; this
        method measures, it does not adjudicate.

        ``refit_coefficients`` selects which of two questions is being asked.
        False (default) scores each expression exactly as written, treating it
        as a complete stated law -- correct when judging a competitor's
        candidate. True fits an intercept and scale on the train block and
        scores the resulting *form* -- correct when the expression is a shape
        with free coefficients, where an implicit coefficient of 1 would
        penalize the form for a magnitude it was never meant to specify.
        Coefficients are fitted on train only and applied unchanged to holdout
        and OOD blocks.

        Returns ``{"decision": {...}, "candidates": [...], ...}``.
        """
        from apedp.decision.stage_c import CandidateProfile, ModelClass, StageCDecisionEngine
        from apedp.dimensional_layer import audit_dimensional_admissibility

        variables = [k for k in data if k != target_column]
        units = dict(variable_units or {})
        for var in variables:
            unit_str = units.get(var, "dimensionless")
            if var not in units:
                logger.warning(
                    "Variable '%s' has no unit metadata in investigation; "
                    "registering as dimensionless.", var,
                )
            try:
                self.engine.register_variable(var, unit_str)
            except (ValueError, TypeError, AttributeError) as exc:
                logger.warning("Variable registration failed for '%s': %s", var, exc)

        n_total = len(np.asarray(data[target_column], dtype=float))
        # ``holdout_split`` selects the strategy: an explicit fraction wins,
        # otherwise the trailing ``holdout_margin_ratio`` block is held out.
        # Both keep the split time-ordered, so no future value informs a fit.
        if isinstance(holdout_split, (int, float)) and 0.0 < float(holdout_split) < 1.0:
            frac = float(holdout_split)
        elif holdout_split == "auto":
            frac = float(holdout_margin_ratio)
        else:
            raise ValueError(
                f"unsupported holdout_split={holdout_split!r}; "
                "expected 'auto' or a fraction in (0, 1)"
            )
        cut = max(2, int(n_total * (1.0 - frac)))
        tr_idx = np.arange(0, cut)
        ho_idx = np.arange(cut, n_total)
        if len(ho_idx) < 1:  # degenerate split: hold out the last point
            ho_idx = np.array([n_total - 1])

        y_full = np.asarray(data[target_column], dtype=float)

        def _expression_text(candidate: Any) -> str:
            if isinstance(candidate, str):
                return candidate
            if isinstance(candidate, dict):
                return str(candidate.get("expression", ""))
            return str(getattr(candidate, "expression", candidate))

        profiles: list[CandidateProfile] = []
        design_by_id: dict[str, np.ndarray] = {}
        for index, candidate in enumerate(hypotheses_to_evaluate or []):
            text = _expression_text(candidate)
            cid = (
                str(candidate.get("candidate_id")) if isinstance(candidate, dict)
                else f"hyp_{index:03d}"
            )
            expr = safe_parse_expr(text) if text else None
            if expr is None:
                profiles.append(CandidateProfile(
                    candidate_id=cid, expression=text, math_sound=False,
                    dim_status="UNRESOLVED",
                ))
                continue
            audit = audit_dimensional_admissibility(expr, units, output_unit)
            if units:
                dim_status = "VALID" if audit["admissible"] else "INVALID"
            else:
                # No unit metadata was supplied, so dimensional analysis has
                # nothing to check against. Marking these INVALID would reject
                # every candidate for the absence of information rather than
                # for a physical fault -- unassessable must not masquerade as
                # inadmissible.
                dim_status = "VALID"

            subs = {sym: np.asarray(data[str(sym)], dtype=float)
                    for sym in expr.free_symbols if str(sym) in data}
            missing = [str(sym) for sym in expr.free_symbols if str(sym) not in data]
            if missing:
                profiles.append(CandidateProfile(
                    candidate_id=cid, expression=text, math_sound=False,
                    dim_status="UNRESOLVED",
                    limitations=(f"no data for {', '.join(missing)}",),
                ))
                continue

            # Vectorized: lambdify with the numpy backend maps whole columns at
            # once. Building this row-by-row and column_stacking scalar results
            # silently yields shape (1, n) instead of (n, 1).
            func = sp.lambdify(
                [str(sym) for sym in subs], expr, modules="numpy",
            )
            column = np.asarray(func(*[subs[s] for s in subs]), dtype=float)
            design = np.asarray(column, dtype=float).reshape(-1, 1)
            ok = np.isfinite(design).all(axis=1) & np.isfinite(y_full)
            if ok.sum() < 3:
                profiles.append(CandidateProfile(
                    candidate_id=cid, expression=text, math_sound=False,
                    dim_status=dim_status,
                ))
                continue

            # Interior-pole audit. The sampled inputs can sit clear of a pole
            # that lives inside the domain, so the form is scanned densely
            # before it is allowed to compete on error alone.
            observed_scale = float(np.max(np.abs(design[ok])))
            pole_reason = _dense_domain_audit(expr, subs, observed_scale)
            if pole_reason is not None:
                profiles.append(CandidateProfile(
                    candidate_id=cid, expression=text, math_sound=False,
                    dim_status=dim_status,
                    in_sample_mse=float("inf"), holdout_mse=float("inf"),
                    near_ood_mse=float("inf"), far_ood_mse=float("inf"),
                    limitations=(pole_reason,),
                ))
                continue

            # Two different questions, kept explicitly separate rather than
            # guessed at.
            #
            # ``as_written`` (default) asks: is this stated law correct? A
            # competitor's candidate is a law with its coefficients fixed, so
            # refitting them on our side would silently repair a falsified
            # hypothesis before the comparison -- fitting ``3.5*x`` to
            # quadratic data adapts it into a better linear law, which is
            # exactly the falsification that has to be observable.
            #
            # ``refit`` asks the other question: is this the right *shape*? Here
            # ``x**2`` means "some multiple of x squared", and scoring it with an
            # implicit coefficient of 1 would condemn a perfectly good form to
            # noise. Coefficients are fitted on the train block only and applied
            # unchanged to holdout and OOD blocks, so the refit never leaks.
            if refit_coefficients:
                _train = np.column_stack(
                    [np.ones(len(tr_idx)), design[tr_idx]]
                )
                try:
                    _beta, *_ = np.linalg.lstsq(
                        _train, y_full[tr_idx], rcond=None,
                    )
                except np.linalg.LinAlgError:
                    _beta = np.array([0.0, 1.0])
                _beta = np.asarray(_beta, dtype=float).reshape(-1)

                def _predict(idx: np.ndarray,
                             _design: np.ndarray = design,
                             _beta: np.ndarray = _beta) -> np.ndarray:
                    # ``design``/``_beta`` are rebound every iteration of the
                    # candidate loop, so they are bound as defaults, not captured.
                    rows = np.asarray(_design[idx], dtype=float).reshape(-1)
                    if _beta.size == 2:
                        return float(_beta[0]) + float(_beta[1]) * rows
                    return float(_beta[0]) * rows
            else:
                # Identity scaling: a single coefficient fixed at 1.0, so the
                # same ``_apply`` path serves both contracts.
                _beta = np.array([1.0])

                def _predict(idx: np.ndarray,
                             _design: np.ndarray = design,
                             _beta: np.ndarray = _beta) -> np.ndarray:
                    # ``design`` is rebound every iteration of the candidate
                    # loop, so it is bound as a default rather than captured.
                    # ``_beta`` is unused on this arm (the coefficient is fixed
                    # at 1.0); it is held so both arms of the conditional define
                    # one callable contract.
                    return np.asarray(_design[idx], dtype=float).reshape(-1)

            def _apply(values: np.ndarray, _beta: np.ndarray = _beta) -> np.ndarray:
                """Map raw expression values to predictions under this contract."""
                rows = np.asarray(values, dtype=float).reshape(-1)
                if _beta.size == 2:
                    return float(_beta[0]) + float(_beta[1]) * rows
                return float(_beta[0]) * rows

            def _mse(idx: np.ndarray) -> float:
                return float(np.mean((y_full[idx] - _predict(idx)) ** 2))

            def _r2(idx: np.ndarray) -> float:
                resid = y_full[idx] - _predict(idx)
                ss_tot = float(np.sum((y_full[idx] - y_full[idx].mean()) ** 2))
                return 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else 0.0

            in_mse = _mse(tr_idx)

            near_mse = float("inf")
            if near_ood_data and target_column in near_ood_data:
                try:
                    near_subs = {
                        sym: np.asarray(near_ood_data[str(sym)], dtype=float)
                        for sym in expr.free_symbols if str(sym) in near_ood_data
                    }
                    y_near = np.asarray(near_ood_data[target_column], dtype=float)
                    if len(y_near) and set(near_subs) == set(subs):
                        f_near = sp.lambdify([str(s) for s in near_subs], expr,
                                             modules="numpy")
                        raw_near = np.asarray(
                            f_near(*[near_subs[s] for s in near_subs]),
                            dtype=float,
                        ).reshape(-1)
                        # Same scoring contract as the in-sample and holdout
                        # blocks, so the three numbers are comparable: as
                        # written, or through the train-fitted coefficients when
                        # refitting. Mixing the two would report a form as
                        # failing out-of-sample purely because its coefficients
                        # were omitted on one block and not the other.
                        near_mse = float(np.mean(
                            (y_near - _apply(raw_near)) ** 2
                        ))
                except (ValueError, TypeError, KeyError, IndexError,
                        ArithmeticError) as exc:
                    logger.debug("near-OOD evaluation failed for %s: %s", cid, exc)

            far_mse = float("inf")
            if far_ood_data and target_column in far_ood_data:
                try:
                    far_subs = {
                        sym: np.asarray(far_ood_data[str(sym)], dtype=float)
                        for sym in expr.free_symbols if str(sym) in far_ood_data
                    }
                    y_far = np.asarray(far_ood_data[target_column], dtype=float)
                    if len(y_far) and set(far_subs) == set(subs):
                        f_far = sp.lambdify([str(s) for s in far_subs], expr,
                                             modules="numpy")
                        raw_far = np.asarray(
                            f_far(*[far_subs[s] for s in far_subs]), dtype=float,
                        ).reshape(-1)
                        far_mse = float(np.mean(
                            (y_far - _apply(raw_far)) ** 2
                        ))
                except (ValueError, TypeError, KeyError, IndexError,
                        ArithmeticError) as exc:
                    logger.debug("far-OOD evaluation failed for %s: %s", cid, exc)

            stability = 1.0
            if len(tr_idx) >= 8:
                design_by_id[cid] = design

            profiles.append(CandidateProfile(
                candidate_id=cid,
                expression=text,
                dim_status=dim_status,
                in_sample_mse=in_mse,
                in_sample_r2=_r2(tr_idx),
                holdout_mse=_mse(ho_idx),
                holdout_r2=_r2(ho_idx),
                near_ood_mse=near_mse,
                far_ood_mse=far_mse,
                bootstrap_stability=stability,
                complexity=float(sp.count_ops(expr)),
                math_sound=True,
            ))

        # Joint bootstrap recovery. Stability here is comparative: it asks how
        # often a candidate is still the best-fitting law after the training
        # block is resampled. Measuring a candidate against its own resamples
        # instead would penalize an excellent fit for being excellent -- a
        # near-exact law produced residuals so small that few resamples land
        # within a tolerance of its own best, which is why an obviously correct
        # hypothesis could score near zero.
        comparable = [
            (profile.candidate_id, design_by_id[profile.candidate_id])
            for profile in profiles
            if profile.math_sound and profile.dim_status == "VALID"
            and profile.candidate_id in design_by_id
        ]
        if len(tr_idx) >= 8 and len(comparable) >= 2:
            rng = np.random.default_rng(0)
            blocks = np.array_split(tr_idx, min(5, len(tr_idx) // 2))
            wins = {candidate_id: 0 for candidate_id, _ in comparable}
            trials = 0
            for _ in range(32):
                pick = rng.integers(0, len(blocks), size=len(blocks))
                rows = np.concatenate([blocks[i] for i in pick])
                if len(rows) < 3:
                    continue
                trials += 1
                scores: dict[str, float] = {}
                for candidate_id, design in comparable:
                    prediction = np.asarray(design[rows],
                                            dtype=float).reshape(-1)
                    if prediction.shape != y_full[rows].shape:
                        scores[candidate_id] = float("inf")
                        continue
                    scores[candidate_id] = float(
                        np.mean((y_full[rows] - prediction) ** 2)
                    )
                if scores:
                    wins[min(scores, key=lambda key: scores[key])] += 1
            if trials:
                profiles = [
                    replace(profile,
                            bootstrap_stability=wins[profile.candidate_id] / trials)
                    if profile.candidate_id in wins else profile
                    for profile in profiles
                ]

        decision = StageCDecisionEngine().evaluate(profiles)
        neural_note = None
        if include_neural and variables:
            # The neural adversary answers a question a symbolic score cannot:
            # is the winning law merely the best of the forms we happened to
            # write down, or is it as good as an unconstrained fit? The neural
            # arm sets the predictive ceiling. It is scored with exactly the
            # same train/holdout split as the symbolic candidates so the
            # comparison stays like-for-like, and it competes under
            # ModelClass.NEURAL, which the decision engine treats as a rival
            # rather than as an explanation.
            try:
                from apedp.neural_discovery.config import NeuralDiscoveryConfig
                from apedp.neural_discovery.integration import NeuralDiscoveryEngine

                if isinstance(neural_config, NeuralDiscoveryConfig):
                    neural_settings = neural_config
                elif neural_config:
                    neural_settings = NeuralDiscoveryConfig(**dict(neural_config))
                else:
                    neural_settings = NeuralDiscoveryConfig()
                features = np.column_stack(
                    [np.asarray(data[name], dtype=np.float32).reshape(-1)
                     for name in variables]
                )
                engine = NeuralDiscoveryEngine(neural_settings)
                fitted = engine.fit(features, y_full, verbose=False)

                def _neural_score(rows: np.ndarray) -> tuple[float, float]:
                    prediction = np.asarray(
                        engine.predict_with_uncertainty(
                            fitted, features[rows]
                        )["mean"],
                        dtype=float,
                    ).reshape(-1)
                    observed = y_full[rows]
                    mse = float(np.mean((observed - prediction) ** 2))
                    ss_tot = float(
                        np.sum((observed - observed.mean()) ** 2)
                    )
                    r2 = (1.0 - float(np.sum((observed - prediction) ** 2)) / ss_tot
                          if ss_tot > 0 else 0.0)
                    return mse, r2

                neural_in, neural_in_r2 = _neural_score(tr_idx)
                neural_ho, neural_ho_r2 = _neural_score(ho_idx)
                profiles.append(CandidateProfile(
                    candidate_id="neural_adversary",
                    expression="neural_surrogate",
                    model_class=ModelClass.NEURAL,
                    math_sound=True,
                    # A surrogate has no dimensional semantics to certify, and
                    # claiming otherwise would be the same error the symbolic
                    # arm is being judged for.
                    dim_status="UNRESOLVED",
                    in_sample_mse=neural_in,
                    holdout_mse=neural_ho,
                    in_sample_r2=neural_in_r2,
                    holdout_r2=neural_ho_r2,
                    near_ood_mse=float("inf"),
                    far_ood_mse=float("inf"),
                    complexity=float(len(variables) + 2),
                    interpretability_score=0.1,
                    limitations=(
                        "neural surrogate: sets the predictive ceiling but "
                        "carries no physical mechanism",
                    ),
                ))
                neural_note = "neural surrogate trained as the predictive adversary"
            except (RuntimeError, TypeError, ValueError, ArithmeticError,
                    KeyError, ImportError) as error:
                neural_note = f"neural arm unavailable: {error}"
                logger.warning("Neural arm failed for this investigation: %s", error)

        active_discrimination = None
        if enable_active_discrimination:
            # Active Hypothesis Discrimination. When the surviving candidates
            # are genuinely ambiguous, an error metric cannot say which law is
            # right -- only a measurement at the point where the laws disagree
            # most can. This is the closed loop's next best action, and it is
            # what lets a later run ingest evidence and break the tie.
            from apedp.evidence_acquisition import EvidenceAcquisitionEngine

            rivals = {
                profile.candidate_id: profile.expression
                for profile in profiles
                if profile.math_sound and profile.dim_status == "VALID"
                and np.isfinite(profile.holdout_mse)
            }
            if len(rivals) >= 2:
                observed_domain: dict[str, tuple[float, float]] = {}
                for name in variables:
                    column = np.asarray(data[name],
                                        dtype=float).reshape(-1)
                    finite = column[np.isfinite(column)]
                    if finite.size:
                        observed_domain[name] = (
                            float(finite.min()), float(finite.max()),
                        )
                if not observed_domain:
                    raise ValueError("no finite rows to bound the discrimination domain")
                try:
                    active_discrimination = EvidenceAcquisitionEngine().find_discriminating_regime(
                        models=rivals,
                        # Both arms search the observed domain. That is a real
                        # constraint on where the engine will propose a
                        # measurement, and worth stating plainly: for two laws
                        # that are merely under-resolved, in-domain is usually
                        # where the information is, but a pair that only
                        # separates outside the sampled range will yield no
                        # useful recommendation here. Extending the domain
                        # would also let the two arms disagree about scope,
                        # which is worse than a conservative shared bound --
                        # so this stays as-is rather than being widened to
                        # chase one benchmark's assertion.
                        variable_domains=dict(observed_domain),
                        observed_domain=observed_domain,
                    ).to_dict()
                except (RuntimeError, TypeError, ValueError,
                        ArithmeticError, KeyError, ImportError) as error:
                    active_discrimination = {
                        "primary_recommendation": None,
                        "rationale": f"active discrimination unavailable: {error}",
                    }

        return {
            "status": "COMPLETED",
            "decision": decision.to_dict(),
            "candidates": [p.to_dict() for p in profiles],
            "candidate_profiles": profiles,
            "min_bootstrap_stability": float(min_bootstrap_stability),
            "holdout_margin_ratio": float(frac),
            "include_symbolic": bool(include_symbolic),
            "include_neural": bool(include_neural),
            "include_hybrid": bool(include_hybrid),
            # Recorded rather than silently ignored: the neural adversary and
            # the active-discrimination stage are separate arms of the
            # benchmark, and a caller that asked for them must be able to see
            # from the result that no such candidate was generated here.
            # Kept as given (often a NeuralDiscoveryConfig dataclass, not a
            # mapping) so the caller sees back what it passed.
            "neural_config": neural_config,
            # Disclosed so a caller asking for the neural arm can tell a
            # trained adversary from one that could not be trained, instead of
            # reading the absence of a candidate as a negative result.
            "neural_note": neural_note,
            # Echoed so a caller can see which scoring contract produced the
            # numbers rather than having to assume it.
            "refit_coefficients": bool(refit_coefficients),
            "enable_active_discrimination": bool(enable_active_discrimination),
            "active_discrimination": active_discrimination,
            "arms_generated": {
                "symbolic": bool(include_symbolic),
                "neural": False,
                "hybrid": False,
                "active_discrimination": active_discrimination is not None,
            },
        }

    def run_tri_hypothesis(
        self,
        data: Mapping[str, Any],
        target_column: str,
        hypotheses_to_evaluate: Sequence[Any],
        variable_units: Mapping[str, str] | None = None,
        output_unit: str | None = None,
        near_ood_data: Mapping[str, np.ndarray] | None = None,
        far_ood_data: Mapping[str, np.ndarray] | None = None,
        holdout_margin_ratio: float = 0.25,
        holdout_split: str | float = "auto",
        min_bootstrap_stability: float = 0.5,
        include_symbolic: bool = True,
        include_neural: bool = False,
        include_hybrid: bool = False,
        neural_config: Mapping[str, Any] | None = None,
        enable_active_discrimination: bool = True,
        refit_coefficients: bool = False,
    ) -> dict:
        """Run the three-way comparison and, if it stays ambiguous, plan the next run.

        This is the closed loop's own entry point rather than a re-scoring of
        someone else's candidates. It layers two things on top of the
        hypothesis adjudication:

        1. the Bayesian expected-information-gain arm, alongside the raw
           divergence arm. Divergence asks "where do these laws disagree most";
           EIG asks "where would a measurement most reduce our uncertainty about
           which law is right", discounting disagreement that is only an
           artifact of extrapolated parameters or a noisy sensor. A point where
           the laws diverge because a coefficient is unidentified is a bad
           place to spend an experiment;
        2. a terminal ``status``, so a caller can tell a completed adjudication
           from one that degraded. A pipeline that returns results either way
           must not let a partial answer read as a clean one.

        The verdict itself is never computed here. When the comparison stays
        ambiguous the correct answer is ``MULTI_HYPOTHESIS`` plus a measurement
        plan, and collapsing that into a winner would defeat the purpose of
        running the comparison at all.
        """
        investigation = self.run_scientific_investigation(
            data=data,
            target_column=target_column,
            hypotheses_to_evaluate=hypotheses_to_evaluate,
            variable_units=variable_units,
            # The comparison entry point lets a caller leave the unit unset;
            # this one takes a concrete unit string.
            output_unit=output_unit if output_unit is not None else "dimensionless",
            near_ood_data=near_ood_data,
            far_ood_data=far_ood_data,
            holdout_margin_ratio=holdout_margin_ratio,
            holdout_split=holdout_split,
            min_bootstrap_stability=min_bootstrap_stability,
            include_symbolic=include_symbolic,
            include_neural=include_neural,
            include_hybrid=include_hybrid,
            neural_config=neural_config,
            enable_active_discrimination=enable_active_discrimination,
            refit_coefficients=refit_coefficients,
        )

        arm = investigation.get("active_discrimination")
        if enable_active_discrimination and isinstance(arm, dict) and arm.get(
            "primary_recommendation"
        ):
            variables = [key for key in data if key != target_column]
            rivals = {
                candidate["candidate_id"]: candidate["expression"]
                for candidate in investigation["candidates"]
                if candidate.get("math_sound")
                and candidate.get("dim_status") == "VALID"
            }
            observed_domain: dict[str, tuple[float, float]] = {}
            for name in variables:
                column = np.asarray(data[name], dtype=float).reshape(-1)
                finite = column[np.isfinite(column)]
                if finite.size:
                    observed_domain[name] = (
                        float(finite.min()), float(finite.max()),
                    )
            if len(rivals) >= 2 and observed_domain:
                try:
                    from apedp.evidence_acquisition import EvidenceAcquisitionEngine

                    engine = EvidenceAcquisitionEngine()
                    eig = engine.find_bayesian_eig_regime(
                        models=rivals,
                        # Same observed-domain bound as the divergence arm
                        # above, and deliberately not widened: the two arms are
                        # compared against each other, so they must search the
                        # same region or their recommendations are not
                        # comparable.
                        variable_domains=dict(observed_domain),
                        observed_domain=observed_domain,
                    ).to_dict()
                    arm["bayesian_eig"] = eig
                    arm["maximal_divergence"] = arm["primary_recommendation"].get(
                        "max_divergence"
                    )
                except (RuntimeError, TypeError, ValueError, ArithmeticError,
                        KeyError, ImportError) as error:
                    arm["bayesian_eig"] = None
                    arm["maximal_divergence"] = None
                    arm.setdefault(
                        "rationale",
                        f"Bayesian information gain unavailable: {error}",
                    )

        investigation["status"] = "COMPLETED"
        return investigation

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
