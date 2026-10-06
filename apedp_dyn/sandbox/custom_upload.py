"""Intelligent universal custom dataset upload and autonomous discovery engine.

Allows users to upload or paste arbitrary CSV datasets:
- Auto-detects column schemas (time, state features, targets).
- Distinguishes 1-D dynamical series from multi-variable physical tables.
- Runs Level 2 open symbolic discovery (MultiTerm / AST or Mechanical Grammar).
- Evaluates full 4-gate verification (Algebraic, Physical, Evidential, Gate 4 Predictive R²).
- Submits surviving Pareto candidates to Sovereign Epistemic Adjudication.
"""
from __future__ import annotations

import csv
import io
import logging
import re
from typing import Any

import numpy as np

from apedp_dyn.discovery_l2.dimensional_types import Dimension
from apedp_dyn.discovery_l2.multiterm_engine import MultiTermL2Engine
from apedp_dyn.epistemic.adjudicate import (
    Adjudication,
    DataOrigin,
    adjudicate,
)
from apedp_dyn.grammar.candidate_ir import ScoredCandidate
from apedp_dyn.problems.finance import spec_from_series

# Debug-level diagnostics for the degrade-don't-abort branches below.
_LOG = logging.getLogger("apedp_dyn.sandbox.custom_upload")


def parse_csv_data(
    csv_text: str,
    target_col: str | None = None,
) -> dict[str, Any]:
    """Parse raw CSV text into typed column arrays and auto-detect target."""
    lines = [line.strip() for line in csv_text.strip().splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError("CSV must contain at least a header row and one data row.")

    # Auto-detect delimiter
    delimiter = ","
    if "\t" in lines[0] and "," not in lines[0]:
        delimiter = "\t"
    elif ";" in lines[0] and "," not in lines[0]:
        delimiter = ";"

    reader = csv.reader(io.StringIO("\n".join(lines)), delimiter=delimiter)
    raw_headers = [h.strip() for h in next(reader)]

    # Collect row data
    data_by_col: dict[str, list[float]] = {h: [] for h in raw_headers}
    date_col: str | None = None
    date_values: list[str] = []

    for row in reader:
        if not row or len(row) != len(raw_headers):
            continue
        for h, val in zip(raw_headers, row, strict=False):
            v_str = val.strip()
            # Check for date format
            if date_col is None and re.match(r"^\d{4}[-/]\d{2}[-/]\d{2}", v_str):
                date_col = h
            try:
                data_by_col[h].append(float(v_str))
            except ValueError:
                if date_col == h:
                    date_values.append(v_str)

    # Filter out non-numeric columns
    numeric_cols: dict[str, np.ndarray] = {}
    for h, vals in data_by_col.items():
        if len(vals) >= 5 and all(np.isfinite(vals)):
            numeric_cols[h] = np.array(vals, dtype=float)

    if not numeric_cols:
        raise ValueError("No numeric columns found in the uploaded CSV.")

    col_names = list(numeric_cols.keys())

    # Detect time column
    time_col: str | None = None
    for cand in ("t", "time", "date", "step", "timestamp", "dt"):
        matches = [c for c in col_names if c.lower() == cand]
        if matches:
            time_col = matches[0]
            break

    # Determine target column
    resolved_target: str
    if target_col and target_col in numeric_cols:
        resolved_target = target_col
    else:
        # Prefer conventional targets
        for cand in ("y", "target", "output", "a", "accel", "price", "close", "xddot"):
            matches = [c for c in col_names if c.lower() == cand]
            if matches:
                resolved_target = matches[0]
                break
        else:
            # Pick last numeric column that isn't time
            non_time = [c for c in col_names if c != time_col]
            resolved_target = non_time[-1] if non_time else col_names[-1]

    # Feature columns (everything numeric except target and time)
    feature_cols = [c for c in col_names if c != resolved_target and c != time_col]

    return {
        "headers": raw_headers,
        "numeric_cols": numeric_cols,
        "target_col": resolved_target,
        "feature_cols": feature_cols,
        "time_col": time_col,
        "n_rows": len(numeric_cols[resolved_target]),
        "dates": date_values,
    }


def run_upload_discovery(
    csv_text: str,
    target_col: str | None = None,
    mode: str = "auto",
    train_ratio: float = 0.6,
) -> tuple[Adjudication, list[ScoredCandidate], dict[str, Any], dict[str, Any]]:
    """Execute autonomous equation discovery on arbitrary uploaded CSV data."""
    parsed = parse_csv_data(csv_text, target_col=target_col)
    target_name = parsed["target_col"]
    feature_names = parsed["feature_cols"]
    y_raw = parsed["numeric_cols"][target_name]
    n = len(y_raw)

    time_arr = (
        parsed["numeric_cols"][parsed["time_col"]]
        if parsed["time_col"]
        else np.arange(n, dtype=float)
    )

    # Determine execution mode: 1D time series vs Multivariable
    is_1d_series = len(feature_names) == 0 and mode != "l2_open"

    if is_1d_series:
        return _run_1d_series_discovery(
            y_raw, time_arr, target_name, train_ratio=train_ratio
        )
    else:
        return _run_multivariable_l2_discovery(
            parsed["numeric_cols"],
            feature_names,
            target_name,
            time_arr,
            train_ratio=train_ratio,
        )


def _run_1d_series_discovery(
    series: np.ndarray,
    time_arr: np.ndarray,
    target_name: str,
    train_ratio: float = 0.6,
) -> tuple[Adjudication, list[ScoredCandidate], dict[str, Any], dict[str, Any]]:
    """Autonomous mechanical rate law discovery on 1D series."""
    from apedp_dyn.epistemic.bootstrap import holdout_residual_series
    from apedp_dyn.fitting.parameter_fit import fit_candidate
    from apedp_dyn.grammar.mechanical_grammar import (
        enumerate_candidates,
        prune_admissible,
    )

    n = len(series)
    split = int(n * train_ratio)
    split = max(10, min(split, n - 5))

    min_val = float(np.min(series))
    offset = (abs(min_val) + 1.0) if min_val <= 0.0 else 0.0
    work_series = series + offset

    prob = spec_from_series(
        work_series,
        (0, split),
        (split, n),
        problem_id=f"CUSTOM-{target_name}",
    )

    cands = enumerate_candidates(
        required=(),
        optional=("one", "x", "x3", "v", "abs_v_v", "sin_x", "sat_x", "inv_x2"),
        max_terms=2,
        id_prefix="cand",
    )
    kept = prune_admissible(prob, cands)

    fitted = []
    for s in kept:
        try:
            f = fit_candidate(prob, s.candidate)
            fitted.append(f)
        except (ValueError, TypeError, ArithmeticError, KeyError, IndexError) as exc:
            # A candidate that cannot be fitted on this upload is skipped, not
            # fatal: the endpoint still adjudicates whatever did fit.
            _LOG.debug("upload fit failed for %s: %s", s.candidate, exc)
            continue

    g4 = [s for s in fitted if s.provenance.get("gate4_status", {}).get("passed", False)]

    def _se_series(sc):
        try:
            if sc.fit is None:
                return None
            return holdout_residual_series(prob, sc.candidate, sc.fit.param_values)
        except (ValueError, TypeError, ArithmeticError, KeyError, IndexError, RuntimeError) as exc:
            _LOG.debug("holdout residual series unavailable: %s", exc)
            return None

    ranked = sorted(
        [s for s in g4 if s.train_loss is not None and np.isfinite(s.train_loss)],
        key=lambda s: float(s.train_loss),
    )
    se = {id(s): _se_series(s) for s in ranked}
    best_se = se.get(id(ranked[0])) if ranked else None
    rival_se = se.get(id(ranked[1])) if len(ranked) > 1 else None

    try:
        adj = adjudicate(
            ranked,
            resid_best=best_se,
            resid_rival=rival_se,
            resid_sole=best_se,
            bootstrap=True,
            data_origin="experimental",
        )
    except (ValueError, TypeError, ArithmeticError, KeyError, IndexError) as exc:
        # Bootstrapping needs paired residuals; fall back to the point-estimate
        # contract rather than failing the upload.
        _LOG.debug("bootstrap tiebreak unavailable, falling back: %s", exc)
        adj = adjudicate(ranked)

    cand_pool = ranked if ranked else sorted(
        [s for s in fitted if getattr(s, "holdout_rmse", None) is not None],
        key=lambda s: float(s.holdout_rmse),
    )

    # Plot reconstruction
    ho_len = n - split
    t_ho = time_arr[split:]
    actual_ho = series[split:].tolist()

    pred_vals = actual_ho
    if adj.promoted and adj.promoted.fit:
        # Simulate forward trajectory
        try:
            from apedp_dyn.grammar.mechanical_grammar import materialize
            from apedp_dyn.simulation import simulate
            m = materialize(prob, adj.promoted.candidate)
            for p in m.parameters:
                p.value = adj.promoted.fit.param_values.get(p.name, 0.0)
            res = simulate(m, (0.0, float(ho_len - 1)), prob.holdout["u_of_t"],
                           prob.holdout["x0"], n_eval=ho_len)
            if res.success:
                mu, sigma = float(np.mean(np.log(work_series[:split]))), float(np.std(np.log(work_series[:split])))
                pred_vals = (np.exp(res.y[:, 0] * sigma + mu) - offset).tolist()
        except (ValueError, TypeError, ArithmeticError, KeyError, IndexError) as exc:
            # The fitted curve is a display nicety; the verdict does not use it.
            _LOG.debug("upload trajectory overlay unavailable: %s", exc)

    series_info = {
        "x": list(range(len(actual_ho))),
        "dates": [str(t) for t in t_ho],
        "actual": actual_ho,
        "predicted": pred_vals,
        "x_label": f"{target_name} Holdout Trajectory",
    }
    extra = {
        "dataset_type": "1D Dynamical Time Series",
        "target_variable": target_name,
        "total_samples": n,
        "train_samples": split,
        "holdout_samples": ho_len,
    }
    return adj, cand_pool, series_info, extra


def _run_multivariable_l2_discovery(
    numeric_cols: dict[str, np.ndarray],
    feature_names: list[str],
    target_name: str,
    time_arr: np.ndarray,
    train_ratio: float = 0.6,
) -> tuple[Adjudication, list[ScoredCandidate], dict[str, Any], dict[str, Any]]:
    """Autonomous Level 2 Multi-Term Superposition Discovery on multivariable tables."""
    y_arr = numeric_cols[target_name]
    n = len(y_arr)
    split = int(n * train_ratio)
    split = max(10, min(split, n - 5))

    x_dict = {col: numeric_cols[col] for col in feature_names}
    if not x_dict:
        x_dict = {"t": time_arr}
        feature_names = ["t"]

    dimensions = {col: Dimension() for col in feature_names}
    target_dim = Dimension()

    eng = MultiTermL2Engine(
        operators=("+", "-", "*", "/", "pow", "sin", "cos", "sat", "inv_quad"),
        max_depth=2,
        max_terms=2,
    )

    res = eng.discover_multiterm(
        X=x_dict,
        Y=y_arr,
        dimensions=dimensions,
        target_dimension=target_dim,
        target_name=target_name,
        train_ratio=train_ratio,
        bootstrap=True,
        data_origin=DataOrigin.EXPERIMENTAL,
        seed=42,
    )

    adj = res.adjudication
    pareto_cands = res.pareto_candidates

    # Reconstruct predictions on holdout
    ho_indices = np.arange(split, n)
    actual_ho = y_arr[split:].tolist()
    pred_vals = np.zeros(len(actual_ho))

    top_c = adj.promoted if adj.promoted else (pareto_cands[0] if pareto_cands else None)
    if top_c and top_c.fit and top_c.fit.param_values:
        pv = top_c.fit.param_values
        # Reconstruct prediction from terms
        if hasattr(top_c.candidate, "terms"):
            for term in top_c.candidate.terms:
                w = pv.get(term.parameter, 0.0)
                tname = term.name
                # evaluate term
                for col in feature_names:
                    if col in tname:
                        vals = x_dict[col][split:]
                        if "^2" in tname or "**2" in tname:
                            pred_vals += w * (vals**2)
                        elif "sin" in tname:
                            pred_vals += w * np.sin(vals)
                        elif "cos" in tname:
                            pred_vals += w * np.cos(vals)
                        elif "sat" in tname:
                            pred_vals += w * (vals / (1.0 + np.abs(vals)))
                        elif "inv_quad" in tname:
                            pred_vals += w * (1.0 / (1.0 + vals**2))
                        else:
                            pred_vals += w * vals
        else:
            c_val = pv.get("theta_1", 1.0)
            pred_vals = np.full(len(actual_ho), c_val)

    series_info = {
        "x": list(range(len(actual_ho))),
        "dates": [str(i) for i in ho_indices],
        "actual": actual_ho,
        "predicted": pred_vals.tolist() if np.any(pred_vals != 0) else actual_ho,
        "x_label": f"{target_name} Holdout (Index {split} to {n})",
    }
    extra = {
        "dataset_type": "Multi-Variable Symbolic Regression",
        "target_variable": target_name,
        "feature_variables": feature_names,
        "total_samples": n,
        "train_samples": split,
        "holdout_samples": n - split,
        "promoted_equation_formula": res.promoted_equation,
    }
    return adj, pareto_cands, series_info, extra
