"""Discovery-registry route body, moved out of :mod:`apedp_dyn.ui.server`.

See :mod:`apedp_dyn.ui.physics_endpoint` for why the shared plumbing is
imported inside the function instead of at module scope.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import HTTPException

from apedp_dyn.discovery_l2.l2_engine import L2DiscoveryResult
from apedp_dyn.discovery_l2.multiterm_engine import MultiTermDiscoveryResult, MultiTermL2Engine


def run_registry_discovery(req: Any) -> dict[str, Any]:
    """Generic L2 discovery on any registered problem module.

    Uses the problem's own ground-truth trajectories: features are the
    state variables, target is the finite-difference derivative of the
    last state (the acceleration/rate law -- the RHS discovery task).
    Same guards, same adjudication, same audit as the demo systems.
    """
    from apedp_dyn.discovery_l2.dimensional_types import Dimension
    from apedp_dyn.discovery_l2.l2_engine import L2DiscoveryEngine
    from apedp_dyn.problems.base import finalize
    from apedp_dyn.ui.server import _LOG, _audit, _json_obj, _opt_float, _problem_modules

    modname = str(req.get("system", ""))
    mods = _problem_modules()
    if modname not in mods:
        raise HTTPException(status_code=404, detail=f"Unknown problem module {modname}.")
    try:
        prob = finalize(mods[modname].spec())
    except (AssertionError, ValueError) as e:
        # Bad problem spec (e.g. truth trajectory blows up): client got a
        # problem the server cannot build truth for -> 422, not engine 500.
        _audit("discover/registry", {"system": modname, "verdict": "error",
                                     "reason": f"truth generation failed: {e}"})
        raise HTTPException(status_code=422,
                            detail=f"Problem {modname} cannot generate ground truth: {e}") from e
    res: MultiTermDiscoveryResult | L2DiscoveryResult
    try:
        Y = np.asarray(prob.train["truth_y"], dtype=float)
        t = np.asarray(prob.train["truth_t"], dtype=float)
        state_names = [s.name for s in prob.states]
        target_state = state_names[-1] if state_names else "y"
        names = list(state_names)
        feats = {n: Y[:, i] for i, n in enumerate(state_names)}
        # Forced systems must offer the drive as evidence: discovering
        # acceleration laws without the input series starves the search
        # of the causing signal (measured 230x RMSE penalty on MSD).
        for idx, u in enumerate(prob.inputs):
            try:
                feats[u.name] = np.asarray(
                    [np.atleast_1d(np.asarray(prob.train["u_of_t"](ti), dtype=float))[idx]
                     for ti in t], dtype=float)
                names.append(u.name)
            except Exception as exc:  # noqa: BLE001 -- broken drive must not hide states
                import logging
                logging.getLogger(__name__).debug("skipping drive %s: %s",
                                                  getattr(u, "name", "?"), exc)
                continue
        target = np.gradient(Y[:, -1], t)
        noise_level = float(req.get("noise", 0.0))
        noise_sigma = None
        if noise_level > 0.0:
            rng = np.random.default_rng(int(req.get("seed", 42)))
            t_std = float(np.std(target))
            noise_sigma = noise_level * (t_std if t_std > 1e-12 else 1.0)
            target = target + rng.normal(0, noise_sigma, size=len(target))

        # Check for explicit holdout data from problem specification
        X_ho = None
        Y_ho = None
        t_ho = None
        has_explicit_holdout = (
            isinstance(getattr(prob, "holdout", None), dict)
            and "truth_y" in prob.holdout
            and "truth_t" in prob.holdout
        )
        if has_explicit_holdout:
            Y_h = np.asarray(prob.holdout["truth_y"], dtype=float)
            t_h = np.asarray(prob.holdout["truth_t"], dtype=float)
            if len(t_h) >= 4:
                feats_ho = {n: Y_h[:, i] for i, n in enumerate(state_names)}
                for idx, u in enumerate(prob.inputs):
                    try:
                        feats_ho[u.name] = np.asarray(
                            [np.atleast_1d(np.asarray(prob.holdout["u_of_t"](ti), dtype=float))[idx]
                             for ti in t_h], dtype=float)
                    except (ValueError, TypeError, ArithmeticError, KeyError,
                            IndexError) as exc:
                        # A malformed input channel must not abort discovery;
                        # the channel is dropped and the candidate simply loses
                        # that feature. Surfaced so a silent drop is greppable.
                        _LOG.debug("dropping input channel %s: %s", u.name, exc)
                X_ho = feats_ho
                t_ho = t_h
                Y_ho = np.gradient(Y_h[:, -1], t_h)

        is_multiterm = bool(req.get("multiterm", False))
        if is_multiterm:
            # Budgets default to the engine's own (2, 4, 3): the RIME-era
            # (3, 8, 4) triple expanded generation ~50x (extra powers) and
            # pushed single requests into minute-scale runs. Callers can still
            # raise any cap explicitly for a heavier search.
            max_terms = int(req.get("max_terms", 3))
            max_depth = int(req.get("max_depth", 2))
            max_term_complexity = int(req.get("max_term_complexity", 4))
            operators = tuple(req.get("operators", ('+', '-', '*', '/', 'pow', 'sin', 'cos', 'abs')))
            m_eng = MultiTermL2Engine(
                max_depth=max_depth,
                max_term_complexity=max_term_complexity,
                max_terms=max_terms,
                operators=operators,
                # The zoo's step/sine regimes starve shape-term screening
                # (see multiterm_engine Step 2), so the registry route runs a
                # wider basis than the 28-term engine default; combinations
                # stay bounded by max_terms=3.
                max_basis_terms=40,
            )
            res = m_eng.discover_multiterm(
                X=feats,
                Y=target,
                dimensions={n: Dimension() for n in names},
                target_dimension=Dimension(),
                target_name=target_state,
                train_ratio=0.6,
                bootstrap=False,
                noise_sigma=noise_sigma,
                seed=42,
                X_holdout=X_ho,
                Y_holdout=Y_ho,
            )
        else:
            eng = L2DiscoveryEngine(operators=("+", "-", "*", "/", "pow", "abs"),
                                    max_depth=2, max_complexity=5)
            res = eng.discover(X=feats, Y=target,
                               dimensions={n: Dimension() for n in names},
                               target_dimension=Dimension(),
                               target_name=target_state,
                               train_ratio=0.6,
                               bootstrap=False, noise_sigma=noise_sigma, seed=42,
                               X_holdout=X_ho,
                               Y_holdout=Y_ho)

    except Exception as e:  # engine failure must be a 500 with reason, never empty
        _audit("discover/registry", {"system": modname, "verdict": "error",
                                     "reason": f"{type(e).__name__}: {e}"})
        raise HTTPException(status_code=500,
                            detail=f"Discovery failed for {modname}: {type(e).__name__}: {e}") from e
    adj = res.adjudication
    pareto_cands = res.pareto_candidates
    n = len(t)
    split_idx = int(n * 0.6)

    # Format Pareto candidates
    formatted_pareto = []
    top_cand = adj.promoted if adj.promoted is not None else (pareto_cands[0] if pareto_cands else None)
    for c in pareto_cands:
        g4 = c.provenance.get("gate4_status", {})
        cid = c.candidate.candidate_id
        is_top = (top_cand is not None and top_cand.candidate.candidate_id == cid)
        formatted_pareto.append({
            "id": cid,
            "complexity": int(c.candidate.metadata.get("complexity", 1)),
            "rmse": float(c.holdout_rmse) if c.holdout_rmse is not None else None,
            "r2": float(g4.get("r2", 0.0)) if g4.get("r2") is not None else None,
            "rho1": float(g4.get("rho1", 1.0)) if g4.get("rho1") is not None else None,
            "is_white": bool(g4.get("is_white", False)),
            "is_promoted": bool(is_top and adj.promoted is not None),
        })

    # Prepare predicted series for the top candidate
    pred_vals = np.zeros(n)
    if top_cand is not None and top_cand.fit and top_cand.fit.param_values:
        nodes = top_cand.candidate.metadata.get("nodes")
        ast_node = top_cand.candidate.metadata.get("ast")
        if nodes:
            try:
                pred_vals = np.zeros(n)
                pv = top_cand.fit.param_values
                for j, node in enumerate(nodes):
                    w = pv.get(f"theta_{j+1}", 0.0)
                    pred_vals += w * node.evaluate(feats)
                if not np.all(np.isfinite(pred_vals)):
                    pred_vals = np.zeros(n)
            except Exception:  # noqa: BLE001 -- fallback to zero prediction on evaluation failure
                pred_vals = np.zeros(n)
        elif ast_node is not None:
            try:
                base_v = ast_node.evaluate(feats)
                coef = top_cand.fit.param_values.get("theta_1", 1.0)
                pred_vals = np.asarray(coef * base_v, dtype=float)
                if not np.all(np.isfinite(pred_vals)):
                    pred_vals = np.zeros(n)
            except Exception:  # noqa: BLE001 -- fallback to zero prediction on evaluation failure
                pred_vals = np.zeros(n)

    plot_t = t
    plot_actual = target
    plot_pred = pred_vals
    if X_ho is not None and Y_ho is not None and t_ho is not None:
        pred_ho = np.zeros(len(t_ho))
        if top_cand is not None and top_cand.fit and top_cand.fit.param_values:
            nodes = top_cand.candidate.metadata.get("nodes")
            ast_node = top_cand.candidate.metadata.get("ast")
            if nodes:
                try:
                    pv = top_cand.fit.param_values
                    for j, node in enumerate(nodes):
                        w = pv.get(f"theta_{j+1}", 0.0)
                        pred_ho += w * node.evaluate(X_ho)
                    if not np.all(np.isfinite(pred_ho)):
                        pred_ho = np.zeros(len(t_ho))
                except (ValueError, TypeError, ArithmeticError, KeyError,
                        IndexError) as exc:
                    _LOG.debug("holdout rollout failed, using flat baseline: %s", exc)
                    pred_ho = np.zeros(len(t_ho))
            elif ast_node is not None:
                try:
                    base_v = ast_node.evaluate(X_ho)
                    coef = top_cand.fit.param_values.get("theta_1", 1.0)
                    pred_ho = np.asarray(coef * base_v, dtype=float)
                    if not np.all(np.isfinite(pred_ho)):
                        pred_ho = np.zeros(len(t_ho))
                except (ValueError, TypeError, ArithmeticError, KeyError,
                        IndexError) as exc:
                    _LOG.debug("AST holdout rollout failed, using flat baseline: %s", exc)
                    pred_ho = np.zeros(len(t_ho))

        dt = float(t[1] - t[0]) if len(t) > 1 else 0.025
        t_ho_offset = t_ho + float(t[-1]) + dt
        plot_t = np.concatenate([t, t_ho_offset])
        plot_actual = np.concatenate([target, Y_ho])
        plot_pred = np.concatenate([pred_vals, pred_ho])
        split_idx = len(t)
        # The multiterm engine stores the prediction it fitted, on the same
        # train-then-holdout axis. Prefer it over rebuilding the curve from term
        # names, but only when it is complete and aligned with the plotted time.
        _stored = (top_cand.provenance.get("gate4_status", {}).get("full_prediction")
                   if top_cand is not None else None)
        if _stored and len(_stored) == len(plot_t) and all(v is not None for v in _stored):
            plot_pred = np.asarray(_stored, dtype=float)

    expr = ""
    last_state = target_state
    if adj.promoted is not None:
        p_cand = adj.promoted
        if is_multiterm and hasattr(res, "promoted_equation") and res.promoted_equation:
            eq = res.promoted_equation
            target_var = target_state
            if eq.startswith(f"{target_var} = "):
                eq = eq[len(f"{target_var} = "):]
            elif " = " in eq:
                eq = eq.split(" = ", 1)[1]
            expr = f"d({last_state})/dt = {eq}"
        else:
            coef = p_cand.fit.param_values.get("theta_1", 1.0) if p_cand.fit else 1.0
            can_expr = (p_cand.provenance.get("canonical_expr")
                        or p_cand.candidate.metadata.get("canonical_expr")
                        or p_cand.candidate.candidate_id)
            expr = f"d({last_state})/dt = {coef:.4g} * ({can_expr})"

    _pc = adj.promoted
    _pcg4 = (_pc.provenance.get("gate4_status", {}) if _pc is not None else {})
    _promoted_record = None
    if _pc is not None:
        _promoted_record = {
            "id": _pc.candidate.candidate_id,
            "holdout_rmse": (float(_pc.holdout_rmse)
                             if _pc.holdout_rmse is not None else None),
            "r2": _opt_float(_pcg4.get("r2")),
            "residual_rho1": _opt_float(_pcg4.get("rho1")),
            "delta_rmse": (float(adj.delta_rmse)
                           if adj.delta_rmse is not None else None),
            "tau_epi": (float(adj.tau_epi)
                        if adj.tau_epi is not None else None),
            # Diagnostic qualification of a promotion. Never changes `verdict`:
            # the frozen gate is unchanged, and these fields only distinguish
            # "the gate permits this" from "this is strong identification".
            "promotion_confidence": adj.promotion_confidence or None,
            "diagnostic_flags": list(adj.diagnostic_flags),
        }

    rival_a = None
    rival_b = None
    if adj.ranking:
        lead = adj.ranking[0]
        rival_a = {
            "id": lead.candidate.candidate_id,
            "rmse": float(lead.holdout_rmse) if lead.holdout_rmse is not None else None,
            "complexity": int(lead.candidate.metadata.get("complexity", 1)),
        }
    if getattr(adj, "rival", None) is not None:
        riv = adj.rival
        rival_b = {
            "id": riv.candidate.candidate_id,
            "rmse": float(riv.holdout_rmse) if riv.holdout_rmse is not None else None,
            "complexity": int(riv.candidate.metadata.get("complexity", 1)),
        }

    resp = _json_obj({
        "system": modname,
        "verdict": adj.status.value,
        "promoted_equation": expr or "No promoted equation (refusal under uncertainty)",
        "promoted_candidate": _promoted_record,
        "reason": adj.reason,
        "tau_epi": float(adj.tau_epi) if adj.tau_epi is not None else 0.05,
        "delta_rmse": float(adj.delta_rmse) if adj.delta_rmse is not None else None,
        # Diagnostics only -- `verdict` above is unchanged by these.
        "promotion_confidence": adj.promotion_confidence or None,
        "diagnostic_flags": list(adj.diagnostic_flags),
        "rival_a": rival_a,
        "rival_b": rival_b,
        "pareto_candidates": formatted_pareto,
        "plot_series": {
            "t": plot_t,
            "actual": plot_actual,
            "predicted": plot_pred,
            "split_index": split_idx,
        },
        "data_origin": "simulation",
    })
    _audit("discover/registry", {"system": modname,
                                 "verdict": resp.get("verdict"),
                                 "promoted_equation": resp.get("promoted_equation"),
                                 "promoted_id": ((resp.get("promoted_candidate") or {}).get("id")),
                                 "data_origin": "registry-ground-truth"})
    return resp
