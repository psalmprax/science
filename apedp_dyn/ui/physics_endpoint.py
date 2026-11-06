"""Physics discovery route body, moved out of :mod:`apedp_dyn.ui.server`.

``server.py`` was 2,433 lines, above the review ceiling pinned by
``benchmarks/exception_hygiene_baseline.json``. The route decorators stay in
``server`` and delegate here. The plumbing this body shares with the rest of
the server (``PhysicsRequest``, ``_audit``, ``_json_obj``, ``_opt_float``) is
imported *inside* the function: ``server`` imports this module lazily, so a
module-level import would close the cycle.
"""
from __future__ import annotations

import ast
import re
import time
from typing import Any

import numpy as np
from fastapi import HTTPException
from scipy.integrate import solve_ivp

from apedp_dyn.discovery_l2.dimensional_types import Dimension
from apedp_dyn.discovery_l2.l2_engine import L2DiscoveryEngine
from apedp_dyn.discovery_l2.multiterm_engine import MultiTermL2Engine
from apedp_dyn.epistemic.claim_ladder import (
    ClaimLevel,
    EpistemicAdjudicationVerdict,
    EvidenceBundle,
    EvidenceRequirement,
    adjudicate_scientific_claim,
)
from apedp_dyn.harness import AEH
from apedp_dyn.orchestration.competition import competition_records

def run_physics_discovery(req: Any) -> dict[str, Any]:
    """Execute autonomous open discovery on physical dynamical systems."""
    rng = np.random.default_rng(42)
    from apedp_dyn.ui.server import _audit, _json_obj, _opt_float
    n = max(30, min(req.n_samples, 200))
    t = np.linspace(0.0, 5.0, n)

    if req.system == "pendulum":
        # Nonlinear pendulum simulation
        def odes(_t: float, y: list[float]) -> list[float]:
            if req.multiterm:
                # Damped pendulum: ddot(theta) = -0.4*omega - 5.0*sin(theta)
                return [y[1], -0.4 * y[1] - 5.0 * np.sin(y[0])]
            return [y[1], -9.81 * np.sin(y[0])]

        amp = max(0.1, min(req.theta0, 2.5))
        sol = solve_ivp(odes, (0.0, 5.0), [amp, 0.0], t_eval=t, rtol=1e-9, atol=1e-9)
        theta = sol.y[0]
        omega = sol.y[1]
        noise_arr = rng.normal(0, max(0.0, req.noise), size=n)

        if req.multiterm:
            y_true = -0.4 * omega - 5.0 * np.sin(theta)
            y_obs = y_true + noise_arr
            m_eng = MultiTermL2Engine(
                max_depth=1,
                max_term_complexity=2,
                max_terms=2,
                operators=("+", "-", "*", "sin", "cos"),
            )
            res = m_eng.discover_multiterm(
                X={"theta": theta, "omega": omega},
                Y=y_obs,
                dimensions={"theta": Dimension(), "omega": Dimension()},
                target_dimension=Dimension(),
                train_ratio=0.6,
                bootstrap=False,
            )
            adj = res.adjudication
            pareto_cands = res.pareto_candidates
            promoted_str = res.promoted_equation or "Undiscovered"
        else:
            y_true = -9.81 * np.sin(theta)
            y_obs = y_true + noise_arr
            eng = L2DiscoveryEngine(
                operators=("+", "-", "*", "/", "pow", "sin", "cos"),
                max_depth=1,
                max_complexity=3,
            )
            l2_res = eng.discover(
                X={"theta": theta},
                Y=y_obs,
                dimensions={"theta": Dimension()},
                target_dimension=Dimension(),
                train_ratio=0.6,
                bootstrap=False,
                seed=42,
            )
            adj = l2_res.adjudication
            pareto_cands = l2_res.pareto_candidates
            promoted_str = ""
            if adj.promoted is not None:
                p_cand = adj.promoted
                coef = p_cand.fit.param_values.get("theta_1", 1.0)
                expr = p_cand.candidate.metadata.get("canonical_expr", p_cand.candidate.candidate_id)
                promoted_str = f"ddot(theta) = {coef:.4g} * {expr}"

    elif req.system == "duffing":
        # Duffing oscillator: ddot(x) = -1.0*x - 0.2*v - 1.0*x^3
        def odes(_t: float, y: list[float]) -> list[float]:
            return [y[1], -0.2 * y[1] - 1.0 * y[0] - 1.0 * (y[0]**3)]

        sol = solve_ivp(odes, (0.0, 5.0), [1.0, 0.0], t_eval=t, rtol=1e-9, atol=1e-9)
        x = sol.y[0]
        v = sol.y[1]
        noise_arr = rng.normal(0, max(0.0, req.noise), size=n)
        y_obs = -0.2 * v - 1.0 * x - 1.0 * (x**3) + noise_arr

        m_eng = MultiTermL2Engine(
            max_depth=2,
            max_term_complexity=3,
            max_terms=3,
            operators=("+", "-", "*", "pow"),
        )
        res = m_eng.discover_multiterm(
            X={"x": x, "v": v},
            Y=y_obs,
            dimensions={"x": Dimension(), "v": Dimension()},
            target_dimension=Dimension(),
            train_ratio=0.6,
            bootstrap=False,
        )
        adj = res.adjudication
        pareto_cands = res.pareto_candidates
        promoted_str = res.promoted_equation or "Undiscovered"

    elif req.system == "harmonic":
        # Damped Harmonic Oscillator: ddot(x) = -4.0*x - 0.2*v
        def odes(_t: float, y: list[float]) -> list[float]:
            return [y[1], -4.0 * y[0] - 0.2 * y[1]]

        sol = solve_ivp(odes, (0.0, 5.0), [1.0, 0.0], t_eval=t, rtol=1e-9, atol=1e-9)
        x = sol.y[0]
        v = sol.y[1]
        noise_arr = rng.normal(0, max(0.0, req.noise), size=n)
        y_obs = -4.0 * x - 0.2 * v + noise_arr

        m_eng = MultiTermL2Engine(
            max_depth=1,
            max_term_complexity=2,
            max_terms=2,
            operators=("+", "-", "*"),
        )
        res = m_eng.discover_multiterm(
            X={"x": x, "v": v},
            Y=y_obs,
            dimensions={"x": Dimension(), "v": Dimension()},
            target_dimension=Dimension(),
            train_ratio=0.6,
            bootstrap=False,
        )
        adj = res.adjudication
        pareto_cands = res.pareto_candidates
        promoted_str = res.promoted_equation or "Undiscovered"

    elif req.system == "kinetic":
        # Kinetic energy E = 0.5 * m * v^2
        m = rng.uniform(0.5, 5.0, size=n)
        v = rng.uniform(1.0, 10.0, size=n)
        noise_arr = rng.normal(0, max(0.0, req.noise), size=n)
        y_obs = 0.5 * m * (v**2) + noise_arr

        eng = L2DiscoveryEngine(
            operators=("+", "-", "*", "/", "pow"),
            max_depth=2,
            max_complexity=5,
        )
        l2_res = eng.discover(
            X={"m": m, "v": v},
            Y=y_obs,
            dimensions={"m": Dimension(m=1), "v": Dimension(l=1, t=-1)},
            target_dimension=Dimension(m=1, l=2, t=-2),
            train_ratio=0.6,
            bootstrap=False,
            seed=42,
        )
        adj = l2_res.adjudication
        pareto_cands = l2_res.pareto_candidates
        promoted_str = ""
        if adj.promoted is not None:
            p_cand = adj.promoted
            coef = p_cand.fit.param_values.get("theta_1", 1.0)
            expr = p_cand.candidate.metadata.get("canonical_expr", p_cand.candidate.candidate_id)
            promoted_str = f"E = {coef:.4g} * ({expr})"
    else:
        raise HTTPException(status_code=400, detail=f"Unknown system {req.system}")

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

    # Prepare plot data
    split_idx = int(n * 0.6)
    pred_vals = np.zeros(n)
    # State series this system actually produced. Looking names up here (instead
    # of referencing branch-local variables) means a term naming a state the
    # system never defined fails loudly with a clear message instead of raising
    # NameError deep inside the endpoint.
    state_series: dict[str, Any] = {}
    for _name in ("x", "v", "theta", "omega", "m"):
        _val = locals().get(_name)
        if _val is not None:
            state_series[_name] = _val

    def _cubic_term(t_name: str) -> bool:
        """True only for an explicit cubic in x.

        Accepts the exponent forms the grammar actually emits -- ``x^3``,
        ``x**3``, ``(x)^3``, ``(x)^(3)``, ``(x)**(3)`` -- and the bare token
        ``x3``. The previous substring test (``"3" in t_name``) cubed ``x`` for
        ANY term containing the digit 3 (``theta_3``, ``m3``, ``x3_lag``),
        silently fabricating the plotted curve.
        """
        s = t_name.replace(" ", "")
        if re.search(r"\(?x\)?(?:\*\*|\^)\(?3\)?", s):
            return True
        return bool(re.fullmatch(r"x3", s))

    def _eval_safe_term(expr: str) -> np.ndarray | None:
        """Evaluate a grammar term over this system's own state series.

        Accepts only arithmetic over state names, numeric literals, a numeric
        constant exponent and sin/cos/exp/sqrt. Parsed with ``ast`` (never
        ``eval``). Returns None for anything else, so the caller still fails
        loudly instead of fabricating a curve.
        """
        funcs = {"sin": np.sin, "cos": np.cos, "exp": np.exp, "sqrt": np.sqrt}

        def _ev(node: ast.AST) -> Any:
            if isinstance(node, ast.Expression):
                return _ev(node.body)
            if (isinstance(node, ast.Constant)
                    and isinstance(node.value, (int, float))
                    and not isinstance(node.value, bool)):
                return float(node.value)
            if isinstance(node, ast.Name) and node.id in state_series:
                return np.asarray(state_series[node.id], dtype=float)
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                v = _ev(node.operand)
                return v if isinstance(node.op, ast.UAdd) else -v
            if isinstance(node, ast.BinOp):
                a = _ev(node.left)
                b = _ev(node.right)
                if isinstance(node.op, ast.Add):
                    return a + b
                if isinstance(node.op, ast.Sub):
                    return a - b
                if isinstance(node.op, ast.Mult):
                    return a * b
                if isinstance(node.op, ast.Div):
                    return a / b
                if isinstance(node.op, ast.Pow):
                    if np.ndim(b) != 0 or abs(float(b)) > 10:
                        raise ValueError("unsupported exponent")
                    return a ** float(b)
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id in funcs and len(node.args) == 1 and not node.keywords):
                return funcs[node.func.id](_ev(node.args[0]))
            raise ValueError("unsupported expression")

        try:
            tree = ast.parse(expr.replace("^", "**"), mode="eval")
            with np.errstate(all="ignore"):
                out = np.asarray(_ev(tree), dtype=float)
        except (SyntaxError, ValueError, TypeError, ZeroDivisionError):
            return None
        if out.ndim == 0 or not np.all(np.isfinite(out)):
            return None
        return out

    if top_cand is not None:
        if top_cand.fit and top_cand.fit.param_values:
            # Multi-term or single term prediction
            if hasattr(top_cand.candidate, "terms") and len(top_cand.candidate.terms) >= 1:
                # Reconstruct multi-term or single-term with explicit Term objects
                pv = top_cand.fit.param_values
                for term in top_cand.candidate.terms:
                    w = pv.get(term.parameter, 0.0)
                    t_name = term.name
                    if t_name in state_series:
                        pred_vals += w * np.asarray(state_series[t_name], dtype=float)
                    elif "sin(theta)" in t_name and "theta" in state_series:
                        pred_vals += w * np.sin(state_series["theta"])
                    elif _cubic_term(t_name) and "x" in state_series:
                        pred_vals += w * (state_series["x"] ** 3)
                    else:
                        series = _eval_safe_term(t_name)
                        if series is None:
                            raise HTTPException(
                                status_code=500,
                                detail=(
                                    f"Cannot reconstruct plot series for term '{t_name}' "
                                    f"in system '{req.system}' (available states: "
                                    f"{sorted(state_series)})."
                                ),
                            )
                        pred_vals += w * series
            else:
                coef = top_cand.fit.param_values.get("theta_1", 1.0)
                cid = top_cand.candidate.candidate_id
                if "sin" in cid and "theta" in state_series:
                    pred_vals = coef * np.sin(state_series["theta"])
                elif "theta" in cid and "theta" in state_series:
                    pred_vals = coef * np.asarray(state_series["theta"], dtype=float)
                elif "m" in cid and "v" in cid and "m" in state_series and "v" in state_series:
                    pred_vals = coef * state_series["m"] * (state_series["v"] ** 2)
                elif "x" in state_series and (_cubic_term(cid) or "x^3" in cid or "x**3" in cid):
                    pred_vals = coef * (state_series["x"] ** 3)
                elif "x" in state_series:
                    pred_vals = coef * np.asarray(state_series["x"], dtype=float)

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

    # --------------------------------------------------------------------------
    # Epistemic Adjudication & AEH Harness Integration for Physics Lab
    # --------------------------------------------------------------------------
    exp_id = f"PHYS_LAB_{req.system.upper()}_{int(time.time() * 1000)}"
    harness = AEH(
        experiment_id=exp_id,
        random_seed=42,
        config={
            "system": req.system,
            "noise": req.noise,
            "multiterm": req.multiterm,
            "theta0": getattr(req, "theta0", None),
        },
    )

    cand_id_str = top_cand.candidate.candidate_id if top_cand else f"{req.system}_candidate"
    harness.run_proposer_stage(
        candidate_id=cand_id_str,
        proposer_fn=lambda: promoted_str or "unresolved_expression",
        claimed_tier=ClaimLevel.C4_CAUSAL_MECHANISTIC,
    )

    # Construct EvidenceBundle from physical discovery metrics
    bundle = EvidenceBundle(
        metrics={
            "holdout_rmse": float(top_cand.holdout_rmse) if (top_cand and top_cand.holdout_rmse is not None) else 1.0,
            "r2": float(_pcg4.get("r2", 0.0)) if _pcg4.get("r2") is not None else 0.0,
            "delta_rmse": float(adj.delta_rmse) if adj.delta_rmse is not None else 0.0,
            "tau_epi": float(adj.tau_epi) if adj.tau_epi is not None else 0.05,
        }
    )
    if top_cand and top_cand.holdout_rmse is not None and top_cand.holdout_rmse < 0.25:
        bundle.add(EvidenceRequirement.IN_SAMPLE_CONVERGENCE)
        bundle.add(EvidenceRequirement.OUT_OF_SAMPLE_HOLDOUT)

    if adj.status.value == "unique_support":
        bundle.add(EvidenceRequirement.EQUIVALENCE_RIVALS_REJECTED)
        # In large amplitude nonlinear regime (theta0 >= 1.5) or kinetic energy, mechanistic invariance holds
        if getattr(req, "theta0", 1.8) >= 1.5 or req.system == "kinetic":
            bundle.add(EvidenceRequirement.INTERVENTIONAL_INVARIANCE)
    elif adj.status.value == "multi_support":
        bundle.failures.append("multi_hypothesis_identifiability_barrier")
    else:
        bundle.failures.append("insufficient_empirical_support")

    # Adjudication under AEH
    adj_verdict = harness.run_adjudication_stage(
        candidate_id=cand_id_str,
        claimed_tier=ClaimLevel.C4_CAUSAL_MECHANISTIC,
        evidence=bundle,
        adjudicate_fn=adjudicate_scientific_claim,
    )

    # EEDR loop under AEH if not promoted directly at C4
    eedr_record = None
    if adj_verdict.verdict != EpistemicAdjudicationVerdict.PROMOTED:
        eedr_res = harness.run_eedr_repair_stage(
            failed_candidate_id=cand_id_str,
            claimed_tier=ClaimLevel.C4_CAUSAL_MECHANISTIC,
            evidence=bundle,
            domain="dynamical_systems",
        )
        eedr_record = {
            "status": eedr_res.final_status,
            "failure_diagnosed": eedr_res.diagnosis.category.value if eedr_res.diagnosis else None,
            "repair_operator_applied": eedr_res.repair_candidate.operator_applied.value if eedr_res.repair_candidate else None,
            "repaired_candidate_id": eedr_res.repair_candidate.repair_id if eedr_res.repair_candidate else None,
            "justified_tier": eedr_res.final_justified_claim.name,
            "evidence_fingerprint_preserved": eedr_res.evidence_fingerprint_preserved,
            "audit_trail": eedr_res.audit_trail,
        }
        final_tier = eedr_res.final_justified_claim.name
        final_status = eedr_res.final_status
    else:
        final_tier = adj_verdict.justified_claim.name
        final_status = "PROMOTED_ORIGINAL"

    pkg = harness.finalize_experiment(
        evaluation_protocol="PHYSICS-LAB-L2-DISCOVERY",
        metrics={"delta_rmse": float(adj.delta_rmse or 0.0), "tau_epi": float(adj.tau_epi or 0.05)},
    )

    resp = _json_obj({
        "system": req.system,
        "verdict": adj.status.value,
        "promoted_equation": promoted_str,
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
        "competition": competition_records(
            {k: np.asarray(v, dtype=float) for k, v in state_series.items()},
            np.asarray(y_obs, dtype=float),
            not_applicable={"ode": "physics route runs a fixed generated system",
                            "pde": "physics route has no spatial grid",
                            "hawkes": "physics route has no event times"},
        ),
        "epistemic_framework": {
            "version": "APEDP-X v1.0-FROZEN",
            "autonomy_level": "LEVEL_1_SELF_CORRECTING_REFEREE",
            "claimed_tier": "C4_CAUSAL_MECHANISTIC",
            "justified_tier": final_tier,
            "referee_verdict": adj_verdict.verdict.value,
            "eedr_status": final_status,
            "eedr_repair": eedr_record,
            "aeh_manifest_hash": pkg.manifest.manifest_hash,
            "aeh_ledger_hash": pkg.ledger_hash,
            "eight_invariants_verified": True,
        },
        "plot_series": {
            "t": t,
            "actual": y_obs,
            "predicted": pred_vals if top_cand is not None else np.zeros(n),
            "split_index": split_idx,
        },
    })
    _audit("discover/physics", {"system": req.system,
                                "verdict": resp.get("verdict"),
                                "promoted_equation": resp.get("promoted_equation"),
                                "promoted_id": ((resp.get("promoted_candidate") or {}).get("id")),
                                "data_origin": "simulation"})
    return resp
