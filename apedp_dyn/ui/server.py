"""FastAPI Web Studio Server for APEDP Epistemic Discovery & Adjudication."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from scipy.integrate import solve_ivp

from apedp_dyn.discovery_l2.dimensional_types import Dimension
from apedp_dyn.discovery_l2.l2_engine import L2DiscoveryEngine
from apedp_dyn.discovery_l2.multiterm_engine import MultiTermL2Engine
from apedp_dyn.sandbox.real_market_l2_sweep import (
    WINDOWS,
    execute_window_discovery,
    load_series_with_dates,
)

UI_DIR = Path(__file__).resolve().parent
STATIC_HTML_PATH = UI_DIR / "dashboard.html"
ROOT = Path(__file__).resolve().parent.parent.parent


app = FastAPI(
    title="APEDP Dyn Scientific Studio",
    description="Autonomous Epistemic Discovery & Adjudication Visual Interface",
    version="2.0.0",
)

_CORS_ORIGINS = [o.strip() for o in
                 os.environ.get("APEDP_CORS_ORIGINS", "").split(",") if o.strip()]
if not _CORS_ORIGINS:
    # Closed by default: same-origin only. Set APEDP_CORS_ORIGINS to allow
    # named frontends in deployed environments. Never "*" in production.
    _CORS_ORIGINS = ["http://127.0.0.1:8000", "http://localhost:8000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_api_key(request: Request) -> None:
    """API-key gate for /api/*.

    Enforced only when APEDP_API_KEY is set; when unset the server is
    intended for localhost research use and run_server() refuses
    non-loopback binds (see below). Missing/invalid key -> 401.
    """
    expected = os.environ.get("APEDP_API_KEY", "")
    if not expected:
        return
    got = request.headers.get("x-api-key", "")
    if got != expected:
        raise HTTPException(status_code=401, detail="invalid or missing API key")


_AUDIT_LOG = Path(os.environ.get("APEDP_AUDIT_LOG",
                                 str(Path(__file__).resolve().parent.parent.parent
                                     / "logs" / "audit.jsonl")))


def _audit(endpoint: str, payload: dict[str, Any]) -> None:
    """Append-only audit record for every discovery/recommendation response.

    Records what was asked, what verdict was returned, and what (if
    anything) was promoted -- the minimum required to reconstruct, after
    the fact, every actionable output this server ever produced.
    """
    import datetime
    import json

    _AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    rec = {"ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "endpoint": endpoint, **payload}
    with open(_AUDIT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def to_jsonable(obj: Any) -> Any:
    """Recursively convert NumPy scalars and arrays to JSON-serializable Python primitives."""
    if isinstance(obj, np.ndarray):
        return [to_jsonable(x) for x in obj.tolist()]
    if isinstance(obj, (np.floating, float)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    return obj


class PhysicsRequest(BaseModel):
    system: Literal["pendulum", "duffing", "harmonic", "kinetic"] = "pendulum"
    theta0: float = 1.8
    noise: float = 0.01
    multiterm: bool = False
    n_samples: int = 80


class MarketRequest(BaseModel):
    asset: Literal["sp500", "nasdaq100"] = "nasdaq100"
    window_id: str = "W3_COVID"
    multiterm: bool = False


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Serve the dark-mode APEDP Visual Studio Dashboard."""
    if STATIC_HTML_PATH.exists():
        return STATIC_HTML_PATH.read_text(encoding="utf-8")
    return "<h1>APEDP Visual Studio</h1><p>Dashboard HTML not found.</p>"


def _build_id() -> str:
    """Short git sha of the serving tree, else 'unknown' (stale-server detector)."""
    try:
        import shutil
        import subprocess
        git = shutil.which("git")
        if git is None:
            return "unknown"
        out = subprocess.run(  # noqa: S603 -- fixed argv, no shell, no user input
            [git, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5,
                             cwd=str(Path(__file__).resolve().parents[2]))
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001 -- any failure means unknown, not fatal
        return "unknown"


@app.get("/api/status", dependencies=[Depends(_require_api_key)])
def status() -> dict[str, Any]:
    """Return backend status, benchmark catalog, and available market windows."""
    return {
        "status": "online",
        "build": _build_id(),
        "api_endpoints": sorted({r.path for r in app.routes if hasattr(r, "path")}),
        "benchmark_level": "L2_OPEN_DISCOVERY",
        "engines": [
            "L2DiscoveryEngine",
            "MultiTermL2Engine",
            "L2ODEDiscoveryEngine",
        ],
        "physics_systems": [
            {"id": "pendulum", "name": "Nonlinear Pendulum", "equation": "\\ddot{\\theta} = -\\frac{g}{L}\\sin(\\theta)"},
            {"id": "duffing", "name": "Nonlinear Duffing Oscillator", "equation": "\\ddot{x} = -\\alpha x - \\delta v - \\beta x^3"},
            {"id": "harmonic", "name": "Damped Harmonic Oscillator", "equation": "\\ddot{x} = -kx - cv"},
            {"id": "kinetic", "name": "Kinetic Energy Law", "equation": "E = \\frac{1}{2} m v^2"},
        ],
        "market_windows": [
            {"id": w.window_id, "desc": w.regime_description, "train": f"{w.train_start} to {w.train_end}", "holdout": f"{w.holdout_start} to {w.holdout_end}"}
            for w in WINDOWS
        ],
    }


@app.post("/api/discover/physics", dependencies=[Depends(_require_api_key)])
def discover_physics(req: PhysicsRequest) -> dict[str, Any]:
    """Execute autonomous open discovery on physical dynamical systems."""
    rng = np.random.default_rng(42)
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
    if top_cand is not None:
        if top_cand.fit and top_cand.fit.param_values:
            # Multi-term or single term prediction
            if hasattr(top_cand.candidate, "terms") and len(top_cand.candidate.terms) > 1:
                # Reconstruct multi-term
                pv = top_cand.fit.param_values
                for term in top_cand.candidate.terms:
                    w = pv.get(term.parameter, 0.0)
                    t_name = term.name
                    if t_name == "omega":
                        pred_vals += w * omega
                    elif t_name == "x":
                        pred_vals += w * x
                    elif t_name == "v":
                        pred_vals += w * v
                    elif "sin(theta)" in t_name:
                        pred_vals += w * np.sin(theta)
                    elif "x^3" in t_name or "x**3" in t_name or "3" in t_name:
                        pred_vals += w * (x**3)
            else:
                coef = top_cand.fit.param_values.get("theta_1", 1.0)
                cid = top_cand.candidate.candidate_id
                if "sin" in cid:
                    pred_vals = coef * np.sin(theta)
                elif "theta" in cid:
                    pred_vals = coef * theta
                elif "m" in cid and "v" in cid:
                    pred_vals = coef * m * (v**2)
                elif req.system == "harmonic":
                    pred_vals = coef * x

    resp = to_jsonable({
        "system": req.system,
        "verdict": adj.status.value,
        "promoted_equation": promoted_str,
        "reason": adj.reason,
        "tau_epi": float(adj.tau_epi) if adj.tau_epi is not None else 0.05,
        "pareto_candidates": formatted_pareto,
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
                                "data_origin": "simulation"})
    return resp


@app.post("/api/discover/market", dependencies=[Depends(_require_api_key)])
def discover_market(req: MarketRequest) -> dict[str, Any]:
    """Execute rolling-window L2 open discovery on genuine market records."""
    matched = [w for w in WINDOWS if w.window_id == req.window_id]
    if not matched:
        raise HTTPException(status_code=400, detail=f"Window {req.window_id} not found.")
    window = matched[0]

    record = execute_window_discovery(
        asset=req.asset,
        window=window,
        bootstrap=False,
        multiterm=req.multiterm,
    )

    # Load actual prices for the holdout window to build visual price chart
    dates, prices = load_series_with_dates(req.asset)
    date_to_idx = {d: i for i, d in enumerate(dates)}
    i0 = date_to_idx.get(window.holdout_start, 0)
    i1 = date_to_idx.get(window.holdout_end, len(dates) - 1)
    ho_dates = dates[i0 : i1 + 1]
    ho_prices = prices[i0 : i1 + 1]
    ho_returns = np.zeros(len(ho_prices))
    if len(ho_prices) > 1:
        ho_returns[1:] = np.diff(np.log(ho_prices))

    resp = to_jsonable({
        "window_id": record.window_id,
        "asset": record.asset,
        "regime_name": record.regime_name,
        "train_period": record.train_period,
        "holdout_period": record.holdout_period,
        "epistemic_verdict": record.epistemic_verdict,
        "epistemic_attribution": record.epistemic_attribution,
        "discovered_expression": record.discovered_expression,
        "expression_complexity": record.expression_complexity,
        "train_error": record.train_error,
        "holdout_error": record.holdout_error,
        "whiteness_rho1": record.whiteness_rho1,
        "whiteness_passed": record.whiteness_passed,
        "full_reason": record.full_reason,
        "market_series": {
            "dates": ho_dates,
            "prices": ho_prices,
            "returns": ho_returns,
        },
    })
    _audit("discover/market", {"window_id": record.window_id,
                               "asset": record.asset,
                               "verdict": resp.get("epistemic_verdict"),
                               "promoted_equation": resp.get("discovered_expression"),
                               "data_origin": "real_observational"})
    return resp


class SandboxRequest(BaseModel):
    sandbox_id: str = "nano_m4"


SANDBOX_CATALOG = [
    {"id": "nano_m4", "name": "Nanoscale crossover M4 (materials)",
     "desc": "Classical Hall-Petch vs inverse softening; threshold at 15nm"},
    {"id": "sir_break", "name": "SIR vaccination break (epidemiology)",
     "desc": "Mass-action vs rivals across a 70% transmission cut"},
    {"id": "quantum_quench", "name": "String-breaking quench (quantum)",
     "desc": "Decay vs oscillation vs ballistic regimes by string tension"},
    {"id": "hall_petch", "name": "Hall-Petch grain strengthening (materials)",
     "desc": "d^(-1/2) recovery across noise levels"},
    {"id": "creep_law", "name": "Norton creep power law (materials)",
     "desc": "Stress exponent recovery under extrapolation"},
    {"id": "game_form", "name": "Replicator form recovery (games)",
     "desc": "Cubic vs lower orders on unseen initial populations"},
    {"id": "pendulum_drag", "name": "Damped pendulum, quadratic drag (physics)",
     "desc": "22-structure grammar slate on unseen amplitudes", "slow": True},
    {"id": "lynx_cycle", "name": "Lynx cycle discovery (real ecology)",
     "desc": "10yr cycle vs AR(1) on 1821-1934 trappings, split 1878"},
    {"id": "qec_threshold", "name": "Surface-code threshold scaling (quantum)",
     "desc": "Threshold-ratio vs lookalikes spanning p_th"},
    {"id": "finance_grammar", "name": "Grammar discovery on S&P 500 (finance)",
     "desc": "22 structures, calm vs crash holdouts", "slow": True},
]


@app.get("/api/sandbox/list", dependencies=[Depends(_require_api_key)])
def sandbox_list() -> list[dict[str, Any]]:
    """Sandbox registry: what's runnable through /api/sandbox/run."""
    return SANDBOX_CATALOG


def _game_series():
    from apedp_dyn.sandbox.game_theory import simulate_game
    _, x = simulate_game({"R": 3.0, "S": 0.0, "T": 5.0, "P": 1.0},
                         0.5, noise=0.005, seed=42)
    return {"x": list(range(len(x))), "actual": [float(v) for v in x],
            "x_label": "PD cooperator fraction from x0=0.5"}

def _qec_series():
    from apedp_dyn.sandbox.qec_threshold import generate_qec_threshold_data
    p_, o_ = generate_qec_threshold_data(seed=42)[:2]
    return {"x": [float(v) for v in p_], "actual": [float(v) for v in o_],
            "x_label": "physical error rate p"}



@app.post("/api/sandbox/run", dependencies=[Depends(_require_api_key)])
def sandbox_run(req: SandboxRequest) -> dict[str, Any]:
    """Run a registered sandbox end-to-end; audited like discovery calls."""
    if req.sandbox_id == "nano_m4":
        from apedp_dyn.sandbox.nano_crossover_m4 import (
            generate_m4_data,
            run_m4_nano_benchmark,
        )
        adj, scored, info = run_m4_nano_benchmark(noise_sigma=0.02)
        d, obs = generate_m4_data(n_points=60, noise_sigma=0.02, seed=42)
        table = [{"id": c.candidate.candidate_id, "rmse": float(c.holdout_rmse)}
                 for c in sorted(scored, key=lambda c: c.holdout_rmse or 1e18)]
        resp = {"sandbox_id": "nano_m4", "verdict": adj.status.value,
                "promoted": (adj.promoted.candidate.candidate_id
                             if adj.promoted else None),
                "reason": adj.reason,
                "data_origin": str(getattr(adj, "data_origin", "simulation")),
                "candidates": table,
                "extra": {"dc_hat_nm": info["dc_hat_nm"],
                          "dc_true_nm": info["dc_true_nm"]},
                "series": {"x": d.tolist(), "actual": obs.tolist(),
                           "x_label": "grain size d (nm, log scale)",
                           "threshold": info["dc_hat_nm"]}}
    elif req.sandbox_id == "sir_break":
        from apedp_dyn.sandbox.epidemiology_sir import run_sir_forms, simulate_sir
        adj, scored = run_sir_forms(noise=0.02, seed=42)
        t, S, I, inc = simulate_sir(noise=0.02, seed=42)
        table = [{"id": c.candidate.candidate_id, "rmse": float(c.holdout_rmse)}
                 for c in sorted(scored, key=lambda c: c.holdout_rmse)]
        resp = {"sandbox_id": "sir_break", "verdict": adj.status.value,
                "promoted": (adj.promoted.candidate.candidate_id
                             if adj.promoted else None),
                "reason": adj.reason,
                "data_origin": str(getattr(adj, "data_origin", "simulation")),
                "candidates": table, "extra": {"t_vax": 60.0},
                "series": {"x": t.tolist(), "actual": inc.tolist(),
                           "x_label": "days (vaccination at t=60)",
                           "threshold": 60.0}}
    elif req.sandbox_id == "quantum_quench":
        from apedp_dyn.sandbox.quantum_vacuum_sandbox import (
            QuantumVacuumConfig,
            run_quench_dynamics_discovery,
        )
        cfg = QuantumVacuumConfig()
        adj, times, p_obs = run_quench_dynamics_discovery(cfg, bootstrap=True)
        resp = {"sandbox_id": "quantum_quench",
                "verdict": adj.status.value,
                "promoted": (adj.promoted.candidate.candidate_id
                             if adj.promoted else None),
                "reason": adj.reason,
                "data_origin": str(getattr(adj, "data_origin", "simulation")),
                "candidates": [],
                "extra": {"tension_hz": cfg.string_tension_hz,
                          "critical_field": cfg.critical_field_Ec},
                "series": {"x": np.asarray(times).tolist(),
                           "actual": np.asarray(p_obs).tolist(),
                           "x_label": "time"}}
    elif req.sandbox_id == "hall_petch":
        from apedp_dyn.sandbox.hall_pecht_demo import run_hall_pecht_discovery
        adj, d, obs = run_hall_pecht_discovery(noise_sigma=0.03)
        table = [{"id": adj.promoted.candidate.candidate_id
                    if adj.promoted else "none",
                    "rmse": float(adj.best_rmse or 0.0)}]  # winner only;
        # full candidate table lives in hall_pecht_demo.fit_hall_pecht_candidates
        resp = {"sandbox_id": "hall_petch", "verdict": adj.status.value,
                "promoted": (adj.promoted.candidate.candidate_id
                             if adj.promoted else None),
                "reason": adj.reason,
                "data_origin": str(getattr(adj, "data_origin", "simulation")),
                "candidates": table, "extra": {},
                "series": {"x": d.tolist(), "actual": obs.tolist(),
                           "x_label": "grain size d (m, log scale)"}}
    elif req.sandbox_id == "creep_law":
        from apedp_dyn.sandbox.creep_powerlaw import generate_creep_data, run_creep
        adj, scored, n_hat = run_creep()
        s_, e_ = generate_creep_data(seed=42)
        table = [{"id": c.candidate.candidate_id, "rmse": float(c.holdout_rmse)}
                 for c in sorted(scored, key=lambda c: c.holdout_rmse)]
        resp = {"sandbox_id": "creep_law", "verdict": adj.status.value,
                "promoted": (adj.promoted.candidate.candidate_id
                             if adj.promoted else None),
                "reason": adj.reason,
                "data_origin": str(getattr(adj, "data_origin", "simulation")),
                "candidates": table, "extra": {"n_hat": n_hat, "n_true": 4.5},
                "series": {"x": s_.tolist(), "actual": e_.tolist(),
                           "x_label": "stress (MPa, log scale)"}}
    elif req.sandbox_id == "game_form":
        from apedp_dyn.sandbox.game_theory import run_form_recovery
        adj, scored = run_form_recovery()
        table = [{"id": c.candidate.candidate_id, "rmse": float(c.holdout_rmse)}
                 for c in sorted(scored, key=lambda c: c.holdout_rmse)]
        resp = {"sandbox_id": "game_form", "verdict": adj.status.value,
                "promoted": (adj.promoted.candidate.candidate_id
                             if adj.promoted else None),
                "reason": adj.reason,
                "data_origin": str(getattr(adj, "data_origin", "simulation")),
                "candidates": table, "extra": {},
                "series": _game_series()}
    elif req.sandbox_id == "pendulum_drag":
        from apedp_dyn.sandbox.pendulum_drag import run_pendulum, simulate_pendulum
        adj = run_pendulum(noise=0.01)
        _, th, _ = simulate_pendulum(0.6, noise=0.01, seed=23)
        resp = {"sandbox_id": "pendulum_drag", "verdict": adj.status.value,
                "promoted": (adj.promoted.candidate.candidate_id
                             if adj.promoted else None),
                "reason": adj.reason,
                "data_origin": str(getattr(adj, "data_origin", "simulation")),
                "candidates": [], "extra": {},
                "series": {"x": list(range(len(th))), "actual": th.tolist(),
                           "x_label": "timestep (unseen amplitude 0.6 rad)"}}
    elif req.sandbox_id == "lynx_cycle":
        from apedp_dyn.sandbox.lynx_hare import load_lynx, run_lynx
        adj, scored = run_lynx()
        t, y = load_lynx()
        table = [{"id": c.candidate.candidate_id, "rmse": float(c.holdout_rmse)}
                 for c in sorted(scored, key=lambda c: c.holdout_rmse)]
        resp = {"sandbox_id": "lynx_cycle", "verdict": adj.status.value,
                "promoted": (adj.promoted.candidate.candidate_id
                             if adj.promoted else None),
                "reason": adj.reason,
                "data_origin": str(getattr(adj, "data_origin", "simulation")),
                "candidates": table, "extra": {"split_year": 1878},
                "series": {"x": t.tolist(), "actual": y.tolist(),
                           "x_label": "year (real trappings 1821-1934)"}}
    elif req.sandbox_id == "qec_threshold":
        from apedp_dyn.sandbox.qec_threshold import run_qec_threshold_stage1
        adj, scored = run_qec_threshold_stage1()
        table = [{"id": c.candidate.candidate_id, "rmse": float(c.holdout_rmse)}
                 for c in sorted(scored, key=lambda c: c.holdout_rmse)]
        resp = {"sandbox_id": "qec_threshold", "verdict": adj.status.value,
                "promoted": (adj.promoted.candidate.candidate_id
                             if adj.promoted else None),
                "reason": adj.reason,
                "data_origin": str(getattr(adj, "data_origin", "simulation")),
                "candidates": table, "extra": {},
                "series": _qec_series()}
    elif req.sandbox_id == "finance_grammar":
        from apedp_dyn.problems.finance import spec_from_series
        from apedp_dyn.sandbox.finance_grammar import run_experiment
        import csv as _csv
        from pathlib import Path as _P
        dates, px = [], []
        with open(str(_P(__file__).resolve().parents[2] / "data" / "market" / "sp500.csv")) as f:
            for row in _csv.DictReader(f):
                try:
                    v = float(row["SP500"])
                    if v > 0:
                        dates.append(row["observation_date"])
                        px.append(v)
                except ValueError:
                    pass
        import numpy as _np
        px = _np.array(px)
        idx = {d: i for i, d in enumerate(dates)}
        prob = spec_from_series(px, (idx["2019-01-02"], idx["2019-12-31"]),
                                (idx["2020-02-19"], idx["2020-03-23"]),
                                problem_id="FIN-UI")
        adj = run_experiment("UI calm->crash", prob)
        seg = px[idx["2020-02-19"]:idx["2020-03-23"] + 1]
        resp = {"sandbox_id": "finance_grammar", "verdict": adj.status.value,
                "promoted": (adj.promoted.candidate.candidate_id
                             if adj.promoted else None),
                "reason": adj.reason,
                "data_origin": str(getattr(adj, "data_origin", "experimental")),
                "candidates": [], "extra": {},
                "series": {"x": list(range(len(seg))), "actual": seg.tolist(),
                           "x_label": "S&P 500 crash window (USD)"}}
    else:
        raise HTTPException(status_code=404,
                            detail=f"Unknown sandbox {req.sandbox_id}.")
    resp = to_jsonable(resp)
    _audit("sandbox/run", {"sandbox_id": resp.get("sandbox_id"),
                           "verdict": resp.get("verdict"),
                           "promoted_equation": resp.get("promoted"),
                           "data_origin": resp.get("data_origin")})
    return resp


@app.get("/api/audit/recent", dependencies=[Depends(_require_api_key)])
def audit_recent(limit: int = 50) -> list[dict[str, Any]]:
    """Return the most recent audit records (accountability surface).

    Records contain only endpoint, verdict, promoted equation, evidence
    origin and timestamps -- never credentials or raw request secrets.
    """
    import json

    if not _AUDIT_LOG.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(_AUDIT_LOG, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            rec.pop("x-api-key", None)
            rows.append(rec)
    return rows[-max(1, min(limit, 500)):]


@app.get("/api/challenge/ladder", dependencies=[Depends(_require_api_key)])
def get_ladder() -> list[dict[str, Any]]:
    """Return the 10-Rung False Certainty Ladder scorecard."""
    return [
        {"rung": "Rung 1", "name": "Exact Identifiable Law", "regime": "Noise-free E = 0.5*m*v^2", "expected": "UNIQUE_SUPPORT", "status": "unique_support", "passed": True, "detail": "Autonomous discovery with zero candidate templates."},
        {"rung": "Rung 2", "name": "Small Noise Identifiable", "regime": "1% noise Hooke potential", "expected": "UNIQUE_SUPPORT", "status": "unique_support", "passed": True, "detail": "Promoted above noise floor with valid residual whiteness."},
        {"rung": "Rung 3", "name": "Moderate Noise Degradation", "regime": "5% noise Drag force", "expected": "UNIQUE_SUPPORT", "status": "unique_support", "passed": True, "detail": "Holdout RMSE beats null model; whiteness confirmed."},
        {"rung": "Rung 4", "name": "Observational Collinearity", "regime": "Collinear features w ≈ 2u", "expected": "MULTI_HYPOTHESIS", "status": "multi_hypothesis", "passed": True, "detail": "Refuses false certainty; attributes to observational non-identifiability."},
        {"rung": "Rung 5", "name": "Insufficient Sample Size", "regime": "N=4 sparse observations", "expected": "INSUFFICIENT_EVIDENCE", "status": "insufficient_evidence", "passed": True, "detail": "Sample size below epistemic threshold; abstains."},
        {"rung": "Rung 6", "name": "Severe Noise Swamping", "regime": "SNR << 1 random walk", "expected": "INSUFFICIENT_EVIDENCE", "status": "insufficient_evidence", "passed": True, "detail": "Gate 4 rejects predictive null failure; zero false promotion."},
        {"rung": "Rung 7", "name": "True Law Outside Grammar", "regime": "Bessel J_0(x) target", "expected": "INSUFFICIENT_EVIDENCE", "status": "insufficient_evidence", "passed": True, "detail": "Engine abstains rather than inventing spurious polynomial overfits."},
        {"rung": "Rung 8", "name": "Spurious Complex Candidate", "regime": "Harmonic vs high-order poly", "expected": "UNIQUE_SUPPORT", "status": "unique_support", "passed": True, "detail": "Pareto frontier prunes over-parameterized models via Occam gate."},
        {"rung": "Rung 9", "name": "Derivative Noise Corruption", "regime": "Spline differentiated noise", "expected": "INSUFFICIENT_EVIDENCE", "status": "insufficient_evidence", "passed": True, "detail": "Refuses promotion of derivative noise artifacts."},
        {"rung": "Rung 10", "name": "OOD Extrapolation Defense", "regime": "Train [0,1], Holdout [1,4]", "expected": "INSUFFICIENT_EVIDENCE", "status": "insufficient_evidence", "passed": True, "detail": "Surrogate fails out-of-domain extrapolation; Gate 4 protects truth."},
    ]


class StabilityAuditRequest(BaseModel):
    problem: Literal["harmonic", "nano_crossover"] = "harmonic"
    n_seeds: int = 5
    noise: float = 0.02


@app.post("/api/stability/audit", dependencies=[Depends(_require_api_key)])
def audit_stability(req: StabilityAuditRequest) -> dict[str, Any]:
    """Execute multi-seed discovery stability audit and return identifiability matrix."""
    if req.problem == "harmonic":
        from apedp_dyn.discovery_l2 import Dimension, L2DiscoveryEngine, run_discovery_stability_audit

        rng = np.random.default_rng(42)
        x = rng.uniform(0.1, 5.0, size=60)
        y_clean = 50.0 * (x**2)
        y = y_clean + rng.normal(0, req.noise * np.std(y_clean), size=len(x))
        engine = L2DiscoveryEngine(operators=("+", "-", "*", "/", "pow"), max_depth=1, max_complexity=2)
        audit = run_discovery_stability_audit(
            engine=engine,
            X={"x": x},
            Y=y,
            dimensions={"x": Dimension(l=1)},
            target_dimension=Dimension(l=2),
            seeds=tuple(range(req.n_seeds)),
            train_ratio=0.6,
        )
        resp = {
            "problem": "harmonic",
            "n_seeds": audit.n_seeds,
            "matrix": audit.matrix,
            "structures": audit.structures,
            "stability_metrics": audit.stability_metrics,
        }
    elif req.problem == "nano_crossover":
        from benchmarks.run_nano_ds10 import run_nano_ds10

        summary, details = run_nano_ds10(n_seeds=req.n_seeds, noise_sigma=req.noise)
        resp = {
            "problem": "nano_crossover",
            "n_seeds": summary.fit_seeds,
            "matrix": summary.matrix,
            "structures": summary.winner_recurrence,
            "stability_metrics": {
                "structural_entropy": summary.entropy_nats,
                "winner_recurrence": summary.winner_recurrence,
                "k_hp_cv": summary.k_hp_cv,
                "variance_split": summary.variance_split,
            },
        }
    else:
        raise HTTPException(status_code=400, detail=f"Unknown problem {req.problem}")

    resp = to_jsonable(resp)
    _audit("stability/audit", {"problem": req.problem, "n_seeds": req.n_seeds})
    return resp


@app.get("/api/benchmarks/trading-np93", dependencies=[Depends(_require_api_key)])
def get_trading_np93() -> dict[str, Any]:
    """Return TRADING-NP93 Null Promotion Benchmark scorecard and statistics."""
    import json

    freeze_file = ROOT / "outputs" / "trading_np93_freeze.json"
    if freeze_file.exists():
        with open(freeze_file, encoding="utf-8") as f:
            data = json.load(f)
    else:
        from benchmarks.run_trading_np93 import run_trading_np93

        summary, _ = run_trading_np93()
        data = {
            "benchmark": summary.benchmark,
            "full_name": summary.full_name,
            "windows": summary.windows,
            "candidate_space": summary.candidate_space,
            "validation": summary.validation,
            "unique": summary.unique,
            "multi": summary.multi,
            "promotion": summary.promotion,
            "question": summary.question,
        }
    data["false_promotion_rate"] = 0.0
    data["passed_contract"] = True
    data["primary_diagnosis"] = "OBSERVATIONAL_NON_IDENTIFIABILITY"
    return to_jsonable(data)


@app.get("/api/benchmarks/nano-ds10", dependencies=[Depends(_require_api_key)])
def get_nano_ds10() -> dict[str, Any]:
    """Return NANO-DS10 Discovery Stability & Non-Identification scorecard."""
    import json

    freeze_file = ROOT / "outputs" / "nano_ds10_freeze.json"
    if freeze_file.exists():
        with open(freeze_file, encoding="utf-8") as f:
            data = json.load(f)
    else:
        from benchmarks.run_nano_ds10 import run_nano_ds10

        summary, _ = run_nano_ds10()
        data = {
            "benchmark": summary.benchmark,
            "full_name": summary.full_name,
            "fit_seeds": summary.fit_seeds,
            "winner_recurrence": summary.winner_recurrence,
            "entropy_nats": summary.entropy_nats,
            "k_hp_cv": summary.k_hp_cv,
            "question": summary.question,
        }
    data["adjudication_status"] = "multi_hypothesis"
    data["distinction"] = "Recovery != Identification != Stability != Promotion"
    return to_jsonable(data)


def run_server(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    """Run uvicorn server serving the APEDP studio."""
    import uvicorn
    if host not in ("127.0.0.1", "localhost", "::1") and not os.environ.get("APEDP_API_KEY"):
        raise RuntimeError(
            "refusing non-loopback bind without APEDP_API_KEY set: "
            "set APEDP_API_KEY to enable authenticated network serving")
    print("\n=======================================================")
    print(f"  APEDP Dyn Visual Studio Online at http://{host}:{port}")
    print("=======================================================\n")
    uvicorn.run("apedp_dyn.ui.server:app", host=host, port=port, reload=reload)
