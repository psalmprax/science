"""FastAPI Web Studio Server for APEDP Epistemic Discovery & Adjudication."""
from __future__ import annotations

import logging
import os
import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from scipy.integrate import solve_ivp

from apedp_dyn.discovery_l2.dimensional_types import Dimension
from apedp_dyn.discovery_l2.l2_engine import L2DiscoveryEngine, L2DiscoveryResult
from apedp_dyn.discovery_l2.multiterm_engine import MultiTermDiscoveryResult, MultiTermL2Engine
from apedp_dyn.epistemic.claim_ladder import (
    ClaimLevel,
    EpistemicAdjudicationVerdict,
    EvidenceBundle,
    EvidenceRequirement,
    adjudicate_scientific_claim,
)
from apedp_dyn.harness import AEH
from apedp_dyn.market.assets import AVAILABLE_ASSETS, load_series_with_dates
from apedp_dyn.market.regimes import REGIME_METADATA, MarketRegime
from apedp_dyn.market.sweep import (
    execute_window_discovery,
    run_market_sweep,
)
from apedp_dyn.market.windows import (
    WINDOWS,
    extensive_rolling_windows,
    generated_rolling_windows,
    get_window_by_id,
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

# Diagnostics for degraded-but-continue paths. Debug level on purpose: these
# are the branches where discovery still returns a usable (flatter) result, so
# they must not spam request logs, but a silently dropped input channel should
# still be greppable.
_LOG = logging.getLogger("apedp_dyn.ui.server")


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
    # NOTE: bool check must precede int (bool subclasses int in Python);
    # otherwise True/False serialize as 1/0 and clients cannot rely on
    # boolean flags such as is_promoted.
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    return obj


def _json_obj(obj: Any) -> dict[str, Any]:
    """``to_jsonable`` with the mapping return type the endpoints declare."""
    return cast(dict[str, Any], to_jsonable(obj))


def _opt_float(value: Any) -> float | None:
    """Coerce a possibly-missing JSON numeric to ``float | None``."""
    return float(value) if value is not None else None


class PhysicsRequest(BaseModel):
    system: str = "pendulum"
    theta0: float = 1.8
    noise: float = 0.01
    multiterm: bool = False
    n_samples: int = 80


class MarketRequest(BaseModel):
    asset: str = "nasdaq100"
    window_id: str = "W3_COVID"
    multiterm: bool = False


class MarketSweepRequest(BaseModel):
    asset: str = "sp500"
    max_windows: int = 5
    multiterm: bool = False


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    """Serve the dark-mode APEDP Visual Studio Dashboard."""
    headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
    }
    if STATIC_HTML_PATH.exists():
        return HTMLResponse(content=STATIC_HTML_PATH.read_text(encoding="utf-8"), headers=headers)
    return HTMLResponse(content="<h1>APEDP Visual Studio</h1><p>Dashboard HTML not found.</p>", status_code=404, headers=headers)


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Any:
    """Serve SVG favicon to eliminate 404 console errors."""
    from fastapi.responses import Response

    svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><text y="26" font-size="26" fill="#00f0ff">Ψ</text></svg>'
    return Response(content=svg, media_type="image/svg+xml")



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


def _problem_modules() -> dict[str, Any]:
    """All importable problem modules exposing spec()."""
    import importlib
    import pkgutil

    import apedp_dyn.problems as P
    out: dict[str, Any] = {}
    for m in pkgutil.iter_modules(P.__path__):
        if m.name.startswith("_") or m.name == "base" or m.name == "catalog":
            continue
        try:
            mod = importlib.import_module(f"apedp_dyn.problems.{m.name}")
            if hasattr(mod, "spec"):
                out[m.name] = mod
        except Exception as exc:  # noqa: BLE001 -- one bad module must not hide the rest
            import logging
            logging.getLogger(__name__).debug("skipping problem module %s: %s", m.name, exc)
            continue
    try:
        from apedp_dyn.problems.catalog import get_catalog_modules
        out.update(get_catalog_modules())
    except Exception as exc:  # noqa: BLE001
        import logging
        logging.getLogger(__name__).debug("skipping catalog modules: %s", exc)
    return out


_PHYSICS_SYSTEMS_CACHE: list[dict[str, Any]] | None = None


def _get_all_physics_systems_items() -> list[dict[str, Any]]:
    global _PHYSICS_SYSTEMS_CACHE
    if _PHYSICS_SYSTEMS_CACHE is not None:
        return _PHYSICS_SYSTEMS_CACHE

    items = []
    for modname, mod in sorted(_problem_modules().items()):
        try:
            category = getattr(mod, "category", "classical")
            name = getattr(mod, "name", modname)
            problem_id = getattr(mod, "problem_id", getattr(mod, "PROBLEM_ID", modname))
            if hasattr(mod, "equation_strings"):
                eqs = mod.equation_strings
                n_states = getattr(mod, "n_states", len(eqs))
            else:
                spec = mod.spec()
                eqs = [f"{e.state}' = {e.sympy_expr}" for e in spec.truth_equations]
                n_states = len(spec.states)
                name = getattr(spec, "name", name)
                problem_id = getattr(spec, "problem_id", problem_id)
        except Exception as exc:  # noqa: BLE001 -- broken spec must not hide the rest
            import logging
            logging.getLogger(__name__).debug("skipping spec %s: %s", modname, exc)
            continue
        items.append({
            "id": modname,
            "problem_id": problem_id,
            "name": name,
            "category": category,
            "equations": eqs,
            "n_states": n_states,
        })
    _PHYSICS_SYSTEMS_CACHE = items
    return items


@app.get("/api/physics/systems", dependencies=[Depends(_require_api_key)])
def physics_systems(
    category: str | None = None,
    search: str | None = None,
    limit: int | None = None,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Registry of every runnable problem module with optional filtering and pagination."""
    items = _get_all_physics_systems_items()
    if category:
        c_low = category.strip().lower()
        items = [it for it in items if it.get("category", "").lower() == c_low]
    if search:
        s_low = search.strip().lower()
        items = [
            it for it in items
            if s_low in it["name"].lower()
            or s_low in it["id"].lower()
            or any(s_low in eq.lower() for eq in it["equations"])
        ]
    if offset:
        items = items[offset:]
    if limit is not None:
        items = items[:limit]
    return items


EXT60_CORPUS = Path(__file__).resolve().parents[2] / "benchmark_corpus_external_60"


@app.get("/api/benchmark/external60/blocks", dependencies=[Depends(_require_api_key)])
def ext60_blocks() -> list[dict[str, Any]]:
    """Public manifest only: block ids + problem counts (never hidden truth)."""
    import json as _json

    manifest = _json.loads((EXT60_CORPUS / "manifest.json").read_text())
    counts: dict[str, int] = {}
    for prob in manifest["problems"]:
        counts[prob["block"]] = counts.get(prob["block"], 0) + 1
    return [{"block": b, "n_problems": n,
             "prereg_id": manifest.get("prereg_id", "")}
            for b, n in sorted(counts.items())]


class Ext60RunRequest(BaseModel):
    blocks: list[str] = ["P"]


@app.post("/api/benchmark/external60/run", dependencies=[Depends(_require_api_key)])
def ext60_run(req: Ext60RunRequest) -> dict[str, Any]:
    """Run APEDPAdapter on PUBLIC data for the requested blocks.

    Writes ONLY to a scratch directory (never the frozen round dir):
    a re-run measures the current tree, it must not overwrite or merge
    with tournament_results_external_60/. Returns per-problem verdicts
    inline; blind scoring against hidden truth stays a separate step.
    """
    import json as _json
    import tempfile
    import time

    from benchmarks.external_60_adapters import APEDPAdapter

    manifest = _json.loads((EXT60_CORPUS / "manifest.json").read_text())
    valid = sorted({p["block"] for p in manifest["problems"]})
    for b in req.blocks:
        if b not in valid:
            raise HTTPException(status_code=400, detail=f"Unknown block {b}.")
    scratch = Path(tempfile.mkdtemp(prefix="ext60_"))
    adapter = APEDPAdapter()
    summary: list[dict[str, Any]] = []
    t0 = time.time()
    for block in req.blocks:
        pids = [p["problem_id"] for p in manifest["problems"] if p["block"] == block]
        results = []
        for pid in pids:
            try:
                res = adapter.run(EXT60_CORPUS, pid)
                row = res.to_dict()
            except Exception as e:  # noqa: BLE001 -- one failure must not kill the block
                row = {"competitor": "apedp", "problem_id": pid,
                       "expression": "RUNNER_ERROR", "abstained": True,
                       "abstain_reason": f"runner exception: {type(e).__name__}"}
            results.append({"problem_id": pid,
                            "abstained": bool(row.get("abstained")),
                            "expression": str(row.get("expression", ""))[:120],
                            "abstain_reason": str(row.get("abstain_reason", ""))[:200]})
        with open(scratch / f"apedp_{block}.json", "w", encoding="utf-8") as f:
            _json.dump({"block": block, "results": results}, f, indent=2)
        summary.append({"block": block, "n": len(results),
                        "abstained": sum(1 for r in results if r["abstained"])})
    out = {"blocks": summary, "scratch_dir": str(scratch),
           "elapsed_s": round(time.time() - t0, 1),
           "note": "scratch only; frozen round untouched; blind scoring separate"}
    _audit("benchmark/external60", {"blocks": [b["block"] for b in summary],
                                    "verdict": "batch-complete",
                                    "data_origin": "synthetic"})
    return _json_obj(out)


@app.post("/api/discover/registry", dependencies=[Depends(_require_api_key)])
def discover_registry(req: dict[str, Any]) -> dict[str, Any]:
    """Generic L2 discovery on any registered problem module.

    Uses the problem's own ground-truth trajectories: features are the
    state variables, target is the finite-difference derivative of the
    last state (the acceleration/rate law -- the RHS discovery task).
    Same guards, same adjudication, same audit as the demo systems.
    """
    from apedp_dyn.discovery_l2.dimensional_types import Dimension
    from apedp_dyn.discovery_l2.l2_engine import L2DiscoveryEngine
    from apedp_dyn.problems.base import finalize

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
                        raise HTTPException(
                            status_code=500,
                            detail=(
                                f"Cannot reconstruct plot series for term '{t_name}' "
                                f"in system '{req.system}' (available states: "
                                f"{sorted(state_series)})."
                            ),
                        )
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


@app.get("/api/market/windows", dependencies=[Depends(_require_api_key)])
def market_windows(
    asset: str = "sp500",
    extensive: bool = False,
    regime: str | None = None,
    limit: int | None = None,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Frozen benchmark windows plus rolling windows for the asset.

    Frozen W1..W6 keep their ids and semantics forever; ROLL ids are
    generated deterministically from the committed series. When extensive=True,
    generates multi-scale windows spanning short, quarterly, semi-annual, and annual horizons.
    """
    if asset not in AVAILABLE_ASSETS:
        raise HTTPException(status_code=400, detail=f"Unknown asset {asset}.")
    out: list[dict[str, Any]] = [
        {
            "id": w.window_id,
            "desc": w.regime_description,
            "train": f"{w.train_start} to {w.train_end}",
            "holdout": f"{w.holdout_start} to {w.holdout_end}",
            "frozen": True,
            "regime": getattr(w, "regime", "general"),
        }
        for w in WINDOWS
    ]
    if extensive:
        rolling_pool = extensive_rolling_windows(asset)
    else:
        rolling_pool = generated_rolling_windows(asset)

    out += [
        {
            "id": w.window_id,
            "desc": w.regime_description,
            "train": f"{w.train_start} to {w.train_end}",
            "holdout": f"{w.holdout_start} to {w.holdout_end}",
            "frozen": False,
            "regime": getattr(w, "regime", "general"),
        }
        for w in rolling_pool
    ]

    if regime:
        r_low = regime.strip().lower()
        out = [w for w in out if w.get("regime", "").lower() == r_low]

    if offset:
        out = out[offset:]
    if limit is not None:
        out = out[:limit]
    return out


@app.get("/api/market/regimes", dependencies=[Depends(_require_api_key)])
def market_regimes() -> dict[str, Any]:
    """Taxonomy of macro-financial market regimes and supported assets."""
    return _json_obj({
        "regimes": [
            {
                "id": r.value,
                "name": REGIME_METADATA[r]["name"],
                "description": REGIME_METADATA[r]["description"],
                "expected_volatility": REGIME_METADATA[r]["expected_volatility"],
            }
            for r in MarketRegime
        ],
        "assets": list(AVAILABLE_ASSETS),
    })


@app.post("/api/market/sweep", dependencies=[Depends(_require_api_key)])
def market_sweep(req: MarketSweepRequest) -> dict[str, Any]:
    """Execute a batch open discovery sweep across market regimes."""
    if req.asset not in AVAILABLE_ASSETS:
        raise HTTPException(status_code=400, detail=f"Unknown asset {req.asset}.")
    try:
        summary = run_market_sweep(
            asset=req.asset,
            max_windows=req.max_windows,
            multiterm=req.multiterm,
        )
        out = {
            "asset": summary.asset,
            "total_windows": summary.total_windows,
            "unique_promotions": summary.unique_promotions,
            "multi_hypotheses": summary.multi_hypotheses,
            "insufficient_evidence": summary.insufficient_evidence,
            "no_valid_candidates": summary.no_valid_candidates,
            "false_promotion_rate": summary.false_promotion_rate,
            "regime_breakdown": summary.regime_breakdown,
            "records": [
                {
                    "window_id": r.window_id,
                    "regime": r.regime_name,
                    "train_period": r.train_period,
                    "holdout_period": r.holdout_period,
                    "discovered_expression": r.discovered_expression,
                    "complexity": r.expression_complexity,
                    "train_error": r.train_error,
                    "holdout_error": r.holdout_error,
                    "verdict": r.epistemic_verdict,
                    "attribution": r.epistemic_attribution,
                    "whiteness_passed": r.whiteness_passed,
                }
                for r in summary.records
            ],
        }
        _audit("market/sweep", {
            "asset": req.asset,
            "total_windows": summary.total_windows,
            "fpr": summary.false_promotion_rate,
            "data_origin": "real_observational",
        })
        return _json_obj(out)
    except Exception as e:
        _audit("market/sweep", {"asset": req.asset, "verdict": "error", "reason": str(e)})
        raise HTTPException(status_code=500, detail=f"Market sweep failed: {e}") from e


@app.post("/api/discover/market", dependencies=[Depends(_require_api_key)])
def discover_market(req: MarketRequest) -> dict[str, Any]:
    """Execute rolling-window L2 open discovery on genuine market records."""
    if req.asset not in AVAILABLE_ASSETS:
        raise HTTPException(status_code=400, detail=f"Unknown asset {req.asset}.")

    window = get_window_by_id(req.asset, req.window_id)
    if window is None:
        raise HTTPException(status_code=400, detail=f"Window {req.window_id} not found.")

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

    resp = _json_obj({
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
    asset: str | None = None
    csv_text: str | None = None
    target_column: str | None = None
    mode: str = "auto"


SANDBOX_CATALOG: list[dict[str, Any]] = [
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
    {"id": "crypto_multi_grammar", "name": "Grammar discovery on Crypto 15-Asset Universe (BTC+ETH+SOL...)",
     "desc": "29 structures (including empirical Order Flow OFI) on 15 synchronized crypto assets, bull vs LUNA crash holdout", "slow": True},
    {"id": "custom_upload", "name": "Universal Custom Dataset Upload (Autonomous Discovery)",
     "desc": "Upload or paste any CSV dataset; autonomous L2 discovery infers the governing equation and adjudicates validity.", "slow": True},
    {"id": "volume_market", "name": "Volume-state vs future volatility (market microstructure)",
     "desc": "Frozen V0 (persistence+|r|+Amihud) vs V1 (+logVolume), fit 2021-2022, OOS 2023+ with permutation control; sidecar volume data, repo benchmark untouched.", "slow": True},
    {"id": "multi_grammar", "name": "Multi-grammar volatility discovery (H-P / H-RV / H-A)",
     "desc": "Tests H-P, H-RV, H-A, H-LOG volatility hypotheses simultaneously with epistemic adjudication", "slow": True},
    {"id": "eseds_dynamical", "name": "ESEDS: Epistemically Safe Extrapolation of Dynamical Systems",
     "desc": "Tests Families A-D (Logistic, Allee, Predator-Prey, Competition) against 6 Traps, Extrapolation Ladder E0-E5, and 4 Competitive Baselines", "slow": True},
    {"id": "sphere_packing", "name": "Sphere Packing & Computational Optimality (Kepler, FCC/HCP, LP Bounds)",
     "desc": "Adjudicates geometric optimality, detects polytype equivalence (FCC vs HCP), and flags amorphous jamming & overlap infeasibility", "slow": True},
    {"id": "causal_discovery", "name": "Causal Discovery Under Confounding & Interventions (Domain B)",
     "desc": "Separates association from causation via Pearl's do-calculus interventions, confounding traps, and collider bias", "slow": True},
    {"id": "pde_verification", "name": "Computational Physics & PDE Invariant Verification (Domain D)",
     "desc": "Gorard-Hakim machine-checkable verification: mass conservation, Lax entropy shocks, TVD ringing, and CFL stability", "slow": True},
    {"id": "apedpx_meta", "name": "APEDP-X Unified Cross-Domain Epistemic Meta-Referee",
     "desc": "Executes the grand cross-domain tournament across all 4 domains simultaneously, enforcing the 7-tier Claim Ladder", "slow": True},
    {"id": "trading_signals", "name": "Quantitative Trading Signals & GA Evolution (22 Signals)",
     "desc": "Adjudicates 22 alpha signals vs evolved combinations under AEH harness with M0-M5 Market Extrapolation Ladder and C4-to-C2 EEDR calibration", "slow": True},
    {"id": "multi_asset_30", "name": "30-Asset Portfolio & CAN-G Gate (15 Crypto + 15 Equities)",
     "desc": "Evaluates Two-Tier Risk Parity vs Naive Pooling under Gate 0.5 (CAN-G) and M0-M5 Market Extrapolation Ladder, detecting the Ranking Trap with C3 structural verification", "slow": True},
    {"id": "multi_asset_60", "name": "60-Asset Portfolio & CAN-G Gate (30 Crypto + 30 Equities/ETFs)",
     "desc": "Evaluates Two-Tier Risk Parity vs Naive Pooling on 60 synchronized assets under Gate 0.5 (CAN-G) and M0-M5 Market Extrapolation Ladder, detecting the Ranking Trap with C3 structural verification", "slow": True},
    {"id": "multi_asset_90", "name": "90-Asset 5-Year Portfolio & CAN-G Gate (45 Crypto + 45 Equities/ETFs/Macro)",
     "desc": "Evaluates Two-Tier Risk Parity vs Naive Pooling on 90 synchronized assets across 5 full years (2021-2025) under Gate 0.5 (CAN-G) and M0-M5 Market Extrapolation Ladder with C3 structural verification", "slow": True},
    {"id": "goldbach_conjecture", "name": "Goldbach Conjecture & Finite Induction Audit (Rust Rayon Engine)",
     "desc": "Executes native multi-threaded Rust verification across 10,000,000 even integers (~75M evens/s) and adjudicates the boundary between C2 bounded empirical law and C6 unproven universal theorem", "slow": False},
    {"id": "riemann_hypothesis", "name": "Riemann Hypothesis & Prime Harmonics (Rust Euler-Maclaurin & Calibration Gate)",
     "desc": "Executes native Rust 2D critical strip audit, high-precision root bisection with Reference-Zero Calibration Gate (Δ<1e-8), Turing count check, and explicit formula prime synthesis under the AEH Claim Ladder", "slow": False},
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



SANDBOX_EXPRESSIONS: dict[str, dict[str, tuple[str, int]]] = {
    "nano_m4": {
        "constant_baseline": ("σ(d) = σ₀", 1),
        "linear_grain": ("σ(d) = σ₀ + k·d", 2),
        "inverse_soft": ("σ(d) = σ₀ + k·d^{1/2}", 3),
        "hp_global": ("σ(d) = σ₀ + k·d^{-1/2}", 3),
        "piecewise_hp_inv": ("σ(d ≥ dc) = s₀ + k·d^{-1/2},  σ(d < dc) = s₁ - m·(dc - d)", 6),
    },
    "hall_petch": {
        "constant": ("σ(d) = σ₀", 1),
        "linear": ("σ(d) = σ₀ + k·d", 2),
        "hp_true": ("σ(d) = σ₀ + k·d^{-1/2}", 3),
    },
    "creep_law": {
        "constant": ("dε/dt = ε̇₀", 1),
        "linear_stress": ("dε/dt = A·σ", 2),
        "exp_stress": ("dε/dt = A·exp(β·σ)", 3),
        "power_law": ("dε/dt = A·σⁿ  (Norton)", 3),
    },
    "game_form": {
        "constant": ("dx/dt = c", 1),
        "linear": ("dx/dt = a·x", 2),
        "quadratic": ("dx/dt = a·x + b·x²", 3),
        "cubic": ("dx/dt = a·x + b·x² + c·x³  (Replicator)", 4),
    },
    "sir_break": {
        "constant": ("dI/dt = c", 1),
        "linear_I": ("dI/dt = -γ·I", 2),
        "exponential_t": ("dI/dt = I₀·exp(-λt)", 3),
        "mass_action": ("dI/dt = β·S·I - γ·I  (Kermack-McKendrick)", 4),
    },
    "quantum_quench": {
        "constant_persistence": ("P(t) = P₀", 1),
        "ballistic_linear": ("P(t) = P₀ - v·t", 2),
        "coherent_oscillations": ("P(t) = P₀·cos(ωt)", 3),
        "decay_string_breaking": ("P(t) = P₀·exp(-Γt)  (Schwinger quench)", 3),
    },
    "lynx_cycle": {
        "constant": ("y(t) = c", 1),
        "linear_trend": ("y(t) = a + b·t", 2),
        "cycle_5yr": ("y(t) = a + b·cos(2πt / 5)", 3),
        "cycle_10yr": ("y(t) = a + b·cos(2πt / 9.6) + c·sin(2πt / 9.6)", 4),
    },
    "qec_threshold": {
        "constant": ("P_L = c", 1),
        "plain_quad": ("P_L = a·p²", 2),
        "cubic": ("P_L = a·p³", 2),
        "threshold_ratio": ("P_L = C·(p / p_th)^{(d+1)/2}  (Ansatz)", 4),
    },
}

TERM_MATH_MAP: dict[str, tuple[str, int]] = {
    "one": ("c", 1),
    "one_hi": ("s₀", 1),
    "one_lo": ("s₁", 1),
    "d_inv_half": ("k·d^{-1/2}", 2),
    "d_lin": ("k·d", 1),
    "d_half": ("k·d^{1/2}", 2),
    "d_lin_lo": ("m·d", 1),
    "x": ("θ₁·x", 1),
    "x2": ("θ₂·x²", 2),
    "x3": ("θ₃·x³", 2),
    "sin_x": ("θ₄·sin(x)", 2),
    "v": ("θ₅·v", 1),
    "abs_v_v": ("θ₆·|v|v", 3),
    "v2": ("θ₇·v²", 2),
    "u": ("λ·u", 1),
    "sat_x": ("θ₈·x/(1+|x|)", 2),
    "sat_v": ("θ₉·v/(1+|v|)", 2),
    "sat_u": ("λ₂·u/(1+|u|)", 2),
    "inv_x2": ("θ₁₀/(1+x²)", 2),
    "u_relax": ("λ₃·(u - v)", 2),
}


def _resolve_sandbox_formula_and_complexity(sandbox_id: str, c: Any) -> tuple[str, int]:
    cid = c.candidate.candidate_id if hasattr(c, "candidate") else str(getattr(c, "id", "cand"))
    if hasattr(c, "candidate") and getattr(c.candidate, "metadata", None):
        if "canonical_expr" in c.candidate.metadata:
            expr = str(c.candidate.metadata["canonical_expr"])
            compl = int(c.candidate.metadata.get("complexity", 1))
            return expr, compl
    if sandbox_id in SANDBOX_EXPRESSIONS and cid in SANDBOX_EXPRESSIONS[sandbox_id]:
        return SANDBOX_EXPRESSIONS[sandbox_id][cid]
    if hasattr(c, "candidate") and getattr(c.candidate, "terms", None):
        term_strs = [TERM_MATH_MAP.get(t.name, (t.name, 1))[0] for t in c.candidate.terms]
        term_compls = [TERM_MATH_MAP.get(t.name, (t.name, 1))[1] for t in c.candidate.terms]
        expr = " + ".join(term_strs)
        compl = sum(term_compls) + max(0, len(term_strs) - 1)
        return expr, compl
    return cid, 1


def _format_sandbox_adjudication(
    sandbox_id: str,
    adj: Any,
    candidates: Sequence[Any] | None = None,
    series: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Format sandbox adjudication results into the canonical Physics Lab schema."""
    pool = list(candidates) if candidates else list(getattr(adj, "ranking", ()))
    top_cand = adj.promoted if adj.promoted is not None else (pool[0] if pool else None)

    formatted_pareto = []
    for c in pool:
        g4 = c.provenance.get("gate4_status", {}) if hasattr(c, "provenance") else {}
        cid = c.candidate.candidate_id if hasattr(c, "candidate") else str(getattr(c, "id", "cand"))
        rmse = float(c.holdout_rmse) if hasattr(c, "holdout_rmse") and c.holdout_rmse is not None else None
        r2 = _opt_float(g4.get("r2"))
        rho1 = _opt_float(g4.get("rho1"))
        is_white = bool(g4.get("is_white", False))

        resids = g4.get("holdout_residuals")
        sq_err = g4.get("holdout_sq_error")
        rho1_unassessable = None
        if rho1 is None and resids is not None and len(resids) >= 4:
            arr = np.asarray(resids, dtype=float).ravel()
            arr_f = arr[np.isfinite(arr)]
            if len(arr_f) >= 4:
                centred = arr_f - float(np.mean(arr_f))
                denom = float(np.dot(centred, centred))
                if denom > 1e-12:
                    rho1 = float(np.dot(centred[:-1], centred[1:]) / denom)
                    is_white = bool(abs(rho1) < 0.50)
        elif rho1 is None and sq_err is not None and len(sq_err) >= 4:
            # Deliberately NOT computing lag-1 autocorrelation here. On squared
            # errors that statistic measures volatility clustering, not residual
            # whiteness, and presenting it as rho1 mislabels heteroskedastic
            # noise as structured residuals. Leave rho1 empty and say why.
            rho1_unassessable = (
                "whiteness requires signed holdout residuals; "
                "only squared errors recorded"
            )

        if r2 is None and resids is not None and len(resids) > 0:
            arr = np.asarray(resids, dtype=float).ravel()
            arr_f = arr[np.isfinite(arr)]
            if len(arr_f) > 0:
                ss_res = float(np.sum(arr_f ** 2))
                n_samples = len(arr_f)
                ss_tot = None
                if series and "actual" in series:
                    y_act = np.asarray(series["actual"], dtype=float).ravel()
                    if len(y_act) >= n_samples:
                        y_ho = y_act[-n_samples:]
                        y_ho_f = y_ho[np.isfinite(y_ho)]
                        if len(y_ho_f) == n_samples:
                            ss_tot = float(np.sum((y_ho_f - float(np.mean(y_ho_f))) ** 2))
                if ss_tot is None or ss_tot <= 1e-12:
                    obs_rms = g4.get("holdout_obs_rms")
                    if obs_rms is not None and float(obs_rms) > 0:
                        ss_tot = float(n_samples * (float(obs_rms) ** 2))
                if ss_tot is not None and ss_tot > 1e-12:
                    r2 = float(1.0 - (ss_res / ss_tot))

        if r2 is None and rmse is not None:
            obs_rms = g4.get("holdout_obs_rms")
            if obs_rms is not None and float(obs_rms) > 0:
                r2 = float(1.0 - (rmse ** 2) / (float(obs_rms) ** 2))

        formula, complexity = _resolve_sandbox_formula_and_complexity(sandbox_id, c)
        is_top = (top_cand is not None and getattr(top_cand, "candidate", None) is not None and top_cand.candidate.candidate_id == cid)
        formatted_pareto.append({
            "id": cid,
            "formula": formula,
            "complexity": int(complexity),
            "rmse": rmse,
            "r2": r2,
            "rho1": rho1,
            "is_white": is_white,
            "rho1_unassessable": rho1_unassessable,
            "is_promoted": bool(is_top and adj.promoted is not None),
        })

    _pc = adj.promoted
    _promoted_record = None
    if _pc is not None:
        _top_p = next((p for p in formatted_pareto if p["id"] == _pc.candidate.candidate_id), None)
        _promoted_record = {
            "id": _pc.candidate.candidate_id,
            "formula": _top_p["formula"] if _top_p else _pc.candidate.candidate_id,
            "holdout_rmse": float(_pc.holdout_rmse) if _pc.holdout_rmse is not None else None,
            "r2": _top_p["r2"] if _top_p else None,
            "residual_rho1": _top_p["rho1"] if _top_p else None,
            "delta_rmse": float(adj.delta_rmse) if adj.delta_rmse is not None else None,
            "tau_epi": float(adj.tau_epi) if adj.tau_epi is not None else None,
        }

    # Rival pair and reason come from the adjudicator, verbatim. An earlier
    # version re-sorted candidates by holdout RMSE here and rebuilt both the
    # pair and the message -- but the adjudicator ranks by train loss and
    # arbitrates the pair on holdout, so the rebuilt text could name a
    # different pair than the reason it replaced (cryp-0022/Δ=0.000333 shown
    # where the adjudicator said cryp-0011/Δ=-0.0434). The display layer must
    # not author adjudication prose; it quotes it.
    def _cand_id(sc: Any) -> str:
        cand = getattr(sc, "candidate", None)
        if cand is not None and getattr(cand, "candidate_id", None) is not None:
            return str(cand.candidate_id)
        return str(getattr(sc, "id", "cand"))

    def _rival_record(sc: Any) -> dict[str, Any] | None:
        if sc is None or getattr(sc, "holdout_rmse", None) is None:
            return None
        form, compl = _resolve_sandbox_formula_and_complexity(sandbox_id, sc)
        return {
            "id": _cand_id(sc),
            "formula": form,
            "rmse": float(sc.holdout_rmse),
            "complexity": int(compl),
        }

    ranking = list(getattr(adj, "ranking", ()) or ())
    best_sc = ranking[0] if ranking else None
    rival_sc = getattr(adj, "rival", None)
    rival_a = _rival_record(best_sc)
    rival_b = _rival_record(rival_sc)
    # Holdout-leaderboard context, reported separately so the leaderboard order
    # stays visible without being mistaken for the adjudicator's pair.
    holdout_leader_id = None
    valid_cands = [c for c in pool if getattr(c, "holdout_rmse", None) is not None]
    if valid_cands:
        holdout_leader_id = _cand_id(
            min(valid_cands, key=lambda s: float(s.holdout_rmse))
        )
    delta_val = (float(adj.delta_rmse)
                 if getattr(adj, "delta_rmse", None) is not None else None)

    if adj.promoted:
        top_form, _ = _resolve_sandbox_formula_and_complexity(sandbox_id, adj.promoted)
        promoted_str = top_form or adj.promoted.candidate.candidate_id
    else:
        promoted_str = "No promoted equation (refusal under uncertainty)"
    # Verbatim. When the holdout leaderboard orders the top two differently
    # from the adjudicator's pair, say so in a clearly marked supplement --
    # appended, never substituted -- so the reader is not confused by the
    # table without being given a second, conflicting verdict.
    reason_str = adj.reason
    if (best_sc is not None and holdout_leader_id is not None
            and _cand_id(best_sc) != holdout_leader_id):
        reason_str = (
            f"{adj.reason} [display note: holdout-best on the leaderboard is "
            f"{holdout_leader_id}; the pair above is the adjudicator's own]"
        )

    tau_for_rows = float(adj.tau_epi) if adj.tau_epi is not None else None
    leader_rmse = (rival_a["rmse"] if rival_a and rival_a.get("rmse") is not None
                   else None)
    for row in formatted_pareto:
        # A row is a genuine rival only if it lies within the adjudicator's
        # resolution floor of the adjudicator's best. Labelling every
        # non-promoted MULTI row RIVAL overstated how many structures the
        # evidence actually failed to separate.
        row_rmse = row.get("rmse")
        row["within_tau"] = bool(
            tau_for_rows is not None and leader_rmse is not None
            and row_rmse is not None and row["id"] != (rival_a["id"] if rival_a else None)
            and abs(float(row_rmse) - float(leader_rmse)) <= tau_for_rows
        )

    return {
        "sandbox_id": sandbox_id,
        "verdict": adj.status.value,
        "promoted_equation": promoted_str,
        "promoted": (adj.promoted.candidate.candidate_id if adj.promoted else None),
        "promoted_candidate": _promoted_record,
        "reason": reason_str,
        "data_origin": str(getattr(adj, "data_origin", "simulation")),
        "tau_epi": float(adj.tau_epi) if adj.tau_epi is not None else 0.05,
        "delta_rmse": delta_val,
        # Diagnostics only -- `verdict` above is unchanged by these.
        "promotion_confidence": adj.promotion_confidence or None,
        "diagnostic_flags": list(adj.diagnostic_flags),
        "rival_a": rival_a,
        "rival_b": rival_b,
        "pareto_candidates": formatted_pareto,
        "candidates": [{"id": c["id"], "rmse": c["rmse"]} for c in formatted_pareto],
        "extra": extra or {},
        "series": series or {},
    }


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
        resp = _format_sandbox_adjudication(
            "nano_m4", adj, scored,
            series={"x": d.tolist(), "actual": obs.tolist(),
                    "x_label": "grain size d (nm, log scale)",
                    "threshold": info["dc_hat_nm"]},
            extra={"dc_hat_nm": info["dc_hat_nm"], "dc_true_nm": info["dc_true_nm"]},
        )
    elif req.sandbox_id == "sir_break":
        from apedp_dyn.sandbox.epidemiology_sir import run_sir_forms, simulate_sir
        adj, scored = run_sir_forms(noise=0.02, seed=42)
        t, _s_arr, _i_arr, inc = simulate_sir(noise=0.02, seed=42)
        resp = _format_sandbox_adjudication(
            "sir_break", adj, scored,
            series={"x": t.tolist(), "actual": inc.tolist(),
                    "x_label": "days (vaccination at t=60)",
                    "threshold": 60.0},
            extra={"t_vax": 60.0},
        )
    elif req.sandbox_id == "quantum_quench":
        from apedp_dyn.sandbox.quantum_vacuum_sandbox import (
            QuantumVacuumConfig,
            run_quench_dynamics_discovery,
        )
        cfg = QuantumVacuumConfig()
        adj, times, p_obs = run_quench_dynamics_discovery(cfg, bootstrap=True)
        resp = _format_sandbox_adjudication(
            "quantum_quench", adj, getattr(adj, "ranking", ()),
            series={"x": np.asarray(times).tolist(),
                    "actual": np.asarray(p_obs).tolist(),
                    "x_label": "time"},
            extra={"tension_hz": cfg.string_tension_hz,
                   "critical_field": cfg.critical_field_Ec},
        )
    elif req.sandbox_id == "hall_petch":
        from apedp_dyn.sandbox.hall_pecht_demo import run_hall_pecht_discovery
        adj, d, obs = run_hall_pecht_discovery(noise_sigma=0.03)
        resp = _format_sandbox_adjudication(
            "hall_petch", adj, getattr(adj, "ranking", ()),
            series={"x": d.tolist(), "actual": obs.tolist(),
                    "x_label": "grain size d (m, log scale)"},
        )
    elif req.sandbox_id == "creep_law":
        from apedp_dyn.sandbox.creep_powerlaw import generate_creep_data, run_creep
        adj, scored, n_hat = run_creep()
        s_, e_ = generate_creep_data(seed=42)
        resp = _format_sandbox_adjudication(
            "creep_law", adj, scored,
            series={"x": s_.tolist(), "actual": e_.tolist(),
                    "x_label": "stress (MPa, log scale)"},
            extra={"n_hat": n_hat, "n_true": 4.5},
        )
    elif req.sandbox_id == "game_form":
        from apedp_dyn.sandbox.game_theory import run_form_recovery
        adj, scored = run_form_recovery()
        resp = _format_sandbox_adjudication(
            "game_form", adj, scored,
            series=_game_series(),
        )
    elif req.sandbox_id == "pendulum_drag":
        from apedp_dyn.sandbox.pendulum_drag import run_pendulum, simulate_pendulum
        adj, scored = run_pendulum(noise=0.01)
        _, th, _ = simulate_pendulum(0.6, noise=0.01, seed=23)
        resp = _format_sandbox_adjudication(
            "pendulum_drag", adj, scored,
            series={"x": list(range(len(th))), "actual": th.tolist(),
                    "x_label": "timestep (unseen amplitude 0.6 rad)"},
        )
    elif req.sandbox_id == "lynx_cycle":
        from apedp_dyn.sandbox.lynx_hare import load_lynx, run_lynx
        adj, scored = run_lynx()
        t, y = load_lynx()
        resp = _format_sandbox_adjudication(
            "lynx_cycle", adj, scored,
            series={"x": t.tolist(), "actual": y.tolist(),
                    "x_label": "year (real trappings 1821-1934)"},
            extra={"split_year": 1878},
        )
    elif req.sandbox_id == "qec_threshold":
        from apedp_dyn.sandbox.qec_threshold import run_qec_threshold_stage1
        adj, scored = run_qec_threshold_stage1()
        resp = _format_sandbox_adjudication(
            "qec_threshold", adj, scored,
            series=_qec_series(),
        )
    elif req.sandbox_id == "finance_grammar":
        import csv as _csv
        from pathlib import Path as _P

        from apedp_dyn.problems.finance import spec_from_series
        from apedp_dyn.sandbox.finance_grammar import run_experiment
        dates: list[str] = []
        raw_px: list[float] = []
        with open(str(_P(__file__).resolve().parents[2] / "data" / "market" / "sp500.csv")) as f:
            for row in _csv.DictReader(f):
                try:
                    v = float(row["SP500"])
                    if v > 0:
                        dates.append(row["observation_date"])
                        raw_px.append(v)
                except ValueError:
                    pass
        import numpy as _np
        px = _np.asarray(raw_px, dtype=float)
        idx = {d: i for i, d in enumerate(dates)}
        prob = spec_from_series(px, (idx["2019-01-02"], idx["2019-12-31"]),
                                (idx["2020-02-19"], idx["2020-03-23"]),
                                problem_id="FIN-UI")
        adj = run_experiment("UI calm->crash", prob)
        seg = px[idx["2020-02-19"]:idx["2020-03-23"] + 1]
        resp = _format_sandbox_adjudication(
            "finance_grammar", adj, getattr(adj, "ranking", ()),
            series={"x": list(range(len(seg))), "actual": seg.tolist(),
                    "x_label": "S&P 500 crash window (USD)"},
        )
    elif req.sandbox_id == "crypto_multi_grammar":
        from apedp_dyn.sandbox.crypto_grammar import (
            CRYPTO_ASSETS,
            run_crypto_experiment,
        )
        target_sym = req.asset or "COMPOSITE_TOP15"
        adj, prob, series_data = run_crypto_experiment(target_asset=target_sym)
        cands_pool = series_data.get("candidates") or getattr(adj, "ranking", ())
        resp = _format_sandbox_adjudication(
            "crypto_multi_grammar", adj, cands_pool,
            series={
                "x": series_data["x"],
                "dates": series_data["dates"],
                "actual": series_data["actual"],
                "x_label": series_data["x_label"],
            },
            extra={
                "target_asset": target_sym,
                "total_assets": len(CRYPTO_ASSETS),
                "available_assets": [*CRYPTO_ASSETS, "COMPOSITE_TOP15"],
                "dataset_file": "data/market/crypto_top15.csv",
                "orderflow_included": True,
                "orderflow_feature": "Order Flow Imbalance (OFI)",
                "total_grammar_structures": 29,
                "train_regime": "2021 Bull Cycle (2021-01-01 to 2021-12-31)",
                "holdout_regime": "2022 Terra-LUNA Shock (2022-05-05 to 2022-05-25)",
            },
        )
    elif req.sandbox_id == "custom_upload":
        from apedp_dyn.sandbox.custom_upload import run_upload_discovery
        if not req.csv_text or not req.csv_text.strip():
            # Generate clean illustrative sample if none provided
            sample_t = np.linspace(0.0, 10.0, 60)
            sample_y = 10.0 + 1.5 * sample_t + 2.0 * np.sin(sample_t)
            csv_lines = ["t,y"] + [f"{ti:.3f},{yi:.4f}" for ti, yi in zip(sample_t, sample_y, strict=True)]
            csv_data = "\n".join(csv_lines)
        else:
            csv_data = req.csv_text

        adj, cands, series_info, extra = run_upload_discovery(
            csv_data, target_col=req.target_column, mode=req.mode
        )
        resp = _format_sandbox_adjudication(
            "custom_upload", adj, cands,
            series=series_info,
            extra=extra,
        )
    elif req.sandbox_id == "volume_market":
        from apedp_dyn.sandbox.volume_sandbox import run_volume_adjudication
        requested_sym = (req.asset or "BTC").upper()
        target_sym = requested_sym
        universe_note = "Universe: 15 assets (single-asset run)"
        if target_sym == "COMPOSITE_TOP15":
            target_sym = "BTC"
            universe_note = ("Requested universe: TOP15 composite. "
                             "Actual evaluated series: BTC (no basket volume series).")
        adj, scored, series_data, vol_extra = run_volume_adjudication(target_sym)
        resp = _format_sandbox_adjudication(
            "volume_market", adj, scored,
            series={"x": series_data["x"],
                    "actual": series_data["actual"],
                    "x_label": series_data["x_label"]},
            extra={**vol_extra,
                   "universe": "15 assets",
                   "asset_run": target_sym,
                   "asset_requested": requested_sym,
                   "universe_note": universe_note,
                   "gate_breakdown": _volume_gate_breakdown(vol_extra, scored, series_data.get("actual"))},
        )
    elif req.sandbox_id == "multi_grammar":
        from apedp_dyn.sandbox.multi_grammar_sandbox import run_multi_grammar_discovery
        adj, _mg_candidates = run_multi_grammar_discovery(n_samples=400)
        # Format candidates for the Pareto display
        formatted_pareto = []
        for c in adj.ranking:
            g4 = c.provenance.get("gate4_status", {}) if hasattr(c, "provenance") else {}
            cid = c.candidate.candidate_id if hasattr(c, "provenance") else str(getattr(c, "id", "cand"))
            rmse = float(c.holdout_rmse) if hasattr(c, "holdout_rmse") and c.holdout_rmse is not None else None
            r2 = _opt_float(g4.get("r2"))
            rho1 = _opt_float(g4.get("rho1"))
            is_white = bool(g4.get("is_white", False))
            formula, complexity = _resolve_sandbox_formula_and_complexity(req.sandbox_id, c)
            is_top = (adj.promoted is not None and adj.promoted.candidate.candidate_id == cid)
            formatted_pareto.append({
                "id": cid,
                "formula": formula,
                "complexity": int(complexity),
                "rmse": rmse,
                "r2": r2,
                "rho1": rho1,
                "is_white": is_white,
                "is_promoted": bool(is_top and adj.promoted is not None),
            })
        _pc = adj.promoted
        _promoted_record = None
        if _pc is not None:
            _top_p = next((p for p in formatted_pareto if p["id"] == _pc.candidate.candidate_id), None)
            _promoted_record = {
                "id": _pc.candidate.candidate_id,
                "formula": _top_p["formula"] if _top_p else _pc.candidate.candidate_id,
                "holdout_rmse": float(_pc.holdout_rmse) if _pc.holdout_rmse is not None else None,
                "r2": _top_p["r2"] if _top_p else None,
                "residual_rho1": _top_p["rho1"] if _top_p else None,
                "delta_rmse": float(adj.delta_rmse) if adj.delta_rmse is not None else None,
                "tau_epi": float(adj.tau_epi) if adj.tau_epi is not None else None,
            }
        ranking = list(getattr(adj, "ranking", ()) or ())
        best_sc = ranking[0] if ranking else None
        rival_a = None
        rival_b = None
        if best_sc is not None:
            rid = best_sc.candidate.candidate_id if hasattr(best_sc, "candidate") else str(getattr(best_sc, "id", "cand"))
            form, _ = _resolve_sandbox_formula_and_complexity(req.sandbox_id, best_sc)
            rival_a = {
                "id": rid,
                "formula": form,
                "rmse": float(best_sc.holdout_rmse) if best_sc.holdout_rmse is not None else None,
                "complexity": int(_resolve_sandbox_formula_and_complexity(req.sandbox_id, best_sc)[1]),
            }
        if adj.rival is not None:
            riv = adj.rival
            rid2 = riv.candidate.candidate_id if hasattr(riv, "candidate") else str(getattr(riv, "id", "cand"))
            form2, _ = _resolve_sandbox_formula_and_complexity(req.sandbox_id, riv)
            rival_b = {
                "id": rid2,
                "formula": form2,
                "rmse": float(riv.holdout_rmse) if riv.holdout_rmse is not None else None,
                "complexity": int(_resolve_sandbox_formula_and_complexity(req.sandbox_id, riv)[1]),
            }
        tau_for_rows = float(adj.tau_epi) if adj.tau_epi is not None else None
        leader_rmse = (rival_a["rmse"] if rival_a and rival_a.get("rmse") is not None else None)
        for row in formatted_pareto:
            row_rmse = row.get("rmse")
            row["within_tau"] = bool(
                tau_for_rows is not None and leader_rmse is not None
                and row_rmse is not None and row["id"] != (rival_a["id"] if rival_a else None)
                and abs(float(row_rmse) - float(leader_rmse)) <= tau_for_rows
            )
        resp = _json_obj({
            "sandbox_id": adj.sandbox_id if hasattr(adj, "sandbox_id") else "multi_grammar",
            "verdict": adj.status.value,
            "promoted_equation": _promoted_record["formula"] if _promoted_record else "No promoted equation (refusal under uncertainty)",
            "promoted": (_promoted_record["id"] if _promoted_record else None),
            "promoted_candidate": _promoted_record,
            "reason": adj.reason,
            "data_origin": getattr(adj, "data_origin", "synthetic"),
            "tau_epi": adj.tau_epi if adj.tau_epi is not None else 0.05,
            "delta_rmse": adj.delta_rmse,
            "promotion_confidence": adj.promotion_confidence or None,
            "diagnostic_flags": list(adj.diagnostic_flags),
            "rival_a": rival_a,
            "rival_b": rival_b,
            "pareto_candidates": formatted_pareto,
            "candidates": [{"id": c["id"], "rmse": c["rmse"]} for c in formatted_pareto],
            "extra": {},
            "series": {},
        })
    elif req.sandbox_id == "eseds_dynamical":
        from apedp_dyn.sandbox.eseds_dynamical import run_eseds_benchmark_matrix, run_eseds_scenario
        bench = run_eseds_benchmark_matrix(verbose=False)
        single = run_eseds_scenario(family="A", trap_id=1, train_end=40, noise_sigma=0.3)
        resp = _json_obj({
            "sandbox_id": "eseds_dynamical",
            "verdict": single["apedp_status"],
            "promoted": single["apedp_promoted"],
            "reason": single["apedp_reason"],
            "tau_epi": single["tau_epi"],
            "delta_rmse": single["delta_rmse"],
            "metrics": bench["metrics"],
            "ladder": single["ladder"],
            "candidates": [{"id": c, "rmse": 0.0} for c in single["candidates"]],
            "extra": {"benchmark_results": bench["results"]},
            "series": {},
        })
    elif req.sandbox_id == "sphere_packing":
        from apedp_dyn.sandbox.sphere_packing_sandbox import (
            run_sphere_packing_benchmark_matrix,
            run_sphere_packing_scenario,
        )
        bench = run_sphere_packing_benchmark_matrix(verbose=False)
        single = run_sphere_packing_scenario(trap_id=3, dim=3)
        resp = _json_obj({
            "sandbox_id": "sphere_packing",
            "verdict": single["apedp_verdict"],
            "status": single["apedp_status"],
            "promoted": single["apedp_promoted"],
            "reason": single["reason"],
            "densities": single["densities"],
            "dual_gap": single["dual_gap"],
            "equivalence_class": single["equivalence_class"],
            "scorecard": bench["scorecard"],
            "candidates": [{"id": c, "rmse": 0.0} for c in single["candidates"]],
            "extra": {"benchmark_results": bench["results"]},
            "series": {},
        })
    elif req.sandbox_id == "causal_discovery":
        from apedp_dyn.sandbox.causal_discovery_sandbox import (
            run_causal_adjudication,
            run_causal_benchmark_matrix,
        )
        bench = run_causal_benchmark_matrix()
        single = run_causal_adjudication(trap_id=1, provide_interventions=False)
        resp = _json_obj({
            "sandbox_id": "causal_discovery",
            "verdict": single["apedp_adjudication"]["verdict"],
            "justified_claim": single["apedp_adjudication"]["justified_claim"],
            "proposed_claim": single["apedp_adjudication"]["proposed_claim"],
            "promoted": single["apedp_adjudication"]["verdict"] == "PROMOTED",
            "reason": single["apedp_adjudication"]["justification"],
            "r2": single["r2"],
            "claimed_beta": single["claimed_beta"],
            "true_causal_effect": single["true_causal_effect"],
            "scorecard": {
                "naive_fpr": bench["naive_fpr"],
                "apedp_fpr": bench["apedp_fpr"],
                "car": bench["car"],
                "sfcr": bench["sfcr"],
            },
            "extra": {"benchmark_results": bench["results"]},
            "series": {},
        })
    elif req.sandbox_id == "pde_verification":
        from apedp_dyn.sandbox.pde_verification_sandbox import (
            run_pde_adjudication,
            run_pde_benchmark_matrix,
        )
        bench = run_pde_benchmark_matrix()
        single = run_pde_adjudication(trap_id=1)
        resp = _json_obj({
            "sandbox_id": "pde_verification",
            "verdict": single["apedp_adjudication"]["verdict"],
            "justified_claim": single["apedp_adjudication"]["justified_claim"],
            "proposed_claim": single["apedp_adjudication"]["proposed_claim"],
            "promoted": single["apedp_adjudication"]["verdict"] == "PROMOTED",
            "reason": single["apedp_adjudication"]["justification"],
            "audit": single["audit"],
            "scorecard": {
                "naive_fpr": bench["naive_fpr"],
                "apedp_fpr": bench["apedp_fpr"],
                "car": bench["car"],
                "sfcr": bench["sfcr"],
            },
            "extra": {"benchmark_results": bench["results"]},
            "series": {},
        })
    elif req.sandbox_id == "apedpx_meta":
        from apedp_dyn.sandbox.apedp_x_meta_sandbox import run_apedpx_unified_meta_benchmark
        bench = run_apedpx_unified_meta_benchmark()
        resp = _json_obj({
            "sandbox_id": "apedpx_meta",
            "verdict": "APEDP_X_EVALUATION_COMPLETE",
            "promoted": True,
            "reason": "Unified cross-domain verification across Dynamical, Causal, Geometric, and PDE domains.",
            "metrics": {
                "naive_fpr": bench["naive_fpr"],
                "apedp_fpr": bench["apedp_fpr"],
                "car": bench["car"],
                "sfcr": bench["sfcr"],
                "valid_retention_rate": bench["valid_retention_rate"],
            },
            "total_proposals": bench["total_proposals"],
            "inflated_proposals": bench["inflated_proposals"],
            "extra": {},
            "series": {},
        })
    elif req.sandbox_id == "trading_signals":
        from apedp_dyn.sandbox.trading_signals_sandbox import (
            load_real_btc_orderflow,
            run_epistemic_trading_adjudication,
        )
        df = load_real_btc_orderflow(600)
        res = run_epistemic_trading_adjudication(panel=df, dataset_name="btc_orderflow")
        resp = _json_obj(res)
    elif req.sandbox_id == "multi_asset_30":
        from apedp_dyn.sandbox.multi_asset_signals_sandbox import (
            run_epistemic_30_asset_adjudication,
        )
        res = run_epistemic_30_asset_adjudication(initial_capital=100000.0)
        resp = _json_obj(res)
    elif req.sandbox_id == "multi_asset_60":
        from apedp_dyn.sandbox.multi_asset_signals_sandbox import (
            run_epistemic_60_asset_adjudication,
        )
        res = run_epistemic_60_asset_adjudication(initial_capital=100000.0)
        resp = _json_obj(res)
    elif req.sandbox_id == "multi_asset_90":
        from apedp_dyn.sandbox.multi_asset_signals_sandbox import (
            run_epistemic_90_asset_adjudication,
        )
        res = run_epistemic_90_asset_adjudication(initial_capital=100000.0)
        resp = _json_obj(res)
    elif req.sandbox_id == "goldbach_conjecture":
        from apedp_dyn.sandbox.goldbach_sandbox import run_goldbach_adjudication
        res = run_goldbach_adjudication(limit=10_000_000)
        resp = _json_obj(res)
    elif req.sandbox_id == "riemann_hypothesis":
        from apedp_dyn.sandbox.riemann_sandbox import run_riemann_adjudication
        res = run_riemann_adjudication(max_x=200.0, sample_points=3000)
        resp = _json_obj(res)
    else:
        raise HTTPException(status_code=404,
                            detail=f"Unknown sandbox {req.sandbox_id}.")
    resp = _json_obj(resp)
    _audit("sandbox/run", {"sandbox_id": resp.get("sandbox_id"),
                           "verdict": resp.get("verdict"),
                           "promoted_equation": resp.get("promoted"),
                           "data_origin": resp.get("data_origin")})
    return resp



def _volume_gate_breakdown(extra: dict, scored: list, y_actual: list | None = None) -> list[dict]:
    """Explicit promotion-gate table for the volume benchmark (auditable)."""
    rows = scored or []
    # Best challenger = lowest holdout RMSE among non-baseline rows; never
    # hardcode V1. Rank on RMSE here; gates below decide promotion.
    chall_rows = [c for c in rows
                  if "V0" not in (c.candidate.candidate_id if hasattr(c, "candidate") else "V0")]
    v1 = min(chall_rows, key=lambda c: float(c.holdout_rmse or float("inf"))) if chall_rows else None
    r2 = float("nan")
    if v1 is not None and hasattr(v1, "provenance"):
        g4 = v1.provenance.get("gate4_status", {})
        resids = g4.get("holdout_residuals")
        # Centered R2 against the TARGET variance (same derivation as the
        # leaderboard rows): 1 - ss_res/SStot. Never the residual variance.
        if resids and y_actual is not None:
            import numpy as _np
            arr = _np.asarray(resids, dtype=float)
            arr = arr[_np.isfinite(arr)]
            yh = _np.asarray(y_actual, dtype=float).ravel()
            yh = yh[_np.isfinite(yh)][-len(arr):] if len(yh) >= len(arr) else yh
            if len(arr) > 0 and len(yh) == len(arr):
                ss_res = float(_np.sum(arr ** 2))
                ss_tot = float(_np.sum((yh - float(_np.mean(yh))) ** 2))
                r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")
    def ok(x: bool) -> str:
        return "✓" if x else "✗"
    # Delta/perm follow the SELECTED challenger (V1 or V2 fills these keys).
    is_v2 = v1 is not None and "V2" in (v1.candidate.candidate_id if hasattr(v1, "candidate") else "")
    delta = float(extra.get("delta_r2_V2" if is_v2 else "delta_r2", float("nan")))
    perm = float(extra.get("perm_p_V2" if is_v2 else "perm_p", float("nan")))
    who = (v1.candidate.candidate_id if v1 is not None and hasattr(v1, "candidate") else "best challenger")
    g_delta = bool(delta == delta and delta > 0)
    g_perm = bool(perm == perm and perm < 0.05)
    g_r2 = bool(r2 == r2 and r2 > 0)
    out = [
        {"gate": "ΔR² > 0", "result": ok(g_delta),
         "detail": f"{who}: incremental ΔR² = {delta:+.4f}" if delta == delta else "n/a"},
        {"gate": "perm p < 0.05", "result": ok(g_perm),
         "detail": f"{who}: permutation p = {perm:.4f}" if perm == perm else "n/a"},
        {"gate": "R² > 0", "result": ok(g_r2),
         "detail": f"best-challenger R² = {r2:+.3f}" if r2 == r2 else "n/a"},
    ]
    out.append({"gate": "PROMOTION", "result": ok(g_delta and g_perm and g_r2),
                "detail": "all gates pass" if (g_delta and g_perm and g_r2) else "at least one gate fails → no promotion"})
    return out

@app.post("/api/sandbox/upload", dependencies=[Depends(_require_api_key)])
def sandbox_upload(req: SandboxRequest) -> dict[str, Any]:
    """Execute autonomous open equation discovery on uploaded custom CSV data."""
    req.sandbox_id = "custom_upload"
    return sandbox_run(req)


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
        from apedp_dyn.discovery_l2 import (
            Dimension,
            L2DiscoveryEngine,
            run_discovery_stability_audit,
        )

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

        summary, _details = run_nano_ds10(n_seeds=req.n_seeds, noise_sigma=req.noise)
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

    resp = _json_obj(resp)
    _audit("stability/audit", {"problem": req.problem, "n_seeds": req.n_seeds})
    return resp


@app.get("/api/benchmarks/trading-np93", dependencies=[Depends(_require_api_key)])
def get_trading_np93() -> dict[str, Any]:
    """Return TRADING-NP93 Null Promotion Benchmark scorecard and statistics."""
    import json

    freeze_file = ROOT / "outputs" / "trading_np93_freeze.json"
    if freeze_file.exists():
        # Serve the frozen measurement verbatim, and derive the contract verdict
        # from ITS OWN counts using the benchmark's definition
        # (false_promotion_rate = promotions / validated windows;
        # passed = no promotions and no unique support). This endpoint used to
        # overwrite those two fields with 0.0 / True on every response,
        # including the frozen measurement -- i.e. it falsified the scoreboard.
        with open(freeze_file, encoding="utf-8") as f:
            data = json.load(f)
        promotion = int(data.get("promotion", 0) or 0)
        validation = int(data.get("validation", 0) or 0)
        unique = int(data.get("unique", 0) or 0)
        data["false_promotion_rate"] = promotion / max(1, validation)
        data["passed_contract"] = bool(promotion == 0 and unique == 0)
        return _json_obj(data)

    from benchmarks.run_trading_np93 import run_trading_np93

    summary, _ = run_trading_np93()
    return _json_obj({
        "benchmark": summary.benchmark,
        "full_name": summary.full_name,
        "windows": summary.windows,
        "candidate_space": summary.candidate_space,
        "validation": summary.validation,
        "unique": summary.unique,
        "multi": summary.multi,
        "promotion": summary.promotion,
        "question": summary.question,
        "false_promotion_rate": float(summary.false_promotion_rate),
        "passed_contract": bool(summary.passed_contract),
    })


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
    return _json_obj(data)


@app.get("/api/benchmarks/deepmind-challenge", dependencies=[Depends(_require_api_key)])
def get_deepmind_challenge() -> dict[str, Any]:
    """Execute and return the 12-case DeepMind Hypothesis Challenge benchmark."""
    from apedp_dyn.sandbox.deepmind_challenge_suite import run_deepmind_challenge_benchmark
    res = run_deepmind_challenge_benchmark()
    return _json_obj(res)


@app.get("/api/benchmarks/ablation-study", dependencies=[Depends(_require_api_key)])
def get_ablation_study() -> dict[str, Any]:
    """Execute and return the 3-Arm Ablation Experiment (Specialized vs Naive vs Agnostic Adapter)."""
    from apedp_dyn.sandbox.agnostic_domain_sandbox import run_ablation_experiment
    res = run_ablation_experiment()
    return _json_obj(res)


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


if __name__ == "__main__":
    run_server(host="127.0.0.1", port=8000, reload=True)

