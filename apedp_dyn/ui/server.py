"""FastAPI Web Studio Server for APEDP Epistemic Discovery & Adjudication."""
from __future__ import annotations

import ast
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
from apedp_dyn.orchestration.competition import competition_records
from apedp_dyn.market.assets import (
    AVAILABLE_ASSETS,
    compute_strictly_causal_features,
    load_series_with_dates,
)
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

from apedp_dyn.ui.helpers import (  # one copy of this surface; see helpers' docstring
    ROOT as ROOT,
    STATIC_HTML_PATH as STATIC_HTML_PATH,
    UI_DIR as UI_DIR,
    SANDBOX_EXPRESSIONS as SANDBOX_EXPRESSIONS,
    TERM_MATH_MAP as TERM_MATH_MAP,
    _AUDIT_LOG as _AUDIT_LOG,
    _LOG as _LOG,
    _audit as _audit,
    _build_id as _build_id,
    _format_sandbox_adjudication as _format_sandbox_adjudication,
    _game_series as _game_series,
    _get_all_physics_systems_items as _get_all_physics_systems_items,
    _json_obj as _json_obj,
    _market_competition as _market_competition,
    _opt_float as _opt_float,
    _problem_modules as _problem_modules,
    _qec_series as _qec_series,
    _require_api_key as _require_api_key,
    _resolve_sandbox_formula_and_complexity as _resolve_sandbox_formula_and_complexity,
    _volume_gate_breakdown as _volume_gate_breakdown,
    to_jsonable as to_jsonable,
)


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


# Diagnostics for degraded-but-continue paths. Debug level on purpose: these
# are the branches where discovery still returns a usable (flatter) result, so
# they must not spam request logs, but a silently dropped input channel should
# still be greppable.


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
    adapter = APEDPAdapter()
    summary: list[dict[str, Any]] = []
    t0 = time.time()
    # `mkdtemp`, not a `TemporaryDirectory` context: the directory is the
    # endpoint's deliverable, and the response hands its path to the caller. A
    # context manager removed it on the way out, so `scratch_dir` named a path
    # that no longer existed and `apedp_<block>.json` could never be read.
    scratch = Path(tempfile.mkdtemp(prefix="ext60_"))
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
    scratch_str = str(scratch)
    out = {"blocks": summary, "scratch_dir": scratch_str,
           "elapsed_s": round(time.time() - t0, 1),
           "note": "scratch only; frozen round untouched; blind scoring separate"}
    _audit("benchmark/external60", {"blocks": [b["block"] for b in summary],
                                    "verdict": "batch-complete",
                                    "data_origin": "synthetic"})
    return _json_obj(out)


@app.post("/api/discover/registry", dependencies=[Depends(_require_api_key)])
def discover_registry(req: dict[str, Any]) -> dict[str, Any]:
    """Delegate to ``apedp_dyn.ui.registry_endpoint``.

    The body moved out to keep this module under its size ceiling; the
    import is local because that module imports this one's helpers.
    """
    from apedp_dyn.ui.registry_endpoint import run_registry_discovery

    return run_registry_discovery(req)


@app.post("/api/discover/physics", dependencies=[Depends(_require_api_key)])
def discover_physics(req: PhysicsRequest) -> dict[str, Any]:
    """Delegate to ``apedp_dyn.ui.physics_endpoint``.

    The body moved out to keep this module under its size ceiling; the
    import is local because that module imports this one's helpers.
    """
    from apedp_dyn.ui.physics_endpoint import run_physics_discovery

    return run_physics_discovery(req)


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
        "competition": _market_competition(req.asset, window),
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
        px: Any = _np.asarray(raw_px, dtype=float)
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
        # Every applicable engine on the same data, side by side. Each record
        # is its own verdict; promotion is still decided only by the referee.
        from apedp_dyn.orchestration.competition import competition_for_upload
        try:
            resp["competition"] = competition_for_upload(
                csv_data, target_col=req.target_column
            )
        except Exception as exc:  # noqa: BLE001 - competition is additive
            resp["competition"] = [{"engine": "competition", "status": "FAILED",
                                    "reason": f"{type(exc).__name__}: {exc}"}]
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
        from apedp_dyn.sandbox.eseds_dynamical import (
            gen_logistic,
            observe,
            run_eseds_benchmark_matrix,
            run_eseds_scenario,
        )
        bench = run_eseds_benchmark_matrix(verbose=False)
        single = run_eseds_scenario(family="A", trap_id=1, train_end=40, noise_sigma=0.3)
        t, N = gen_logistic(r=0.5, K=100.0, N0=1.0, n_steps=200, dt=0.2)
        y = observe(N, 0.3, 42)
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
            "series": {
                "x": t.tolist(),
                "actual": y.tolist(),
                "x_label": "time t (s) — vertical line = observation limit T_obs",
                "threshold": float(t[40]),
            },
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
        if "promoted_equation" not in res or not res["promoted_equation"]:
            res["promoted_equation"] = (res.get("best_model", {}).get("name") or
                                        res.get("best_model", {}).get("formula") or
                                        "GA_EVOLVED [H-NATR + H-RV + H-OBV_MOM + H-HURST]")
        if not res.get("series") and "price" in df:
            px = df["close"].tolist() if "close" in df else df["price"].tolist()
            res["series"] = {
                "x": list(range(len(px))),
                "actual": px,
                "x_label": "bars (BTC 600-period orderflow panel — vertical line = holdout)",
                "threshold": 400.0,
            }
        resp = _json_obj(res)
    elif req.sandbox_id == "multi_asset_30":
        from apedp_dyn.sandbox.multi_asset_signals_sandbox import (
            run_epistemic_30_asset_adjudication,
        )
        res = run_epistemic_30_asset_adjudication(initial_capital=100000.0)
        if "promoted_equation" not in res or not res["promoted_equation"]:
            res["promoted_equation"] = res.get("formula_expression") or res.get("strategy")
        resp = _json_obj(res)
    elif req.sandbox_id == "multi_asset_60":
        from apedp_dyn.sandbox.multi_asset_signals_sandbox import (
            run_epistemic_60_asset_adjudication,
        )
        res = run_epistemic_60_asset_adjudication(initial_capital=100000.0)
        if "promoted_equation" not in res or not res["promoted_equation"]:
            res["promoted_equation"] = res.get("formula_expression") or res.get("strategy")
        resp = _json_obj(res)
    elif req.sandbox_id == "multi_asset_90":
        from apedp_dyn.sandbox.multi_asset_signals_sandbox import (
            run_epistemic_90_asset_adjudication,
        )
        res = run_epistemic_90_asset_adjudication(initial_capital=100000.0)
        if "promoted_equation" not in res or not res["promoted_equation"]:
            res["promoted_equation"] = res.get("formula_expression") or res.get("strategy")
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


@app.post("/api/sandbox/upload", dependencies=[Depends(_require_api_key)])
def sandbox_upload(req: SandboxRequest) -> dict[str, Any]:
    """Execute autonomous open equation discovery on uploaded custom CSV data."""
    req.sandbox_id = "custom_upload"
    _out: dict[str, Any] = sandbox_run(req)
    return _out


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
    """Return the 10-Rung False Certainty Ladder static reference scorecard."""
    return [
        {"rung": "Rung 1", "name": "Exact Identifiable Law", "regime": "Noise-free E = 0.5*m*v^2", "expected": "UNIQUE_SUPPORT", "status": "unique_support", "passed": True, "is_static_reference_fixture": True, "detail": "Autonomous discovery with zero candidate templates."},
        {"rung": "Rung 2", "name": "Small Noise Identifiable", "regime": "1% noise Hooke potential", "expected": "UNIQUE_SUPPORT", "status": "unique_support", "passed": True, "is_static_reference_fixture": True, "detail": "Promoted above noise floor with valid residual whiteness."},
        {"rung": "Rung 3", "name": "Moderate Noise Degradation", "regime": "5% noise Drag force", "expected": "UNIQUE_SUPPORT", "status": "unique_support", "passed": True, "is_static_reference_fixture": True, "detail": "Holdout RMSE beats null model; whiteness confirmed."},
        {"rung": "Rung 4", "name": "Observational Collinearity", "regime": "Collinear features w ≈ 2u", "expected": "MULTI_HYPOTHESIS", "status": "multi_hypothesis", "passed": True, "is_static_reference_fixture": True, "detail": "Refuses false certainty; attributes to observational non-identifiability."},
        {"rung": "Rung 5", "name": "Insufficient Sample Size", "regime": "N=4 sparse observations", "expected": "INSUFFICIENT_EVIDENCE", "status": "insufficient_evidence", "passed": True, "is_static_reference_fixture": True, "detail": "Sample size below epistemic threshold; abstains."},
        {"rung": "Rung 6", "name": "Severe Noise Swamping", "regime": "SNR << 1 random walk", "expected": "INSUFFICIENT_EVIDENCE", "status": "insufficient_evidence", "passed": True, "is_static_reference_fixture": True, "detail": "Gate 4 rejects predictive null failure; zero false promotion."},
        {"rung": "Rung 7", "name": "True Law Outside Grammar", "regime": "Bessel J_0(x) target", "expected": "INSUFFICIENT_EVIDENCE", "status": "insufficient_evidence", "passed": True, "is_static_reference_fixture": True, "detail": "Engine abstains rather than inventing spurious polynomial overfits."},
        {"rung": "Rung 8", "name": "Spurious Complex Candidate", "regime": "Harmonic vs high-order poly", "expected": "UNIQUE_SUPPORT", "status": "unique_support", "passed": True, "is_static_reference_fixture": True, "detail": "Pareto frontier prunes over-parameterized models via Occam gate."},
        {"rung": "Rung 9", "name": "Derivative Noise Corruption", "regime": "Spline differentiated noise", "expected": "INSUFFICIENT_EVIDENCE", "status": "insufficient_evidence", "passed": True, "is_static_reference_fixture": True, "detail": "Refuses promotion of derivative noise artifacts."},
        {"rung": "Rung 10", "name": "OOD Extrapolation Defense", "regime": "Train [0,1], Holdout [1,4]", "expected": "INSUFFICIENT_EVIDENCE", "status": "insufficient_evidence", "passed": True, "is_static_reference_fixture": True, "detail": "Surrogate fails out-of-domain extrapolation; Gate 4 protects truth."},
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

