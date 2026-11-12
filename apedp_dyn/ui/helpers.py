"""Helpers and shared constants extracted from ``apedp_dyn/ui/server.py``.

``server.py`` was 2,433 lines, mixing routes with the machinery that feeds
them, which is more than a reviewer can hold in one sitting. Everything here
is a pure helper or a shared constant, moved byte-for-byte; ``server`` imports
them back, so ``apedp_dyn.ui.server.<name>`` still resolves for callers that
import or monkeypatch these names directly.

``_AUDIT_LOG`` derives its default path from ``__file__``, and this module sits
beside ``server.py``, so the resolved location is unchanged.
"""
from __future__ import annotations

import hmac
import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
from fastapi import HTTPException, Request

from apedp_dyn.market.assets import (
    compute_strictly_causal_features,
    load_series_with_dates,
)
from apedp_dyn.orchestration.competition import competition_records

UI_DIR = Path(__file__).resolve().parent
STATIC_HTML_PATH = UI_DIR / "dashboard.html"
ROOT = Path(__file__).resolve().parent.parent.parent




_LOOPBACK_PEERS = frozenset({"127.0.0.1", "::1", "localhost", "testclient"})


def _require_api_key(request: Request) -> None:
    """API-key gate for /api/*.

    Enforced only when APEDP_API_KEY is set; when unset the server is
    intended for localhost research use and run_server() refuses
    non-loopback binds (see below). Missing/invalid key -> 401.
    """
    expected = os.environ.get("APEDP_API_KEY", "")
    if not expected:
        # No key means local research use. run_server() refuses a non-loopback
        # bind, but `uvicorn apedp_dyn.ui.server:app --host 0.0.0.0` never calls
        # run_server(), so the bind check alone left the API open to the
        # network. The peer address is checked here, where every launch path
        # passes. "testclient" is Starlette's in-process test transport.
        peer = request.client.host if request.client else ""
        if peer not in _LOOPBACK_PEERS:
            raise HTTPException(
                status_code=401,
                detail="APEDP_API_KEY is not set; only loopback clients are served",
            )
        return
    auth = request.headers.get("authorization", "")
    bearer = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
    got = request.headers.get("x-api-key", "") or bearer
    if not hmac.compare_digest(got.encode("utf-8"), expected.encode("utf-8")):
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



def _market_competition(asset: str, window: Any) -> list[dict[str, Any]]:
    """Engine competition on the window's TRAIN slice (holdout never enters)."""
    dates, prices = load_series_with_dates(asset)
    date_to_idx = {d: i for i, d in enumerate(dates)}
    causal = compute_strictly_causal_features(prices)
    i0 = date_to_idx[window.train_start]
    i1 = date_to_idx[window.train_end]
    sl = slice(i0, i1 + 1)
    X = {"r_lag": causal["r_lag"][sl], "vol": causal["vol"][sl],
         "ma_dev": causal["ma_dev"][sl]}
    return competition_records(X, causal["returns"][sl], train_ratio=0.6)



def _game_series() -> Any:
    from apedp_dyn.sandbox.game_theory import simulate_game
    _, x = simulate_game({"R": 3.0, "S": 0.0, "T": 5.0, "P": 1.0},
                         0.5, noise=0.005, seed=42)
    return {"x": list(range(len(x))), "actual": [float(v) for v in x],
            "x_label": "PD cooperator fraction from x0=0.5"}

def _qec_series() -> Any:
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



def _volume_gate_breakdown(extra: dict[str, Any], scored: list[Any],
                          y_actual: list[Any] | None = None) -> list[dict[str, Any]]:
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
