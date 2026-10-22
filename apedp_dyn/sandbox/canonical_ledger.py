"""Canonical execution-and-accounting path for policy evaluation.

Single implementation from which every reported metric is aggregated.
No competing harnesses: features, validity masks, scores, positions,
costs, and aggregates all derive from one per-observation ledger.

Accounting space: additive log-returns. Per observation:
  gross = pos * fwd, fee = FEE * |dpos|, net = gross - fee.
Identity gross - fees == net holds exactly at row level and in sum.

Missing data: validity mask computed BEFORE any imputation. An invalid
score forces flat with reason code; never silently becomes zero signal.
"""
from __future__ import annotations

import hashlib
import sys
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, '.')

from apedp_dyn.conf.canonical_spec import (
    CONTINUOUS_FACTOR_ACTIVE_MIN,
    STOCK_SPLIT_INVARIANCE_TOLERANCE,
    TACTICAL_ACTIVE_MAX,
    TACTICAL_ACTIVE_MIN,
    TAIL_RISK_MIN_EVENTS,
)
from apedp_dyn.sandbox.grammar_space import causal_zscore, trailing_rank

FEE = 0.0005  # 5 bps per unit position change

#: Holdout R^2 floor for ``dynamical_system`` archetype candidates (the same
#: floor the epistemic ODE referee uses for worst-state holdout fit).
PHYSICAL_ROLLOUT_MIN_R2 = 0.85


def raw_features(d: pd.DataFrame) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Raw (pre-standardization) feature components + log returns."""
    p = d["price"].to_numpy(float)
    v = d["V"].to_numpy(float)
    tb = d["taker_buy"].to_numpy(float) if "taker_buy" in d else v / 2.0
    r = np.append(0.0, np.diff(np.log(np.maximum(p, 1e-12))))
    sma20 = pd.Series(p).rolling(20, min_periods=20).mean().to_numpy()
    std20 = pd.Series(p).rolling(20, min_periods=20).std().to_numpy()
    out = {
        "H-BB": (p - (sma20 - 2 * std20)) / np.maximum(4 * std20, 1e-12),
        "H-MOM": pd.Series(p).pct_change(5, fill_method=None).to_numpy(),
    }
    obv = np.zeros(len(p))
    for t in range(1, len(p)):
        obv[t] = obv[t - 1] + (v[t] if p[t] > p[t - 1] else (-v[t] if p[t] < p[t - 1] else 0.0))
    out["H-OBV_MOM"] = pd.Series(obv).pct_change(10, fill_method=None).to_numpy()
    vc = np.maximum(v, 1e-12)
    out["CVD"] = (2.0 * tb - vc) / vc
    sa = np.maximum(0.01 + 0.3 * np.abs(r) + 0.5 * np.where(r < 0, np.abs(r), 0.0), 1e-4)
    out["SIG_ASYM"] = sa
    out["RV"] = v  # ranked below (trailing window, causal)
    h = np.full(len(p), np.nan)
    for t in range(30, len(p)):
        sub = r[t - 30:t]
        if not np.all(np.isfinite(sub)):
            continue
        cd = np.cumsum(sub - np.mean(sub))
        rr = np.max(cd) - np.min(cd)
        s = np.std(sub)
        if s > 1e-9 and rr > 1e-9:
            h[t] = (np.log(rr / s) / np.log(30)) - 0.5
    out["HURST"] = h
    ema12 = pd.Series(p).ewm(span=12, adjust=False).mean()
    ema26 = pd.Series(p).ewm(span=26, adjust=False).mean()
    ml = ema12 - ema26
    out["MACD"] = (ml - ml.ewm(span=9, adjust=False).mean()).to_numpy()
    return out, r


def zscore_features(raw: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Causal z-scores + validity masks (finite raw input). Masks first."""
    z, valid = {}, {}
    for k, arr in raw.items():
        if k == "RV":
            continue
        a = np.asarray(arr, dtype=float)
        valid[k] = np.isfinite(a)
        z[k] = np.nan_to_num(causal_zscore(np.nan_to_num(a, nan=0.0)), nan=0.0)
    return z, valid


def run_ledger(syms: list[str], data: dict[str, pd.DataFrame],
               weights: dict[str, float], rule: str,
               th_in: float, th_out: float,
               eval_slice: slice, fee: float = FEE,
               ablate_feature_rank: bool = False,
               check_binding: bool = True,
               strategy_archetype: str = "tactical",
               min_tail_events: int = 30) -> tuple[list[dict], dict]:
    """Execute one policy over assets; return (ledger rows, summary).

    rule: 'base' (short iff S<0) or 'hyst' (enter iff S<-th_in,
    hold while S<-th_out). Invalid scores force flat with reason.
    """
    rows: list[dict] = []
    for s in syms:
        raw, r = raw_features(data[s])
        z, valid = zscore_features(raw)
        tm = np.nan_to_num(causal_zscore(
            np.nan_to_num(z["CVD"], nan=0.0) *
            np.nan_to_num(z["H-MOM"], nan=0.0)), nan=0.0)
        snr = np.nan_to_num(causal_zscore(
            np.nan_to_num(z["H-MOM"], nan=0.0) /
            (np.clip(np.nan_to_num(raw["SIG_ASYM"], nan=1.0), -1e100, 1e100) ** 2)), nan=0.0)
        if not ablate_feature_rank:
            rv = np.nan_to_num(trailing_rank(np.asarray(raw["RV"], dtype=float)), nan=0.0)
        else:
            rv = np.nan_to_num(causal_zscore(np.asarray(raw["RV"], dtype=float)), nan=0.0)
        macd_z = np.nan_to_num(causal_zscore(np.asarray(raw["MACD"], dtype=float)), nan=0.0)
        hurst_ok = np.isfinite(np.asarray(raw["HURST"], dtype=float))
        had = np.nan_to_num(causal_zscore(
            np.where(np.nan_to_num(hurst_ok, nan=0.0) > 0, macd_z, -z["H-BB"])), nan=0.0)
        comp = {"H-BB": z["H-BB"], "H-OBV_MOM": z["H-OBV_MOM"],
                "NL_TAKER_MOM": tm, "NL_MASTER_SNR": snr,
                "H-MOM": z["H-MOM"], "H-RV": rv, "NL_HURST_ADAPT": had}
        ok = np.ones(len(r), bool)
        for t in ("H-BB", "H-OBV_MOM"):
            ok &= valid[t]
        # Every consumed raw input must be finite; imputed zeros never
        # count as valid signal (mask precedes imputation).
        for t in ("H-MOM", "CVD", "SIG_ASYM"):
            ok &= np.isfinite(np.asarray(raw[t], dtype=float))
        ok &= np.isfinite(np.asarray(raw["RV"], dtype=float))
        ok &= hurst_ok & np.isfinite(np.asarray(raw["MACD"], dtype=float)) \
            if "NL_HURST_ADAPT" in weights else ok
        S = sum(weights[k] * comp[k] for k in weights)
        idx = np.arange(len(r))[eval_slice]
        st = 0.0
        for i in idx:
            if not ok[i]:
                pos, reason = 0.0, "invalid_score"
            elif rule == "base":
                pos, reason = (-1.0 if S[i] < 0 else 0.0), "base_rule"
            else:
                if st == 0 and S[i] < -th_in:
                    st = -1.0
                elif st == -1 and S[i] >= -th_out:
                    st = 0.0
                pos, reason = st, "hyst_rule"
            fwd = float(r[i + 1]) if (i + 1 < len(r) and np.isfinite(r[i + 1])) else 0.0
            gross = 0.0 if pos == 0.0 else float(pos * fwd)
            rows.append({"asset": s, "t": int(i), "score": float(S[i]),
                         "valid": bool(ok[i]), "reason": reason,
                         "pos": float(pos), "fwd": float(fwd),
                         "gross": float(gross), "dpos": 0.0, "fee": 0.0,
                         "net": float(gross)})

    # Mandate mask-first compliance: zero-imputed features may never silently become signals
    assert_mask_first_compliance(rows)

    # second pass: position changes need per-asset previous position
    out: list[dict] = []
    by_asset: dict[str, list[dict]] = {}
    for row in rows:
        by_asset.setdefault(row["asset"], []).append(row)
    for _s, rs in by_asset.items():
        prev = 0.0
        for row in rs:
            dpos = abs(row["pos"] - prev)
            fee_c = fee * dpos
            row["dpos"] = float(dpos)
            row["fee"] = float(fee_c)
            row["net"] = float(row["gross"] - fee_c)
            prev = row["pos"]
            out.append(row)
    gross = sum(x["gross"] for x in out)
    fees = sum(x["fee"] for x in out)
    net = sum(x["net"] for x in out)
    assert abs((gross - fees) - net) < 1e-9, "accounting identity violated"

    # Non-binding threshold check: assert active fraction lies within operational bounds
    is_binding = True
    active_fraction = sum(1 for x in out if abs(x["pos"]) > 0) / max(len(out), 1)
    status = "GATE_BINDING"
    notes = "Valid binding policy run."

    if len(syms) > 1:
        # Group by timestamp for contemporaneous multi-asset portfolio Sharpe
        by_t: dict[int, list[float]] = {}
        for x in out:
            by_t.setdefault(x["t"], []).append(x["net"])
        net_returns = [float(np.mean(vals)) for _, vals in sorted(by_t.items())]
    else:
        net_returns = [x["net"] for x in out]
    computed_sharpe = float(np.mean(net_returns) / np.std(net_returns) * np.sqrt(365)) if (len(net_returns) > 1 and np.std(net_returns) > 1e-9) else 0.0

    if check_binding:
        is_binding, active_fraction, gate_status, gate_reason = assert_strategy_class_compliance(
            out, strategy_archetype=strategy_archetype, min_events=min_tail_events
        )
        if not is_binding:
            status = gate_status
            sharpe = None  # Refused when strategy violates operational exposure or statistical power
            notes = gate_reason
        else:
            status = gate_status
            sharpe = computed_sharpe
    else:
        sharpe = computed_sharpe

    summary = {
        "gross": gross,
        "fees": fees,
        "net": net,
        "turnover": sum(x["dpos"] for x in out),
        "n_obs": len(out),
        "n_invalid": sum(1 for x in out if not x["valid"]),
        "status": status,
        "gate_binding": is_binding,
        "active_fraction": active_fraction,
        "sharpe": sharpe,
        "notes": notes,
    }
    return out, summary


def assert_mask_first_compliance(
    rows: list[dict],
    raise_on_violation: bool = True,
) -> bool:
    """Assert that invalid/missing data strictly forces flat positions with reason.

    Zero-imputed features may never silently become active trade signals.
    """
    for r in rows:
        if not r["valid"]:
            if r["pos"] != 0.0 or r["reason"] != "invalid_score":
                msg = (
                    f"MASK-FIRST COMPLIANCE VIOLATION at t={r['t']}, asset={r['asset']}: "
                    f"invalid score yielded non-zero position {r['pos']} with reason '{r['reason']}'. "
                    f"Zero-imputed features must strictly force flat positions."
                )
                if raise_on_violation:
                    raise ValueError(msg)
                return False
    return True


def assert_threshold_binding(
    rows: list[dict],
    min_active: float = TACTICAL_ACTIVE_MIN,
    max_active: float = TACTICAL_ACTIVE_MAX,
) -> tuple[bool, float, str]:
    """Assert that threshold-conditioned positions actually bind within tactical window.

    Outside that interval, the gate is purely decorative and does not actively condition trades.
    Returns: (is_binding, active_fraction, status_str)
    """
    n = len(rows)
    if n == 0:
        return False, 0.0, "GATE_NON_BINDING"
    active_count = sum(1 for r in rows if abs(r["pos"]) > 0.0)
    active_fraction = float(active_count / n)

    if active_fraction < min_active or active_fraction > max_active:
        return False, active_fraction, "GATE_NON_BINDING"
    return True, active_fraction, "GATE_BINDING"


def assert_strategy_class_compliance(
    rows: list[dict],
    strategy_archetype: str = "tactical",
    min_events: int = TAIL_RISK_MIN_EVENTS,
    min_active_tactical: float = TACTICAL_ACTIVE_MIN,
    max_active_tactical: float = TACTICAL_ACTIVE_MAX,
    min_active_continuous: float = CONTINUOUS_FACTOR_ACTIVE_MIN,
    declared_invariant: str | None = None,
    invariant_drift: float | None = None,
    rollout_stable: bool | None = None,
    holdout_r2: float | None = None,
) -> tuple[bool, float, str, str]:
    """Evaluate candidate operational exposure and statistical power per strategy archetype.

    Strategy Archetypes:
      1. 'tail_risk': Rare-event crisis alpha. Active rate may be < 5%, but requires >= min_events
         independent trades/episodes in sample to guarantee statistical power.
      2. 'continuous_factor': Broad cross-sectional factor. Requires active fraction >= min_active_continuous.
      3. 'tactical': Dynamic regime/swing signal. Requires active fraction in [5%, 90%].
      4. 'physical_invariant': Conservation/bound invariant. Financial active-rate gates are
         meaningless here; compliance requires declared physical evidence: an invariant name and a
         measured drift bound (``invariant_drift`` <= 1e-6), passed by the caller.
      5. 'dynamical_system': ODE/PDE law. Compliance requires forward-rollout stability and a
         holdout R^2 >= 0.85 (``rollout_stable`` / ``holdout_r2``), passed by the caller.

    The two STEM archetypes FAIL CLOSED (``MISSING_PHYSICAL_EVIDENCE``) when their dedicated
    evidence is not supplied: a physical law can no longer sail through financial turnover
    gates, and it cannot pass the physics gates without measured evidence either.

    Returns: (is_compliant, active_fraction, status_str, reason_str)
    """
    n = len(rows)
    if n == 0:
        return False, 0.0, "EMPTY_SAMPLE", "Zero observations in evaluation sample."
    active_count = sum(1 for r in rows if abs(r["pos"]) > 0.0)
    active_fraction = float(active_count / n)

    # Count distinct active trading events/episodes (transitions from flat to active)
    active_events = 0
    prev_pos = 0.0
    prev_asset = None
    for r in rows:
        cur_pos = r["pos"]
        cur_asset = r.get("asset")
        if cur_asset != prev_asset:
            prev_pos = 0.0
            prev_asset = cur_asset
        if abs(cur_pos) > 0.0 and abs(prev_pos) == 0.0:
            active_events += 1
        prev_pos = cur_pos

    arch = strategy_archetype.lower()
    if arch == "physical_invariant":
        if not declared_invariant:
            return (False, active_fraction, "MISSING_PHYSICAL_EVIDENCE",
                    "physical_invariant candidate must declare the invariant; "
                    "financial active-rate gates do not apply to physical laws.")
        if invariant_drift is None or invariant_drift > 1e-6:
            return (False, active_fraction, "INVARIANT_DRIFT_EXCEEDED",
                    f"Declared invariant '{declared_invariant}' drift "
                    f"{invariant_drift if invariant_drift is not None else float('nan'):.3e} exceeds the 1e-6 bound.")
        return (True, active_fraction, "GATE_BINDING",
                f"Physical invariant contract satisfied: '{declared_invariant}' drift within bound.")

    if arch == "dynamical_system":
        if rollout_stable is None or holdout_r2 is None:
            return (False, active_fraction, "MISSING_PHYSICAL_EVIDENCE",
                    "dynamical_system candidate must supply rollout stability and holdout R^2; "
                    "financial turnover gates do not apply to dynamical laws.")
        if not rollout_stable:
            return (False, active_fraction, "ROLLOUT_UNSTABLE",
                    "Forward rollout of the discovered dynamical law diverged.")
        if holdout_r2 < PHYSICAL_ROLLOUT_MIN_R2:
            return (False, active_fraction, "HOLDOUT_R2_BELOW_FLOOR",
                    f"Dynamical law holdout R^2 {holdout_r2:.3f} < {PHYSICAL_ROLLOUT_MIN_R2:.2f}.")
        return (True, active_fraction, "GATE_BINDING",
                "Dynamical system contract satisfied: stable rollout with holdout R^2 floor met.")

    if arch == "tail_risk":
        if active_events < min_events:
            return (
                False,
                active_fraction,
                "INSUFFICIENT_STATISTICAL_POWER",
                f"Tail-risk alpha observed only {active_events} events (minimum {min_events} required for statistical power).",
            )
        if active_fraction > 0.20:
            return (
                False,
                active_fraction,
                "TAIL_RISK_OVER_EXPOSURE",
                f"Tail-risk alpha active fraction {active_fraction:.1%} exceeds 20% ceiling for crisis protection.",
            )
        return True, active_fraction, "GATE_BINDING", f"Tail-risk contract satisfied: {active_events} crisis events."

    elif arch == "continuous_factor":
        if active_fraction < min_active_continuous:
            return (
                False,
                active_fraction,
                "NON_CONTINUOUS_FACTOR_EXPOSURE",
                f"Continuous factor active fraction {active_fraction:.1%} falls below {min_active_continuous:.0%} continuous threshold.",
            )
        return True, active_fraction, "GATE_BINDING", "Continuous factor exposure contract satisfied."

    else:  # tactical / default
        is_binding, _, _ = assert_threshold_binding(rows, min_active=min_active_tactical, max_active=max_active_tactical)
        if not is_binding:
            return (
                False,
                active_fraction,
                "GATE_NON_BINDING",
                f"Gate is non-binding/decorative (active fraction {active_fraction:.1%} outside [{min_active_tactical:.0%}, {max_active_tactical:.0%}]); Sharpe refused.",
            )
        return True, active_fraction, "GATE_BINDING", "Tactical threshold binding contract satisfied."


def assert_stock_split_invariance(
    compute_signal_fn: Callable[[pd.DataFrame], np.ndarray],
    df: pd.DataFrame,
    split_ratio: float = 2.0,
    price_cols: tuple[str, ...] = ("price", "close", "high", "low", "open"),
    vol_cols: tuple[str, ...] = ("v", "vol", "volume", "taker_buy"),
    atol: float = STOCK_SPLIT_INVARIANCE_TOLERANCE,
) -> tuple[bool, float, str]:
    """Test mathematically consistent stock-split invariance on candidate signal.

    A k-for-1 split maps: P' = P / k, V' = k * V. Dollar volume P' * V' = P * V is conserved.
    Signals claiming price-scale invariance must yield identical signals S(P', V') == S(P, V).
    """
    df_orig = df.copy()
    s_orig = np.asarray(compute_signal_fn(df_orig), dtype=float)

    df_split = df.copy()
    for col in df_split.columns:
        col_lower = col.lower()
        if any(p == col_lower or col_lower.endswith(f"_{p}") or col_lower.startswith(f"{p}_") for p in price_cols):
            df_split[col] = df_split[col] / split_ratio
        elif any(v == col_lower or col_lower.endswith(f"_{v}") or col_lower.startswith(f"{v}_") for v in vol_cols):
            df_split[col] = df_split[col] * split_ratio

    s_split = np.asarray(compute_signal_fn(df_split), dtype=float)
    valid = np.isfinite(s_orig) & np.isfinite(s_split)
    if not np.any(valid):
        return False, 0.0, "NO_FINITE_SIGNALS"

    diffs = np.abs(s_orig[valid] - s_split[valid])
    max_diff = float(np.max(diffs))
    is_inv = bool(max_diff <= atol)
    status = "SPLIT_INVARIANT" if is_inv else "SPLIT_VARIANT"
    return is_inv, max_diff, status



def run_feature_rank_ablation(
    syms: list[str],
    data: dict[str, pd.DataFrame],
    weights: dict[str, float],
    rule: str,
    th_in: float,
    th_out: float,
    eval_slice: slice,
    fee: float = FEE,
) -> dict[str, Any]:
    """Execute live feature-rank ablation comparing ranked vs unranked feature representations.

    Arm A (Rank-Transformed Feature): Feature X^*_{H-RV} utilizes causal trailing_rank.
    Arm B (Ablated / Unranked): Feature X^*_{H-RV} utilizes causal z-score (removes ranking).

    Returns exact live deltas (Δ Net Return, Δ Sharpe, Δ Turnover).
    """
    _rows_ranked, sum_ranked = run_ledger(
        syms, data, weights, rule, th_in, th_out, eval_slice, fee=fee, ablate_feature_rank=False, check_binding=False
    )
    _rows_unranked, sum_unranked = run_ledger(
        syms, data, weights, rule, th_in, th_out, eval_slice, fee=fee, ablate_feature_rank=True, check_binding=False
    )

    delta_net = sum_ranked["net"] - sum_unranked["net"]
    delta_turnover = sum_ranked["turnover"] - sum_unranked["turnover"]
    sr_ranked = sum_ranked.get("sharpe") or 0.0
    sr_unranked = sum_unranked.get("sharpe") or 0.0
    delta_sharpe = sr_ranked - sr_unranked

    return {
        "ranked_summary": sum_ranked,
        "unranked_summary": sum_unranked,
        "delta_net_return": delta_net,
        "delta_turnover": delta_turnover,
        "delta_sharpe": delta_sharpe,
        "is_noop": bool(abs(delta_net) < 1e-12 and abs(delta_turnover) < 1e-12),
        "ablation_verdict": "FEATURE_RANK_INFORMATIVE" if abs(delta_net) >= 1e-6 else "FEATURE_RANK_EQUIVALENT",
    }


def ledger_hash(rows: list[dict]) -> str:
    h = hashlib.sha256()
    for x in rows:
        h.update(repr(sorted(x.items())).encode())
    return h.hexdigest()



# --------------------------------------------------------------------------
# REG-F14b per-caller wiring (docs/SYSTEM_FLAWS_AUDIT.md register): this
# production caller publishes a LEDGER-DERIVED DSR-adjusted multiplicity
# record (mode b, never hand-written), validated at publish time. See
# apedp_dyn/epistemic/caller_registry.py for the census and discipline.
# --------------------------------------------------------------------------
def publish_caller_compliance_record(
    ledger=None,
    candidate_id: str = "",
    dataset_id: str | None = None,
    nominal_variants: int | None = None,
):
    "Publish this caller's REG-F14b mode-(b) compliance record."
    from apedp_dyn.epistemic.caller_registry import publish_compliance_record

    return publish_compliance_record(
        ledger=ledger,
        caller_id="canonical_ledger",
        candidate_id=candidate_id,
        dataset_id=dataset_id,
        nominal_variants=nominal_variants,
    )
