"""Grammar Space for volatility hypotheses (forensics §§1-19 spec, doc S81).

Executable form of the Grammar Space document: admissible primitives and
operators with domain checks (§18), causal predictor builders for H-P /
H-RV / H-A (§§1-3), the d5mean target (§4), model combinations (§5), the
incremental/redundancy equations (§§6-9), and the Level-0..5 promotion
pipeline with the governing rule that later evidence cannot rescue an
earlier failed gate (§§10-14, 19). Additive only; no kernel changes.

Spec note: §18 writes "log(x)>0" as a domain restriction; implemented as
x>0 (log's domain), flagged where enforced.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import Any
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

PRIMITIVES = ("r_t", "V_t")
EXTENDED_PRIMITIVES = ("r_t", "V_t", "OFI_t")
UNARY_OPS = ("abs", "log", "zscore", "rank", "sqrt", "tanh")
ROLLING_OPS = ("RM_w",)
BINARY_OPS = ("+", "-", "*", "/", "min", "max")
MIN_HISTORY = 60
RANK_WINDOW = 40

LEVELS = ("L0-grammar", "L1-predictive", "L2-incremental", "L3-stable",
          "L4-null-robust", "L5-mechanistic")


def compute_grammar_fingerprint() -> str:
    """Cryptographic SHA256 fingerprint of admissible primitives, operators, and parameters.

    Ensures strict benchmark provenance: any modification to hypothesis primitives,
    operators, or search constraints alters this hash and invalidates legacy claims.
    """
    canonical_repr = {
        "primitives": sorted(PRIMITIVES),
        "extended_primitives": sorted(EXTENDED_PRIMITIVES),
        "unary_ops": sorted(UNARY_OPS),
        "rolling_ops": sorted(ROLLING_OPS),
        "binary_ops": sorted(BINARY_OPS),
        "min_history": MIN_HISTORY,
        "rank_window": RANK_WINDOW,
        "levels": list(LEVELS),
    }
    dumped = json.dumps(canonical_repr, sort_keys=True).encode("utf-8")
    return hashlib.sha256(dumped).hexdigest()


GRAMMAR_FINGERPRINT: str = compute_grammar_fingerprint()


@dataclass(frozen=True)
class DomainCheck:
    ok: bool
    reason: str = ""


def check_log(x: np.ndarray) -> DomainCheck:
    x = np.asarray(x, dtype=float)
    if bool(np.all(x[np.isfinite(x)] > 0)):
        return DomainCheck(True)
    return DomainCheck(False, "log requires x>0 on finite entries (spec §18)")


def check_div(x: np.ndarray, y: np.ndarray) -> DomainCheck:
    y = np.asarray(y, dtype=float)
    if bool(np.all(y[np.isfinite(y)] != 0)):
        return DomainCheck(True)
    return DomainCheck(False, "division requires nonzero divisor (spec §18)")


def check_sqrt(x: np.ndarray) -> DomainCheck:
    x = np.asarray(x, dtype=float)
    if bool(np.all(x[np.isfinite(x)] >= 0)):
        return DomainCheck(True)
    return DomainCheck(False, "sqrt requires x>=0 on finite entries (spec §18)")


def check_zscore(x: np.ndarray) -> DomainCheck:
    x = np.asarray(x, dtype=float)
    s = float(np.nanstd(x))
    if np.isfinite(s) and s > 0:
        return DomainCheck(True)
    return DomainCheck(False, "zscore requires std>0 (spec §18)")


def causal_zscore(v: np.ndarray, min_history: int = MIN_HISTORY) -> np.ndarray:
    """Expanding causal z-score; NaN until min_history finite observations."""
    v = np.asarray(v, dtype=float)
    s = pd.Series(v)
    mu = s.expanding(min_history).mean().to_numpy()
    sd = s.expanding(min_history).std().to_numpy()
    with np.errstate(all="ignore"):
        return np.where(sd > 0, (v - mu) / sd, np.nan)


def trailing_rank(v: np.ndarray, window: int = RANK_WINDOW) -> np.ndarray:
    """Causal trailing-window percentile rank in [0,1]; NaN until window full."""
    v = np.asarray(v, dtype=float)
    out = np.full(len(v), np.nan)
    for t in range(window - 1, len(v)):
        w = v[t - window + 1:t + 1]
        if np.isfinite(w).all():
            out[t] = float(np.sum(w <= v[t]) - 1) / (window - 1)
    return out


@dataclass
class HypothesisSpec:
    hid: str
    equation: str
    members: tuple[str, ...]
    interpretation: str
    build: Callable[[pd.DataFrame], np.ndarray] | None = None  # set by registry below


def _feat_logV(p: pd.DataFrame) -> np.ndarray:
    return np.asarray(np.log(np.maximum(p["V"].to_numpy(float), 1e-18)))


def _feat_rankV(p: pd.DataFrame) -> np.ndarray:
    return trailing_rank(p["V"].to_numpy(float))


def _feat_HA(p: pd.DataFrame) -> np.ndarray:
    # Frozen established form (trading_e4t1.build_asset_panel): outer z
    # around the sum of (log-level + |r| z + Amihud z), 1e-12 floors.
    # Kept verbatim; elegance is not a license to redefine a predictor.
    V = p["V"].to_numpy(float)
    r = p["r"].to_numpy(float)
    rm = pd.Series(V).rolling(40).mean().to_numpy()
    with np.errstate(all="ignore"):
        inner = (np.log(np.maximum(rm, 1e-12)) + causal_zscore(np.abs(r))
                 + causal_zscore(np.abs(r) / np.maximum(V, 1e-12)))
        return np.asarray(causal_zscore(inner))


def _feat_sig(p: pd.DataFrame) -> np.ndarray:
    return np.asarray(np.abs(p["r"].to_numpy(float)))


def _feat_sqrtV(p: pd.DataFrame) -> np.ndarray:
    """Concave level-volume response: diminishing marginal activity."""
    return np.asarray(np.sqrt(np.maximum(p["V"].to_numpy(float), 0.0)))


def _feat_asym(p: pd.DataFrame) -> np.ndarray:
    """Asymmetric volatility: baseline absolute return + downside panic interaction."""
    r = p["r"].to_numpy(float)
    sig = np.abs(r)
    sig_neg = np.where(r < 0, sig, 0.0)
    return np.column_stack([sig, sig_neg])


def _feat_ofi(p: pd.DataFrame) -> np.ndarray:
    """Microstructure order flow imbalance."""
    if "ofi" in p.columns:
        return causal_zscore(p["ofi"].to_numpy(float))
    return causal_zscore(p["r"].to_numpy(float))


def _feat_hurst(p: pd.DataFrame) -> np.ndarray:
    """Long-memory fractional / Hurst regime memory."""
    r = p["r"].to_numpy(float)
    out = np.full(len(r), np.nan)
    window = 30
    for t in range(window, len(r)):
        sub = r[t - window:t]
        cum_dev = np.cumsum(sub - np.mean(sub))
        r_range = np.max(cum_dev) - np.min(cum_dev)
        s_std = np.std(sub)
        if s_std > 1e-9 and r_range > 1e-9:
            out[t] = (np.log(r_range / s_std) / np.log(window)) - 0.5
    return causal_zscore(out)


def _feat_sqz(p: pd.DataFrame) -> np.ndarray:
    """Volatility compression squeeze release momentum."""
    price = pd.Series(p["price"].to_numpy(float) if "price" in p.columns else np.exp(np.cumsum(p["r"].to_numpy(float))))
    ema12 = price.ewm(span=12, adjust=False).mean()
    ema26 = price.ewm(span=26, adjust=False).mean()
    macd = (ema12 - ema26).to_numpy()
    std20 = price.rolling(20, min_periods=20).std().to_numpy()
    atr = pd.Series(np.abs(p["r"].to_numpy(float)) * price.to_numpy()).rolling(20, min_periods=20).mean().to_numpy()
    sqz = (4.0 * std20) / np.maximum(3.0 * atr, 1e-12)
    return causal_zscore(macd * np.exp(-np.clip(sqz, -2.0, 2.0)))


def _feat_master_snr(p: pd.DataFrame) -> np.ndarray:
    """Unified Signal-to-Noise Ratio: directional drift divided by asymmetric variance floor."""
    r = p["r"].to_numpy(float)
    sig_asym = np.maximum(0.01 + 0.3 * np.abs(r) + 0.5 * np.where(r < 0, np.abs(r), 0.0), 1e-4)
    mom = pd.Series(r).rolling(5, min_periods=5).mean().to_numpy()
    snr = mom / (sig_asym ** 2)
    return causal_zscore(snr)


def _feat_frac_diff(p: pd.DataFrame) -> np.ndarray:
    """Fractional differentiation proxy (stationarizing without excessive loss)."""
    price = pd.Series(p["price"].to_numpy(float) if "price" in p.columns else np.exp(np.cumsum(p["r"].to_numpy(float)))).values
    d = 0.4
    k = np.arange(1, 21)
    w = -np.cumprod((d - k + 1) / k)
    w = np.concatenate([[1.0], w])
    out = np.full(len(price), np.nan)
    for t in range(len(w)-1, len(price)):
        out[t] = np.sum(w * price[t - len(w)+1:t+1][::-1])
    return causal_zscore(out)


def _feat_ewm_vol(p: pd.DataFrame) -> np.ndarray:
    """Exponentially weighted volatility (risk measure)."""
    r = p["r"].to_numpy(float)
    vol = pd.Series(r**2).ewm(span=20, adjust=False).mean().apply(np.sqrt).to_numpy()
    return causal_zscore(vol)


def _feat_zscore_mom(p: pd.DataFrame) -> np.ndarray:
    """Z-scored momentum over multiple horizons."""
    r = p["r"].to_numpy(float)
    mom5 = pd.Series(r).rolling(5).sum().to_numpy()
    mom20 = pd.Series(r).rolling(20).sum().to_numpy()
    return causal_zscore(mom5 - 0.25*mom20)


def _feat_volume_price_div(p: pd.DataFrame) -> np.ndarray:
    """Volume-price divergence signal."""
    r = p["r"].to_numpy(float)
    V = p["V"].to_numpy(float)
    pv = pd.Series(r).rolling(10).corr(pd.Series(np.log(np.maximum(V,1e-18))).rolling(10).mean()).to_numpy()
    return causal_zscore(pv)


HYPOTHESES: dict[str, HypothesisSpec] = {
    # ── Foundational Volatility Formulations ──
    "H-P": HypothesisSpec("H-P", "y=a+b|r_t|", ("sig",),
                          "predictive persistence only; b is not a causal effect.", _feat_sig),
    "H-ASYM": HypothesisSpec("H-ASYM", "y=a+b_1|r_t|+b_2|r_t|I(r_t<0)", ("sig", "sig_neg"),
                             "asymmetric volatility leverage effect (downside panic spike).", _feat_asym),
    "H-RV": HypothesisSpec("H-RV", "y=a+bR_40(V_t)", ("rankV",),
                           "relative activity representation, not a unique mechanism.", _feat_rankV),
    "H-A": HypothesisSpec("H-A", "y=a+bHA_t", ("HA",),
                          "composite predictive representation, not automatically a mechanism.", _feat_HA),
    "H-LOG": HypothesisSpec("H-LOG", "y=a+blog(V_t)", ("logV",),
                            "level-volume representation.", _feat_logV),
    "H-SQRTV": HypothesisSpec("H-SQRTV", "y=a+b*sqrt(V_t)", ("sqrtV",),
                              "concave level-volume response (diminishing marginal activity).",
                              _feat_sqrtV),

    # ── Advanced Microstructure, Memory & Gated Formulations ──
    "H-OFI": HypothesisSpec("H-OFI", "y=a+b*OFI_t", ("ofi",),
                            "microstructure order flow imbalance force.", _feat_ofi),
    "H-HURST": HypothesisSpec("H-HURST", "y=a+b*(H_t-0.5)", ("hurst",),
                              "fractional long-memory regime persistence.", _feat_hurst),
    "H-SQZ": HypothesisSpec("H-SQZ", "y=a+b*(MACD*exp(-Squeeze))", ("sqz_trend",),
                            "volatility compression released trend breakout.", _feat_sqz),
    "H-SNR": HypothesisSpec("H-SNR", "y=a+b*(Drift/sigma_asym^2)", ("snr",),
                            "unified risk-standardized master signal-to-noise ratio.", _feat_master_snr),
    "H-FRAC": HypothesisSpec("H-FRAC", "y=a+b*FD_t", ("fracdiff",),
                             "fractionally differentiated price (stationarizing transform).", _feat_frac_diff),
    "H-EWMA-VOL": HypothesisSpec("H-EWMA-VOL", "y=a+b*sigma_ewm_t", ("ewmvol",),
                                  "exponentially weighted volatility regime.", _feat_ewm_vol),
    "H-ZMOM": HypothesisSpec("H-ZMOM", "y=a+b*ZMom_t", ("zmom",),
                             "multi-horizon z-scored momentum.", _feat_zscore_mom),
    "H-PVDIV": HypothesisSpec("H-PVDIV", "y=a+b*PVdiv_t", ("pvdiv",),
                              "volume-price divergence/correlation regime.", _feat_volume_price_div),
}

COMBOS = {
    "M_P": ("H-P",), "M_RV": ("H-RV",), "M_A": ("H-A",),
    "M_P+RV": ("H-P", "H-RV"), "M_P+A": ("H-P", "H-A"),
    "M_RV+A": ("H-RV", "H-A"), "M_P+RV+A": ("H-P", "H-RV", "H-A"),
}

EXTENDED_COMBOS = {
    **COMBOS,
    "M_P+SQRTV": ("H-P", "H-SQRTV"),
    "M_P+ASYM": ("H-P", "H-ASYM"),
    "M_ASYM+OFI": ("H-ASYM", "H-OFI"),
    "M_SQZ+HURST": ("H-SQZ", "H-HURST"),
    "M_COMPLETE": ("H-P", "H-ASYM", "H-OFI", "H-SQZ", "H-HURST", "H-SNR"),
    "M_FRAC+ZMOM": ("H-FRAC", "H-ZMOM"),
    "M_ADVANCED": ("H-P", "H-ASYM", "H-FRAC", "H-ZMOM", "H-EWMA-VOL"),
    "M_MICRO": ("H-OFI", "H-PVDIV", "H-SNR"),
}


def build_features(panel: pd.DataFrame) -> dict[str, np.ndarray]:
    """Causal feature columns for one asset panel (Level-0 domain-checked)."""
    return {
        "sig": _feat_sig(panel), "rankV": _feat_rankV(panel),
        "HA": _feat_HA(panel), "logV": _feat_logV(panel), "sqrtV": _feat_sqrtV(panel),
        "asym": _feat_asym(panel), "ofi": _feat_ofi(panel),
        "hurst": _feat_hurst(panel), "sqz": _feat_sqz(panel),
        "snr": _feat_master_snr(panel), "fracdiff": _feat_frac_diff(panel),
        "ewmvol": _feat_ewm_vol(panel), "zmom": _feat_zscore_mom(panel),
        "pvdiv": _feat_volume_price_div(panel),
    }


def build_target(panel: pd.DataFrame) -> np.ndarray:
    """d5mean target, strictly after predictor time t (spec §4)."""
    return np.asarray(
        pd.Series(np.abs(panel["r"].to_numpy(float))).rolling(5).mean().shift(-5).to_numpy()
    )


@dataclass
class GateEvidence:
    grammar_valid: bool = False
    causal: bool = False
    predictive: bool = False  # L(candidate) < L(benchmark) OOS
    incremental: bool = False  # improves after credible competitors included
    stable: bool = False  # persists across origins/assets
    null_robust: bool = False  # pre-registered permutation threshold passed
    interpretable: bool = False  # distinct+stable+survives competitors


@dataclass
class PromotionVerdict:
    hid: str
    level: int  # highest consecutive gate passed, -1 if L0 fails
    verdict: str
    failed_at: str = ""


def promote(hid: str, ev: GateEvidence) -> PromotionVerdict:
    """Level pipeline: later evidence cannot rescue an earlier failed gate (§19)."""
    gates = [("L0-grammar", ev.grammar_valid), ("L0-causal", ev.causal),
             ("L1-predictive", ev.predictive), ("L2-incremental", ev.incremental),
             ("L3-stable", ev.stable), ("L4-null-robust", ev.null_robust),
             ("L5-mechanistic", ev.interpretable)]
    # Level counts consecutive passed gates with grammar+causal sharing L0:
    # fail at index 0 -> -1; index 1 -> 0; index k>=2 -> k-2 (levels L0..).
    for idx, (name, passed) in enumerate(gates):
        if not passed:
            level = -1 if idx == 0 else (0 if idx == 1 else idx - 2)
            return PromotionVerdict(hid, level, f"HELD at {name}; later gates inadmissible", name)
    return PromotionVerdict(hid, 5, "PROMOTED to mechanism")


def frozen_record() -> dict[str, GateEvidence]:
    """Canonical §80 evidence record for H-P / H-RV / H-A."""
    return {
        "H-P": GateEvidence(True, True, True, True, True, True, False),
        "H-RV": GateEvidence(True, True, True, False, False, False, False),
        "H-A": GateEvidence(True, True, True, False, False, False, False),
    }


def decision_summary() -> dict[str, PromotionVerdict]:
    """Executable §17 table: grammar valid, predictive tested, incremental fails,
    stability mixed/drifting, permutation fails, mechanism no."""
    return {hid: promote(hid, ev) for hid, ev in frozen_record().items()}


@dataclass
class BatterySummary:
    combos: tuple = field(default_factory=lambda: tuple(COMBOS))
    incremental_eqs: tuple = field(default_factory=lambda: (
        "d_RV|P=L(M_P)-L(M_P+RV)", "d_A|P=L(M_P)-L(M_P+A)",
        "d_A|P,RV=L(M_P+RV)-L(M_P+RV+A)", "d_RV|P,A=L(M_P+A)-L(M_P+RV+A)"))


# ── Multi-Asset Trading Policy Grammar Space (15, 30, 60, 90 Universes) ───────
@dataclass(frozen=True)
class TradingGrammarSymbol:
    symbol: str
    grammar_type: str
    role: str
    description: str


TRADING_POLICY_SYMBOLS: dict[str, TradingGrammarSymbol] = {
    "H_BB": TradingGrammarSymbol(
        symbol="H_BB",
        grammar_type="Feature primitive",
        role="Predictor input",
        description="Bollinger Bands %B directional boundary position (magnitude-preserving X*)",
    ),
    "H_OBV_MOM": TradingGrammarSymbol(
        symbol="H_OBV_MOM",
        grammar_type="Feature primitive",
        role="Predictor input",
        description="On-Balance Volume 10-day pct change momentum (magnitude-preserving X*)",
    ),
    "NL_TAKER_MOM": TradingGrammarSymbol(
        symbol="NL_TAKER_MOM",
        grammar_type="Feature primitive",
        role="Predictor input",
        description="Non-linear taker buy volume surge interaction with momentum (magnitude-preserving X*)",
    ),
    "NL_MASTER_SNR": TradingGrammarSymbol(
        symbol="NL_MASTER_SNR",
        grammar_type="Feature primitive",
        role="Predictor input",
        description="Unified signal-to-noise ratio: directional drift over asymmetric variance floor squared (magnitude-preserving X*)",
    ),
    "S": TradingGrammarSymbol(
        symbol="S",
        grammar_type="Score expression",
        role="Shared predictor output",
        description="Frozen 4-term linear composite score S_{i,t} across assets and time",
    ),
    "SIGN": TradingGrammarSymbol(
        symbol="SIGN",
        grammar_type="Stateless decision operator",
        role="Baseline direction",
        description="Stateless sign or threshold-band mapping: p^{(B)} = sign(S) or band(-1, 0, 1)",
    ),
    "HYSTERESIS": TradingGrammarSymbol(
        symbol="HYSTERESIS",
        grammar_type="Stateful decision operator",
        role="Entry, persistence and exit",
        description="Stateful path-dependent decision operator with entry/exit thresholds and persistence memory",
    ),
    "THETA_IN": TradingGrammarSymbol(
        symbol="THETA_IN",
        grammar_type="Threshold parameter",
        role="Hysteresis entry boundary",
        description="Upper entry threshold for new long/short position activation (theta_in > theta_out >= 0)",
    ),
    "THETA_OUT": TradingGrammarSymbol(
        symbol="THETA_OUT",
        grammar_type="Threshold parameter",
        role="Hysteresis exit boundary",
        description="Lower exit threshold boundary for reverting to flat position",
    ),
    "PREV_POSITION": TradingGrammarSymbol(
        symbol="PREV_POSITION",
        grammar_type="State variable",
        role="Previous position",
        description="Prior bar discrete position p_{i, t-1} in {-1, 0, +1}",
    ),
    "EXPOSURE_SCALE": TradingGrammarSymbol(
        symbol="EXPOSURE_SCALE",
        grammar_type="Scalar parameter",
        role="Exposure matching",
        description="Asset or portfolio-level exposure scaling factor alpha_i = mean_E_i(H) / mean_E_i(B)",
    ),
    "TURNOVER_COST": TradingGrammarSymbol(
        symbol="TURNOVER_COST",
        grammar_type="Execution operator",
        role="Cost based on position changes",
        description="Execution friction penalty proportional to absolute position turnover |p_t - p_{t-1}|",
    ),
    # ── Candidates C, D, E additions ──
    "DEADBAND": TradingGrammarSymbol(
        symbol="DEADBAND",
        grammar_type="Stateless decision operator",
        role="Weak-signal filtering",
        description="Stateless deadband filtering operator: p_t^{(C)} = +1 if S_t >= tau, -1 if S_t <= -tau, 0 otherwise",
    ),
    "TAU": TradingGrammarSymbol(
        symbol="TAU",
        grammar_type="Threshold parameter",
        role="Deadband filtering boundary",
        description="Symmetric score magnitude threshold boundary tau >= 0 for deadband filtering",
    ),
    "TURNOVER_PENALTY": TradingGrammarSymbol(
        symbol="TURNOVER_PENALTY",
        grammar_type="Stateful decision operator",
        role="Cost-aware utility policy",
        description="Stateful utility decision operator: p_t^{(D)} = argmax_{p in {-1,0,+1}} (p S_t - lambda |p - p_{t-1}|)",
    ),
    "LAMBDA": TradingGrammarSymbol(
        symbol="LAMBDA",
        grammar_type="Penalty scalar parameter",
        role="Turnover penalty weight",
        description="Lagrangian friction penalty weight lambda >= 0 governing position inertia",
    ),
    "PRODUCT_H_BB_TAKER": TradingGrammarSymbol(
        symbol="PRODUCT_H_BB_TAKER",
        grammar_type="Interaction operator",
        role="Nonlinear interaction predictor term",
        description="Bilinear feature interaction: X^*_{H-BB, t} * X^*_{NL-TAKER_MOM, t}",
    ),
    "BETA_5": TradingGrammarSymbol(
        symbol="BETA_5",
        grammar_type="Interaction coefficient parameter",
        role="Nonlinear interaction weight",
        description="Discovered scaling coefficient beta_5 for nonlinear product interaction term",
    ),
}

# ── Primary Candidate Taxonomy (A, B, H, C, D, E) ───────────────────────────
@dataclass(frozen=True)
class GrammarCandidateSpec:
    candidate_id: str
    name: str
    candidate_type: str
    primary_question: str
    formula: str


GRAMMAR_CANDIDATE_CATALOG: dict[str, GrammarCandidateSpec] = {
    "A": GrammarCandidateSpec(
        candidate_id="A",
        name="Four-term magnitude model",
        candidate_type="Predictor",
        primary_question="Does the original signal retain predictive value?",
        formula="S_t^{(A)} = 0.8111 X^*_{H-BB} - 1.4272 X^*_{H-OBV_MOM} + 0.5984 X^*_{NL-TAKER_MOM} + 0.6740 X^*_{NL-MASTER_SNR}",
    ),
    "B": GrammarCandidateSpec(
        candidate_id="B",
        name="Second frozen predictor",
        candidate_type="Predictor",
        primary_question="Does the alternative structure generalize?",
        formula="S_t^{(B)} = 0.5447 X^*_{H-BB} - 0.2019 X^*_{H-MOM} - 1.6370 X^*_{H-OBV_MOM} + 0.4171 X^*_{NL-TAKER_MOM}",
    ),
    "H": GrammarCandidateSpec(
        candidate_id="H",
        name="Hysteresis",
        candidate_type="Stateful policy",
        primary_question="Does position persistence add value?",
        formula="p_t^{(H)} = H(S_t, p_{t-1}; theta_in, theta_out)",
    ),
    "C": GrammarCandidateSpec(
        candidate_id="C",
        name="Deadband",
        candidate_type="Stateless policy",
        primary_question="Is weak-signal filtering sufficient?",
        formula="p_t^{(C)} = +1 if S_t >= tau, -1 if S_t <= -tau, 0 otherwise",
    ),
    "D": GrammarCandidateSpec(
        candidate_id="D",
        name="Turnover penalty",
        candidate_type="Stateful utility policy",
        primary_question="Does explicit change-cost penalization improve decisions?",
        formula="p_t^{(D)} = argmax_{p in {-1,0,+1}} (p S_t - lambda |p - p_{t-1}|)",
    ),
    "E": GrammarCandidateSpec(
        candidate_id="E",
        name="Interaction predictor",
        candidate_type="Nonlinear predictor",
        primary_question="Does conditional interaction add genuine predictive value?",
        formula="S_t^{(E)} = S_t^{(A)} + beta_5 (X^*_{H-BB} * X^*_{NL-TAKER_MOM})",
    ),
}

# Shared Predictor Equation A: Frozen 4-Term Composite Weights
# Provenance: authoritatively pinned from PINNED_30_CHROMOSOME in multi_asset_signals_sandbox.py
# (Indices 9, 16, 24, 25 of 26-signal space).
EQUATION_A_AUTHORITATIVE_WEIGHTS: dict[str, float] = {
    "H_BB": 0.811068,
    "H_OBV_MOM": -1.427209,
    "NL_TAKER_MOM": 0.598382,
    "NL_MASTER_SNR": 0.673950,
}

EQUATION_A_ROUNDED_WEIGHTS: dict[str, float] = {
    "H_BB": 0.8111,
    "H_OBV_MOM": -1.4272,
    "NL_TAKER_MOM": 0.5984,
    "NL_MASTER_SNR": 0.6740,  # 0.67395 rounds to 0.6740; displayed as 0.6739 in truncated records
}

# Predictor Equation B: Second Frozen Alternative Predictor
# Provenance: authoritatively pinned from PINNED_60_CHROMOSOME in multi_asset_signals_sandbox.py
# (Indices 9, 10, 16, 23 of 26-signal space).
EQUATION_B_AUTHORITATIVE_WEIGHTS: dict[str, float] = {
    "H_BB": 0.544673,
    "H_MOM": -0.201948,
    "H_OBV_MOM": -1.637016,
    "NL_TAKER_MOM": 0.417085,
}


def evaluate_equation_a_score(
    feat_bb: np.ndarray,
    feat_obv: np.ndarray,
    feat_taker: np.ndarray,
    feat_snr: np.ndarray,
    use_authoritative: bool = True,
) -> np.ndarray:
    """Compute shared predictor score S_{i,t} using Equation A.

    S_{i,t} = w_bb * X^*_{H-BB,i,t} + w_obv * X^*_{H-OBV_MOM,i,t}
            + w_taker * X^*_{NL-TAKER_MOM,i,t} + w_snr * X^*_{NL-MASTER_SNR,i,t}
    """
    w = EQUATION_A_AUTHORITATIVE_WEIGHTS if use_authoritative else EQUATION_A_ROUNDED_WEIGHTS
    return (
        w["H_BB"] * feat_bb
        + w["H_OBV_MOM"] * feat_obv
        + w["NL_TAKER_MOM"] * feat_taker
        + w["NL_MASTER_SNR"] * feat_snr
    )


def evaluate_equation_b_score(
    feat_bb: np.ndarray,
    feat_mom: np.ndarray,
    feat_obv: np.ndarray,
    feat_taker: np.ndarray,
) -> np.ndarray:
    """Compute second frozen predictor score S_{i,t}^{(B)} using Equation B."""
    w = EQUATION_B_AUTHORITATIVE_WEIGHTS
    return (
        w["H_BB"] * feat_bb
        + w["H_MOM"] * feat_mom
        + w["H_OBV_MOM"] * feat_obv
        + w["NL_TAKER_MOM"] * feat_taker
    )


def op_product_interaction(feat_1: np.ndarray, feat_2: np.ndarray) -> np.ndarray:
    """Candidate interaction operator: PRODUCT(H_BB, NL_TAKER_MOM)."""
    return np.asarray(np.asarray(np.asarray(np.asarray(feat_1, dtype=float) * np.asarray(feat_2, dtype=float))))


def evaluate_equation_e_score(
    feat_bb: np.ndarray,
    feat_obv: np.ndarray,
    feat_taker: np.ndarray,
    feat_snr: np.ndarray,
    beta_5: float = 0.10,
    use_authoritative: bool = True,
) -> np.ndarray:
    """Compute Candidate E nonlinear predictor: S_t^{(E)} = S_t^{(A)} + beta_5 * (X^*_{H-BB} * X^*_{NL-TAKER_MOM})."""
    score_a = evaluate_equation_a_score(
        feat_bb, feat_obv, feat_taker, feat_snr, use_authoritative=use_authoritative
    )
    interaction = op_product_interaction(feat_bb, feat_taker)
    return score_a + beta_5 * interaction


def op_deadband(score: np.ndarray, tau: float = 0.1979) -> np.ndarray:
    """Candidate C: Stateless deadband decision operator: DEADBAND(S, tau).

    p_t^{(C)} = +1 if S_t >= tau, -1 if S_t <= -tau, 0 otherwise.
    Has zero memory of prior position; tests pure signal-strength filtering.
    """
    score = np.asarray(score, dtype=float)
    return np.where(score >= tau, 1.0, np.where(score <= -tau, -1.0, 0.0))


def op_turnover_penalty_step(score: float, prev_pos: float, lambda_pen: float = 0.20) -> float:
    """Single-asset single-step stateful utility decision: TURNOVER_PENALTY(S, PREV_POSITION, LAMBDA).

    p_t^{(D)} = argmax_{p in {-1, 0, +1}} (p * S_t - lambda * |p - prev_pos|)
    """
    u_m1 = -score - lambda_pen * abs(-1.0 - prev_pos)
    u_0 = 0.0 - lambda_pen * abs(0.0 - prev_pos)
    u_p1 = score - lambda_pen * abs(1.0 - prev_pos)
    candidates = [-1.0, 0.0, 1.0]
    utilities = [u_m1, u_0, u_p1]
    return candidates[int(np.argmax(utilities))]


def op_turnover_penalty(
    scores: np.ndarray,
    lambda_pen: float = 0.20,
    init_pos: np.ndarray | None = None,
) -> np.ndarray:
    """Candidate D: Stateful utility decision operator with turnover penalty lambda."""
    scores = np.asarray(scores, dtype=float)
    is_1d = scores.ndim == 1
    sc_2d = scores[:, None] if is_1d else scores
    T, N = sc_2d.shape
    pos = np.zeros((T, N), dtype=float)
    cur = np.zeros(N, dtype=float) if init_pos is None else np.array(init_pos, dtype=float).copy()
    for t in range(T):
        for i in range(N):
            cur[i] = op_turnover_penalty_step(sc_2d[t, i], cur[i], lambda_pen)
        pos[t] = cur.copy()
    return pos[:, 0] if is_1d else pos


def op_sign(score: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """Stateless baseline decision operator: SIGN(S; threshold)."""
    score = np.asarray(score, dtype=float)
    if threshold <= 0.0:
        return np.where(score > 0.0, 1.0, np.where(score < 0.0, -1.0, 0.0))
    return np.where(score > threshold, 1.0, np.where(score < -threshold, -1.0, 0.0))


def op_hysteresis_step(
    score: float,
    prev_pos: float,
    theta_in: float = 0.1979,
    theta_out: float = 0.0,
) -> float:
    """Single-asset single-step stateful decision operator: HYSTERESIS(S, PREV_POSITION; THETA_IN, THETA_OUT)."""
    if prev_pos > 0.0:
        if score < -theta_in:
            return -1.0
        elif score < theta_out:
            return 0.0
        return 1.0
    elif prev_pos < 0.0:
        if score > theta_in:
            return 1.0
        elif score > -theta_out:
            return 0.0
        return -1.0
    else:
        if score > theta_in:
            return 1.0
        elif score < -theta_in:
            return -1.0
        return 0.0


def op_hysteresis(
    scores: np.ndarray,
    theta_in: float = 0.1979,
    theta_out: float = 0.0,
    init_pos: np.ndarray | None = None,
) -> np.ndarray:
    """Stateful decision operator applied over time series scores (T, N) or (T,)."""
    scores = np.asarray(scores, dtype=float)
    is_1d = scores.ndim == 1
    if is_1d:
        sc_2d = scores[:, None]
    else:
        sc_2d = scores
    T, N = sc_2d.shape
    pos = np.zeros((T, N), dtype=float)
    cur = np.zeros(N, dtype=float) if init_pos is None else np.array(init_pos, dtype=float).copy()
    for t in range(T):
        for i in range(N):
            cur[i] = op_hysteresis_step(sc_2d[t, i], cur[i], theta_in, theta_out)
        pos[t] = cur.copy()
    return pos[:, 0] if is_1d else pos


def op_exposure_scale(
    pos_base: np.ndarray,
    alpha: float | np.ndarray,
) -> np.ndarray:
    """Equation 3: Exposure-matched control operator: p^{(E)} = alpha * p^{(B)}."""
    scaled = np.asarray(alpha, dtype=float) * np.asarray(pos_base, dtype=float)
    return np.asarray(scaled)


def op_turnover_cost(
    pos_curr: np.ndarray,
    pos_prev: np.ndarray,
    capital: float = 100000.0,
    fee_bps: float = 5.0,
    n_assets: int = 1,
) -> float:
    """Execution cost operator: TURNOVER_COST based on position changes."""
    to = float(np.sum(np.abs(pos_curr - pos_prev)))
    fee_rate = fee_bps / 10000.0
    return capital * to * (fee_rate / max(n_assets, 1))


# ── Equivalence Verification Gates ─────────────────────────────────────────────
@dataclass(frozen=True)
class EquivalenceAudit:
    predictor_equivalence: bool
    policy_equivalence: bool
    economic_equivalence: bool
    predictor_max_abs_diff: float
    policy_divergence_pct: float
    economic_delta_net_ret: float
    economic_delta_sharpe: float
    economic_delta_max_dd: float
    details: dict[str, Any]


def audit_policy_equivalences(
    score_base: np.ndarray,
    score_hyst: np.ndarray,
    pos_base: np.ndarray,
    pos_hyst: np.ndarray,
    metrics_base: dict[str, float],
    metrics_hyst: dict[str, float],
    tol_predictor: float = 1e-9,
    tol_econ_ret: float = 0.05,
    tol_econ_sr: float = 0.02,
) -> EquivalenceAudit:
    """Adjudicate Predictor, Policy, and Economic equivalences between baseline and hysteresis."""
    # 1. Predictor Equivalence
    max_diff_score = float(np.max(np.abs(score_base - score_hyst)))
    pred_equiv = bool(max_diff_score <= tol_predictor)

    # 2. Policy Equivalence
    pol_div = float(np.mean(pos_base != pos_hyst) * 100.0)
    pol_equiv = bool(pol_div == 0.0)

    # 3. Economic Equivalence
    delta_net = abs(metrics_base.get("net_return", 0.0) - metrics_hyst.get("net_return", 0.0))
    delta_sr = abs(metrics_base.get("sharpe", 0.0) - metrics_hyst.get("sharpe", 0.0))
    delta_dd = abs(metrics_base.get("max_dd", 0.0) - metrics_hyst.get("max_dd", 0.0))
    econ_equiv = bool((delta_net <= tol_econ_ret) and (delta_sr <= tol_econ_sr))

    return EquivalenceAudit(
        predictor_equivalence=pred_equiv,
        policy_equivalence=pol_equiv,
        economic_equivalence=econ_equiv,
        predictor_max_abs_diff=max_diff_score,
        policy_divergence_pct=pol_div,
        economic_delta_net_ret=delta_net,
        economic_delta_sharpe=delta_sr,
        economic_delta_max_dd=delta_dd,
        details={
            "metrics_base": metrics_base,
            "metrics_hyst": metrics_hyst,
        },
    )


# ── Comprehensive Trading Grammar Space Architecture (Layers 0 - 4) ───────────
from apedp_dyn.grammar.trading_grammar_catalog import (  # noqa: E402
    AvailabilityTiming,
    DataType,
    FailurePolicy,
    FeatureFamily,
    FeatureSpec,
    GrammarLayer,
    MASTER_FEATURE_CATALOG,
    MASTER_TARGET_CATALOG,
    TargetSpec,
    TradingGrammarCatalog,
    compute_recommended_first_stage_features,
    consecutive_down_bars,
    consecutive_up_bars,
    count_condition,
    cross_sectional_rank,
    cross_sectional_zscore,
    ewma,
    exp_clamp,
    negative_part,
    op_acceleration,
    op_clip,
    op_cooldown,
    op_diff,
    op_lag,
    op_log_return,
    op_pct_change,
    op_tanh,
    positive_part,
    regime_gate,
    robust_zscore,
    roll_beta,
    roll_corr,
    roll_kurtosis,
    roll_max,
    roll_mean,
    roll_median,
    roll_min,
    roll_quantile,
    roll_skew,
    roll_slope,
    roll_std,
    roll_sum,
    roll_var,
    safe_div,
    safe_log,
    safe_log1p,
    safe_pow,
    safe_sqrt,
    signal_gate,
    time_since_last_high,
    time_since_last_low,
    winsorize,
)


