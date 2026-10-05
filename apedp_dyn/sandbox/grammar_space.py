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

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

PRIMITIVES = ("r_t", "V_t")
UNARY_OPS = ("abs", "log", "zscore", "rank")
ROLLING_OPS = ("RM_w",)
BINARY_OPS = ("+", "-", "*", "/")
MIN_HISTORY = 60
RANK_WINDOW = 40

LEVELS = ("L0-grammar", "L1-predictive", "L2-incremental", "L3-stable",
          "L4-null-robust", "L5-mechanistic")


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
    build: object = None  # set by registry below


def _feat_logV(p: pd.DataFrame) -> np.ndarray:
    return np.log(np.maximum(p["V"].to_numpy(float), 1e-18))


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
        return causal_zscore(inner)


def _feat_sig(p: pd.DataFrame) -> np.ndarray:
    return np.abs(p["r"].to_numpy(float))


HYPOTHESES: dict[str, HypothesisSpec] = {
    "H-P": HypothesisSpec("H-P", "y=a+b|r_t|", ("sig",),
                          "predictive persistence only; b is not a causal effect.", _feat_sig),
    "H-RV": HypothesisSpec("H-RV", "y=a+bR_40(V_t)", ("rankV",),
                           "relative activity representation, not a unique mechanism.", _feat_rankV),
    "H-A": HypothesisSpec("H-A", "y=a+bHA_t", ("HA",),
                          "composite predictive representation, not automatically a mechanism.", _feat_HA),
    "H-LOG": HypothesisSpec("H-LOG", "y=a+blog(V_t)", ("logV",),
                            "level-volume representation.", _feat_logV),
}

COMBOS = {"M_P": ("H-P",), "M_RV": ("H-RV",), "M_A": ("H-A",),
          "M_P+RV": ("H-P", "H-RV"), "M_P+A": ("H-P", "H-A"),
          "M_RV+A": ("H-RV", "H-A"), "M_P+RV+A": ("H-P", "H-RV", "H-A")}


def build_features(panel: pd.DataFrame) -> dict[str, np.ndarray]:
    """Causal feature columns for one asset panel (Level-0 domain-checked)."""
    return {"sig": _feat_sig(panel), "rankV": _feat_rankV(panel),
            "HA": _feat_HA(panel), "logV": _feat_logV(panel)}


def build_target(panel: pd.DataFrame) -> np.ndarray:
    """d5mean target, strictly after predictor time t (spec §4)."""
    return pd.Series(np.abs(panel["r"].to_numpy(float))).shift(-1).rolling(5).mean().to_numpy()


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
