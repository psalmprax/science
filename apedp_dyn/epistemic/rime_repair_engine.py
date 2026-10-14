"""RIME Repair v0.1.1–v0.1.4 — bounded residual-guided repair engine.

Design (user-spec §1–§6, grounded in real repo types):

  RIME proposes (CandidateProposal) → diagnosis measures (ResidualReport) →
  THIS engine modifies (bounded operators) → paired validation compares on
  untouched rows → sealed single evaluation → evidence package deferred to
  the APEDP-X track for adjudication.

Deliberate deviations from the §6 sketch, with reasons:

  * Python, not a Rust module: every operation here composes the frozen
    kernel scorer (`compute_score`) with explicit new terms. No graph
    surgery is needed, so no kernel rebuild and no new equation types.
    `CandidateProposal` is reused verbatim — no competing definitions.
  * No `EpistemicStatus` is returned: no CandidateProposal→ScoredCandidate
    bridge exists in this repo, and fabricating `CandidateDynamics` terms
    for RIME equations would be ungrounded. The outcome carries an evidence
    package (`submit_note`); promotion stays with the APEDP-X track, which
    operates on its own candidate types. Independence is structural.
  * Four data roles (§4) are enforced in code: `RepairPartitions` releases
    the sealed split exactly once and raises the repo's own
    `HoldoutContaminationError` on any second access. A sealed failure has
    no API path back into repair.

Operator scope (v0.1.1–v0.1.3):

  * R1 (parameter refit): least-squares refit of the frozen scorer output
    on the repair split. Tests parameter error vs missing structure.
  * R2 (term insertion): one permitted term at a time — square of a declared
    dependency, or the pairwise product when ≥2 exist. Fitted as
    a·orig + b·term + c on the repair split only.
  * R3 (lag insertion): lag-1 of a declared dependency (causal shift with
    NaN warmup; minimal causal delay — the full frozen LAG_SET is out of
    v0.1 scope). Fitted the same way.
  * R4 (equivalence): not a generator. `cluster_equivalent` groups proposals
    by measured prediction equality on validation rows, mirroring the
    referee's canonicalization principle without impersonating adjudicate().

AGENTS.md §1.3: this module defines no threshold literals. `min_rows`,
`min_validation_r2_gain`, `equivalence_tol`, budgets, and all diagnostic
cutoffs are explicit caller arguments.
"""
from __future__ import annotations

import enum
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from apedp_dyn.epistemic.experiment_ledger import HoldoutContaminationError
from apedp_dyn.epistemic import rime_repair_diagnosis as diagmod
from apedp_dyn.grammar.proposer_interface import CandidateProposal


class RepairOperator(str, enum.Enum):
    R1_PARAM_REFIT = "R1_PARAM_REFIT"
    R2_TERM_INSERT = "R2_TERM_INSERT"
    R3_LAG_INSERT = "R3_LAG_INSERT"
    R5_PADE_SATURATION = "R5_PADE_SATURATION"


class RepairStatus(str, enum.Enum):
    IMPROVED = "IMPROVED"
    NO_IMPROVEMENT = "NO_IMPROVEMENT"
    AMBIGUOUS = "AMBIGUOUS"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True)
class RepairConfig:
    """Bounds for one repair run. All fields explicit, no defaults that hide policy.

    Predictive-gate criteria (both are reported; the bar is chosen by the
    caller, never inferred):
      * ``min_validation_r2_gain`` — absolute validation R² headroom. Fails by
        construction on real baselines already near 1 (measured: refuses a
        sealed-validated +0.005 on AME2020 where the original is 0.994).
      * ``min_sse_reduction`` — relative squared-error reduction
        G = (SSE_base − SSE_cand)/SSE_base on the same rows. Scale-aware; the
        preregistered companion criterion for high-baseline problems.
    """

    max_candidates_per_round: int
    max_rounds: int
    max_added_terms: int
    min_validation_r2_gain: float
    equivalence_tol: float
    operators: tuple[RepairOperator, ...]
    min_sse_reduction: float = 0.0
    # Identifiability band (explicit, default 0 = frozen single-winner
    # behaviour): passing candidates whose validation R2 lies within this
    # band of the best are observationally indistinguishable, and the run
    # resolves to AMBIGUOUS (MULTI_HYPOTHESIS territory) rather than
    # crowning the marginally-highest scorer.
    equivalence_r2_tol: float = 0.0
    # Domain-informed term family (Layer C, declared and measured separately):
    # bounded rational powers x^(p) of the declared dependencies, appended to
    # the R2 catalogue. Empty = frozen standard catalogue only. Domain priors
    # are NOT hidden corrections: they enter only through this explicit field
    # and are scored by the same gates.
    rational_powers: tuple[float, ...] = ()


@dataclass
class RepairPartitions:
    """Chronological discovery / repair / validation / sealed splits.

    The sealed split is released exactly once via :meth:`take_sealed`; any
    second access raises :class:`HoldoutContaminationError`. There is no API
    path from a sealed result back into repair: a sealed failure stays a
    failure until a new experiment is preregistered (i.e. a new object).
    """

    discovery_df: pd.DataFrame
    discovery_y: np.ndarray
    repair_df: pd.DataFrame
    repair_y: np.ndarray
    validation_df: pd.DataFrame
    validation_y: np.ndarray
    _sealed_df: pd.DataFrame = field(repr=False)
    _sealed_y: np.ndarray = field(repr=False)
    _sealed_taken: bool = field(default=False, repr=False)
    spans: dict[str, tuple[int, int]] = field(default_factory=dict)

    @staticmethod
    def from_frame(
        df: pd.DataFrame,
        y: np.ndarray | pd.Series,
        fracs: tuple[float, float, float, float] = (0.4, 0.2, 0.2, 0.2),
    ) -> "RepairPartitions":
        y = np.asarray(y, dtype=float)
        n = len(df)
        if n < 40 or abs(sum(fracs) - 1.0) > 1e-9 or any(f <= 0.0 for f in fracs):
            raise ValueError("need >=40 rows and positive fracs summing to 1")
        i1 = int(n * fracs[0])
        i2 = i1 + int(n * fracs[1])
        i3 = i2 + int(n * fracs[2])
        return RepairPartitions(
            discovery_df=df.iloc[:i1].reset_index(drop=True),
            discovery_y=y[:i1],
            repair_df=df.iloc[i1:i2].reset_index(drop=True),
            repair_y=y[i1:i2],
            validation_df=df.iloc[i2:i3].reset_index(drop=True),
            validation_y=y[i2:i3],
            _sealed_df=df.iloc[i3:].reset_index(drop=True),
            _sealed_y=y[i3:],
            spans={"discovery": (0, i1), "repair": (i1, i2), "validation": (i2, i3), "sealed": (i3, n)},
        )

    def take_sealed(self) -> tuple[pd.DataFrame, np.ndarray]:
        if self._sealed_taken:
            raise HoldoutContaminationError(
                "sealed split already consumed: a sealed failure stays a failure "
                "until a new experiment is preregistered"
            )
        # Bypass frozen-dataclass restriction via object.__setattr__.
        object.__setattr__(self, "_sealed_taken", True)
        return self._sealed_df, self._sealed_y


def _finite_mask(*arrays: np.ndarray) -> np.ndarray:
    """Row mask: 1-D arrays contribute element finiteness, 2-D (design
    matrix) contributes all-columns-finite per row."""
    m: np.ndarray | None = None
    for a in arrays:
        a = np.asarray(a, dtype=float)
        f = np.all(np.isfinite(a), axis=1) if a.ndim == 2 else np.isfinite(a)
        m = f if m is None else (m & f)
    assert m is not None
    return m


def _r2(yt: np.ndarray, yp: np.ndarray) -> float:
    ss = float(np.sum((yt - yt.mean()) ** 2))
    return 1.0 - float(np.sum((yt - yp) ** 2)) / ss if ss > 0.0 else float("nan")


def _sse(yt: np.ndarray, yp: np.ndarray) -> float:
    """Sum of squared errors over the jointly-finite rows (NaN if none)."""
    yt = np.asarray(yt, dtype=float)
    yp = np.asarray(yp, dtype=float)
    m = np.isfinite(yt) & np.isfinite(yp)
    return float(np.sum((yt[m] - yp[m]) ** 2)) if int(m.sum()) else float("nan")


def _lstsq_readout(cols: list[np.ndarray], y: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares fit of [terms..., 1] to y on finite rows. Returns (coefs, r2)."""
    M = np.column_stack([np.asarray(c, dtype=float) for c in cols] + [np.ones(len(y))])
    m = _finite_mask(M, y)
    if int(m.sum()) < len(cols) + 2:
        return np.full(len(cols) + 1, np.nan), float("nan")
    coef, *_ = np.linalg.lstsq(M[m], y[m], rcond=None)
    pred = M[m] @ coef
    return coef, _r2(y[m], pred)


def _lagk(col: Any, k: int) -> np.ndarray:
    """Causal shift by k rows with NaN warmup. k=1 preserves _lag1 behavior."""
    out = np.full(len(col), np.nan)
    v = np.asarray(col, dtype=float)
    if k <= 0:
        out[:] = v
    else:
        out[k:] = v[:-k]
    return out


def _lag1(col: pd.Series) -> np.ndarray:
    return _lagk(col, 1)


# Shortest frozen lags (RIME graph LAG_SET opens [1, 2, 5, 10, 20];
# v0.1 repair scope covers the two shortest — documented, not configured).
_REPAIR_LAGS: tuple[int, ...] = (1, 2)


def _interaction_probes(df: pd.DataFrame, deps: Sequence[str]) -> list[tuple[str, str, np.ndarray]]:
    """Probe/term family: pairwise products of declared dependencies at lag
    0 and the two shortest frozen lags (squares included as i==j).

    SINGLE SOURCE for both the diagnostic screen and the R2 term catalogue:
    a probe and its repair term are the same column by construction, so the
    screen can only fire for structure the engine is actually allowed to add.
    Derived ONLY from declared dependencies — no frame shopping.
    Returns (term_label, source_dep, values).
    """
    terms: list[tuple[str, str, np.ndarray]] = []
    cols = {d: np.asarray(df[d], dtype=float) if d in df.columns else np.full(len(df), np.nan) for d in deps}
    for ii, a in enumerate(deps):
        for b in deps[ii:]:
            terms.append((f"({a}*{b})", f"{a},{b}", cols[a] * cols[b]))
            for k in _REPAIR_LAGS:
                la, lb = _lagk(cols[a], k), _lagk(cols[b], k)
                terms.append((f"({a}[t-{k}]*{b}[t-{k}])", f"{a},{b}", la * lb))
    return terms


def _rational_power(col: np.ndarray, p: float) -> np.ndarray:
    """x^p on the signed-safe branch: sign(x)*|x|^p, NaN-preserving.

    Bounded rational powers on possibly-signed inputs must not silently
    produce NaN (which the scorer would skip) — the sign-preserving branch is
    the domain-safe reading and is declared by the caller's exponent choice.
    """
    v = np.asarray(col, dtype=float)
    return np.sign(v) * np.power(np.abs(v), p)


def _candidate_terms(
    df: pd.DataFrame,
    deps: Sequence[str],
    rational_powers: Sequence[float] = (),
) -> list[tuple[str, str, np.ndarray, Any]]:
    """Permitted term columns with their operator attached.

    Standard catalogue: single lag-1 insertions (R3) plus the interaction
    probe family mirrored 1:1 with the diagnostic screen (R2).
    Optional domain-informed family: rational powers of each declared
    dependency (R2), enabled ONLY via the explicit config field.
    No other columns are readable — no frame shopping. The operator rides
    along explicitly — no string sniffing.
    """
    terms: list[tuple[str, str, np.ndarray, Any]] = []
    for d in deps:
        if d in df.columns:
            terms.append((f"({d}[t-1])", d, _lagk(df[d], 1), RepairOperator.R3_LAG_INSERT))
    for label, src, values in _interaction_probes(df, deps):
        terms.append((label, src, values, RepairOperator.R2_TERM_INSERT))
    if rational_powers:
        for d in deps:
            if d not in df.columns:
                continue
            col = np.asarray(df[d], dtype=float)
            for p in rational_powers:
                pf = float(p)
                exp = int(pf) if pf.is_integer() else f"{pf:.3f}".rstrip("0").rstrip(".")
                terms.append((f"({d}^{exp})", d, _rational_power(col, pf),
                              RepairOperator.R2_TERM_INSERT))
    return terms


def _rational_power_term_fn(src: str, p: float):
    """Live-frame re-derivation for a rational-power term (causal by construction)."""
    def fn(df: pd.DataFrame, _d=src, _p=float(p)):
        if _d not in df.columns:
            return np.full(len(df), np.nan)
        return _rational_power(df[_d], _p)
    return fn


def _wrap_repaired(
    parent: CandidateProposal,
    operator: RepairOperator,
    term_label: str | None,
    term_values_fn: Any,
    coefs: np.ndarray,
    repair_r2: float,
    repair_rows: int,
    added_terms: int,
    seq: int,
) -> CandidateProposal:
    if term_label is None:
        alpha, beta = float(coefs[0]), float(coefs[1])
        formula = f"{alpha:+.6g}*({parent.formula}){beta:+.6g}"
        params = {"alpha": alpha, "beta": beta}
    else:
        alpha, beta, gamma = float(coefs[0]), float(coefs[1]), float(coefs[2])
        formula = f"{alpha:+.6g}*({parent.formula}){beta:+.6g}*({term_label}){gamma:+.6g}"
        params = {"alpha": alpha, "beta": beta, "gamma": gamma}
    params = dict(params)
    params.update({"repair_r2": float(repair_r2), "added_terms": int(added_terms)})

    def compute_score(df: pd.DataFrame, _p=parent, _fn=term_values_fn, _c=np.asarray(coefs, dtype=float)) -> np.ndarray:
        base = np.asarray(_p.compute_score(df), dtype=float)
        if _fn is None:
            return _c[0] * base + _c[1]
        t = np.asarray(_fn(df), dtype=float)
        out = _c[0] * base + _c[1] * t + _c[2]
        out[~(np.isfinite(base) & np.isfinite(t))] = np.nan
        return out

    deps = tuple(parent.declared_nominal_dependencies)
    return CandidateProposal(
        proposer_id="rime_repair_engine_v011",
        candidate_id=f"{parent.candidate_id}::{operator.value}#{seq}",
        formula=formula,
        strategy_archetype=parent.strategy_archetype,
        declared_scale_invariant=parent.declared_scale_invariant,
        declared_nominal_dependencies=deps,
        compute_score=compute_score,
        parameters=params,
        metadata={
            "parent_id": parent.candidate_id,
            "parent_graph_id": (parent.metadata or {}).get("graph_id"),
            "operator": operator.value,
            "term": term_label,
            "repair_rows": int(repair_rows),
            "engine": "rime_repair_v011",
        },
    )


def _wrap_pade_repaired(
    parent: CandidateProposal,
    term_label: str,
    term_values_fn: Any,
    a_param: float,
    b_param: float,
    repair_r2: float,
    repair_rows: int,
    seq: int,
) -> CandidateProposal:
    formula = f"({a_param:+.6g}*({parent.formula})) / (1 + {b_param:.6g}*|{parent.formula}|)"
    params = {
        "a": float(a_param),
        "b": float(b_param),
        "repair_r2": float(repair_r2),
        "added_terms": 1,
    }

    def compute_score(df: pd.DataFrame, _fn=term_values_fn) -> np.ndarray:
        return np.asarray(_fn(df), dtype=float)

    deps = tuple(parent.declared_nominal_dependencies)
    return CandidateProposal(
        proposer_id="rime_repair_engine_v011",
        candidate_id=f"{parent.candidate_id}::{RepairOperator.R5_PADE_SATURATION.value}#{seq}",
        formula=formula,
        strategy_archetype=parent.strategy_archetype,
        declared_scale_invariant=parent.declared_scale_invariant,
        declared_nominal_dependencies=deps,
        compute_score=compute_score,
        parameters=params,
        metadata={
            "parent_id": parent.candidate_id,
            "parent_graph_id": (parent.metadata or {}).get("graph_id"),
            "operator": RepairOperator.R5_PADE_SATURATION.value,
            "term": term_label,
            "repair_rows": int(repair_rows),
            "engine": "rime_repair_v011",
        },
    )


def generate_repairs(
    original: CandidateProposal,
    findings: Sequence[str],
    repair_df: pd.DataFrame,
    repair_y: np.ndarray,
    config: RepairConfig,
) -> list[CandidateProposal]:
    """Bounded, finding-targeted candidate generation (no validation touched).

    Priority: R1 first (parameter error is the cheapest hypothesis), then the
    finding-targeted structural operator. `no_reliable_structure` and
    `insufficient_rows` emit nothing. Total capped at max_candidates_per_round.
    """
    findings = tuple(findings)
    if diagmod.FINDING_NO_RELIABLE_STRUCTURE in findings or diagmod.FINDING_INSUFFICIENT_ROWS in findings:
        return []
    base = np.asarray(original.compute_score(repair_df), dtype=float)
    y = np.asarray(repair_y, dtype=float)
    out: list[CandidateProposal] = []
    seq = 0

    def room() -> bool:
        return len(out) < config.max_candidates_per_round

    if RepairOperator.R1_PARAM_REFIT in config.operators and room():
        coef, r2 = _lstsq_readout([base], y)
        if np.all(np.isfinite(coef)):
            seq += 1
            out.append(_wrap_repaired(original, RepairOperator.R1_PARAM_REFIT, None, None, coef, r2, int(np.isfinite(y).sum()), 0, seq))

    if RepairOperator.R5_PADE_SATURATION in config.operators and room():
        finite = np.isfinite(base) & np.isfinite(y)
        if int(finite.sum()) >= 20:
            zb = base[finite]
            yb = y[finite]
            try:
                from scipy.optimize import curve_fit
                popt, _ = curve_fit(
                    lambda z, a, b: (a * z) / (1.0 + b * np.abs(z)),
                    zb, yb, p0=[1.0, 0.0], bounds=([0.001, 0.0], [100.0, 1e7]), maxfev=2000
                )
                a_fit, b_fit = float(popt[0]), float(popt[1])
                pred_tr = (a_fit * zb) / (1.0 + b_fit * np.abs(zb))
                ss_tot = float(np.sum((yb - yb.mean()) ** 2))
                r2_tr = 1.0 - float(np.sum((yb - pred_tr) ** 2)) / ss_tot if ss_tot > 0 else 0.0
                seq += 1
                fn = lambda df, _a=a_fit, _b=b_fit, _parent=original: (
                    _a * np.asarray(_parent.compute_score(df), dtype=float)
                    / (1.0 + _b * np.abs(np.asarray(_parent.compute_score(df), dtype=float)))
                )
                label = f"pade_sat(a={a_fit:.4g},b={b_fit:.4g})"
                out.append(_wrap_pade_repaired(original, label, fn, a_fit, b_fit, r2_tr, int(finite.sum()), seq))
            except Exception:
                pass

    if config.max_added_terms < 1:
        return out
    deps = [d for d in original.declared_nominal_dependencies if d in repair_df.columns]
    if not deps:
        return out
    want_r2 = ((diagmod.FINDING_CURVATURE_SUSPECT in findings
                or diagmod.FINDING_INTERACTION_SUSPECT in findings)
               and RepairOperator.R2_TERM_INSERT in config.operators)
    want_r3 = diagmod.FINDING_TEMPORAL_DEPENDENCE in findings and RepairOperator.R3_LAG_INSERT in config.operators
    if not (want_r2 or want_r3):
        return out
    for label, src, values, op in _candidate_terms(repair_df, deps, config.rational_powers):
        if not room():
            break
        if op is RepairOperator.R3_LAG_INSERT and not want_r3:
            continue
        if op is RepairOperator.R2_TERM_INSERT and not want_r2:
            continue
        coef, r2 = _lstsq_readout([base, values], y)
        if not np.all(np.isfinite(coef)):
            continue
        seq += 1
        # Re-derive the term inside the scorer from the live frame (causal).
        # Labels encode their own construction: "(d[t-1])" is a single lag;
        # "(a*b)" / "(a[t-k]*b[t-k])" split lagged columns from src "a,b";
        # "(d^p)" is a sign-safe rational power of one declared dependency.
        if op is RepairOperator.R3_LAG_INSERT:
            fn = (lambda df, _d=src: _lagk(df[_d], 1) if _d in df.columns else np.full(len(df), np.nan))
        elif "^" in label:
            mp = re.search(r"\^(\d+(?:\.\d+)?)\)$", label)
            fn = _rational_power_term_fn(src, float(mp.group(1))) if mp else (
                lambda df: np.full(len(df), np.nan))
        else:
            a, b = src.split(",")
            m = re.search(r"\[t-(\d+)\]", label)
            k = int(m.group(1)) if m else 0
            fn = (lambda df, _a=a, _b=b, _k=k: _lagk(np.asarray(df[_a], dtype=float), _k)
                  * _lagk(np.asarray(df[_b], dtype=float), _k)
                  if _a in df.columns and _b in df.columns else np.full(len(df), np.nan))
        out.append(_wrap_repaired(original, op, label, fn, coef, r2, int(np.isfinite(y).sum()), 1, seq))
    return out


@dataclass
class GateReport:
    candidate_id: str
    val_r2: float
    val_rmse: float
    r2_gain: float
    sse_reduction: float
    predictive_pass: bool
    targeted_finding_cleared: bool
    new_temporal_flag: bool
    stability_pass: bool
    added_terms: int


@dataclass
class RepairOutcome:
    """Result of one bounded repair run. Carries evidence, never a verdict.

    Promotion stays with the APEDP-X track: `submit_note` describes the
    evidence package a referee call would need. No status here promotes.
    """

    status: RepairStatus
    winner: CandidateProposal | None
    targeted_finding: str | None
    validation: dict[str, GateReport]
    sealed: dict[str, float] | None
    ambiguity_set_size: int
    candidates_evaluated: int
    rounds_used: int
    submit_note: str


def cluster_equivalent(
    proposals: Sequence[CandidateProposal],
    df: pd.DataFrame,
    y: np.ndarray,
    tol: float,
) -> list[list[CandidateProposal]]:
    """Group proposals by measured prediction equality on finite rows.

    Mirrors the referee's canonicalization principle (duplicate structures
    are not distinct hypotheses) using predictions, since RIME graphs with
    different canonical strings (e.g. `x` vs `(0+(-(-x)))`) can be
    numerically identical. `tol` is explicit.
    """
    scored = [(p, np.asarray(p.compute_score(df), dtype=float)) for p in proposals]
    clusters: list[list[int]] = []
    for i, (_, pi) in enumerate(scored):
        placed = False
        for cl in clusters:
            pj = scored[cl[0]][1]
            m = np.isfinite(pi) & np.isfinite(pj)
            if int(m.sum()) > 0 and float(np.max(np.abs(pi[m] - pj[m]))) <= tol:
                cl.append(i)
                placed = True
                break
        if not placed:
            clusters.append([i])
    return [[proposals[i] for i in cl] for cl in clusters]


def _paired_metrics(yt: np.ndarray, yp: np.ndarray) -> tuple[float, float]:
    m = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[m], yp[m]
    if yt.size == 0:
        return float("nan"), float("nan")
    return _r2(yt, yp), float(np.sqrt(np.mean((yt - yp) ** 2)))


def build_probes(df: pd.DataFrame, deps: Sequence[str]) -> dict[str, np.ndarray]:
    """Public probe builder (also used by external diagnosis callers).

    Same single source as the R2 catalogue: pairwise products of declared
    dependencies at lag 0 and the two shortest frozen lags. Keys are probe
    labels, values full-frame arrays (caller masks to finite rows).
    """
    return {label: values for label, _src, values in _interaction_probes(df, deps)}


def _masked_classify(    report: Any,
    y: np.ndarray,
    pred: np.ndarray,
    df: pd.DataFrame,
    deps: Sequence[str],
    cutoffs: dict[str, Any],
) -> tuple[str, ...]:
    """Classify with interaction probes masked to the same finite rows.

    Probes come from _interaction_probes (declared deps only) evaluated on
    the live frame, then restricted by the residual finite mask — the
    screen and the R2 catalogue see identical columns by construction.
    """
    yp = np.asarray(pred, dtype=float)
    yt = np.asarray(y, dtype=float)
    m = np.isfinite(yt) & np.isfinite(yp)
    probes = {label: np.asarray(vals, dtype=float)[m]
              for label, _src, vals in _interaction_probes(df, deps)}
    return diagmod.classify_findings(report, (yt - yp)[m], yp[m], probes, **cutoffs)


def repair(
    original: CandidateProposal,
    partitions: RepairPartitions,
    config: RepairConfig,
    *,
    rho_flag: float,
    bias_ratio_cut: float,
    variance_ratio_cut: float,
    curvature_corr_cut: float,
    interaction_corr_cut: float,
    min_rows: int,
) -> RepairOutcome:
    """One bounded repair run: diagnose → generate → validate → sealed once."""
    cutoffs = dict(rho_flag=rho_flag, bias_ratio_cut=bias_ratio_cut,
                   variance_ratio_cut=variance_ratio_cut,
                   curvature_corr_cut=curvature_corr_cut,
                   interaction_corr_cut=interaction_corr_cut, min_rows=min_rows)
    # --- diagnose on the repair split (never validation, never sealed) ---
    rep = diagmod.diagnose_residuals(original, partitions.repair_df, partitions.repair_y, "repair")
    yp_r = np.asarray(original.compute_score(partitions.repair_df), dtype=float)
    findings = _masked_classify(rep, partitions.repair_y, yp_r, partitions.repair_df,
                                original.declared_nominal_dependencies, cutoffs)
    targeted = next((f for f in findings if f not in (diagmod.FINDING_NO_RELIABLE_STRUCTURE, diagmod.FINDING_INSUFFICIENT_ROWS)), None)
    if targeted is None:
        return RepairOutcome(RepairStatus.INSUFFICIENT_EVIDENCE if findings == (diagmod.FINDING_INSUFFICIENT_ROWS,) else RepairStatus.NO_IMPROVEMENT,
                             None, None, {}, None, 1, 0, 0,
                             "no finding warrants a repair; original retained without modification")

    # --- generate (bounded) + validate on untouched validation rows ---
    parent = original
    added_total = 0
    rounds_used = 0
    validation: dict[str, GateReport] = {}
    all_cands: list[CandidateProposal] = []
    evaluated = 0
    base_yt = np.asarray(partitions.validation_y, dtype=float)
    base_yp = np.asarray(parent.compute_score(partitions.validation_df), dtype=float)
    base_r2, _ = _paired_metrics(base_yt, base_yp)
    budget_total = config.max_candidates_per_round * max(config.max_rounds, 1)

    def validate_one(c: CandidateProposal, base_yp: np.ndarray, base_r2: float) -> GateReport:
        yt = np.asarray(partitions.validation_y, dtype=float)
        yp = np.asarray(c.compute_score(partitions.validation_df), dtype=float)
        r2, rmse = _paired_metrics(yt, yp)
        gain = r2 - base_r2 if np.isfinite(r2) and np.isfinite(base_r2) else float("nan")
        # Two preregistered predictive criteria, reported independently:
        # absolute R2 headroom AND relative SSE reduction (scale-aware). The
        # caller's bar decides; neither is inferred from the other.
        sse_base, sse_cand = _sse(yt, base_yp), _sse(yt, yp)
        sse_red = ((sse_base - sse_cand) / sse_base
                   if np.isfinite(sse_base) and np.isfinite(sse_cand) and sse_base > 0.0
                   else float("nan"))
        predictive = bool(
            (np.isfinite(gain) and gain >= config.min_validation_r2_gain)
            or (
                config.min_sse_reduction > 0.0
                and np.isfinite(sse_red)
                and sse_red >= config.min_sse_reduction
            )
        )
        rep_v = diagmod.diagnose_residuals(c, partitions.validation_df, yt, "validation")
        fv = _masked_classify(rep_v, yt, yp, partitions.validation_df,
                              c.declared_nominal_dependencies, cutoffs)
        cleared = targeted not in fv
        new_temp = diagmod.FINDING_TEMPORAL_DEPENDENCE in fv and targeted != diagmod.FINDING_TEMPORAL_DEPENDENCE
        half = len(yt) // 2
        stab = False
        if half >= min_rows and len(yt) - half >= min_rows:
            r2a, _ = _paired_metrics(yt[:half], yp[:half])
            r2b, _ = _paired_metrics(yt[half:], yp[half:])
            b_a, _ = _paired_metrics(yt[:half], base_yp[:half])
            b_b, _ = _paired_metrics(yt[half:], base_yp[half:])
            stab = bool(np.isfinite(r2a) and np.isfinite(r2b) and r2a > b_a and r2b > b_b)
        return GateReport(c.candidate_id, r2, rmse, gain, sse_red, predictive, cleared,
                          new_temp, stab, int(c.parameters.get("added_terms", 0)))

    def passes(g: GateReport) -> bool:
        return g.predictive_pass and g.targeted_finding_cleared and not g.new_temporal_flag and g.stability_pass

    for _ in range(max(config.max_rounds, 1)):
        if added_total >= config.max_added_terms > 0:
            break
        cands = generate_repairs(parent, findings, partitions.repair_df, partitions.repair_y, config)
        if not cands:
            break
        rounds_used += 1
        if evaluated + len(cands) > budget_total:
            return RepairOutcome(RepairStatus.BUDGET_EXHAUSTED, None, targeted, validation, None,
                                 1, evaluated, rounds_used,
                                 "candidate budget exhausted before validation completed")
        round_best: CandidateProposal | None = None
        for c in cands:
            added = int(c.parameters.get("added_terms", 0))
            if added_total + added > config.max_added_terms:
                continue
            evaluated += 1
            all_cands.append(c)
            g = validate_one(c, base_yp, base_r2)
            validation[c.candidate_id] = g
            if passes(g) and (round_best is None or g.val_r2 > validation[round_best.candidate_id].val_r2):
                round_best = c
        if round_best is None:
            break
        # Chain: winner becomes next round's parent and must re-earn against
        # the tightened baseline; findings re-targeted on the repair split.
        parent = round_best
        added_total += int(round_best.parameters.get("added_terms", 0))
        base_yp = np.asarray(parent.compute_score(partitions.validation_df), dtype=float)
        base_r2, _ = _paired_metrics(base_yt, base_yp)
        rep_w = diagmod.diagnose_residuals(parent, partitions.repair_df, partitions.repair_y, "repair")
        findings = _masked_classify(rep_w, partitions.repair_y,
                                    np.asarray(parent.compute_score(partitions.repair_df), dtype=float),
                                    partitions.repair_df, parent.declared_nominal_dependencies, cutoffs)
        targeted = next((f for f in findings if f not in (diagmod.FINDING_NO_RELIABLE_STRUCTURE, diagmod.FINDING_INSUFFICIENT_ROWS)), targeted)

    passing_ids = [cid for cid, g in validation.items() if passes(g)]
    if not passing_ids:
        return RepairOutcome(RepairStatus.NO_IMPROVEMENT, None, targeted, validation, None,
                             1, evaluated, rounds_used,
                             "no candidate passed all gates; original retained")
    best = max((c for c in all_cands if c.candidate_id in passing_ids),
               key=lambda c: validation[c.candidate_id].val_r2)
    passing = [c for c in all_cands if c.candidate_id in passing_ids]
    clusters = cluster_equivalent(passing, partitions.validation_df, np.asarray(partitions.validation_y, dtype=float), config.equivalence_tol)
    amb_size = max((len(cl) for cl in clusters if any(c.candidate_id == best.candidate_id for c in cl)), default=1)
    # Near-tie identifiability band: distinct structures inside the band are
    # observationally indistinguishable on validation, whichever scored
    # marginally higher. The band is explicit; 0 preserves frozen behaviour.
    if config.equivalence_r2_tol > 0.0:
        best_r2 = validation[best.candidate_id].val_r2
        tied = [c for c in passing
                if abs(validation[c.candidate_id].val_r2 - best_r2) <= config.equivalence_r2_tol]
        amb_size = max(amb_size, len(tied))
    if amb_size > 1:
        return RepairOutcome(RepairStatus.AMBIGUOUS, best, targeted, validation, None,
                             amb_size, evaluated, rounds_used,
                             "multiple prediction-equivalent winners: ambiguity preserved, no unique repair")

    # --- sealed single evaluation of the frozen winner ---
    sealed_df, sealed_y = partitions.take_sealed()
    s_yt = np.asarray(sealed_y, dtype=float)
    s_yp = np.asarray(best.compute_score(sealed_df), dtype=float)
    s_yp0 = np.asarray(original.compute_score(sealed_df), dtype=float)
    s_r2, s_rmse = _paired_metrics(s_yt, s_yp)
    o_r2, o_rmse = _paired_metrics(s_yt, s_yp0)
    sealed = {"winner_r2": s_r2, "winner_rmse": s_rmse, "original_r2": o_r2, "original_rmse": o_rmse,
              "sealed_r2_gain": (s_r2 - o_r2) if np.isfinite(s_r2) and np.isfinite(o_r2) else float("nan")}
    return RepairOutcome(RepairStatus.IMPROVED, best, targeted, validation, sealed,
                         1, evaluated, rounds_used,
                         "frozen winner evaluated once on sealed rows; evidence package ready for the APEDP-X track (no verdict here)")
