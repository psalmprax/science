"""RIME Repair v0.1.0 — residual DIAGNOSIS only (no equation modification).

Boundary discipline (AGENTS.md §1.1):

  * This module never modifies an equation, never fits parameters, never
    promotes a candidate, and never touches a holdout slice. It consumes a
    frozen ``CandidateProposal`` (real kernel scorer, frozen readout) plus an
    explicitly supplied repair-split frame, and emits a frozen
    ``ResidualReport`` of measured statistics.
  * Diagnosis is a hypothesis label, not a conclusion: a finding names the
    repair family worth investigating, it does not assert the repair is true.
  * All decision cutoffs are explicit caller-supplied arguments (no default
    threshold literals in this module, per AGENTS.md §1.3). The residual
    whiteness flag is imported from its single existing definition rather
    than re-declared.

Pipeline position: RIME proposes → this module diagnoses → the repair
engine modifies → independent validation compares → APEDP-X adjudicates.
This module covers diagnosis; modification lives in rime_repair_engine.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

# Finding vocabulary (frozen). Each label names a repair family to
# investigate; none asserts a repair is correct.
FINDING_INTERCEPT_BIAS = "intercept_bias"
FINDING_TEMPORAL_DEPENDENCE = "temporal_dependence"
FINDING_HETEROSCEDASTICITY_SUSPECT = "heteroscedasticity_suspect"
FINDING_CURVATURE_SUSPECT = "curvature_suspect"
FINDING_INTERACTION_SUSPECT = "interaction_suspect"
FINDING_NO_RELIABLE_STRUCTURE = "no_reliable_structure"
FINDING_INSUFFICIENT_ROWS = "insufficient_rows"

_FROZEN_FINDINGS = frozenset(
    {
        FINDING_INTERCEPT_BIAS,
        FINDING_TEMPORAL_DEPENDENCE,
        FINDING_HETEROSCEDASTICITY_SUSPECT,
        FINDING_CURVATURE_SUSPECT,
        FINDING_INTERACTION_SUSPECT,
        FINDING_NO_RELIABLE_STRUCTURE,
        FINDING_INSUFFICIENT_ROWS,
    }
)


def _finite_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask]


def _lag1_corr(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3 or float(np.std(x)) == 0.0:
        return float("nan")
    c = np.corrcoef(x[:-1], x[1:])[0, 1]
    return float(c)


@dataclass(frozen=True)
class ResidualReport:
    """Measured residual statistics for one frozen proposal on one data split.

    All values are measurements, never judgments. ``partition_label`` records
    which split was used (e.g. ``"repair"``); sealed-test splits must never be
    passed here — the module cannot enforce that, the caller protocol must.
    """

    candidate_id: str
    formula: str
    graph_id: str | None
    partition_label: str
    n_rows: int
    n_finite: int
    mean_resid: float
    std_resid: float
    std_target: float
    rmse: float
    mae: float
    r2: float
    corr: float
    rho1: float
    max_abs_resid: float
    frac_finite: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "formula": self.formula,
            "graph_id": self.graph_id,
            "partition_label": self.partition_label,
            "n_rows": self.n_rows,
            "n_finite": self.n_finite,
            "mean_resid": self.mean_resid,
            "std_resid": self.std_resid,
            "std_target": self.std_target,
            "rmse": self.rmse,
            "mae": self.mae,
            "r2": self.r2,
            "corr": self.corr,
            "rho1": self.rho1,
            "max_abs_resid": self.max_abs_resid,
            "frac_finite": self.frac_finite,
        }

    def sha256(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, allow_nan=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def diagnose_residuals(
    proposal: Any,
    df: pd.DataFrame,
    y: np.ndarray | pd.Series,
    partition_label: str,
) -> ResidualReport:
    """Score a frozen proposal on an explicitly supplied frame; report stats.

    ``proposal.compute_score`` is the real kernel scorer with the readout
    frozen on development data (past-information-only by construction).
    ``y`` is passed explicitly because a proposal does not carry its target.
    """
    y_pred = np.asarray(proposal.compute_score(df), dtype=float)
    y_true = np.asarray(y, dtype=float)
    n_rows = len(y_true)
    yt, yp = _finite_pair(y_true, y_pred)
    n_finite = int(yt.size)
    if n_finite == 0:
        return ResidualReport(
            candidate_id=proposal.candidate_id,
            formula=proposal.formula,
            graph_id=(proposal.metadata or {}).get("graph_id"),
            partition_label=partition_label,
            n_rows=n_rows,
            n_finite=0,
            mean_resid=float("nan"),
            std_resid=float("nan"),
            std_target=float("nan"),
            rmse=float("nan"),
            mae=float("nan"),
            r2=float("nan"),
            corr=float("nan"),
            rho1=float("nan"),
            max_abs_resid=float("nan"),
            frac_finite=0.0,
        )
    resid = yt - yp
    ss = float(np.sum((yt - yt.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid**2)) / ss if ss > 0.0 else float("nan")
    corr = float(np.corrcoef(yt, yp)[0, 1]) if n_finite > 2 else float("nan")
    return ResidualReport(
        candidate_id=proposal.candidate_id,
        formula=proposal.formula,
        graph_id=(proposal.metadata or {}).get("graph_id"),
        partition_label=partition_label,
        n_rows=n_rows,
        n_finite=n_finite,
        mean_resid=float(np.mean(resid)),
        std_resid=float(np.std(resid)),
        std_target=float(np.std(yt)),
        rmse=float(np.sqrt(np.mean(resid**2))),
        mae=float(np.mean(np.abs(resid))),
        r2=r2,
        corr=corr,
        rho1=_lag1_corr(resid),
        max_abs_resid=float(np.max(np.abs(resid))),
        frac_finite=n_finite / n_rows if n_rows else 0.0,
    )


def classify_findings(
    report: ResidualReport,
    resid: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    probes: dict[str, np.ndarray],
    *,
    rho_flag: float,
    bias_ratio_cut: float,
    variance_ratio_cut: float,
    curvature_corr_cut: float,
    interaction_corr_cut: float,
    min_rows: int,
) -> tuple[str, ...]:
    """Map measured statistics + residuals to finding labels.

    Every cutoff is an explicit argument (no threshold literals in this
    module, per AGENTS.md §1.3). `probes` maps probe labels to value arrays
    aligned with the residual rows (built by the caller, e.g. the repair
    engine, from declared dependencies only); an empty dict disables the
    interaction screen. Screens used:

      * intercept bias: |mean(resid)| > bias_ratio_cut * std(target)
      * temporal dependence: |rho1| > rho_flag
      * heteroscedasticity: var(second-half resid) / var(first-half resid)
        outside [1/cut, cut]
      * curvature: corr(resid, (y_pred - mean)^2), the missing-square screen
      * interaction: max over probes of |corr(resid, probe)| >
        interaction_corr_cut — the missing-interaction screen. Omitted
        interactions of independent drivers leave white residuals, so no
        other screen here can see them; the probe family must contain the
        missing structure for this to fire (detection coverage limit).

    Labels are repair hypotheses to investigate, never conclusions.
    """
    if report.n_finite < min_rows:
        return (FINDING_INSUFFICIENT_ROWS,)
    findings: list[str] = []
    std_t = report.std_target
    if std_t > 0.0 and abs(report.mean_resid) > bias_ratio_cut * std_t:
        findings.append(FINDING_INTERCEPT_BIAS)
    if np.isfinite(report.rho1) and abs(report.rho1) > rho_flag:
        findings.append(FINDING_TEMPORAL_DEPENDENCE)
    yp = np.asarray(y_pred, dtype=float)
    r = np.asarray(resid, dtype=float)
    m = np.isfinite(r) & np.isfinite(yp)
    r, yp = r[m], yp[m]
    if r.size >= min_rows:
        half = r.size // 2
        v0, v1 = float(np.var(r[:half])), float(np.var(r[half:]))
        if v0 > 0.0 and v1 > 0.0 and (v1 / v0 > variance_ratio_cut or v0 / v1 > variance_ratio_cut):
            findings.append(FINDING_HETEROSCEDASTICITY_SUSPECT)
        centered = yp - yp.mean()
        quad = centered**2
        if float(np.std(quad)) > 0.0 and r.size > 2:
            cc = float(np.corrcoef(r, quad)[0, 1])
            if np.isfinite(cc) and abs(cc) > curvature_corr_cut:
                findings.append(FINDING_CURVATURE_SUSPECT)
        for _label, pv in probes.items():
            pv = np.asarray(pv, dtype=float)
            if pv.shape != r.shape:
                continue
            mm = np.isfinite(pv)
            if int(mm.sum()) >= min_rows and float(np.std(pv[mm])) > 0.0 and r[mm].size > 2:
                pc = float(np.corrcoef(r[mm], pv[mm])[0, 1])
                if np.isfinite(pc) and abs(pc) > interaction_corr_cut:
                    findings.append(FINDING_INTERACTION_SUSPECT)
                    break
    if not findings:
        return (FINDING_NO_RELIABLE_STRUCTURE,)
    for f in findings:
        assert f in _FROZEN_FINDINGS
    return tuple(findings)


def finding_vocabulary() -> Sequence[str]:
    """Frozen finding vocabulary (for audits and schema checks)."""
    return tuple(sorted(_FROZEN_FINDINGS))
