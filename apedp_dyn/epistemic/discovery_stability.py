"""Discovery stability: recurrence, entropy, and variance decomposition.

Orthogonal to identification/transfer/promotion. Answers "what kind of
MULTI is this" by separating:
  - structural recurrence P(structure = H) over discovery seeds,
  - structural entropy H = -sum p log p (low = search finds the same thing),
  - parameter dispersion (CV per coefficient),
  - fit variance (Var of train RMSE across seeds),
  - transfer variance split: fit-seed vs scoring-seed components.

A structure can be recurrent yet unstable in parameters; a verdict can
be MULTI from observational equivalence or from search instability --
this module tells those apart with numbers instead of adjectives.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def structural_distribution(structures: list[str]) -> dict[str, float]:
    """Empirical P(structure) over discovery seeds."""
    from collections import Counter

    n = len(structures)
    if n == 0:
        return {}
    counts = Counter(structures)
    return {s: count / n for s, count in counts.items()}



def structural_entropy(structures: list[str]) -> float:
    """Shannon entropy of the discovery distribution (nats)."""
    dist = structural_distribution(structures)
    val = float(-sum(p * np.log(p) for p in dist.values() if p > 0))
    return 0.0 if abs(val) < 1e-12 else val


def coefficient_cv(values: list[float]) -> float | None:
    """Coefficient of variation |std/mean|; None when mean ~ 0."""
    a = np.asarray(values, dtype=float)
    m = float(np.mean(a))
    if abs(m) < 1e-12:
        return None
    return float(np.std(a) / abs(m))


def variance_split(transfer_matrix: np.ndarray) -> dict[str, float]:
    """Decompose transfer RMSE variance into fit-seed vs scoring-seed parts.

    transfer_matrix[i, j] = error of model fit on seed i, scored on seed j.
    Row means vary with fit quality; column means vary with scoring noise.
    """
    m = np.asarray(transfer_matrix, dtype=float)
    row_means = m.mean(axis=1)
    col_means = m.mean(axis=0)
    total = float(np.var(m))
    return {"total": total,
            "fit_seed": float(np.var(row_means)),
            "scoring_seed": float(np.var(col_means)),
            "fit_share": float(np.var(row_means) / total) if total > 0 else 0.0}


def stability_report(structures: list[str],
                     param_samples: dict[str, list[float]] | None = None,
                     transfer_matrix=None) -> dict:
    """One diagnostic block for a multi-seed discovery experiment."""
    rep = {"n_seeds": len(structures),
           "distribution": structural_distribution(structures),
           "entropy_nats": structural_entropy(structures)}
    if param_samples:
        rep["parameter_cv"] = {k: coefficient_cv(v) for k, v in param_samples.items()}
    if transfer_matrix is not None:
        rep["transfer_variance"] = variance_split(np.asarray(transfer_matrix))
    return rep


def compose_identifiability_stability_matrix(
    search_scope: dict[str, Any],
    stability_rep: dict[str, Any],
    adjudication: Any,
    transfer_scope: Any | None = None,
) -> dict[str, Any]:
    """Compose the Identifiability & Discovery Stability Matrix across orthogonal dimensions."""
    dist = stability_rep.get("distribution", {})
    max_rec = max(dist.values()) if dist else 1.0

    status = getattr(adjudication.status, "value", str(adjudication.status))
    promoted = getattr(adjudication, "promoted", None)
    attribution = str(getattr(adjudication, "epistemic_attribution", "unknown"))

    delta_rmse = getattr(adjudication, "delta_rmse", None)
    tau_epi = getattr(adjudication, "tau_epi", None)

    # Determine transfer status
    if transfer_scope is not None:
        if hasattr(transfer_scope, "status"):
            t_status = transfer_scope.status
        elif isinstance(transfer_scope, dict) and "status" in transfer_scope:
            t_status = transfer_scope["status"]
        else:
            t_status = "NOT_ESTABLISHED"
    else:
        t_status = "NOT_ESTABLISHED"

    # Secondary diagnoses
    secondaries: list[str] = []
    # If entropy is high (> 0.5) or recurrence < 0.7: discovery instability
    if stability_rep.get("entropy_nats", 0.0) > 0.5 or max_rec < 0.7:
        secondaries.append("DISCOVERY_INSTABILITY")

    # If parameter CV is high (> 0.3 for any param)
    param_cvs = stability_rep.get("parameter_cv", {})
    if any(cv is not None and cv > 0.3 for cv in param_cvs.values()):
        secondaries.append("PARAMETER_INSTABILITY")

    if t_status == "cross_domain_failed":
        secondaries.append("CROSS_DOMAIN_TRANSFER_FAILURE")

    return {
        "search_scope": search_scope,
        "discovery": {
            "structural_recurrence": float(max_rec),
            "structural_entropy": float(stability_rep.get("entropy_nats", 0.0)),
            "parameter_dispersion": param_cvs,
            "transfer_variance": stability_rep.get("transfer_variance", {}),
        },
        "identification": {
            "status": status,
            "rival_margin": delta_rmse,
            "tau_epi": tau_epi,
        },
        "transfer": {
            "status": t_status,
        },
        "promotion": {
            "status": "PROMOTED" if promoted is not None else "NOT_PROMOTED",
            "promoted_candidate": getattr(promoted.candidate, "candidate_id", None) if promoted else None,
        },
        "diagnosis": {
            "primary": attribution,
            "secondary": secondaries,
        },
    }

