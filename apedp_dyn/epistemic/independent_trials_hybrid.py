"""Hybrids of the ways of counting independent trials for the deflated Sharpe ratio (development).

Status: deprecated (retired 27 Nov 2026 on the owner's word: no hybrid beat the cluster count, one was unsafe; kept to replicate its record)

Not wired into the referee. ``apedp_dyn/epistemic/independent_trials.py`` deflates the best of many
strategies three ways: against every trial (the referee's count today), against clusters of correlated
trials (López de Prado and Lewis) and with an eigenvalue count of effective trials (Li and Ji). On the
owner's word of 26 November ("test hybrid approach in all the suggested solutions") this adds four
combinations, every one read against the same canonical DSR threshold:

* ``cluster_then_eigen``: the selected trial's cluster series, deflated with the eigenvalue count of the
  cluster series (clusters for families, eigenvalues for what correlation is left between them);
* ``own_with_cluster_count``: the selected trial's own Sharpe ratio and the referee's variance across
  trials, with the number of clusters as the count (the referee's statistic, the clusters' count);
* ``cluster_and_eigen``: significant only when both the cluster and the eigenvalue counts say so;
* ``cluster_or_eigen``: significant when either does (more power, at a cost in false alarms to measure).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from apedp_dyn.conf.canonical_spec import DSR_THRESHOLD
from apedp_dyn.epistemic import independent_trials as it
from apedp_referee.trial_ledger import compute_deflated_sharpe_ratio

METHODS = ("as_referee", "by_cluster", "by_eigenvalues", "cluster_then_eigen", "own_with_cluster_count",
           "cluster_and_eigen", "cluster_or_eigen")


def deflated(returns: np.ndarray, selected: int | None = None, *, method: str = "average", seed: int = 0) -> dict[str, Any]:
    """``independent_trials.deflated`` with the four hybrids added, on the same clusters."""
    r = np.asarray(returns, dtype=float)
    out: dict[str, Any] = it.deflated(r, selected, method=method, seed=seed)
    t, n = r.shape
    labels = np.asarray(out["labels"])
    k = int(labels.max()) + 1
    srs = it.sharpe(r)
    pick = int(out["selected"])
    cr = it.cluster_returns(r, labels)
    csr = it.sharpe(cr)
    mine = int(labels[pick])
    cskew, ckurt = it._moments(cr[:, mine])
    m_clusters = max(1, math.ceil(it.eigen_effective(cr))) if k > 1 else 1
    a = compute_deflated_sharpe_ratio(float(csr[mine]), m_clusters, float(np.var(csr, ddof=1)) if k > 1 else 0.0, t,
                                      cskew, ckurt)
    out["cluster_then_eigen"] = {"dsr": a["dsr"], "n_trials": m_clusters, "significant": a["dsr"] >= DSR_THRESHOLD}
    skew, kurt = it._moments(r[:, pick])
    b = compute_deflated_sharpe_ratio(float(srs[pick]), k, float(np.var(srs, ddof=1)) if n > 1 else 0.0, t, skew, kurt)
    out["own_with_cluster_count"] = {"dsr": b["dsr"], "n_trials": k, "significant": b["dsr"] >= DSR_THRESHOLD}
    pair = (out["by_cluster"]["dsr"], out["by_eigenvalues"]["dsr"])
    counts = f'{out["by_cluster"]["n_trials"]} and {out["by_eigenvalues"]["n_trials"]}'
    out["cluster_and_eigen"] = {"dsr": min(pair), "n_trials": counts, "significant": bool(min(pair) >= DSR_THRESHOLD)}
    out["cluster_or_eigen"] = {"dsr": max(pair), "n_trials": counts, "significant": bool(max(pair) >= DSR_THRESHOLD)}
    return out
