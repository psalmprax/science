"""How many independent trials a search made, from its trials' returns (development).

Status: experimental

The referee's deflated Sharpe ratio (``apedp_referee.trial_ledger``) charges a search one trial for every
candidate it recorded. Bailey and López de Prado meant the number of *independent* trials: when the
candidates are near copies of one another (one signal with ten look-backs), counting each overstates
the multiplicity, the hurdle comes out too high, and a real effect is refused. López de Prado and Lewis
("Detection of false investment strategies using unsupervised learning methods", Quantitative Finance,
2019) estimate the number by clustering the trials' returns on the distance sqrt((1 - rho) / 2), choosing
the number of clusters by the silhouette ("ONC"), and deflating the selected strategy's cluster against
the clusters. The same procedure is claimed in US patent application US20190294990A1, "Detecting false
positives in statistical models"; its legal status has not been checked, and nothing here is used in a
product.

What is here: ``cluster_labels`` (k-means on the distance matrix as in ONC's base step, or average-linkage
clustering for speed, either way the number of clusters chosen by the mean over the spread of the
silhouette), ``eigen_effective`` (Li and Ji's effective number of tests from the eigenvalues of the
correlation matrix, as a second estimate), ``cluster_returns`` (inverse-variance weights within a
cluster), and ``deflated`` (the referee's own ``compute_deflated_sharpe_ratio`` called three ways: as the
referee does now, with the selected strategy's cluster against the clusters, and with the eigenvalue
count). **Nothing here is wired into the referee.** Changing how the referee counts trials changes what
is in force and waits for the owner's word, a preregistered test and an independent review
(AGENTS.md 1.5 and 1.6).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from apedp_dyn.conf.canonical_spec import DSR_THRESHOLD
from apedp_referee.trial_ledger import compute_deflated_sharpe_ratio


def _corr(returns: np.ndarray) -> np.ndarray:
    """Correlation of the columns; a constant column is uncorrelated with everything else."""
    r = np.asarray(returns, dtype=float)
    sd = r.std(axis=0)
    z = np.where(sd > 0, (r - r.mean(axis=0)) / np.where(sd > 0, sd, 1.0), 0.0)
    c = (z.T @ z) / max(r.shape[0], 1)
    np.fill_diagonal(c, 1.0)
    clipped: np.ndarray = np.clip(c, -1.0, 1.0)
    return clipped


def distance(corr: np.ndarray) -> np.ndarray:
    d: np.ndarray = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))
    return d


def _silhouette(dist: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Silhouette of every trial on a precomputed distance matrix (0 for a cluster of one)."""
    n = len(labels)
    groups, inverse = np.unique(labels, return_inverse=True)
    k = len(groups)
    if k < 2:
        return np.zeros(n)
    onehot = np.zeros((n, k))
    onehot[np.arange(n), inverse] = 1.0
    counts = onehot.sum(axis=0)
    sums = dist @ onehot                                   # sum of distances from each trial to each cluster
    own_count = counts[inverse]
    a = np.where(own_count > 1, sums[np.arange(n), inverse] / np.maximum(own_count - 1, 1), 0.0)
    means = sums / counts
    means[np.arange(n), inverse] = np.inf
    b = means.min(axis=1)
    top = np.maximum(a, b)
    out: np.ndarray = np.where((own_count > 1) & (top > 0), (b - a) / np.where(top > 0, top, 1.0), 0.0)
    return out


def _quality(dist: np.ndarray, labels: np.ndarray) -> float:
    s = _silhouette(dist, labels)
    sd = s.std()
    return float(s.mean() / sd) if sd > 0 else (float("inf") if s.mean() > 0 else -float("inf"))


def cluster_labels(returns: np.ndarray, *, method: str = "average", max_k: int | None = None, n_init: int = 3,
                   seed: int = 0, min_corr: float | None = None) -> np.ndarray:
    """Clusters of trials whose returns move together; the number chosen by the silhouette's mean over its spread.

    ``method="kmeans"`` is ONC's base step (k-means on the rows of the distance matrix, ``n_init`` starts for each
    k); ``"average"`` cuts one average-linkage tree at every k, much faster. A family whose best split is no better
    than none (every silhouette at or below zero) is returned as independent trials, one cluster each, and so is
    every cluster whose mean correlation between members is below ``min_corr`` (by default 4 / sqrt(T), a few times
    what chance gives between independent trials).
    """
    r = np.asarray(returns, dtype=float)
    n = r.shape[1]
    if n < 3:
        return np.arange(n)
    dist = distance(_corr(r))
    top = min(max_k or n - 1, n - 1)
    best_q, best = -float("inf"), np.arange(n)
    if method == "average":
        tree = linkage(squareform(dist, checks=False), method="average")
        for k in range(2, top + 1):
            labels = fcluster(tree, k, criterion="maxclust") - 1
            if len(np.unique(labels)) < 2:
                continue
            q = _quality(dist, labels)
            if q > best_q:
                best_q, best = q, labels
    elif method == "kmeans":
        from sklearn.cluster import KMeans
        rng = np.random.default_rng(seed)
        for _ in range(n_init):
            for k in range(2, top + 1):
                labels = KMeans(n_clusters=k, n_init=1, random_state=int(rng.integers(2**31))).fit(dist).labels_
                q = _quality(dist, labels)
                if q > best_q:
                    best_q, best = q, labels
    else:
        raise ValueError(f"unknown method {method!r}")
    if _silhouette(dist, best).mean() <= 0:
        return np.arange(n)
    # A cluster whose members barely move together is chance: among many independent trials some pairs
    # correlate at a few times 1/sqrt(T). Such a cluster is split back into single trials.
    floor = (4.0 / math.sqrt(r.shape[0])) if min_corr is None else min_corr
    corr = _corr(r)
    out = best.copy()
    nxt = int(best.max()) + 1
    for g in np.unique(best):
        idx = np.flatnonzero(best == g)
        if len(idx) < 2:
            continue
        block = corr[np.ix_(idx, idx)]
        mean = (block.sum() - len(idx)) / (len(idx) * (len(idx) - 1))
        if mean < floor:
            for i in idx[1:]:
                out[i] = nxt
                nxt += 1
    _, relabelled = np.unique(out, return_inverse=True)
    labels_out: np.ndarray = np.asarray(relabelled, dtype=int)
    return labels_out


def eigen_effective(returns: np.ndarray) -> float:
    """Li and Ji's (2005) effective number of tests: sum over the correlation's eigenvalues of
    [lambda >= 1] + (lambda - floor(lambda))."""
    lam = np.clip(np.linalg.eigvalsh(_corr(np.asarray(returns, dtype=float))), 0.0, None)
    return float(np.sum((lam >= 1.0).astype(float) + (lam - np.floor(lam))))


def cluster_returns(returns: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """One series per cluster: its members' returns with inverse-variance weights."""
    r = np.asarray(returns, dtype=float)
    out = []
    for g in range(int(labels.max()) + 1):
        idx = np.flatnonzero(labels == g)
        var = r[:, idx].var(axis=0)
        w = np.where(var > 0, 1.0 / np.where(var > 0, var, 1.0), 0.0)
        w = w / w.sum() if w.sum() > 0 else np.full(len(idx), 1.0 / len(idx))
        out.append(r[:, idx] @ w)
    stacked: np.ndarray = np.column_stack(out)
    return stacked


def sharpe(series: np.ndarray) -> np.ndarray:
    s = np.asarray(series, dtype=float)
    sd = s.std(axis=0, ddof=1)
    ratio: np.ndarray = np.where(sd > 0, s.mean(axis=0) / np.where(sd > 0, sd, 1.0), 0.0)
    return ratio


def _moments(x: np.ndarray) -> tuple[float, float]:
    return float(stats.skew(x)), float(stats.kurtosis(x, fisher=False))


def deflated(returns: np.ndarray, selected: int | None = None, *, method: str = "average", seed: int = 0) -> dict[str, Any]:
    """The selected trial (by default the one with the highest Sharpe ratio) deflated three ways, per bar.

    ``as_referee``: its Sharpe ratio against every trial (the count and the variance the referee uses today).
    ``by_cluster``: its cluster's inverse-variance series against the clusters (López de Prado and Lewis).
    ``by_eigenvalues``: its own Sharpe ratio, with Li and Ji's effective number of trials.
    Each with ``significant`` meaning the canonical DSR threshold is met.
    """
    r = np.asarray(returns, dtype=float)
    t, n = r.shape
    srs = sharpe(r)
    pick = int(np.argmax(srs)) if selected is None else int(selected)
    skew, kurt = _moments(r[:, pick])
    out: dict[str, Any] = {"selected": pick, "trials": n, "bars": t, "threshold": DSR_THRESHOLD}
    raw = compute_deflated_sharpe_ratio(float(srs[pick]), n, float(np.var(srs, ddof=1)) if n > 1 else 0.0, t, skew, kurt)
    out["as_referee"] = {"dsr": raw["dsr"], "n_trials": n, "significant": raw["dsr"] >= DSR_THRESHOLD}
    labels = cluster_labels(r, method=method, seed=seed)
    k = int(labels.max()) + 1
    cr = cluster_returns(r, labels)
    csr = sharpe(cr)
    mine = int(labels[pick])
    cskew, ckurt = _moments(cr[:, mine])
    clus = compute_deflated_sharpe_ratio(float(csr[mine]), k, float(np.var(csr, ddof=1)) if k > 1 else 0.0, t, cskew, ckurt)
    out["by_cluster"] = {"dsr": clus["dsr"], "n_trials": k, "significant": clus["dsr"] >= DSR_THRESHOLD,
                         "cluster_of_selected": mine, "cluster_size": int(np.sum(labels == mine))}
    m = max(1, math.ceil(eigen_effective(r)))
    eig = compute_deflated_sharpe_ratio(float(srs[pick]), m, float(np.var(srs, ddof=1)) if n > 1 else 0.0, t, skew, kurt)
    out["by_eigenvalues"] = {"dsr": eig["dsr"], "n_trials": m, "significant": eig["dsr"] >= DSR_THRESHOLD}
    out["labels"] = labels.tolist()
    return out
