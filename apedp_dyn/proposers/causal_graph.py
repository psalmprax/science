"""The whole lagged causal graph of a multivariate time series, with two plain baselines to measure it against.

Status: experimental

:class:`apedp_dyn.proposers.pcmci_proposer.PcmciProposer` asks PCMCI+ for the
parents of one target. ``discover_graph`` runs PCMCI+ once and reads every
link ``source[t - lag] -> target[t]`` (autoregressive links included), which
is what a graph engine is for and costs no more than one target does.

Two baselines use no conditional-independence search, so the value of that
search can be measured: ``var_links`` (a full vector autoregression, each
coefficient tested by its t statistic) and ``correlation_links`` (each lagged
pair tested by its marginal correlation). On a linear system with few
variables the vector autoregression is a strong method; marginal correlation
is the naive one that reports indirect and confounded links.

Only lagged links are returned. PCMCI+ with partial correlation assumes
linear dependence, no hidden common cause and stationarity; a link that acts
through an even function (the square of the source) is invisible to it.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from apedp_dyn.proposers.pcmci_proposer import PcmciUnavailableError, is_available
from apedp_referee.canonical_spec import DIEBOLD_MARIANO_P_VALUE_THRESHOLD

Link = tuple[str, int, str]  # source, lag, target


def discover_graph(frame: pd.DataFrame, tau_max: int = 3, pc_alpha: float = DIEBOLD_MARIANO_P_VALUE_THRESHOLD) -> list[dict[str, Any]]:
    """Every lagged link PCMCI+ keeps, for all targets, from one run."""
    if not is_available():
        raise PcmciUnavailableError("tigramite is not installed (pip install tigramite)")
    from tigramite import data_processing as pp
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite.pcmci import PCMCI

    names = list(frame.columns)
    data = pp.DataFrame(frame.to_numpy(dtype=float), var_names=names)
    result = PCMCI(dataframe=data, cond_ind_test=ParCorr(), verbosity=0).run_pcmciplus(tau_min=1, tau_max=tau_max, pc_alpha=pc_alpha)
    links = []
    for source in range(len(names)):
        for target in range(len(names)):
            for lag in range(1, tau_max + 1):
                if result["graph"][source, target, lag] == "-->":
                    links.append({"source": names[source], "lag": lag, "target": names[target],
                                  "p_value": float(result["p_matrix"][source, target, lag]),
                                  "strength": float(result["val_matrix"][source, target, lag])})
    return sorted(links, key=lambda link: (link["target"], link["source"], link["lag"]))


def even_links(frame: pd.DataFrame, links: list[dict[str, Any]], tau_max: int = 3,
               alpha: float = DIEBOLD_MARIANO_P_VALUE_THRESHOLD) -> list[dict[str, Any]]:
    """Links that act through an even function of the source, which partial correlation cannot see.

    For every target, the linear effect of the parents already found is removed,
    and what is left is tested against the centred square of each lagged
    variable that is not yet a parent. The level is ``alpha`` divided by the
    number of tests for that target, so the extra pass cannot add more than
    about ``alpha`` false links per target. A link found here says the target's
    mean depends on the size of the source, whatever its sign.
    """
    names = list(frame.columns)
    x = frame.to_numpy(dtype=float)
    rows = np.arange(tau_max, len(x))
    found = []
    for target in names:
        parents = [(link["source"], link["lag"]) for link in links if link["target"] == target]
        design = np.column_stack([x[rows - lag, names.index(source)] for source, lag in parents] + [np.ones(len(rows))])
        y = x[rows, names.index(target)]
        resid = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
        candidates = [(source, lag) for source in names for lag in range(1, tau_max + 1) if (source, lag) not in parents]
        for source, lag in candidates:
            square = x[rows - lag, names.index(source)] ** 2
            square = square - design @ np.linalg.lstsq(design, square, rcond=None)[0]
            r, p = stats.pearsonr(square, resid)
            if p < alpha / max(1, len(candidates)):
                found.append({"source": source, "lag": lag, "target": target, "p_value": float(p), "strength": float(r), "kind": "even"})
    return found


def discover_graph_with_even_links(frame: pd.DataFrame, tau_max: int = 3,
                                   pc_alpha: float = DIEBOLD_MARIANO_P_VALUE_THRESHOLD) -> list[dict[str, Any]]:
    """PCMCI+ links, plus the links through an even function that it cannot see (``even_links``)."""
    links = discover_graph(frame, tau_max, pc_alpha)
    return [*links, *even_links(frame, links, tau_max, pc_alpha)]


def var_links(frame: pd.DataFrame, tau_max: int = 3, alpha: float = DIEBOLD_MARIANO_P_VALUE_THRESHOLD) -> list[dict[str, Any]]:
    """Links whose coefficient in a full vector autoregression of order ``tau_max`` is significant at ``alpha``."""
    names = list(frame.columns)
    x = frame.to_numpy(dtype=float)
    n = len(x) - tau_max
    design = np.column_stack([x[tau_max - lag : len(x) - lag, j] for j in range(len(names)) for lag in range(1, tau_max + 1)]
                             + [np.ones(n)])
    gram_inverse = np.linalg.pinv(design.T @ design)
    dof = n - design.shape[1]
    links = []
    for t, target in enumerate(names):
        y = x[tau_max:, t]
        coef = gram_inverse @ design.T @ y
        resid = y - design @ coef
        se = np.sqrt(np.maximum(np.diag(gram_inverse) * float(resid @ resid) / dof, 1e-300))
        p = 2.0 * stats.t.sf(np.abs(coef / se), dof)
        for j, source in enumerate(names):
            for lag in range(1, tau_max + 1):
                k = j * tau_max + lag - 1
                if p[k] < alpha:
                    links.append({"source": source, "lag": lag, "target": target, "p_value": float(p[k]), "strength": float(coef[k])})
    return links


def correlation_links(frame: pd.DataFrame, tau_max: int = 3, alpha: float = DIEBOLD_MARIANO_P_VALUE_THRESHOLD) -> list[dict[str, Any]]:
    """Links whose marginal lagged correlation is significant at ``alpha``."""
    names = list(frame.columns)
    x = frame.to_numpy(dtype=float)
    links = []
    for j, source in enumerate(names):
        for t, target in enumerate(names):
            for lag in range(1, tau_max + 1):
                r, p = stats.pearsonr(x[:-lag, j], x[lag:, t])
                if p < alpha:
                    links.append({"source": source, "lag": lag, "target": target, "p_value": float(p), "strength": float(r)})
    return links


def as_set(links: list[dict[str, Any]]) -> set[Link]:
    return {(link["source"], int(link["lag"]), link["target"]) for link in links}


def score(found: set[Link], truth: set[Link]) -> dict[str, float]:
    """Precision, recall and F1 of a set of links against the true set."""
    hit = len(found & truth)
    precision = hit / len(found) if found else 1.0
    recall = hit / len(truth) if truth else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"found": float(len(found)), "true": float(len(truth)), "hit": float(hit), "precision": precision, "recall": recall, "f1": f1}
