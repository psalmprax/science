"""PCMCI+ parent-set proposer for lagged time series.

Status: experimental

Wraps Tigramite's PCMCI+ (Runge 2020) as a proposer: it finds which lagged
variables are direct parents of the target after conditioning on the others,
fits a linear model on those parents, and emits one ``CandidateProposal``. It
decides nothing: the referee adjudicates the proposal like any other.

What it is for: removing indirect and autocorrelation-induced links that a
correlation screen or a regression over all lags keeps. What it does not do:
with the partial-correlation test used here it sees linear dependence only (a
pure interaction such as x1[t-1] * x2[t-1] is invisible to it), and PCMCI+
assumes no hidden common cause.

Tigramite is an optional dependency; ``is_available()`` says whether it is
installed, and the proposer raises ``PcmciUnavailableError`` if it is not.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from apedp_dyn.grammar.proposer_interface import CandidateProposal, ProposerBase, StrategyArchetype
from apedp_referee.canonical_spec import DIEBOLD_MARIANO_P_VALUE_THRESHOLD


class PcmciUnavailableError(RuntimeError):
    """Tigramite is not installed."""


def is_available() -> bool:
    try:
        import tigramite  # noqa: F401
    except ImportError:
        return False
    return True


def _lagged(frame: pd.DataFrame, parents: Sequence[tuple[str, int]]) -> np.ndarray:
    """Columns ``name[t - lag]`` (NaN where the past is not available), plus an intercept."""
    cols = [frame[name].shift(lag).to_numpy(dtype=float) for name, lag in parents]
    out: np.ndarray = np.column_stack([*cols, np.ones(len(frame))])
    return out


class PcmciProposer(ProposerBase):
    """Direct lagged parents of the target by PCMCI+, then a linear fit on them."""

    def __init__(self, target_col: str = "target", tau_max: int = 3,
                 pc_alpha: float = DIEBOLD_MARIANO_P_VALUE_THRESHOLD,
                 strategy_archetype: StrategyArchetype = StrategyArchetype.TACTICAL) -> None:
        self._target_col = target_col
        self._tau_max = int(tau_max)
        self._pc_alpha = float(pc_alpha)
        self._strategy_archetype = strategy_archetype

    @property
    def proposer_id(self) -> str:
        return "pcmci_plus_parcorr"

    def discover_parents(self, frame: pd.DataFrame) -> list[dict[str, Any]]:
        """Every link ``source[t - lag] -> target[t]`` PCMCI+ keeps, with its p-value and strength."""
        if not is_available():
            raise PcmciUnavailableError("tigramite is not installed (pip install tigramite)")
        from tigramite import data_processing as pp
        from tigramite.independence_tests.parcorr import ParCorr
        from tigramite.pcmci import PCMCI

        names = list(frame.columns)
        target = names.index(self._target_col)
        data = pp.DataFrame(frame.to_numpy(dtype=float), var_names=names)
        result = PCMCI(dataframe=data, cond_ind_test=ParCorr(), verbosity=0).run_pcmciplus(
            tau_min=1, tau_max=self._tau_max, pc_alpha=self._pc_alpha)
        links = []
        for source in range(len(names)):
            for lag in range(1, self._tau_max + 1):
                if result["graph"][source, target, lag] == "-->":
                    links.append({"source": names[source], "lag": lag,
                                  "p_value": float(result["p_matrix"][source, target, lag]),
                                  "strength": float(result["val_matrix"][source, target, lag])})
        return sorted(links, key=lambda link: (link["source"], link["lag"]))

    def fit_and_propose(self, train_data: Any, target_col: str | None = None, **kwargs: Any) -> Sequence[CandidateProposal]:
        if target_col is not None:
            self._target_col = target_col
        frame = train_data if isinstance(train_data, pd.DataFrame) else pd.DataFrame(train_data)
        links = self.discover_parents(frame)
        if not links:
            return ()
        parents = [(link["source"], link["lag"]) for link in links]
        design = _lagged(frame, parents)
        y = frame[self._target_col].to_numpy(dtype=float)
        ok = np.all(np.isfinite(design), axis=1) & np.isfinite(y)
        coef = np.linalg.lstsq(design[ok], y[ok], rcond=None)[0]
        resid = y[ok] - design[ok] @ coef
        formula = " + ".join([f"{c:.6g}*{n}[t-{lag}]" for c, (n, lag) in zip(coef, parents, strict=False)]
                             + [f"{coef[-1]:.6g}"])
        feature_cols = [c for c in frame.columns if c != self._target_col or any(n == c for n, _ in parents)]

        def compute_score(df: pd.DataFrame, parents: list[tuple[str, int]] = parents, coef: np.ndarray = coef) -> np.ndarray:
            out: np.ndarray = _lagged(df, parents) @ coef
            return out

        sst = float(np.sum((y[ok] - y[ok].mean()) ** 2))
        return (CandidateProposal(
            proposer_id=self.proposer_id, candidate_id=f"pcmci:{'|'.join(f'{n}@{lag}' for n, lag in parents)}",
            formula=formula, strategy_archetype=self._strategy_archetype, declared_scale_invariant=False,
            compute_score=compute_score,
            parameters={"dev_r2": 1.0 - float(np.sum(resid ** 2)) / sst if sst > 0 else 0.0,
                        "coefficients": [float(c) for c in coef]},
            metadata={"links": links, "tau_max": self._tau_max, "pc_alpha": self._pc_alpha,
                      "independence_test": "ParCorr (linear)", "assumes": "no hidden common cause",
                      "selection_pool_size": len(feature_cols) * self._tau_max},
        ),)
