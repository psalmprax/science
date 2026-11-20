"""Drop a candidate the path cannot fit, before the path sees it.

Status: experimental

**Why (audit M-24).** The main search fits every candidate by least squares
(``contrast_resolution.fitted_candidate``). ``numpy.linalg.lstsq`` raises when
its SVD does not converge, the path does not catch that, and the whole
discovery ends. Three preregistered runs ended that way on 2026-11-20: two of
234 in PREREG-ODE-REFINED and one of 266 in PREREG-PORTFOLIO-SEARCH, each on a
law the earlier route recovers. More searches hand over more candidates, so
the portfolio made it more likely. The modules that would have to catch it
are frozen by several preregistrations.

**What.** ``FitGuardProposer`` wraps a search stage. Each proposal is parsed
and fitted here, on the same development rows, with the path's own fitting
function, under both noise models. One that raises, or whose coefficients or
predictions are not finite, is dropped and recorded in ``dropped``. The rest
pass through unchanged and in order.

**Limit.** The path polishes a candidate's constants before fitting it, so
the expression it fits can differ from the one checked here. The guard
removes the candidates that fail as proposed; it does not prove the path
cannot fail. Not yet measured in a preregistered run.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import sympy as sp

from apedp_dyn.discovery_l2 import contrast_resolution as cr
from apedp_dyn.grammar.proposer_interface import CandidateProposal, ProposerBase

NOISE_MODELS = ("relative", "absolute")


def fit_problem(formula: str, names: Sequence[str], X: np.ndarray, y: np.ndarray) -> str:
    """Why the path could not fit this formula on these rows, or ``""`` if it can."""
    try:
        expr = sp.sympify(formula, locals={n: sp.Symbol(n) for n in names})
    except (sp.SympifyError, TypeError, SyntaxError) as exc:
        return f"not parseable ({type(exc).__name__})"
    for noise in NOISE_MODELS:
        try:
            with np.errstate(all="ignore"):
                fitted = cr.fitted_candidate(expr, names, X, y, noise)
                finite = bool(np.all(np.isfinite(fitted.coef)) and np.all(np.isfinite(fitted.dev_pred)))
        except Exception as exc:  # noqa: BLE001 - any failure of the fit is the thing being screened for
            return f"fit raised {type(exc).__name__} ({noise} noise)"
        if not finite:
            return f"fit not finite ({noise} noise)"
    return ""


class FitGuardProposer(ProposerBase):
    """A search stage whose unfittable proposals never reach the path."""

    def __init__(self, inner: ProposerBase, target_col: str = "target") -> None:
        self._inner = inner
        self._target_col = target_col
        #: (formula, reason) for every proposal dropped by the last call
        self.dropped: list[tuple[str, str]] = []

    @property
    def proposer_id(self) -> str:
        return str(self._inner.proposer_id)

    @property
    def failures(self) -> list[str]:
        """The wrapped stage's own failures, if it records any."""
        return list(getattr(self._inner, "failures", []) or [])

    @property
    def last_run(self) -> dict[str, Any]:
        return {**dict(getattr(self._inner, "last_run", {}) or {}), "fit_guard_dropped": len(self.dropped)}

    def fit_and_propose(self, train_data: Any, target_col: str | None = None, guesses: Sequence[str] | None = None,
                        **kwargs: Any) -> Sequence[CandidateProposal]:
        frame = train_data if isinstance(train_data, pd.DataFrame) else pd.DataFrame(train_data)
        target = target_col or self._target_col
        names = [str(c) for c in frame.columns if c != target]
        X = frame[names].to_numpy(dtype=float)
        y = frame[target].to_numpy(dtype=float)
        self.dropped = []
        kept: list[CandidateProposal] = []
        for proposal in self._inner.fit_and_propose(frame, target_col=target, guesses=guesses, **kwargs):
            reason = fit_problem(str(proposal.formula), names, X, y)
            if reason:
                self.dropped.append((str(proposal.formula)[:200], reason))
            else:
                kept.append(proposal)
        return kept
