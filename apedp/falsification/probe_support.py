"""Shared probe plumbing: recorded probe errors and envelope sampling."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import sympy as sp

from .constants import INVARIANT_SAMPLES

logger = logging.getLogger("apedp.falsification")

class _ProbeSupportMixin:
    """Shared probe plumbing: recorded probe errors and envelope sampling."""

    @staticmethod
    def _record_probe_error(
        metrics: dict[str, Any],
        probe: str,
        exc: Exception | str,
        detail: str = "",
        measured_alternative: bool = False,
    ) -> None:
        """Record that a probe could not decide, instead of letting it pass silently.

        A probe that crashed or timed out produced no evidence about the model, so
        it must never be counted as a check that passed. Every such event is
        published in ``metrics["probe_errors"]`` with the probe's name, the error
        type and a detail string.

        ``measured_alternative`` distinguishes the two kinds of gap. Set it when a
        different measurement in the same axis still answered the question the
        probe was asked (a symbolic derivative that will not evaluate, where the
        finite-difference gradient is the declared fallback). Such an error is
        reported -- it is a degradation, and it is in the record -- but it does not
        by itself void a clean verdict. Errors left ``unresolved`` mean nothing
        measured that property, and :meth:`falsify` then refuses to report
        ``SURVIVED_PROVISIONAL``. This is the rule the invariant axis was fixed to
        follow, applied across all five axes.
        """
        error = exc if isinstance(exc, str) else type(exc).__name__
        metrics.setdefault("probe_errors", []).append({
            "probe": probe,
            "error": error,
            "detail": detail or ("" if isinstance(exc, str) else str(exc)),
            "unresolved": not measured_alternative,
        })
        logger.warning(
            "Falsification probe %s could not decide: %s",
            probe,
            detail or error,
            exc_info=not isinstance(exc, str),
        )

    def _sample_envelope(
        self,
        fn: Any,
        syms: list[sp.Symbol],
        domains: dict[str, tuple[float, float]],
    ) -> tuple[list[tuple[dict[str, float], float | None]], list[dict[str, Any]]]:
        """Evaluate the model across its declared envelope.

        For every symbol the model is swept across that symbol's declared interval
        (other symbols held at their midpoint); one further sweep moves all symbols
        together along the domain diagonal. Returns ``(samples, errors)`` where each
        sample is ``(point, value)`` and ``value`` is ``None`` when the model is not
        real-valued there, and ``errors`` records any sweep that could not be
        evaluated at all -- an unevaluated invariant is an unknown, never a pass.
        """
        bounds = {str(s): tuple(domains.get(str(s), (-1.0, 1.0))) for s in syms}
        mids = {name: (lo + hi) / 2.0 for name, (lo, hi) in bounds.items()}
        n = INVARIANT_SAMPLES

        sweeps: list[dict[str, np.ndarray]] = []
        for s in syms:
            lo, hi = bounds[str(s)]
            sweep = {name: np.full(n, mids[name], dtype=float) for name in bounds}
            sweep[str(s)] = np.linspace(lo, hi, n)
            sweeps.append(sweep)
        if syms:
            t = np.linspace(0.0, 1.0, n)
            sweeps.append({name: lo + t * (hi - lo) for name, (lo, hi) in bounds.items()})

        samples: list[tuple[dict[str, float], float | None]] = []
        errors: list[dict[str, Any]] = []
        for sweep in sweeps:
            args = [sweep[str(s)] for s in syms]
            try:
                with np.errstate(all="ignore"):
                    values = np.ravel(np.asarray(fn(*args), dtype=float))
            except (TypeError, ValueError, ArithmeticError, AttributeError, NotImplementedError) as exc:
                errors.append({
                    "probe": "EnvelopeSweep",
                    "error": type(exc).__name__,
                    "detail": str(exc),
                    "unresolved": True,
                })
                continue
            length = len(args[0]) if args else 0
            if values.size == 1 and length > 1:
                values = np.full(length, float(values[0]))
            for i in range(min(length, values.size)):
                point = {name: float(np.ravel(arr)[i]) for name, arr in sweep.items()}
                v = float(values[i])
                samples.append((point, v if np.isfinite(v) else None))
        return samples, errors
