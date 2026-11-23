"""Whether a table lies inside the conditions under which the route in force was measured.

Status: experimental

**Why this exists.** The table route in force (``vouched_route_v2``) was
measured on tables of up to three inputs, 700 rows to fit on, Gaussian noise
of up to 5 % of the target's size and rows spread evenly
(PREREG-VOUCHING-V2-2026-11). The UI applies it to any table. Above 5 % noise
the last checks are known to be weaker (audit M-36), and on 60 to 300 rows the
first version was weaker (M-34). Nothing told the user when a table lay
outside what was measured. This module says so. It **marks**; it refuses
nothing and changes no law, so what the route reports is what was measured.

**How the noise is estimated.** The benchmarks set noise as a share of the
root mean square of the clean target. The reported law's residuals on the
rows it was fitted on give the noise's size, ``sigma``. The clean target's
root mean square is ``sqrt(mean(y**2) - sigma**2)``. A wrong law leaves its
error in the residuals and reads as noisier, so the mark errs towards
"outside". The estimate's standard error is ``share / sqrt(2 n)``. Within two
standard errors of 5 % the table is "at the edge", above that "outside". The
two-standard-error band is a convention for a mark, not a threshold that
decides anything.

**What cannot be checked from one table:** whether the noise is Gaussian and
whether the rows are spread evenly. The record says so every time.

The measured conditions are constants here. ``tests/test_measured_range.py``
checks them against the blind test's laws file, so they cannot drift from it.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import sympy as sp

#: the largest noise in PREREG-VOUCHING-V2's runs, as a share of the clean target's root mean square
MEASURED_NOISE = 0.05
#: the most inputs of any law in that run
MEASURED_INPUTS = 3
#: the rows each law was fitted on in that run (1,000 rows, 70 % to develop on)
MEASURED_ROWS = 700
EDGE_ERRORS = 2.0
SOURCE = "preregistration/RESULT-VOUCHING-V2-2026-11.md"
UNCHECKED = "Gaussian noise and evenly spread rows were measured too; one table cannot show either"
INSIDE, EDGE, OUTSIDE, UNKNOWN = "inside", "edge", "outside", "unknown"


def noise_share(law: str, names: Sequence[str], X: np.ndarray, y: np.ndarray) -> tuple[float, float] | None:
    """The noise as a share of the clean target's root mean square, and its standard error; ``None`` if it cannot be read."""
    symbols = {name: sp.Symbol(name) for name in names}
    try:
        expr = sp.sympify(law, locals=symbols)
        f = sp.lambdify([symbols[name] for name in names], expr, "numpy")
        with np.errstate(all="ignore"):
            values = np.broadcast_to(np.asarray(f(*[X[:, i] for i in range(X.shape[1])]), dtype=float), y.shape)
    except Exception:  # noqa: BLE001 - a law that cannot be evaluated gives no estimate; the mark says unknown
        return None
    keep = np.isfinite(values) & np.isfinite(y)
    n = int(keep.sum())
    if n < 10:
        return None
    residual = y[keep] - values[keep]
    sigma2 = float(np.mean(residual ** 2))
    clean2 = float(np.mean(y[keep] ** 2)) - sigma2
    if clean2 <= 0.0:
        return math.inf, math.inf
    share = math.sqrt(sigma2 / clean2)
    return share, share / math.sqrt(2.0 * n)


def assess(law: str | None, names: Sequence[str], X: np.ndarray, y: np.ndarray, *, route: str | None = None,
           measured_route: str = "refined") -> dict[str, Any]:
    """The mark for one table: ``state`` is inside, edge, outside or unknown, with the reasons and the numbers behind it."""
    X = np.asarray(X, dtype=float).reshape(len(y), -1)
    y = np.asarray(y, dtype=float)
    out: dict[str, Any] = {"state": UNKNOWN, "reasons": [], "inputs": int(X.shape[1]), "rows": len(y),
                           "noise": None, "noise_error": None, "unchecked": UNCHECKED, "source": SOURCE,
                           "measured": {"noise_up_to": MEASURED_NOISE, "inputs_up_to": MEASURED_INPUTS,
                                        "rows_at_least": MEASURED_ROWS, "route": measured_route}}
    if route is not None and route != measured_route:
        out["reasons"].append(f"the route in force was measured on the {measured_route} path, not on {route}")
        return out
    outside: list[str] = []
    edge: list[str] = []
    if X.shape[1] > MEASURED_INPUTS:
        outside.append(f"{X.shape[1]} inputs; measured up to {MEASURED_INPUTS}")
    if len(y) < MEASURED_ROWS:
        outside.append(f"{len(y)} rows to fit on; measured on {MEASURED_ROWS}")
    if not law:
        out["reasons"] = [*outside, "no law to estimate the noise from"]
        out["state"] = OUTSIDE if outside else UNKNOWN
        return out
    read = noise_share(law, names, X, y)
    if read is None:
        out["reasons"] = [*outside, "the law could not be evaluated on these rows, so the noise is not estimated"]
        out["state"] = OUTSIDE if outside else UNKNOWN
        return out
    share, error = read
    out["noise"] = None if not math.isfinite(share) else round(share, 5)
    out["noise_error"] = None if not math.isfinite(error) else round(error, 5)
    if share - EDGE_ERRORS * error > MEASURED_NOISE:
        outside.append(f"noise about {share:.1%} of the target's size; measured up to {MEASURED_NOISE:.0%}")
    elif share + EDGE_ERRORS * error > MEASURED_NOISE:
        edge.append(f"noise about {share:.1%} of the target's size, at the edge of the {MEASURED_NOISE:.0%} measured")
    out["reasons"] = outside + edge
    out["state"] = OUTSIDE if outside else EDGE if edge else INSIDE
    return out
