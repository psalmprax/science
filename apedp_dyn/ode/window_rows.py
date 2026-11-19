"""Regression rows from a sampled trajectory: one state and one rate of change per disjoint window.

Status: experimental

The main search path (``apedp_dyn.discovery_l2.main_search``) is a regression
path: rows of inputs, one noisy target per row, noise independent from row to
row. Its adequacy check asks whether a law's residuals can be predicted from the
residuals of neighbouring rows, and a trajectory differentiated point by point
fails that for the true law: a smoothed derivative's error is shared by
neighbouring samples, and on a trajectory the neighbours in state space are the
neighbours in time (audit M-07 found the derivative route limited by selection,
not by the derivative).

Here the trajectory is cut into disjoint windows. Inside each window every
variable is fitted with a polynomial in time; the fitted value and slope at the
window's centre are that window's state and rate. Windows share no samples, so
with independent measurement noise their errors are independent, which is what
the path assumes, and every window has the same noise level, which is the
path's ``noise="absolute"`` model.

The polynomial's degree is chosen per variable, the same for all windows, by
leave-one-out prediction error summed over the windows. Clean data takes the
highest degree and its rates are exact to many digits; noisy data takes a low
one. Nothing has to be told the noise level.

What this costs: a trajectory of ``n`` samples becomes ``n // window`` rows, and
a window has to be short against the time scale of the dynamics or the
polynomial cannot follow it (the fit residual, returned as ``fit_rms``, shows
when it does not).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial import legendre


@dataclass(frozen=True)
class WindowRows:
    states: np.ndarray       #: (windows, variables) fitted value at each window's centre
    rates: np.ndarray        #: (windows, variables) fitted time derivative there
    times: np.ndarray        #: (windows,) time of each centre
    degree: tuple[int, ...]  #: polynomial degree chosen for each variable
    fit_rms: np.ndarray      #: (variables,) residual of the window fits: the measurement noise if the fit is adequate
    rate_sd: np.ndarray      #: (variables,) standard error of a rate implied by ``fit_rms``


def window_rows(t: np.ndarray, x: np.ndarray, window: int = 100, max_degree: int = 9) -> WindowRows:
    """Cut ``x`` (samples by variables, uniform times ``t``) into disjoint windows of ``window`` samples."""
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if len(t) != len(x):
        raise ValueError("t and x must have the same number of samples")
    steps = np.diff(t)
    if len(t) < window or not np.allclose(steps, steps[0], rtol=1e-6, atol=0.0):
        raise ValueError("window_rows needs uniformly sampled times and at least one full window")
    if not 1 <= max_degree < window - 2:
        raise ValueError("max_degree must be at least 1 and well below the window length")
    count = len(t) // window
    blocks = x[: count * window].reshape(count, window, x.shape[1])
    u = np.linspace(-1.0, 1.0, window)                 # time inside a window, scaled
    half_width = 0.5 * (window - 1) * float(steps[0])  # so d/dt = (d/du) / half_width
    centres: np.ndarray = t[: count * window].reshape(count, window).mean(axis=1)

    states = np.empty((count, x.shape[1]))
    rates = np.empty((count, x.shape[1]))
    degrees, fit_rms, rate_sd = [], [], []
    for j in range(x.shape[1]):
        best: tuple[float, int, np.ndarray, float, float] | None = None
        for degree in range(1, max_degree + 1):
            basis = legendre.legvander(u, degree)
            q, r = np.linalg.qr(basis)
            leverage = np.sum(q ** 2, axis=1)
            coef = np.linalg.solve(r, q.T @ blocks[:, :, j].T)  # (degree + 1, windows)
            resid = blocks[:, :, j].T - basis @ coef
            press = float(np.sum((resid / (1.0 - leverage)[:, None]) ** 2))
            if best is None or press < best[0]:
                at_centre = legendre.legvander(np.zeros(1), degree)[0]
                slope_at_centre = np.array([legendre.legval(0.0, legendre.legder(np.eye(degree + 1)[k])) for k in range(degree + 1)])
                sigma = float(np.sqrt(np.sum(resid ** 2) / (count * (window - degree - 1))))
                functional = np.linalg.solve(r.T, slope_at_centre)
                best = (press, degree, np.vstack([at_centre @ coef, slope_at_centre @ coef / half_width]),
                        sigma, sigma * float(np.linalg.norm(functional)) / half_width)
        assert best is not None
        states[:, j], rates[:, j] = best[2][0], best[2][1]
        degrees.append(best[1])
        fit_rms.append(best[3])
        rate_sd.append(best[4])
    return WindowRows(states, rates, centres, tuple(degrees), np.array(fit_rms), np.array(rate_sd))
