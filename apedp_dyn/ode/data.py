"""Data structures and typed containers for continuous ODE discovery."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OdeTrajectoryData:
    """Multi-variable continuous dynamical system trajectory data x(t).

    Attributes:
        t: Time vector of shape (N_t,).
        states: Dictionary mapping state variable names to 1D trajectory arrays.
        dt: Average temporal step size.
    """
    t: np.ndarray
    states: dict[str, np.ndarray]
    dt: float

    @classmethod
    def from_dict(
        cls,
        series_dict: Mapping[str, Sequence[float]],
        dt: float,
        t: Sequence[float] | None = None,
    ) -> OdeTrajectoryData:
        states_dict: dict[str, np.ndarray] = {}
        n_t = None
        for k, v in series_dict.items():
            arr = np.ascontiguousarray(v, dtype=np.float64)
            if n_t is None:
                n_t = len(arr)
            elif len(arr) != n_t:
                raise ValueError(f"State '{k}' has length {len(arr)}, expected {n_t}")
            states_dict[k] = arr

        if n_t is None or n_t == 0:
            raise ValueError("No state series provided in series_dict")

        if t is not None:
            t_arr = np.ascontiguousarray(t, dtype=np.float64)
            if len(t_arr) != n_t:
                raise ValueError(f"Time vector length {len(t_arr)} != state trajectory length {n_t}")
        else:
            t_arr = np.arange(n_t, dtype=np.float64) * float(dt)

        return cls(t=t_arr, states=states_dict, dt=float(dt))
