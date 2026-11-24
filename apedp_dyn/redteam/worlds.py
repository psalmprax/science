"""A world for the referee: a law, what a search is said to have returned, how rows are drawn, what noise they carry.

Status: experimental

Everything a decision under test is given comes from one ``World``, and
everything random in it from the world's own seed, so a world found to defeat
a referee can be written down in a line and run again.

``sampling`` says where the rows fall:

- ``uniform``: independently and uniformly over the box (the published
  benchmarks' way);
- ``narrow``: uniformly over a part of the box, ``spread`` of its width in
  every input;
- ``clustered``: around three points, with standard deviation ``spread`` of the
  box's width;
- ``correlated``: the first input uniform, every other one equal to it plus a
  deviation of standard deviation ``spread`` of the box's width (kept inside
  the box).

``noise_shape`` says what is added to the targets, always with root mean square
``noise`` times that of the targets:

- ``gaussian``: independent, equal variance (what the path under test assumes
  when told the noise is absolute);
- ``student``: Student's t with three degrees of freedom, heavy tails;
- ``proportional``: standard deviation proportional to the size of the target;
- ``outliers``: Gaussian, and on one row in fifty a further error of ten
  standard deviations.

The rows a law is scored on are drawn the same way as the rows it was found
on, from a separate stream, and carry no noise. A law can therefore only be
called wrong for what it gets wrong where the data were, which is the
benchmarks' own standard; what it does elsewhere is not tested here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import sympy as sp

SAMPLINGS = ("uniform", "narrow", "clustered", "correlated")
NOISE_SHAPES = ("gaussian", "student", "proportional", "outliers")
TEST_ROWS = 500
CLUSTERS = 3
OUTLIER_SHARE, OUTLIER_SIZE = 0.02, 10.0


@dataclass(frozen=True)
class World:
    law: str
    inputs: int
    candidates: tuple[str, ...]          #: what the search stage is said to have returned, in its order
    truth_listed: bool = False           #: whether ``law`` itself is among them
    rows: int = 1000                     #: development and pool rows together
    noise: float = 0.01                  #: RMS of the noise as a share of the RMS of the targets
    noise_shape: str = "gaussian"
    sampling: str = "uniform"
    spread: float = 1.0                  #: the sampling's parameter (unused by ``uniform``)
    seed: int = 0
    low: float = 1.0
    high: float = 5.0

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "candidates": list(self.candidates)}

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> World:
        return cls(**{**record, "candidates": tuple(record["candidates"])})


@dataclass
class Rows:
    X: np.ndarray            #: development and pool inputs
    y: np.ndarray            #: their targets, with the world's noise
    clean: np.ndarray        #: the same targets without noise
    X_test: np.ndarray
    y_test: np.ndarray       #: test targets, without noise


def inputs_of(world: World, count: int, rng: np.random.Generator) -> np.ndarray:
    """``count`` rows of inputs drawn the world's way. The layout (which part, which centres) is the world's, not the draw's."""
    width = world.high - world.low
    layout = np.random.default_rng([world.seed, 7])        # the same part of the box for every draw of this world
    rows: np.ndarray
    if world.sampling == "uniform":
        rows = rng.uniform(world.low, world.high, (count, world.inputs))
    elif world.sampling == "narrow":
        share = float(np.clip(world.spread, 0.01, 1.0))
        start = world.low + layout.uniform(0.0, 1.0 - share, world.inputs) * width
        rows = start + rng.uniform(0.0, share * width, (count, world.inputs))
    elif world.sampling == "clustered":
        centres = layout.uniform(world.low + 0.1 * width, world.high - 0.1 * width, (CLUSTERS, world.inputs))
        rows = np.clip(centres[rng.integers(0, CLUSTERS, count)]
                       + world.spread * width * rng.standard_normal((count, world.inputs)), world.low, world.high)
    elif world.sampling == "correlated":
        first = rng.uniform(world.low, world.high, count)
        rows = first[:, None] + world.spread * width * rng.standard_normal((count, world.inputs))
        rows[:, 0] = first
        rows = np.clip(rows, world.low, world.high)
    else:
        raise ValueError(f"unknown sampling {world.sampling!r}")
    return rows


def noise_of(world: World, clean: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """The world's noise for targets ``clean``: root mean square ``noise`` times theirs (before any outliers)."""
    size = world.noise * float(np.sqrt(np.mean(clean ** 2)))
    noise: np.ndarray
    if size == 0.0:
        noise = np.zeros(len(clean))
    elif world.noise_shape == "gaussian":
        noise = size * rng.standard_normal(len(clean))
    elif world.noise_shape == "student":
        noise = size * rng.standard_t(3, len(clean)) / np.sqrt(3.0)
    elif world.noise_shape == "proportional":
        weight = np.abs(clean) / max(float(np.sqrt(np.mean(clean ** 2))), 1e-300)
        noise = size * weight * rng.standard_normal(len(clean))
    elif world.noise_shape == "outliers":
        hit = rng.random(len(clean)) < OUTLIER_SHARE
        noise = size * rng.standard_normal(len(clean)) + hit * OUTLIER_SIZE * size * rng.choice([-1.0, 1.0], len(clean))
    else:
        raise ValueError(f"unknown noise shape {world.noise_shape!r}")
    return noise


def law_values(world: World, X: np.ndarray) -> np.ndarray:
    symbols = [sp.Symbol(f"x{i}") for i in range(world.inputs)]
    fn = sp.lambdify(symbols, sp.sympify(world.law, locals={str(s): s for s in symbols}), "numpy")
    with np.errstate(all="ignore"):
        out: np.ndarray = np.asarray(fn(*X.T), dtype=float) * np.ones(len(X))
    return out


def rows_of(world: World) -> Rows:
    """The world's rows: found on ``X, y``; scored on ``X_test, y_test``."""
    if world.sampling not in SAMPLINGS or world.noise_shape not in NOISE_SHAPES:
        raise ValueError("unknown sampling or noise shape")
    X = inputs_of(world, world.rows, np.random.default_rng([world.seed, 1]))
    X_test = inputs_of(world, TEST_ROWS, np.random.default_rng([world.seed, 2]))
    clean, y_test = law_values(world, X), law_values(world, X_test)
    if not (np.all(np.isfinite(clean)) and np.all(np.isfinite(y_test))):
        raise ValueError("the law is not finite on the world's rows")
    return Rows(X, clean + noise_of(world, clean, np.random.default_rng([world.seed, 3])), clean, X_test, y_test)
