"""The discovery task: observed trajectories, a library, and a concealed answer.

Status: experimental

Experiment QED-001 asks whether a proposer recovers a known governing equation
from noisy observations. A task therefore carries four things:

* the **development** observations and their measured derivative -- all a
  proposer ever sees;
* the **holdout** observations, which only the referee sees;
* the **library** the equation has to be written in;
* the **answer**: the support and coefficients the data were generated from.

The split of responsibility is the platform's usual one (AGENTS.md 1.1): a
proposer receives a :class:`~apedp_dyn.qed.objective.SupportObjective` built from
development rows only and cannot reach the task object, the holdout rows or the
answer. The referee receives the task and does the grading. Because this is a
benchmark with a known answer, the *referee* is allowed to know it; the search
is not.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from apedp_dyn.qed.library import LibraryRefusal, TermLibrary, oscillator_library
from apedp_dyn.qed.objective import SupportObjective


class TaskRefusal(ValueError):
    """Raised when a task or a trajectory cannot be used as declared."""


class RolloutRefusal(ValueError):
    """Raised when a candidate does not say how some coordinate evolves."""


@dataclass(frozen=True)
class RolloutRun:
    """An integrated trajectory, and whether it stayed finite.

    A wrong equation usually diverges rather than merely mispredicting, and a
    library of polynomials reaches overflow quickly. The failure is therefore
    represented, not raised: the rows after divergence are NaN and
    ``diverged`` is true, so the referee can refuse the candidate on its merits
    instead of recording a crash as an incident.
    """

    states: np.ndarray
    diverged: bool
    limit: float

    @property
    def finite_rows(self) -> int:
        return int(np.sum(np.all(np.isfinite(self.states), axis=1)))


def simulate_oscillator(
    x0: float,
    v0: float,
    *,
    steps: int,
    dt: float,
    stiffness: float,
    damping: float,
) -> np.ndarray:
    """Integrate ``x'' = -stiffness*x - damping*v`` with classical RK4."""

    def acceleration(position: float, velocity: float) -> float:
        return -stiffness * position - damping * velocity

    states: np.ndarray = np.empty((steps, 2), dtype=float)
    states[0] = (x0, v0)
    for index in range(steps - 1):
        x, v = states[index]
        k1x, k1v = v, acceleration(x, v)
        k2x, k2v = v + dt / 2 * k1v, acceleration(x + dt / 2 * k1x, v + dt / 2 * k1v)
        k3x, k3v = v + dt / 2 * k2v, acceleration(x + dt / 2 * k2x, v + dt / 2 * k2v)
        k4x, k4v = v + dt * k3v, acceleration(x + dt * k3x, v + dt * k3v)
        states[index + 1, 0] = x + dt / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
        states[index + 1, 1] = v + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
    return states


@dataclass(frozen=True)
class DiscoveryTask:
    """One equation-discovery problem, with the answer kept on the task object."""

    name: str
    library: TermLibrary
    dev_states: np.ndarray
    dev_target: np.ndarray
    holdout_states: np.ndarray
    holdout_target: np.ndarray
    dt: float
    discovered: int
    kinematic_pairs: tuple[tuple[int, int], ...]
    truth_support: tuple[int, ...]
    truth_coefficients: tuple[float, ...]
    penalty: float
    budget: int
    noise_level: float
    seed: int
    derivation: str = field(default="synthetic", compare=False)

    def __post_init__(self) -> None:
        for label, states in (("development", self.dev_states), ("holdout", self.holdout_states)):
            if states.ndim != 2:
                raise TaskRefusal(f"{label} states must be 2-D, got {states.shape}")
            if not np.all(np.isfinite(states)):
                raise TaskRefusal(f"{label} states must be finite")
        if self.dev_target.shape != (self.dev_states.shape[0],):
            raise TaskRefusal("development target must have one row per development state")
        if self.holdout_target.shape != (self.holdout_states.shape[0],):
            raise TaskRefusal("holdout target must have one row per holdout state")
        width = self.dev_states.shape[1]
        if self.holdout_states.shape[1] != width:
            raise TaskRefusal("development and holdout states must have the same width")
        if not 0 <= self.discovered < width:
            raise TaskRefusal(f"discovered coordinate {self.discovered} is out of range")
        for source, derivative in self.kinematic_pairs:
            if not (0 <= source < width and 0 <= derivative < width):
                raise TaskRefusal(f"kinematic pair {(source, derivative)} is out of range")
            if source == self.discovered:
                raise TaskRefusal(
                    "the discovered coordinate cannot also be given by a kinematic identity"
                )
        if len(self.truth_support) != len(self.truth_coefficients):
            raise TaskRefusal("the truth support and the truth coefficients disagree")
        if self.penalty < 0 or self.budget < 1:
            raise TaskRefusal("penalty must be non-negative and the budget at least one fit")

    @property
    def truth_names(self) -> tuple[str, ...]:
        return tuple(self.library.names[index] for index in self.truth_support)

    @property
    def width(self) -> int:
        _out: int = self.dev_states.shape[1]
        return _out

    def objective(self, *, budget: int | None = None) -> SupportObjective:
        """A fresh budgeted objective over development rows only.

        Every proposer gets its own instance, so no arm can warm another's cache
        or spend another's budget.
        """
        design = self.library.design_matrix(self.dev_states)
        return SupportObjective(
            self.library,
            design,
            self.dev_target,
            penalty=self.penalty,
            budget=self.budget if budget is None else budget,
        )

    def design_rows(self, states: np.ndarray) -> np.ndarray:
        return self.library.design_matrix(states)

    def render(self, support: Any, coefficients: Any) -> str:
        return self.library.render(support, coefficients, lhs=f"d state[{self.discovered}]/dt")

    def derivatives(self, state: np.ndarray, support: Any, coefficients: Any) -> np.ndarray:
        """``d state/dt`` for one sample: the fitted law plus the given identities."""
        state = np.asarray(state, dtype=float).reshape(1, -1)
        out: np.ndarray = np.full(self.width, np.nan, dtype=float)
        if support:
            columns = self.library.select(self.library.design_matrix(state), support)
            out[self.discovered] = float(columns[0] @ np.asarray(coefficients, dtype=float))
        else:
            out[self.discovered] = 0.0
        for source, derivative in self.kinematic_pairs:
            out[source] = float(state[0, derivative])
        if not np.all(np.isfinite(out)):
            unknown = [index for index, value in enumerate(out) if not np.isfinite(value)]
            raise RolloutRefusal(f"no evolution rule for coordinates {unknown}")
        return out

    def rollout(
        self,
        initial: np.ndarray,
        support: Any,
        coefficients: Any,
        *,
        steps: int,
        divergence_limit: float = 1e6,
    ) -> RolloutRun:
        """Integrate the candidate forward from ``initial`` with RK4."""
        initial = np.asarray(initial, dtype=float).ravel()
        if initial.shape != (self.width,):
            raise TaskRefusal(f"initial state must have {self.width} entries")
        states: np.ndarray = np.full((steps, self.width), np.nan, dtype=float)
        states[0] = initial

        def rhs(vector: np.ndarray) -> np.ndarray:
            return self.derivatives(vector, support, coefficients)

        diverged = False
        for index in range(steps - 1):
            here = states[index]
            try:
                k1 = rhs(here)
                k2 = rhs(here + self.dt / 2 * k1)
                k3 = rhs(here + self.dt / 2 * k2)
                k4 = rhs(here + self.dt * k3)
                following = here + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            except (FloatingPointError, LibraryRefusal, OverflowError):
                diverged = True
                break
            if not np.all(np.isfinite(following)) or np.max(np.abs(following)) > divergence_limit:
                diverged = True
                break
            states[index + 1] = following
        return RolloutRun(states=states, diverged=diverged, limit=divergence_limit)


OSCILLATOR_STIFFNESS = 1.44
OSCILLATOR_DAMPING = 0.15
OSCILLATOR_DEV_IC = (2.5, 0.0)
OSCILLATOR_HOLDOUT_IC = (2.5, 1.2)
OSCILLATOR_STEPS = 400
OSCILLATOR_DT = 0.02
OSCILLATOR_TRUTH = ((-1.44, "x"), (-0.15, "v"))


def damped_oscillator_task(
    *,
    seed: int = 11,
    noise_level: float = 0.01,
    penalty: float = 0.002,
    budget: int = 256,
) -> DiscoveryTask:
    """QED-001: recover ``v' = -1.44 x - 0.15 v`` from noisy ``(x, v)``.

    Both trajectories use RK4 at ``dt = 0.02``; noise is Gaussian with a
    standard deviation of ``noise_level`` times the development velocity scale,
    added independently to both coordinates, and the target is the centred
    finite-difference derivative of the *observed* velocity. The development and
    holdout trajectories start from different initial conditions, so the holdout
    is an extrapolation in state space rather than a saturated tail.
    """
    if not 0 < noise_level < 1:
        raise TaskRefusal(f"noise_level must be a fraction, got {noise_level}")
    clean_dev = simulate_oscillator(
        OSCILLATOR_DEV_IC[0],
        OSCILLATOR_DEV_IC[1],
        steps=OSCILLATOR_STEPS,
        dt=OSCILLATOR_DT,
        stiffness=OSCILLATOR_STIFFNESS,
        damping=OSCILLATOR_DAMPING,
    )
    clean_holdout = simulate_oscillator(
        OSCILLATOR_HOLDOUT_IC[0],
        OSCILLATOR_HOLDOUT_IC[1],
        steps=OSCILLATOR_STEPS,
        dt=OSCILLATOR_DT,
        stiffness=OSCILLATOR_STIFFNESS,
        damping=OSCILLATOR_DAMPING,
    )
    rng = np.random.default_rng(seed)
    sigma = noise_level * float(np.std(clean_dev[:, 1]))
    dev = clean_dev + rng.normal(0.0, sigma, clean_dev.shape)
    holdout = clean_holdout + rng.normal(0.0, sigma, clean_holdout.shape)

    library = oscillator_library()
    truth_support = tuple(sorted(library.index_of(name) for _value, name in OSCILLATOR_TRUTH))
    truth_coefficients = tuple(float(value) for value, _name in OSCILLATOR_TRUTH)
    return DiscoveryTask(
        name="qed-001/damped-oscillator",
        library=library,
        dev_states=dev,
        dev_target=np.gradient(dev[:, 1], OSCILLATOR_DT),
        holdout_states=holdout,
        holdout_target=np.gradient(holdout[:, 1], OSCILLATOR_DT),
        dt=OSCILLATOR_DT,
        discovered=1,
        kinematic_pairs=((0, 1),),
        truth_support=truth_support,
        truth_coefficients=truth_coefficients,
        penalty=penalty,
        budget=budget,
        noise_level=noise_level,
        seed=seed,
        derivation="synthetic: x'' = -1.44 x - 0.15 v, RK4, Gaussian observation noise",
    )


__all__ = [
    "OSCILLATOR_DAMPING",
    "OSCILLATOR_DEV_IC",
    "OSCILLATOR_DT",
    "OSCILLATOR_HOLDOUT_IC",
    "OSCILLATOR_STEPS",
    "OSCILLATOR_STIFFNESS",
    "OSCILLATOR_TRUTH",
    "DiscoveryTask",
    "RolloutRefusal",
    "RolloutRun",
    "TaskRefusal",
    "damped_oscillator_task",
    "simulate_oscillator",
]
