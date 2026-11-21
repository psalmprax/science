"""Draw worlds inside an arena, and change them towards the outcome a referee must not produce.

Status: experimental

An *arena* fixes which settings of a world may vary: how much noise, how many
rows, how they are sampled, what shape the noise has. The first arena is the
conditions of the preregistered runs, so a wrong law reported there is a
counterexample to what those runs measured; the others each leave those
conditions in one direction, so a wrong law there marks where the measured
record stops applying.

``draw`` makes a world at random inside an arena. ``change`` alters one thing
about a world and stays inside the arena. ``climb`` keeps the worlds whose
outcome is worst and changes those again: a search over data for what defeats
the decision it is given. How bad an outcome is comes from the caller (for a
law-discovery route: one minus the test R^2 of a reported law that is not the
law, zero otherwise), so the same search serves any decision.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from apedp_dyn.redteam.worlds import World

SPREADS: Mapping[str, tuple[float, float]] = {"uniform": (1.0, 1.0), "narrow": (0.15, 0.6), "clustered": (0.02, 0.1),
                                              "correlated": (0.02, 0.3)}
LOWEST_NOISE = 1e-3          # below this a noise level is drawn as none at all
STEP = 1.5                   # a change multiplies or divides noise and rows by this
MAX_COUNTERFEITS = 3


@dataclass(frozen=True)
class Arena:
    name: str
    noise: tuple[float, float]
    rows: tuple[int, int]
    samplings: tuple[str, ...]
    noise_shapes: tuple[str, ...]
    what: str


ARENAS: Mapping[str, Arena] = {arena.name: arena for arena in (
    Arena("measured", (0.0, 0.01), (1000, 1000), ("uniform",), ("gaussian",),
          "the conditions of the preregistered runs; a wrong law here contradicts what they measured"),
    Arena("few_rows", (0.0, 0.01), (60, 300), ("uniform",), ("gaussian",), "fewer rows than the benchmarks give"),
    Arena("sampling", (0.0, 0.01), (1000, 1000), ("narrow", "clustered", "correlated"), ("gaussian",),
          "rows that do not cover the box evenly"),
    Arena("noise_shape", (LOWEST_NOISE, 0.01), (1000, 1000), ("uniform",), ("student", "proportional", "outliers"),
          "noise of another shape than the path is told"),
    Arena("noise_level", (0.01, 0.1), (1000, 1000), ("uniform",), ("gaussian",), "more noise than the benchmarks used"),
)}


def _noise(arena: Arena, rng: np.random.Generator) -> float:
    low, high = arena.noise
    if low == 0.0 and rng.random() < 0.25:
        return 0.0
    return float(np.exp(rng.uniform(np.log(max(low, LOWEST_NOISE)), np.log(high))))


def _rows(arena: Arena, rng: np.random.Generator) -> int:
    low, high = arena.rows
    return round(float(np.exp(rng.uniform(np.log(low), np.log(high)))))


def _sampling(arena: Arena, rng: np.random.Generator) -> tuple[str, float]:
    sampling = str(rng.choice(arena.samplings))
    low, high = SPREADS[sampling]
    return sampling, float(rng.uniform(low, high))


def _candidates(entry: Mapping[str, Any], fakes: Sequence[str], listed: bool, rng: np.random.Generator) -> tuple[str, ...]:
    texts = [*fakes, *([str(entry["law"])] if listed else [])]
    return tuple(texts[int(i)] for i in rng.permutation(len(texts)))


def draw(arena: Arena, laws: Sequence[Mapping[str, Any]], rng: np.random.Generator) -> World:
    """A world at random inside ``arena``. ``laws`` are entries with ``law``, ``inputs`` and ``counterfeits`` (each a ``text``)."""
    entry = laws[int(rng.integers(0, len(laws)))]
    available = [str(fake["text"]) for fake in entry["counterfeits"]]
    listed = bool(rng.random() < 0.5) or not available
    count = min(len(available), int(rng.integers(1, MAX_COUNTERFEITS + 1)))
    fakes = [available[int(i)] for i in rng.choice(len(available), size=count, replace=False)] if available else []
    sampling, spread = _sampling(arena, rng)
    return World(law=str(entry["law"]), inputs=int(entry["inputs"]), candidates=_candidates(entry, fakes, listed, rng),
                 truth_listed=listed, rows=_rows(arena, rng), noise=_noise(arena, rng),
                 noise_shape=str(rng.choice(arena.noise_shapes)), sampling=sampling, spread=spread,
                 seed=int(rng.integers(0, 2 ** 31 - 1)))


def change(world: World, arena: Arena, laws: Mapping[str, Mapping[str, Any]], rng: np.random.Generator) -> World:
    """``world`` with one thing altered, still inside ``arena``. ``laws`` maps a law's text to its entry."""
    entry = laws[world.law]
    available = [str(fake["text"]) for fake in entry["counterfeits"]]
    fakes = [text for text in world.candidates if text != world.law]
    unused = [text for text in available if text not in fakes]
    moves = ["seed", "order"]
    if arena.noise[0] < arena.noise[1]:
        moves.append("noise")
    if arena.rows[0] < arena.rows[1]:
        moves.append("rows")
    if len(arena.samplings) > 1 or SPREADS[arena.samplings[0]][0] < SPREADS[arena.samplings[0]][1]:
        moves.append("sampling")
    if len(arena.noise_shapes) > 1:
        moves.append("shape")
    if unused and fakes:
        moves.append("swap")
    if unused and len(fakes) < MAX_COUNTERFEITS:
        moves.append("add")
    if len(fakes) > 1 or (fakes and world.truth_listed):
        moves.append("drop")
    if fakes:
        moves.append("truth")
    move = str(rng.choice(moves))
    fresh = int(rng.integers(0, 2 ** 31 - 1))
    if move == "seed":
        return replace(world, seed=fresh)
    if move == "order":
        return replace(world, candidates=tuple(world.candidates[int(i)] for i in rng.permutation(len(world.candidates))))
    if move == "noise":
        low, high = arena.noise
        factor = STEP if rng.random() < 0.5 else 1.0 / STEP
        level = max(world.noise, LOWEST_NOISE) * factor
        if level < max(low, LOWEST_NOISE):
            level = 0.0 if low == 0.0 else low
        return replace(world, noise=float(min(level, high)))
    if move == "rows":
        factor = STEP if rng.random() < 0.5 else 1.0 / STEP
        return replace(world, rows=int(np.clip(round(world.rows * factor), arena.rows[0], arena.rows[1])))
    if move == "sampling":
        sampling, spread = _sampling(arena, rng)
        return replace(world, sampling=sampling, spread=spread)
    if move == "shape":
        return replace(world, noise_shape=str(rng.choice([s for s in arena.noise_shapes if s != world.noise_shape])))
    if move == "swap":
        fakes[int(rng.integers(0, len(fakes)))] = unused[int(rng.integers(0, len(unused)))]
    elif move == "add":
        fakes.append(unused[int(rng.integers(0, len(unused)))])
    elif move == "drop":
        fakes.pop(int(rng.integers(0, len(fakes))))
    listed = (not world.truth_listed) if move == "truth" else world.truth_listed
    return replace(world, candidates=_candidates(entry, fakes, listed, rng), truth_listed=listed)


def keep_worst(scored: Sequence[tuple[float, World]], keep: int) -> list[tuple[float, World]]:
    """The ``keep`` distinct worlds with the worst outcomes, worst first (ties: the earlier one)."""
    seen: set[World] = set()
    out: list[tuple[float, World]] = []
    for badness, world in sorted(scored, key=lambda pair: -pair[0]):
        if world not in seen:
            seen.add(world)
            out.append((badness, world))
        if len(out) == keep:
            break
    return out


def climb(start: Sequence[World], badness: Callable[[World], float], arena: Arena, laws: Mapping[str, Mapping[str, Any]],
          rounds: int, children: int = 4, keep: int = 8, seed: int = 0) -> list[tuple[float, World]]:
    """Worlds changed step by step towards a worse outcome; the worst ``keep`` after ``rounds`` rounds.

    One process and one world at a time: the reference version. A campaign runs the same
    selection with its worlds evaluated side by side.
    """
    rng = np.random.default_rng(seed)
    held = keep_worst([(badness(world), world) for world in start], keep)
    for _ in range(rounds):
        tried = [change(world, arena, laws, rng) for _score, world in held for _ in range(children)]
        known = {world for _score, world in held}
        held = keep_worst([*held, *((badness(world), world) for world in tried if world not in known)], keep)
    return held
