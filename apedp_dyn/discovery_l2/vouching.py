"""What a route can vouch for: a law is reported only if an error of the size it would be blamed for could have been seen.

Status: experimental

**Why this module exists.** The red-team campaign of 2026-11-22
(``benchmarks/red_team_campaign.json``) searched for data on which the route in
force reports a wrong law. It found two weaknesses, and the blind runs of the
same day showed both on data nobody had designed:

* **When candidates cannot be told apart on the rows the path may measure, it
  reports the one with the shortest description.** Nearly every wrong law of
  the campaign came out of such a class, and so did the approximations that
  make up most of what the route reports at 1 % noise.
* **The path's last check is weak.** It asks whether each residual can be
  predicted from its eight nearest rows' residuals better than from nothing.
  For noise alone that statistic is about -1/8, and the check passes anything
  up to 0, so it only sees structure worth an eighth of the residual variance:
  an error a third the size of the noise goes through.

**What is done here**, to a law the route is about to report and to the
candidates it could not tell from it:

1. *A structure test with a known null* (``structure_test``). The same
   statistic at three neighbourhood sizes, each compared with its own
   distribution when the signs of the residuals are flipped at random, which is
   what residuals of a right law with errors symmetric about zero look like,
   whatever their sizes. A law whose residuals are structured at the canonical
   test level is refused.
2. *The class rule.* Members of an unseparated class that pass the test must
   agree with each other to the accuracy a reported law is held to. If they
   do, the simplest is reported, as before. If they do not, the data could not
   tell materially different laws apart and nothing is reported: the class is
   named instead.
3. *The power rule* (``power_to_see``). An error just large enough to make the
   law count as wrong is planted in its own residuals, in the shape of each
   rival's difference from it and of low-order polynomials of the inputs, and
   the test is run again. If it would have missed such an error more often
   than the canonical power minimum allows, the route cannot vouch for the law
   at that accuracy, and does not report it. The same goes if the constants
   alone cannot be fitted that accurately from these rows at this noise.

Every refusal carries its reason. Nothing here reads the truth, a test set or
rows the path has not already been given.

**What this cannot do.** It can only withhold a report or pass it on to
another member of the same class; it finds no law. A right law is refused at
the test's level, one time in a hundred. Errors that are not symmetric about
zero, or that depend on each other from row to row, break the null. An error
smaller than the accuracy line is, by that line, not an error. And every
setting here was chosen on the campaign's worlds, whose answers were known:
what it does on data nobody designed is for a run fixed in advance to say.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import sympy as sp

from apedp_dyn.conf import canonical_spec

#: test R^2 a reported law is held to by the benchmarks (SRBench's accuracy criterion); below it a law counts as wrong
ACCURACY_R2 = 0.999
LEVEL = float(canonical_spec.DIEBOLD_MARIANO_P_VALUE_THRESHOLD)        # the level at which structure refuses a law
POWER = float(canonical_spec.TEMPORAL_RATE_CALIBRATION_POWER_MIN)      # how often a planted error has to be seen
SCALES = (8, 32, 128)             # neighbours a residual is predicted from: the path's own, and two coarser
FLIPS = 2047                      # random sign patterns behind each null distribution
TRIALS = 32                       # times each shape of error is planted
NUMERIC_FLOOR = 1e-6              # relative residual below which a fit is exact (the path's own floor)
MAX_ROWS = 1500                   # every pair of rows is compared; larger samples are subsampled, as the path does


@dataclass
class Checked:
    """One candidate after the structure test."""

    law: str
    usable: bool                               #: finite on every row
    exact: bool = False                        #: fits to numerical precision
    p_value: float = 1.0                       #: smallest over the scales, corrected for their number
    refused: bool = False
    scales: list[dict[str, float]] = field(default_factory=list)
    size: tuple[int, int] = (0, 0)             #: operations and constants: what "simplest" goes by here


@dataclass
class Vouched:
    """What is reported after the checks, and why."""

    law: str | None
    reason: str                                #: "" when the route's own law is reported unchanged
    switched: bool = False                     #: another member of the class is reported in its place
    members: list[Checked] = field(default_factory=list)
    power: float | None = None                 #: the least chance, over the shapes tried, of seeing a planted error
    record: dict[str, Any] = field(default_factory=dict)


def _symbols(names: Sequence[str]) -> list[sp.Symbol]:
    return [sp.Symbol(name) for name in names]


def _values(text: str, names: Sequence[str], X: np.ndarray) -> np.ndarray | None:
    try:
        expr = sp.sympify(text, locals={name: sp.Symbol(name) for name in names})
        fn = sp.lambdify(_symbols(names), expr, "numpy")
        with np.errstate(all="ignore"):
            out = np.asarray(fn(*X.T), dtype=float) * np.ones(len(X))
    except (sp.SympifyError, TypeError, ValueError, ArithmeticError, NameError, AttributeError):
        return None
    return out if np.all(np.isfinite(out)) else None


def _size(text: str, names: Sequence[str]) -> tuple[int, int]:
    try:
        expr = sp.sympify(text, locals={name: sp.Symbol(name) for name in names})
    except (sp.SympifyError, TypeError, ValueError):
        return (10 ** 6, 10 ** 6)
    return int(sp.count_ops(expr)), len(expr.atoms(sp.Float))


def differences(X: np.ndarray, scales: Sequence[int] = SCALES) -> list[tuple[int, np.ndarray]]:
    """For each scale, the matrix that turns residuals into residual minus the mean of the nearest rows' residuals."""
    Z = (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), 1e-12)
    d2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(axis=2)
    np.fill_diagonal(d2, np.inf)
    order = np.argsort(d2, axis=1)
    n = len(X)
    out: list[tuple[int, np.ndarray]] = []
    for scale in scales:
        k = max(1, min(int(scale), n // 4))
        if out and out[-1][0] == k:
            continue
        matrix = np.eye(n)
        matrix[np.repeat(np.arange(n), k), order[:, :k].ravel()] -= 1.0 / k
        out.append((k, matrix))
    return out


def _statistic(matrix: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    """The path's own statistic for each column: leave-one-out R^2 of a residual from its neighbours' residuals."""
    centred = residuals - residuals.mean(axis=0)
    out: np.ndarray = 1.0 - np.sum((matrix @ residuals) ** 2, axis=0) / np.maximum(np.sum(centred ** 2, axis=0), 1e-300)
    return out


def structure_test(residuals: np.ndarray, operators: Sequence[tuple[int, np.ndarray]], *, flips: int = FLIPS,
                   seed: int = 0) -> tuple[float, list[dict[str, float]]]:
    """Whether the residuals are structured: the smallest p over the scales, corrected for their number.

    For each scale the statistic is compared with its values when the residuals' signs are flipped
    at random. Each scale's record also carries the value above which that scale alone refuses.
    """
    rng = np.random.default_rng(seed)
    flipped = rng.choice([-1.0, 1.0], size=(len(residuals), flips)) * residuals[:, None]
    scales: list[dict[str, float]] = []
    smallest = 1.0
    for k, matrix in operators:
        seen = float(_statistic(matrix, residuals[:, None])[0])
        null = _statistic(matrix, flipped)
        p = (1 + int(np.sum(null >= seen))) / (flips + 1)
        critical = float(np.quantile(null, 1.0 - LEVEL / len(operators)))
        scales.append({"neighbours": float(k), "statistic": seen, "null_mean": float(null.mean()),
                       "null_sd": float(null.std()), "critical": critical, "p": p})
        smallest = min(smallest, p)
    return min(1.0, smallest * len(operators)), scales


def _shapes(X: np.ndarray, rivals: Sequence[np.ndarray]) -> list[np.ndarray]:
    """Shapes an error could take: each rival's difference from the law, then low-order polynomials of the inputs."""
    Z = (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), 1e-12)
    columns = [*rivals, *(Z[:, i] for i in range(Z.shape[1])), *(Z[:, i] ** 2 for i in range(Z.shape[1])),
               *(Z[:, i] * Z[:, j] for i in range(Z.shape[1]) for j in range(i + 1, Z.shape[1]))]
    shapes = []
    for column in columns:
        centred = column - column.mean()
        spread = float(centred.std())
        if np.isfinite(spread) and spread > 0.0:
            shapes.append(centred / spread)
    return shapes


def power_to_see(residuals: np.ndarray, X: np.ndarray, operators: Sequence[tuple[int, np.ndarray]],
                 scales: Sequence[dict[str, float]], amplitude: float, rivals: Sequence[np.ndarray] = (), *,
                 trials: int = TRIALS, seed: int = 0) -> tuple[float, list[float]]:
    """How often the structure test would have seen an error of root mean square ``amplitude``.

    The error is planted in the residuals with their signs flipped afresh, once per trial and shape.
    Returns the least chance over the shapes and every shape's chance.
    """
    shapes = _shapes(X, rivals)
    if not shapes:
        return 0.0, []
    rng = np.random.default_rng([seed, 1])
    chances = []
    for shape in shapes:
        signs = rng.choice([-1.0, 1.0], size=(len(residuals), trials))
        planted = signs * residuals[:, None] + amplitude * (shape[:, None] * rng.choice([-1.0, 1.0], size=(1, trials)))
        seen: np.ndarray = np.zeros(trials, dtype=bool)
        for (_, matrix), scale in zip(operators, scales, strict=True):
            seen |= _statistic(matrix, planted) > scale["critical"]
        chances.append(float(seen.mean()))
    return min(chances), chances


def vouch(law: str, others: Sequence[str], names: Sequence[str], X: np.ndarray, y: np.ndarray,
          X_more: np.ndarray | None = None, *, accuracy: float = ACCURACY_R2, level: float = LEVEL, power: float = POWER,
          flips: int = FLIPS, trials: int = TRIALS, seed: int = 0) -> Vouched:
    """The law to report in place of ``law``, or none, with the reason.

    ``others`` are the candidates the path could not tell from ``law``. ``X`` and ``y`` are the rows
    the law was fitted on; ``X_more`` are further inputs (no targets) on which members of a class
    are compared with each other.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(y) > MAX_ROWS:
        keep = np.sort(np.random.default_rng(seed).choice(len(y), MAX_ROWS, replace=False))
        X, y = X[keep], y[keep]
    box = X if X_more is None or not len(X_more) else np.vstack([X, np.asarray(X_more, dtype=float)])
    operators = differences(X)
    scale = float(np.sqrt(np.mean(y ** 2))) or 1.0
    members: list[Checked] = []
    fitted: dict[str, np.ndarray] = {}
    for text in dict.fromkeys([law, *others]):
        values = _values(text, names, X)
        member = Checked(text, values is not None, size=_size(text, names))
        members.append(member)
        if values is None:
            member.refused = True
            continue
        fitted[text] = values
        residuals = y - values
        if float(np.sqrt(np.mean(residuals ** 2))) < NUMERIC_FLOOR * scale:
            member.exact = True
            continue
        member.p_value, member.scales = structure_test(residuals, operators, flips=flips, seed=seed)
        member.refused = member.p_value < level
    record: dict[str, Any] = {"members": len(members), "refused": sum(m.refused for m in members), "level": level,
                              "accuracy": accuracy, "rows": len(y)}
    kept = sorted((m for m in members if not m.refused), key=lambda m: (m.size, members.index(m)))
    if not kept:
        return Vouched(None, "every candidate left structure in its residuals", members=members, record=record)
    chosen = kept[0] if members[0].refused else members[0]
    on_box = {m.law: _values(m.law, names, box) for m in kept}
    mine = on_box[chosen.law]
    if mine is None:
        return Vouched(None, "the law is not finite on the inputs at hand", members=members, record=record)
    spread = float(np.var(mine))
    apart = []
    for member in kept:
        theirs = on_box[member.law]
        if member is chosen:
            continue
        if theirs is None or float(np.mean((theirs - mine) ** 2)) > (1.0 - accuracy) * max(spread, 1e-300):
            apart.append(member.law)
    record["apart"] = len(apart)
    if apart:
        return Vouched(None, f"{len(apart) + 1} candidates passed every check and differ by more than the accuracy claimed",
                       members=members, record={**record, "class": [chosen.law, *apart]})
    switched = chosen is not members[0]
    if chosen.exact:
        return Vouched(chosen.law, "another member of the class is reported: the first left structure" if switched else "",
                       switched, members, 1.0, record)
    residuals = y - fitted[chosen.law]
    signal = float(np.var(fitted[chosen.law]))
    allowed = (1.0 - accuracy) * signal                               # the squared error at which a law counts as wrong
    fit_error = chosen.size[1] * float(np.var(residuals)) / len(y)    # what fitting its constants on these rows costs
    record.update({"allowed_error": allowed, "fit_error": fit_error})
    if fit_error >= allowed:
        return Vouched(None, "at this noise and with these rows its constants cannot be fitted to the accuracy claimed",
                       members=members, record=record)
    rivals = [fitted[m.law] - fitted[chosen.law] for m in members if m.law in fitted and m is not chosen]
    least, chances = power_to_see(residuals, X, operators, chosen.scales, float(np.sqrt(allowed)), rivals, trials=trials,
                                  seed=seed)
    record["chances"] = [round(c, 3) for c in chances]
    if least < power:
        return Vouched(None, f"an error large enough to make it wrong would have been seen {least:.0%} of the time, "
                             f"not {power:.0%}", members=members, power=least, record=record)
    return Vouched(chosen.law, "another member of the class is reported: the first left structure" if switched else "",
                   switched, members, least, record)
