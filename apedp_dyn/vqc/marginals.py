"""Exact Z-basis marginals of a stabilizer specification, at any width.

Status: experimental

The sampling comparison was refereed by the **exact full distribution**, which is
enumerable only up to 12 qubits: 2**12 outcomes are readable, 2**24 are not. That put a
ceiling on the comparison that had nothing to do with either sampler. This module removes
the ceiling by computing the exact distribution of a *subset* of wires, which stays small
even when the state does not.

Why a marginal is exact and cheap
---------------------------------
For a pure stabilizer state, the Z-basis distribution is **uniform over an affine subspace**
of GF(2)^n. Expanding `prod (I + g_i) / 2` and taking the diagonal element gives a character
sum over the subsets of generators whose X parts cancel; the sum is zero or `2**d`, so every
supported outcome has probability `2**(d - n)` with `d` the dimension of that subspace (the
dimension of the kernel of the X-part map). Marginalising is then the marginal of a uniform
distribution over an affine set: for each outcome on the subset, the complement either has
`2**((n-k) - rank(B_S))` solutions or none, and "none" is a linear condition on the subset's
outcome -- one per dependent row of the complement's part of the constraint matrix.

So the budget is Gaussian elimination over GF(2) -- `O(n**3)` bits at 1,000 qubits -- plus
enumerating `2**|T|` outcomes, and nothing exponential in the *state's* width.

What this referee can and cannot see
------------------------------------
A marginal over `k` wires is a projection of the distribution, not the distribution: a
sampler could be wrong in a way that no chosen subset exposes. `Marginal.is_uniform`
names the sharpest case -- a marginal that is uniform over every outcome cannot separate
two states that both have it, and every graph state is uniform in the Z basis, so no
Z-basis marginal tells one graph state from another. It still tests the sampler: an
empirical distribution that is not the uniform one it should be fails.

:func:`adjudicate_marginal` therefore tests several subsets, reports each one's
chi-square p-value and total-variation distance separately, and Bonferroni-corrects the
family verdict for the number of subsets tested rather than quoting the best one. That
correction is recorded per subset as the product it is, not applied silently.

The complement is traced out here by linear algebra over the *specification*, so the result
does not depend on either sampler, on a circuit, or on the sampling code path.
"""
from __future__ import annotations

import dataclasses
import math
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from apedp_dyn.vqc.external import Sampling
from apedp_dyn.vqc.synthesis import I, TargetSpecification, X, Y, Z

__all__ = [
    "Marginal",
    "MarginalUnsupported",
    "adjudicate_marginal",
    "chi_square_sf",
    "exact_marginal",
    "marginal_counts",
    "marginal_pair",
]

#: The X part and the Z part of each Pauli label, as bits.
_X_BIT = {I: 0, X: 1, Y: 1, Z: 0}
_Z_BIT = {I: 0, X: 0, Y: 1, Z: 1}


class MarginalUnsupported(ValueError):
    """A marginal this referee will not enumerate, with the reason."""


@dataclasses.dataclass(frozen=True)
class Marginal:
    """The exact Z-basis distribution of one subset of wires, and how it was derived."""

    subset: tuple[int, ...]
    num_qubits: int
    target_sha256: str
    probabilities: Mapping[tuple[bool, ...], float]
    #: Dimension of the full distribution's affine support, `n - rank(X parts)`.
    dimension: int
    rank_x: int
    #: The probability of each supported outcome (the distribution is uniform on it).
    probability: float
    seconds: float

    @property
    def support(self) -> int:
        return len(self.probabilities)

    @property
    def cells(self) -> int:
        return 1 << len(self.subset)

    @property
    def is_uniform(self) -> bool:
        """True when every outcome is supported: the marginal carries no information."""
        return self.support == self.cells

    def as_dict(self, *, top: int = 4) -> dict[str, Any]:
        ordered = sorted(self.probabilities.items(), key=lambda item: (-item[1], item[0]))
        return {
            "subset": list(self.subset),
            "subset_size": len(self.subset),
            "qubits": self.num_qubits,
            "target_sha256": self.target_sha256,
            "cells": self.cells,
            "support": self.support,
            "dimension": self.dimension,
            "probability": self.probability,
            "is_uniform": self.is_uniform,
            "rank_x": self.rank_x,
            "seconds": self.seconds,
            "top_outcomes": [
                {"bits": "".join("1" if bit else "0" for bit in bits),
                 "probability": probability}
                for bits, probability in ordered[:top]],
        }


def _echelon(rows: list[int]) -> dict[int, int]:
    """Reduced row echelon form over GF(2): pivot column -> row, as bitmasks.

    The reduction happens in both directions, which is what makes the nullspace below a
    one-liner: a new row is cleared at every existing pivot, and the new pivot is cleared
    from every existing row. A one-directional version leaves pivot bits inside other
    rows, and its "nullspace" is not a nullspace.
    """
    basis: list[tuple[int, int]] = []
    for row in rows:
        for pivot, existing in basis:
            if (row >> pivot) & 1:
                row ^= existing
        if not row:
            continue
        pivot = row.bit_length() - 1
        basis = [(other_pivot, other ^ row if (other >> pivot) & 1 else other)
                 for other_pivot, other in basis]
        basis.append((pivot, row))
        basis.sort(key=lambda item: -item[0])
    return dict(basis)


def _rank(rows: list[int]) -> int:
    return len(_echelon(rows))


def _nullspace(rows: list[int], ncols: int) -> tuple[int, list[int]]:
    """A basis of `{x : rows @ x == 0}` over GF(2), as bitmasks over `ncols` bits.

    Every free column gives one basis vector with that column set and the pivot columns
    solved for from the reduced form.
    """
    echelon = _echelon(rows)
    free = [column for column in range(ncols) if column not in echelon]
    vectors: list[int] = []
    for column in free:
        vector = 1 << column
        for pivot, row in echelon.items():
            if (row >> column) & 1:
                vector |= 1 << pivot
        vectors.append(vector)
    return len(free), vectors


def _bits_spanning(values: list[int], bitmask: int) -> int:
    """The XOR of `values[i]` over the bits set in `bitmask`."""
    total = 0
    index = 0
    while bitmask:
        if bitmask & 1:
            total ^= values[index]
        bitmask >>= 1
        index += 1
    return total


def _restrict(value: int, order: Sequence[int]) -> int:
    """`value` restricted to `order`, renumbered positionally within that order."""
    total = 0
    for position, bit in enumerate(order):
        if (value >> bit) & 1:
            total |= 1 << position
    return total


def _checked_subset(target: TargetSpecification, subset: Sequence[int],
                    *, max_qubits: int) -> tuple[int, ...]:
    wires = tuple(int(wire) for wire in subset)
    if not wires:
        raise MarginalUnsupported("a marginal needs at least one wire")
    if len(set(wires)) != len(wires):
        raise MarginalUnsupported(f"the subset repeats a wire: {wires}")
    if min(wires) < 0 or max(wires) >= target.num_qubits:
        raise MarginalUnsupported(
            f"the subset {wires} is not inside a {target.num_qubits}-qubit state")
    if len(wires) > max_qubits:
        raise MarginalUnsupported(
            f"a {len(wires)}-wire marginal enumerates 2**{len(wires)} outcomes; this "
            f"referee stops at {max_qubits} wires because beyond that the exact "
            f"distribution has more cells than the shot budget can resolve")
    return tuple(sorted(wires))


def exact_marginal(target: TargetSpecification, subset: Sequence[int], *,
                   max_qubits: int = 12) -> Marginal:
    """The exact Z-basis distribution of `subset`, from the specification alone.

    See the module docstring for the derivation. The result is checked to sum to one and
    to have the support size the rank predicts before it is returned -- a wrong marginal
    is a wrong referee, and a silent one would be worse than an abstention.
    """
    start = time.perf_counter()
    wires = _checked_subset(target, subset, max_qubits=max_qubits)
    num_qubits = target.num_qubits
    inside = set(wires)
    outside = [qubit for qubit in range(num_qubits) if qubit not in inside]

    x_parts: list[int] = []
    z_parts: list[int] = []
    signs: list[int] = []
    for labels, phase in target.stabilizers:
        if int(phase) % 2:
            raise MarginalUnsupported(
                f"a stabilizer has phase {phase}, which is not ±1; the specification is "
                f"not a valid stabilizer state")
        x_part = z_part = 0
        for qubit, label in enumerate(labels):
            if _X_BIT[int(label)]:
                x_part |= 1 << qubit
            if _Z_BIT[int(label)]:
                z_part |= 1 << qubit
        x_parts.append(x_part)
        z_parts.append(z_part)
        signs.append(0 if int(phase) % 4 == 0 else 1)
    generators = len(x_parts)
    if generators != num_qubits:
        raise MarginalUnsupported(
            f"the specification has {generators} stabilizers for {num_qubits} qubits; "
            f"a pure state has one per qubit")

    # The full distribution is uniform over {z : B z = c}, where the rows of B are the Z
    # parts of the combinations whose X parts cancel everywhere -- the kernel of the
    # X-part map. Its dimension fixes the probability of every supported outcome.
    x_rows = [sum(1 << index for index in range(generators)
                  if (x_parts[index] >> qubit) & 1)
              for qubit in range(num_qubits)]
    rank_x = _rank(x_rows)
    dimension, combinations = _nullspace(x_rows, generators)
    if dimension != num_qubits - rank_x:  # pragma: no cover - rank-nullity
        raise MarginalUnsupported("the X-part kernel does not have the expected size")
    z_rows = [_bits_spanning(z_parts, combination) for combination in combinations]
    sign_rows = [_bits_spanning(signs, combination) for combination in combinations]

    # Marginalising: sum the uniform distribution over the complement. For each outcome
    # on the subset the complement has 2**((n-k) - rank(B_S)) solutions or none, and
    # "none" is a linear condition on the subset's outcome.
    z_on_subset = [_restrict(row, wires) for row in z_rows]
    z_on_complement = [_restrict(row, outside) for row in z_rows]
    rank_complement = _rank(z_on_complement)
    column_rows = [sum(1 << index for index in range(dimension)
                       if (z_on_complement[index] >> position) & 1)
                   for position in range(len(outside))]
    nullity, weights = _nullspace(column_rows, dimension)
    if nullity != dimension - rank_complement:  # pragma: no cover - rank-nullity
        raise MarginalUnsupported("the complement's null space has the wrong dimension")
    parity_constraints = [(_bits_spanning(z_on_subset, weight),
                           _bits_spanning(sign_rows, weight))
                          for weight in weights]
    probability = float(2.0 ** (-(num_qubits - dimension)
                                + (num_qubits - len(wires)) - rank_complement))

    probabilities: dict[tuple[bool, ...], float] = {}
    for outcome in range(1 << len(wires)):
        if any((((z_part & outcome).bit_count() & 1) != sign)
               for z_part, sign in parity_constraints):
            continue
        probabilities[tuple(bool((outcome >> position) & 1)
                            for position in range(len(wires)))] = probability

    total = sum(probabilities.values())
    expected_support = 1 << (len(wires) - nullity)
    if abs(total - 1.0) > 1e-9 or len(probabilities) != expected_support:
        raise MarginalUnsupported(
            f"the derived marginal supports {len(probabilities)} outcomes, expected "
            f"{expected_support}, and sums to {total}; refusing to report it")
    return Marginal(
        subset=wires,
        num_qubits=num_qubits,
        target_sha256=target.sha256(),
        probabilities=probabilities,
        dimension=dimension,
        rank_x=rank_x,
        probability=probability,
        seconds=time.perf_counter() - start,
    )


def marginal_counts(counts: Mapping[Sequence[bool], int],
                    subset: Sequence[int]) -> dict[tuple[bool, ...], int]:
    """Project a sample table onto `subset`, keeping the subset's wire order."""
    wires = tuple(int(wire) for wire in subset)
    projected: dict[tuple[bool, ...], int] = {}
    width = max((len(bits) for bits in counts), default=0)
    if wires and max(wires) >= width:
        raise MarginalUnsupported(
            f"the sample table is {width} bits wide, the subset reaches {max(wires)}")
    for bits, count in counts.items():
        key = tuple(bool(bits[wire]) for wire in wires)
        projected[key] = projected.get(key, 0) + int(count)
    return projected


# ------------------------------------------------------------------ adjudication
#: The family-wise false-positive rate the marginal block is decided at: one run makes
#: dozens of subset decisions, so the rate is a budget for the whole family rather than
#: for one lucky subset. Bonferroni is the correction, and it is stated with the numbers.
FAMILY_ALPHA = 1e-3

#: A chi-square goodness-of-fit is only quoted where it is valid: every supported cell
#: must have five or more expected counts. Below that the subset abstains by name
#: instead of producing a number nobody should trust.
MIN_EXPECTED_COUNT = 5.0


def _upper_regularised_gamma(a: float, x: float) -> float:
    """``Q(a, x)``, the regularised upper incomplete gamma, to ~1e-14 relative.

    Series for the lower function and a continued fraction for the upper, as in
    Numerical Recipes; the two branches meet around ``x = a + 1`` where both converge
    quickly. No SciPy: the referee's statistics should not depend on an optional wheel.
    """
    if x < 0.0 or a <= 0.0:
        raise ValueError(f"Q({a}, {x}) is not defined")
    if x == 0.0:
        return 1.0
    log_prefix = a * math.log(x) - x - math.lgamma(a)
    if x < a + 1.0:
        term = 1.0 / a
        total = term
        denominator = a
        for _ in range(10_000):
            denominator += 1.0
            term *= x / denominator
            total += term
            if abs(term) < abs(total) * 1e-16:
                break
        return max(0.0, 1.0 - total * math.exp(log_prefix))
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for index in range(1, 10_000):
        an = -index * (index - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-16:
            break
    return min(1.0, math.exp(log_prefix) * h)


def chi_square_sf(statistic: float, dof: int) -> float:
    """``P(X >= statistic)`` for ``X ~ chi-square(dof)``.

    The referee's p-values are this function; it is pinned against textbook critical
    values in the tests rather than assumed.
    """
    if dof < 1:
        raise ValueError(f"a chi-square needs at least one degree of freedom, got {dof}")
    if statistic <= 0.0:
        return 1.0
    return _upper_regularised_gamma(dof / 2.0, statistic / 2.0)


def _goodness_of_fit(counts: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    """The chi-square statistic and p-value for one empirical marginal.

    Cells the exact marginal gives zero probability are excluded (they are counted
    separately as outcomes outside the support); a chi-square over them would divide by
    zero and is not the question. The minimum expected count is reported so a reader can
    see whether the approximation was valid where the verdict was taken.
    """
    supported = expected > 0.0
    statistic = float((((counts[supported] - expected[supported]) ** 2)
                       / expected[supported]).sum())
    dof = int(supported.sum()) - 1
    return {
        "statistic": statistic,
        "dof": dof,
        "p_value": chi_square_sf(statistic, dof) if dof >= 1 else 1.0,
        "min_expected": float(expected[supported].min()) if supported.any() else 0.0,
    }


def _two_sample(counts: np.ndarray, other: np.ndarray) -> dict[str, Any]:
    """The chi-square statistic for two sample sets over the same cells."""
    total = counts + other
    live = total > 0.0
    statistic = float((((2.0 * counts[live] - total[live]) ** 2) / total[live]).sum())
    dof = int(live.sum()) - 1
    return {
        "statistic": statistic,
        "dof": dof,
        "p_value": chi_square_sf(statistic, dof) if dof >= 1 else 1.0,
        "min_expected": float((total[live] / 2.0).min()) if live.any() else 0.0,
    }


def _tv(counts: np.ndarray, shots: int, expected: np.ndarray) -> float:
    """Total variation between the empirical marginal and the exact one (effect size)."""
    return float(0.5 * np.abs(counts / shots - expected).sum())


def _z_scores(counts: np.ndarray, probabilities: np.ndarray,
              shots: int) -> tuple[float, int]:
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(probabilities > 0,
                     (counts - shots * probabilities)
                     / np.sqrt(shots * probabilities * (1.0 - probabilities)),
                     np.where(counts > 0, np.inf, 0.0))
    finite = z[np.isfinite(z)]
    return (float(np.abs(finite).max()) if finite.size else 0.0,
            int((np.abs(finite) > 4.0).sum()))


def _checked_pair(first: Sampling, second: Sampling, target: TargetSpecification) -> None:
    for samples in (first, second):
        if samples.target_sha256 != target.sha256():
            raise ValueError("both sample sets must be drawn from the target")
        if samples.num_qubits != target.num_qubits:
            raise ValueError(
                f"a sample set has {samples.num_qubits} qubits, the target has "
                f"{target.num_qubits}")
    if first.shots != second.shots:
        raise ValueError(
            f"the two sample sets used {first.shots} and {second.shots} shots; a "
            f"matched comparison uses one budget")


def adjudicate_marginal(samples: Sampling, target: TargetSpecification,
                        subsets: Sequence[Sequence[int]], *, alpha: float = FAMILY_ALPHA,
                        top: int = 4) -> dict[str, Any]:
    """Decide one sample set against the exact distribution of several wire subsets.

    Each subset is decided by a chi-square goodness-of-fit against its exact marginal,
    and the family is decided by the Bonferroni-corrected p-value ``min(1, m * p)`` over
    the ``m`` subsets tested. The correction is the point: testing dozens of projections
    and quoting the closest one is how a projection test becomes a flattering number.

    A subset whose supported cells have fewer than :data:`MIN_EXPECTED_COUNT` expected
    counts abstains as `UNRESOLVED` with the count that failed, because a chi-square
    there is not a small error but the wrong test. Total variation is reported as the
    effect size; the verdicts come from the p-values.

    `is_uniform` is a limitation, not a verdict: a uniform marginal cannot tell two
    states with that same marginal apart -- every graph state is uniform in the Z basis,
    so no Z-basis marginal separates one graph state from another. It still tests the
    sampler: an empirical distribution that is not the uniform one it should be fails.
    """
    if samples.target_sha256 != target.sha256():
        raise ValueError(
            "the sample set was drawn from a different specification than the one it "
            "is being adjudicated against")
    if samples.num_qubits != target.num_qubits:
        raise ValueError(
            f"the sample set has {samples.num_qubits} qubits, the target has "
            f"{target.num_qubits}")
    if not subsets:
        raise ValueError("a marginal decision needs at least one subset")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    rows: list[dict[str, Any]] = []
    for subset in subsets:
        marginal = exact_marginal(target, subset)
        projected = marginal_counts(samples.counts, marginal.subset)
        ordered = sorted(marginal.probabilities)
        counts = np.array([projected.get(bits, 0) for bits in ordered], dtype=float)
        exact = np.array([marginal.probabilities[bits] for bits in ordered], dtype=float)
        test = _goodness_of_fit(counts, samples.shots * exact)
        max_abs_z, beyond = _z_scores(counts, exact, samples.shots)
        valid = test["min_expected"] >= MIN_EXPECTED_COUNT
        family_p = min(1.0, test["p_value"] * len(subsets)) if valid else None
        rows.append({
            **marginal.as_dict(top=top),
            "tv": _tv(counts, samples.shots, exact),
            "chi_square": test["statistic"],
            "dof": test["dof"],
            "p_value": test["p_value"],
            "family_p_value": family_p,
            "min_expected": test["min_expected"],
            "verdict": ("CONSISTENT" if family_p >= alpha else "INCONSISTENT")
            if valid else "UNRESOLVED",
            "reason": None if valid else (
                f"{test['min_expected']:.2f} expected counts per supported cell, below "
                f"the {MIN_EXPECTED_COUNT:g} a chi-square needs"),
            "max_abs_z": max_abs_z,
            "cells_beyond_4_sigma": beyond,
            "outcomes_outside_the_exact_support": int(
                sum(1 for bits, count in projected.items()
                    if count > 0 and bits not in marginal.probabilities)),
        })
    payload = samples.as_dict()
    resolved = [row for row in rows if row["verdict"] != "UNRESOLVED"]
    payload.update({
        "referee": "exact Z-basis marginals of the specification",
        "test": "chi-square goodness-of-fit per subset, Bonferroni-corrected over subsets",
        "alpha": alpha,
        "family_size": len(rows),
        "subsets": rows,
        "resolved_subsets": len(resolved),
        "unresolved_subsets": [row["subset"] for row in rows
                               if row["verdict"] == "UNRESOLVED"],
        "verdict": (
            "UNRESOLVED" if not resolved
            else "INCONSISTENT" if any(row["verdict"] == "INCONSISTENT" for row in rows)
            else "CONSISTENT"),
        "uniform_subsets": [row["subset"] for row in rows if row["is_uniform"]],
    })
    return payload


def marginal_pair(first: Sampling, second: Sampling, target: TargetSpecification,
                  subsets: Sequence[Sequence[int]], *, alpha: float = FAMILY_ALPHA
                  ) -> dict[str, Any]:
    """Compare two sample sets on the same subsets, with the exact marginal as the scale.

    The statistic is the two-sample chi-square over the marginal's cells, corrected
    Bonferroni over the subsets tested, so `DISAGREE` means the two engines differ by
    more than sampling noise explains rather than that they are two finite samples.
    """
    _checked_pair(first, second, target)
    if not subsets:
        raise ValueError("a marginal comparison needs at least one subset")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    rows: list[dict[str, Any]] = []
    for subset in subsets:
        marginal = exact_marginal(target, subset)
        ordered = sorted(marginal.probabilities)
        # Project once per sample set, then index: projecting inside the cell loop is
        # quadratic in the outcome count, which is what a wide row cannot afford.
        first_projected = marginal_counts(first.counts, marginal.subset)
        second_projected = marginal_counts(second.counts, marginal.subset)
        first_counts = np.array([first_projected.get(bits, 0) for bits in ordered],
                                dtype=float)
        second_counts = np.array([second_projected.get(bits, 0) for bits in ordered],
                                 dtype=float)
        test = _two_sample(first_counts, second_counts)
        valid = test["min_expected"] >= MIN_EXPECTED_COUNT
        family_p = min(1.0, test["p_value"] * len(subsets)) if valid else None
        rows.append({
            "subset": list(marginal.subset),
            "cells": marginal.cells,
            "support": marginal.support,
            "is_uniform": marginal.is_uniform,
            "tv": float(0.5 * np.abs(first_counts / first.shots
                                     - second_counts / second.shots).sum()),
            "chi_square": test["statistic"],
            "dof": test["dof"],
            "p_value": test["p_value"],
            "family_p_value": family_p,
            "min_expected": test["min_expected"],
            "verdict": ("AGREE" if family_p >= alpha else "DISAGREE")
            if valid else "UNRESOLVED",
            "reason": None if valid else (
                f"{test['min_expected']:.2f} expected counts per cell, below the "
                f"{MIN_EXPECTED_COUNT:g} a chi-square needs"),
        })
    resolved = [row for row in rows if row["verdict"] != "UNRESOLVED"]
    return {
        "engines": [first.tool, second.tool],
        "shots_each": first.shots,
        "target_sha256": target.sha256(),
        "test": "two-sample chi-square per subset, Bonferroni-corrected over subsets",
        "alpha": alpha,
        "family_size": len(rows),
        "subsets": rows,
        "resolved_subsets": len(resolved),
        "verdict": (
            "UNRESOLVED" if not resolved
            else "DISAGREE" if any(row["verdict"] == "DISAGREE" for row in rows)
            else "AGREE"),
    }
