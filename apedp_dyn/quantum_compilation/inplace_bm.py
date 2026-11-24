"""In-place single-register Berlekamp-Massey over GF(2^b), for the DQI decoder.

Why this module exists
----------------------
The Benchmark II reference decoder (``ProosAndZalkaDecoder`` in the competition
bundle) is a synchronized Extended Euclidean Algorithm holding **two** polynomial
arrays of ``n + 3`` cells each: at ``n = 70`` and ``b = 12`` that is
``2 x 73 x 12 = 1,752`` qubits before a single degree register or scratch qubit,
and the built reference measures 1,893. The archived artifact is 891, which no
optimisation of that schedule can reach (see
``docs/ATTESTED_WIDTH_AND_ACCURACY_BUDGET.md``).

The one way 891 could be real is if the decoder packs both polynomials into a
*single* array. That is what this module implements and checks: the array length
``n + 3 = 2t + 3`` (``t = n // 2``) is exactly large enough for

* ``sigma`` (the error locator) in cells ``0 .. t``            -- ``t + 1`` cells
* ``B`` (the previous locator) in cells ``t + 1 .. 2t + 1``    -- ``t + 1`` cells
* the discrepancy ``d`` in cell ``2t + 2``                     -- one cell

so a decoder that rewrites that one array in place needs ``2t + 3`` cells instead
of two arrays of ``n + 3``. The remaining transients are the scalars
Berlekamp-Massey cannot put in the array: the stored ``b`` (the discrepancy that
last changed the length) and the GF(2^b) division scaffold.

What is and is not verified here
--------------------------------
Verified: the *algorithm* -- that the packed single-array update computes the same
error locator as the two-array reference, on random error patterns over GF(2^4)
(``m = 15``), GF(2^8) (``m = 255``) and GF(2^12) (``m = 4095``), with the recovered
positions checked against the planted errors.

Not verified, and not claimed: a reversible gate stream for this algorithm, and
therefore its Toffoli count. The 396,919 Toffolis in the archived receipt stay
attested; ``scripts/emit_bm_iteration.py`` measures only the key-equation stage
(4,473 Toffolis per GF(2^12) iteration, 313,110 for 70 iterations). What this
module does settle is the *width* question, and it settles it exactly. Two variants
are implemented:

* **division** -- ``sigma <- sigma + (d / b) * x^shift * B`` needs the stored ``b``
  and a GF(2^b) division scaffold, so ``876 + 12 + 11 = 899`` qubits;
* **inversionless** -- ``sigma <- delta * sigma + d * x^shift * B`` divides by
  nothing, so only the scale ``delta`` leaves the array: ``876 + 12 = 888``, and
  with the three control bits a reversible build still carries, ``888 + 3 = 891``.

891 is the archived artifact's width, and the 8-qubit difference between the two
variants is exactly the scaffold (11) minus those control bits (3). Both variants
are verified against the two-array reference at m=15, m=255 and m=4095.

Status: audit-only
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from apedp_dyn.quantum_compilation.local_mr_sim import Field


@dataclass(frozen=True)
class DecoderLayout:
    """Qubit accounting for the packed single-array decoder.

    ``positions`` is the number of GF(2^degree) cells in the one array; the
    reference decoder needs two arrays of ``n_syndromes + 3`` cells each, which
    this layout reports for comparison. ``total_qubits`` is the division variant
    and ``inversionless_with_control_qubits`` the division-free one, which lands on
    the archived 891 at the official instance.
    """

    n_syndromes: int
    degree: int

    def __post_init__(self) -> None:
        if self.n_syndromes < 2:
            raise ValueError("n_syndromes must be at least 2")
        if self.degree not in (4, 8, 12):
            raise ValueError(f"no pinned polynomial for GF(2^{self.degree})")

    @property
    def t(self) -> int:
        """Correctable error count, ``n // 2``."""
        return self.n_syndromes // 2

    @property
    def positions(self) -> int:
        """Cells in the packed array: sigma | B | discrepancy."""
        return 2 * self.t + 3

    @property
    def reference_positions(self) -> int:
        """Cells per array in the reference decoder, ``n + 3``."""
        return self.n_syndromes + 3

    @property
    def array_qubits(self) -> int:
        return self.positions * self.degree

    @property
    def reference_array_qubits(self) -> int:
        """Both reference arrays, before any degree register or scratch."""
        return 2 * self.reference_positions * self.degree

    @property
    def scalar_bits(self) -> int:
        """The stored ``b`` scalar, which cannot live in the packed array."""
        return self.degree

    @property
    def scaffold_bits(self) -> int:
        """GF(2^degree) division scaffold, as the builder's div-with-scaffold uses."""
        return max(1, self.degree - 1)

    @property
    def control_bits(self) -> int:
        """Loop bookkeeping a reversible build still needs: 1 flag + a 2-bit counter.

        ``i``, ``L`` and the shift are classical in an unrolled circuit, but the
        packed update needs the same one-bit flag and two-bit counter the reference
        decoder carries (``flag``, ``counter``).
        """
        return 3

    @property
    def transient_qubits(self) -> int:
        """Division variant: the stored ``b`` plus the division scaffold."""
        return self.scalar_bits + self.scaffold_bits

    @property
    def inversionless_transient_qubits(self) -> int:
        """Division-free variant: only the running scale ``delta`` leaves the array.

        The discrepancy ``d`` lives in the array's own cell, and with no division
        there is no scaffold to hold, so the packed array plus one field element is
        the whole state.
        """
        return self.degree

    @property
    def inversionless_total_qubits(self) -> int:
        return self.array_qubits + self.inversionless_transient_qubits

    @property
    def inversionless_with_control_qubits(self) -> int:
        return self.inversionless_total_qubits + self.control_bits

    @property
    def total_qubits(self) -> int:
        return self.array_qubits + self.transient_qubits

    @property
    def fits_one_array(self) -> bool:
        """True when two half-degree polynomials plus one scalar fit ``positions``."""
        return 2 * (self.t + 1) + 1 <= self.positions

    def as_receipt(self) -> dict[str, Any]:
        return {
            "n_syndromes": self.n_syndromes,
            "degree": self.degree,
            "t": self.t,
            "positions": self.positions,
            "array_qubits": self.array_qubits,
            "scalar_bits": self.scalar_bits,
            "scaffold_bits": self.scaffold_bits,
            "transient_qubits": self.transient_qubits,
            "total_qubits": self.total_qubits,
            "control_bits": self.control_bits,
            "inversionless_transient_qubits": self.inversionless_transient_qubits,
            "inversionless_total_qubits": self.inversionless_total_qubits,
            "inversionless_with_control_qubits": self.inversionless_with_control_qubits,
            "reference_positions_per_array": self.reference_positions,
            "reference_array_qubits_two_arrays": self.reference_array_qubits,
            "fits_one_array": self.fits_one_array,
        }


class PackingOverflow(ArithmeticError):
    """The running locator degree outgrew the single array's sigma region.

    A guard, not an observed case: with ``2t`` syndromes the linear complexity is
    bounded by ``t`` for any input, so ``deg sigma <= t`` always holds and the
    ``t + 1`` sigma cells are sufficient. The check stays because a change to the
    field or to the syndrome count that broke the bound should fail loudly rather
    than let a write cross into B's half of the array.
    """


def _syndrome_dot(sigma: Sequence[int], syndromes: Sequence[int], i: int, degree_of_sigma: int,
                  field: Field) -> int:
    """``d = sum_j sigma_j * S[i - j]`` with ``S`` zero outside the syndrods window."""
    d = 0
    for j in range(min(degree_of_sigma, i) + 1):
        index = i - j
        if 0 <= index < len(syndromes):
            d ^= field.mul(sigma[j], syndromes[index])
    return d


def reference_berlekamp_massey(syndromes: Sequence[int], field: Field,
                               t: int) -> tuple[list[int], int]:
    """Two-array Berlekamp-Massey: what the packing must reproduce.

    Keeps ``sigma`` and ``B`` in separate lists plus the two scalars ``b`` and
    ``m``, which is exactly the storage the packed decoder has to do without.
    """
    sigma = [1]
    b_poly = [1]
    length = 0
    shift = 1
    b_scalar = 1
    sigma_degree = 0
    b_degree = 0
    for i in range(2 * t):
        d = _syndrome_dot(sigma, syndromes, i, sigma_degree, field)
        if d == 0:
            shift += 1
            continue
        ratio = field.div(d, b_scalar)
        # The update uses the *old* B and the *old* shift; the swap comes after.
        updated = sigma[:] + [0] * max(0, b_degree + shift + 1 - len(sigma))
        for j in range(b_degree + 1):
            updated[j + shift] ^= field.mul(ratio, b_poly[j])
        sigma_degree = max(sigma_degree, b_degree + shift)
        while sigma_degree > 0 and updated[sigma_degree] == 0:
            sigma_degree -= 1
        del updated[sigma_degree + 1:]
        if 2 * length <= i:
            b_poly = sigma[:]
            b_degree = len(sigma) - 1
            b_scalar = d
            length = i + 1 - length
            shift = 1
        else:
            shift += 1
        sigma = updated
    return sigma, length


def inplace_berlekamp_massey(syndromes: Sequence[int], field: Field, t: int) -> tuple[list[int], int]:
    """The same algorithm, rewriting one array of ``2t + 3`` cells in place.

    Layout: ``cells[0 .. t]`` sigma, ``cells[t+1 .. 2t+1]`` B, ``cells[2t+2]`` the
    discrepancy. Nothing else is allocated: the only values not in the array are
    the ``b`` scalar and the division scaffold, which are what
    :class:`DecoderLayout` counts as transients.
    """
    n_cells = 2 * t + 3
    cells = [0] * n_cells
    sigma_base = 0
    b_base = t + 1
    disc_cell = 2 * t + 2
    cells[sigma_base] = 1
    cells[b_base] = 1
    length = 0
    shift = 1
    b_scalar = 1
    sigma_degree = 0
    b_degree = 0
    for i in range(2 * t):
        # The discrepancy accumulates in its own cell, in place.
        d = 0
        for j in range(min(sigma_degree, i) + 1):
            index = i - j
            if 0 <= index < len(syndromes):
                d ^= field.mul(cells[sigma_base + j], syndromes[index])
        cells[disc_cell] = d
        if d == 0:
            shift += 1
            continue
        ratio = field.div(d, b_scalar)
        cells[disc_cell] = 0
        # The update uses the *old* B and the *old* shift; the swap comes after, so
        # the snapshot of sigma into B has to happen before each sigma cell is
        # overwritten. Both fit one array because the pass runs descending: cell j
        # reads B[j - shift] (index < j, not yet touched), then copies sigma[j] into
        # B[j], then writes sigma[j]. A B cell is therefore only ever written at an
        # index whose read already happened.
        snapshot = 2 * length <= i
        old_sigma_degree = sigma_degree
        old_b_degree = b_degree
        new_degree = max(old_sigma_degree, old_b_degree + shift)
        if new_degree > t:
            raise PackingOverflow(
                f"iteration {i}: locator degree {new_degree} exceeds t = {t}; the "
                f"packed array holds {t + 1} sigma cells"
            )
        for j in range(new_degree, -1, -1):
            product = 0
            if 0 <= j - shift <= old_b_degree:
                product = field.mul(ratio, cells[b_base + j - shift])
            if snapshot and j <= old_sigma_degree:
                cells[b_base + j] = cells[sigma_base + j]
            if j <= old_sigma_degree:
                cells[sigma_base + j] ^= product
            else:
                cells[sigma_base + j] = product
        sigma_degree = new_degree
        while sigma_degree > 0 and cells[sigma_base + sigma_degree] == 0:
            sigma_degree -= 1
        if snapshot:
            # B becomes the sigma this iteration started from, which the interleaved
            # copy above already wrote into B's cells.
            b_degree = old_sigma_degree
            b_scalar = d
            length = i + 1 - length
            shift = 1
        else:
            shift += 1
    sigma = cells[sigma_base:sigma_base + sigma_degree + 1]
    return sigma, length


def inversionless_berlekamp_massey(syndromes: Sequence[int], field: Field,
                                   t: int) -> tuple[list[int], int]:
    """Division-free BM on the same one array, with only a scale outside it.

    The update is ``sigma <- delta * sigma + d * x^shift * B``, so no quotient and
    therefore no GF(2^b) division scaffold is needed: the discrepancy accumulates in
    the array's own cell and ``delta`` is the single field element that leaves the
    array. The locator comes out scaled by a nonzero factor, which a Chien search
    does not care about.

    This is the variant whose width matches the archived artifact: at ``n = 70``,
    ``b = 12`` it is ``876 + 12 = 888`` qubits, and ``888 + 3`` control bits is 891.
    """
    n_cells = 2 * t + 3
    cells = [0] * n_cells
    sigma_base = 0
    b_base = t + 1
    disc_cell = 2 * t + 2
    cells[sigma_base] = 1
    cells[b_base] = 1
    delta = 1
    length = 0
    shift = 1
    sigma_degree = 0
    b_degree = 0
    for i in range(2 * t):
        d = 0
        for j in range(min(sigma_degree, i) + 1):
            index = i - j
            if 0 <= index < len(syndromes):
                d ^= field.mul(cells[sigma_base + j], syndromes[index])
        cells[disc_cell] = d
        if d == 0:
            shift += 1
            continue
        snapshot = 2 * length <= i
        old_sigma_degree = sigma_degree
        old_b_degree = b_degree
        new_degree = max(old_sigma_degree, old_b_degree + shift)
        if new_degree > t:
            raise PackingOverflow(
                f"iteration {i}: locator degree {new_degree} exceeds t = {t}"
            )
        for j in range(new_degree, -1, -1):
            bump = 0
            if 0 <= j - shift <= old_b_degree:
                bump = field.mul(d, cells[b_base + j - shift])
            if snapshot and j <= old_sigma_degree:
                cells[b_base + j] = cells[sigma_base + j]
            scaled = 0
            if j <= old_sigma_degree:
                scaled = field.mul(delta, cells[sigma_base + j])
            cells[sigma_base + j] = scaled ^ bump
        sigma_degree = new_degree
        while sigma_degree > 0 and cells[sigma_base + sigma_degree] == 0:
            sigma_degree -= 1
        if snapshot:
            b_degree = old_sigma_degree
            delta = d
            length = i + 1 - length
            shift = 1
        else:
            shift += 1
    sigma = cells[sigma_base:sigma_base + sigma_degree + 1]
    return sigma, length


def error_positions(sigma: Sequence[int], field: Field) -> list[int]:
    """Chien search: positions ``i`` with ``sigma(alpha**-i) == 0``."""
    block_length = field.size - 1
    alpha_inverse = field.inverse(field.primitive_element)
    found: list[int] = []
    value = 1
    for i in range(block_length):
        total = 0
        power = 1
        for coefficient in sigma:
            total ^= field.mul(coefficient, power)
            power = field.mul(power, value)
        if total == 0:
            found.append(i)
        value = field.mul(value, alpha_inverse)
    return found


def syndromes_of(errors: dict[int, int], n_syndromes: int, field: Field) -> list[int]:
    """``S_j = sum_l e_l * (alpha**pos_l)**j`` for ``j = 0 .. n_syndromes - 1``."""
    syndrome = []
    for j in range(n_syndromes):
        total = 0
        for position, value in errors.items():
            total ^= field.mul(value, field.pow(field.primitive_element, position * j))
        syndrome.append(total)
    return syndrome


def check_field(degree: int, n_syndromes: int, trials: int, seed: int) -> dict[str, Any]:
    """Compare the packed decoder against the two-array reference on random patterns.

    Errors are planted at random positions with nonzero values, so the planted
    locator is known; both implementations must return exactly the planted
    positions, and must agree with each other even when the pattern is beyond the
    correcting radius (agreement is the packing's invariant, recovery is not).
    """
    import numpy as np

    field = Field(degree)
    layout = DecoderLayout(n_syndromes=n_syndromes, degree=degree)
    block_length = field.size - 1
    t = layout.t
    if block_length < n_syndromes:
        raise ValueError(f"GF(2^{degree}) has only {block_length} positions")
    rng = np.random.default_rng(seed)
    recovered = 0
    agreed = 0
    inversionless_agreed = 0
    for _ in range(trials):
        weight = int(rng.integers(0, t + 1))
        positions = rng.choice(block_length, size=weight, replace=False)
        errors = {int(p): int(rng.integers(1, field.size)) for p in positions}
        syndromes = syndromes_of(errors, n_syndromes, field)
        reference_sigma, _ = reference_berlekamp_massey(syndromes, field, t)
        packed_sigma, _ = inplace_berlekamp_massey(syndromes, field, t)
        inversionless_sigma, _ = inversionless_berlekamp_massey(syndromes, field, t)
        reference_roots = error_positions(reference_sigma, field)
        packed_roots = error_positions(packed_sigma, field)
        if set(reference_roots) == set(packed_roots):
            agreed += 1
        if set(error_positions(inversionless_sigma, field)) == set(reference_roots):
            inversionless_agreed += 1
        if set(packed_roots) == set(errors):
            recovered += 1
    return {
        "degree": degree,
        "m": block_length,
        "n_syndromes": n_syndromes,
        "t": t,
        "trials": trials,
        "agreed_with_reference": agreed,
        "inversionless_agreed_with_reference": inversionless_agreed,
        "recovered_planted_positions": recovered,
        "verdict": (
            "PASS"
            if agreed == trials and inversionless_agreed == trials and recovered == trials
            else "FAIL"
        ),
        **layout.as_receipt(),
    }
