#!/usr/bin/env python3
"""The 54-AND GF(2^12) field multiply, emitted as gates.

`benchmarks/quantum_compilation/designer_directions.json` established, by exhaustive
search, that a GF(2) bilinear product of three bits needs **6** multiplications
(not the 9 schoolbook and not the 9 Karatsuba's uneven split gives), and that two
bits need **3**. Composed by Karatsuba those base cases give a 12-bit product in
`3 * 3 * 6 = 54` ANDs, against the `63` of the Karatsuba recursion `ixor_gf2_mul`
is built from.

That was an algorithm, not a circuit. This module is the circuit: a base-case
table, the Karatsuba composition of it, a classical evaluator for checking it, and
a gate emitter that produces the same instruction stream `ixor_gf2_mul` does.

Why it is cheaper, exactly
--------------------------
`ixor_gf2_mul` accumulates the product *in place* into the target: it needs no
ancillas, so an uncontrolled call costs 63 Toffolis and nothing else. A *controlled*
call cannot be done that way without turning all 63 into three-control gates, so the
built-in pays 63 to compute the product into a 12-qubit temporary, 12 more to add
that in under the control, and gets the temporary's uncomputation for free by
measurement (`gen_gf_unmul`). That is 75 Toffolis per controlled call.

This emitter computes the product into 12 clean qubits with 54 ANDs and 12 clean
qubits' worth of accumulation (CNOTs), adds it in under the control with the same
12 Toffolis, and clears its workspace with `mr` — the Z-basis measure-and-reset,
free and phase-safe on a qubit that is a function of the rest of the state. So a
controlled call costs 54 + 12 = **66** Toffolis, and the multiply that is 59.48% of
the gated arm's executed count gets 12% cheaper rather than 14.3%: the 12-Toffoli
add is in both circuits, so it does not scale with the AND count. Measured, the arm
goes 1,325,230.0 → 1,230,640.0 per shot: 4.320× → **4.652×**.

The AND count is not the only cost. The workspace is larger than the built-in's
12 qubits (scratch windows for the three sub-products at each level), and the CNOT
ladder is longer, because the Karatsuba identity has to be assembled from products
that are no longer accumulated in place. Both are reported by
`scripts/measure_gf54_mul_kickmix.py` rather than assumed.

Usage
-----
As a builder mixin, which is how the measurement scripts swap it in::

    class Builder(Gf54MulMixin, runner.builder_class(km)):
        pass

The mixin replaces `ixor_gf2_mul` and nothing else, so every other instruction the
decoder emits is the known-good one.
"""
from __future__ import annotations

#: Base-case bilinear algorithms, transcribed from the exhaustive search in
#: `scripts/study_circuit_designer_directions.py` (record:
#: `benchmarks/quantum_compilation/designer_directions.json`, `mu(2) = 3`,
#: `mu(3) = 6`, both minima proved). `pairs[t] = (u, v)` means product `t` is
#: `(u . a) * (v . b)`, with bit `i` of `u` selecting `a[i]`; `weights[k][t]` is the
#: coefficient of product `t` in output bit `k`. `tests/test_dyn_mr_gated_arms_kickmix.py`
#: re-derives both tables rather than trusting this copy.
BASE_ALGORITHMS: dict[int, tuple[tuple[tuple[int, int], ...], tuple[tuple[int, ...], ...]]] = {
    # The recursion's own floor: an uneven split leaves a one-bit sub-product.
    1: (
        ((1, 1),),
        ((1,),),
    ),
    2: (
        ((1, 1), (2, 2), (3, 3)),
        ((1, 0, 0), (1, 1, 1), (0, 1, 0)),
    ),
    3: (
        ((1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)),
        ((1, 0, 0, 0, 0, 0),
         (1, 1, 1, 0, 0, 0),
         (1, 1, 0, 1, 1, 0),
         (0, 1, 0, 1, 0, 1),
         (0, 0, 0, 1, 0, 0)),
    ),
}


def and_count(n: int) -> int:
    """ANDs for an n-bit product: the base cases, composed by Karatsuba."""
    if n in BASE_ALGORITHMS:
        return len(BASE_ALGORITHMS[n][0])
    k = (n + 1) // 2
    return 3 * and_count(k)


def terms(bits: tuple[frozenset[int], ...]) -> tuple[list[tuple[frozenset[int], frozenset[int]]],
                                                     list[set[int]]]:
    """The bilinear form of the product over operands that are sums of original bits.

    `bits[i]` is the set of original bit positions whose XOR forms operand bit `i`, so
    a Karatsuba split can represent "the sum of the two halves" without touching the
    recursion's shape. Returns `(products, outputs)`: product `t` is the AND of the
    XORs named by `products[t][0]` and `products[t][1]`, and output bit `k` is the XOR
    of the products in `outputs[k]`.
    """
    n = len(bits)
    if n in BASE_ALGORITHMS:
        pairs, weights = BASE_ALGORITHMS[n]
        products = [(_form(bits, u), _form(bits, v)) for u, v in pairs]
        outputs = [{t for t, c in enumerate(row) if c} for row in weights]
        return products, outputs

    k = (n + 1) // 2
    low_bits, high_bits = bits[:k], bits[k:]
    low_products, low_outputs = terms(low_bits)
    high_products, high_outputs = terms(high_bits)
    # The middle product's operands are the sums of the halves, over the low half's
    # width: where the high half is shorter, the low half's own bits stand unchanged.
    mid_bits = tuple(low_bits[i] ^ high_bits[i] if i < len(high_bits) else low_bits[i]
                     for i in range(k))
    mid_products, mid_outputs = terms(mid_bits)

    products = low_products + high_products + mid_products
    high_base = len(low_products)
    mid_base = len(low_products) + len(high_products)
    outputs: list[set[int]] = [set() for _ in range(2 * n - 1)]

    def place(source: list[set[int]], offset: int, base: int) -> None:
        for i, row in enumerate(source):
            outputs[offset + i] ^= {t + base for t in row}

    # product = low + x^k (mid + low + high) + x^2k high
    place(low_outputs, 0, 0)
    place(low_outputs, k, 0)
    place(mid_outputs, k, mid_base)
    place(high_outputs, k, high_base)
    place(high_outputs, 2 * k, high_base)
    return products, outputs


def _form(bits, selection: int) -> frozenset[int]:
    """The XOR of the operand bits `selection` names, as a set of original positions.

    Over GF(2) a sum of sums is the symmetric difference of their position sets.
    """
    out: frozenset[int] = frozenset()
    for i, form in enumerate(bits):
        if (selection >> i) & 1:
            out = out ^ form
    return out


def _mask(form: frozenset[int]) -> int:
    return sum(1 << i for i in form)


def poly_product(a: int, b: int, n: int) -> int:
    """Carry-less product, via the composed bilinear form (for checking it)."""
    products, outputs = terms(tuple(frozenset([i]) for i in range(n)))
    p = [
        ((bin(_mask(u) & a).count("1") & 1) & (bin(_mask(v) & b).count("1") & 1))
        for u, v in products
    ]
    out = 0
    for k, row in enumerate(outputs):
        bit = 0
        for t in row:
            bit ^= p[t]
        out |= bit << k
    return out


def x_pow_mod(field, i: int) -> int:
    """x^i reduced modulo the field polynomial, via the field's own multiplication."""
    value = 1
    for _ in range(i):
        value = field.mul(value, 2)
    return value


def reduction_map(field, n: int) -> list[int]:
    """Bit `i` of the unreduced product maps to this field element."""
    return [1 << i if i < n else x_pow_mod(field, i) for i in range(2 * n - 1)]


def field_product(a: int, b: int, field, n: int | None = None) -> int:
    """The reduced product, as an integer, from the bilinear form."""
    n = n or field.degree
    poly = poly_product(a, b, n)
    mapping = reduction_map(field, n)
    out = 0
    for i in range(2 * n - 1):
        if (poly >> i) & 1:
            out ^= mapping[i]
    return out


# ---------------------------------------------------------------- gate emission


def _emit_linear(builder, bits):
    """XOR of `bits` into one qubit; returns (qubit, undo) — `undo` restores it."""
    qubits = list(bits)
    if len(qubits) == 1:
        return qubits[0], None
    target = builder.alloc_qubits(1)[0]
    for q in qubits:
        builder.cx(q, target)
    return target, target


def _undo_linear(builder, bits, target) -> None:
    if target is None:
        return
    for q in reversed(list(bits)):
        builder.cx(q, target)
    builder.free(target)


def _emit_base(builder, a, b, out) -> None:
    """Accumulate the base-case product of `a`, `b` into `out`; 6 ANDs at n = 3."""
    products, outputs = terms(tuple(frozenset([i]) for i in range(len(a))))
    product = builder.alloc_qubits(1)[0]
    for t, (u, v) in enumerate(products):
        left_bits = [a[i] for i in sorted(u)]
        right_bits = [b[i] for i in sorted(v)]
        left, left_temp = _emit_linear(builder, left_bits)
        right, right_temp = _emit_linear(builder, right_bits)
        builder.ccx(left, right, product)
        for k in outputs_indexed(outputs, t):
            builder.cx(product, out[k])
        # Measurement-based uncompute: the product qubit holds `left & right`, so
        # clearing it costs nothing and leaves the phase alone.
        builder.del_and(left, right, product)
        _undo_linear(builder, right_bits, right_temp)
        _undo_linear(builder, left_bits, left_temp)
    builder.free(product)


def outputs_indexed(outputs, t: int) -> list[int]:
    """The output bits product `t` feeds."""
    return [k for k, row in enumerate(outputs) if t in row]


def _clear(builder, qubits) -> None:
    """Measure-and-reset a scratch register whose bits are functions of the state."""
    bit = builder.alloc_bits(1)[0]
    for q in qubits:
        builder.mr(q, bit)
    builder.free(bit)


def _emit_product(builder, a, b, out) -> None:
    """Accumulate the polynomial product of `a`, `b` into `out` (2n-1 qubits)."""
    n = len(a)
    if n in BASE_ALGORITHMS:
        _emit_base(builder, a, b, out)
        return

    k = (n + 1) // 2
    width_low = 2 * k - 1
    width_high = 2 * (n - k) - 1

    low = builder.alloc_qubits(width_low)
    _emit_product(builder, a[:k], b[:k], low)
    for i in range(width_low):
        builder.cx(low[i], out[i])          # + low
        builder.cx(low[i], out[k + i])      # + x^k low
    # Each window is a function of the state while it is live, so it is cleared for
    # free as soon as it has been added in rather than kept to the end of the level.
    _clear(builder, low)
    builder.free(low)

    high = builder.alloc_qubits(width_high)
    _emit_product(builder, a[k:], b[k:], high)
    for i in range(width_high):
        builder.cx(high[i], out[k + i])     # + x^k high
        builder.cx(high[i], out[2 * k + i])  # + x^2k high
    _clear(builder, high)
    builder.free(high)

    # Middle product on the sums of the halves, written into the halves and undone.
    for i in range(n - k):
        builder.cx(a[k + i], a[i])
        builder.cx(b[k + i], b[i])
    mid = builder.alloc_qubits(width_low)
    _emit_product(builder, a[:k], b[:k], mid)
    for i in range(width_low):
        builder.cx(mid[i], out[k + i])
    _clear(builder, mid)
    builder.free(mid)
    for i in range(n - k):
        builder.cx(b[k + i], b[i])
        builder.cx(a[k + i], a[i])


def emit_gf54_mul(builder, lhs, rhs, *, target, field, control=True) -> None:
    """`target ^= lhs * rhs` (mod the field polynomial), controlled if asked.

    Drop-in replacement for `ixor_gf2_mul`, at 54 ANDs + 12 controlled-add Toffolis
    instead of the built-in 63 + 12.
    """
    degree = field.degree
    if len(lhs) != degree or len(rhs) != degree or len(target) != degree:
        raise ValueError(f"all three registers must be {degree} qubits wide")

    # The product lives in a clean window of its own: `target` keeps accumulating and
    # must stay disjoint from the workspace. Below the field polynomial's degree x^i
    # reduces to itself, so the window's low half is the field element and the high
    # half folds into it with CNOTs alone.
    product = builder.alloc_qubits(2 * degree - 1)
    _emit_product(builder, list(lhs), list(rhs), list(product))

    mapping = reduction_map(field, degree)
    for i in range(degree - 1):
        fold = mapping[degree + i]
        for k in range(degree):
            if (fold >> k) & 1:
                builder.cx(product[degree + i], product[k])

    if control is True:
        for k in range(degree):
            builder.cx(product[k], target[k])
    else:
        for k in range(degree):
            builder.ccx(control, product[k], target[k])

    # The window holds `lhs * rhs`, a function of the state: measure and reset is free
    # and leaves the tracked phase alone.
    bit = builder.alloc_bits(1)[0]
    for q in list(product):
        builder.mr(q, bit)
    builder.free(product)
    builder.free(bit)


class Gf54MulMixin:
    """Replaces `ixor_gf2_mul` with the 54-AND circuit; every other op is untouched."""

    def ixor_gf2_mul(self, lhs, rhs, *, target, field=None, control=True):
        if field is None:
            raise ValueError("the 54-AND multiply needs an explicit GF2Field")
        return emit_gf54_mul(self, lhs, rhs, target=target, field=field, control=control)
