#!/usr/bin/env python3
"""The 54-AND multiply with a *phase-safe* clear, so the pinned binary can emit it.

`scripts/gf54_mul.py` emits the 54-AND product and then clears its windows with
`mr`. That is the whole reason the pinned competition binary cannot use it
(Experiment D), and Experiment D2 shows the obvious substitution — `hmr` for `mr` —
is not a fix: an X-basis measurement of `|1>` puts a factor of −1 on the branch that
measures 1, so an `hmr` clear *dephases* (37 of 64 shots at the pinned instance) and
the verifier charges every one of them.

The pinned binary already contains the correct primitive, and it is the same idea as
`del_and`, generalised from one AND to a whole product: `del_gf2_mul`
(`gen_gf_unmul`) measures the product register in the X basis and applies
**classically conditioned `CZ` fixups** whose parity is computed from the two
factors. The phase is restored exactly, the clear costs **0 Toffolis**, and it needs
no workspace of its own. Its contract is that the register really does hold
`lhs * rhs` for two live, disjoint, unchanged 12-bit factors.

This module emits the multiply so that the only window that needs clearing is
exactly that: a **12-qubit reduced-product** register.

How
---
`gf54_mul.py` accumulates the base-case products into an unreduced `2n-1`-bit
polynomial window and folds it with CNOTs afterwards, so the window holds the
*polynomial* product — a sum of products no `del_gf2_mul` can clear (and `mr` is the
only clear it has). Here the reduction is applied **as the products are
accumulated**: each base-case product bit lands at a polynomial index `i`, and that
index contributes to the field bits in `reduction_map(field)[i]`, so the product is
CNOT-added straight into a 12-bit register. CNOTs are free, and the register then
holds exactly `lhs * rhs`, which is `del_gf2_mul`'s precondition.

Two consequences worth stating plainly:

* every AND target is still cleared with `del_and` (0 Toffolis, phase-safe: the
  target really is `left & right` of two live operands), and
* the scratch is **smaller**, not larger: a 12-qubit window where the old emitter
  needed a rolling 19-qubit one, because no unreduced window exists at all. The
  "clear a product by keeping all 54 ANDs live" route would have cost 54 scratch
  qubits; this route costs 12.

Cost: 54 ANDs + the 12-Toffoli controlled add = **66 Toffolis per controlled call**,
against the built-in `ixor_gf2_mul`'s 63 + 12 = 75, with no `mr` and no `hmr`.

Example::

    class Builder(Gf54MulReducedMixin, runner.builder_class(km)):
        pass
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_MUL = Path(__file__).resolve().parent / "gf54_mul.py"
_spec = importlib.util.spec_from_file_location("gf54_mul", _MUL)
gf54 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gf54)

BASE_ALGORITHMS = gf54.BASE_ALGORITHMS
and_count = gf54.and_count
terms = gf54.terms
reduction_map = gf54.reduction_map
_emit_linear = gf54._emit_linear
_undo_linear = gf54._undo_linear


def _field_bits(mapping: list[int], index: int) -> list[int]:
    """Which field bits a polynomial product bit at `index` contributes to."""
    mask = mapping[index]
    return [k for k in range(mask.bit_length()) if (mask >> k) & 1]


def _emit_reduced_product(builder, a, b, add, offsets: tuple[int, ...] = (0,)) -> None:
    """Accumulate `a * b` through `add` at each polynomial offset in `offsets`.

    `add(polynomial_index, qubit)` is the sink: it decides which field bits that
    polynomial index feeds (a 12-bit reduced window, or the target register itself).
    A sub-product that is needed at two offsets is computed **once** and added twice,
    which is what the `(1 + x^k)` pre/post ladders of `gen_gf_mul` do with CNOTs; here
    it is the sink's argument list, so it costs nothing at all.
    """
    n = len(a)
    if n in BASE_ALGORITHMS:
        products, outputs = terms(tuple(frozenset([i]) for i in range(n)))
        product = builder.alloc_qubits(1)[0]
        for t, (u, v) in enumerate(products):
            left_bits = [a[i] for i in sorted(u)]
            right_bits = [b[i] for i in sorted(v)]
            left, left_temp = _emit_linear(builder, left_bits)
            right, right_temp = _emit_linear(builder, right_bits)
            builder.ccx(left, right, product)
            for k in gf54.outputs_indexed(outputs, t):
                for offset in offsets:
                    add(offset + k, product)
            # The target really is `left & right`, so the clear is a measurement with
            # a classically conditioned fixup: no Toffoli, and the phase is restored.
            builder.del_and(left, right, product)
            _undo_linear(builder, right_bits, right_temp)
            _undo_linear(builder, left_bits, left_temp)
        builder.free(product)
        return

    k = (n + 1) // 2
    # `a b = low (1 + x^k) + x^k high (1 + x^k) + x^k (low + high)` over GF(2), so
    # each half lands at two offsets and the middle product at one.
    shifted = tuple(sorted({o + k for o in offsets}))
    low_offsets = tuple(sorted({*offsets, *shifted}))
    high_offsets = tuple(sorted({*(o + 2 * k for o in offsets), *shifted}))
    _emit_reduced_product(builder, a[:k], b[:k], add, low_offsets)
    _emit_reduced_product(builder, a[k:], b[k:], add, high_offsets)

    # The middle product on the sums of the halves; the halves are restored before
    # returning, so at the end every product's factors are the originals again.
    for i in range(n - k):
        builder.cx(a[k + i], a[i])
        builder.cx(b[k + i], b[i])
    _emit_reduced_product(builder, a[:k], b[:k], add, shifted)
    for i in range(n - k):
        builder.cx(b[k + i], b[i])
        builder.cx(a[k + i], a[i])


def emit_gf54_mul_reduced(builder, lhs, rhs, *, target, field, control=True) -> None:
    """`target ^= lhs * rhs` (mod the field polynomial), 54 ANDs, phase-safe clears.

    Uncontrolled: the products are accumulated straight into `target`, so there is no
    window to clear at all. Controlled: they accumulate into a 12-bit window which is
    added in under the control and then cleared with `del_gf2_mul`.
    """
    degree = field.degree
    if len(lhs) != degree or len(rhs) != degree or len(target) != degree:
        raise ValueError(f"all three registers must be {degree} qubits wide")
    mapping = reduction_map(field, degree)

    def sink(register):
        def add(index: int, qubit) -> None:
            for k in _field_bits(mapping, index):
                builder.cx(qubit, register[k])

        return add

    if control is True:
        _emit_reduced_product(builder, list(lhs), list(rhs), sink(target))
        return

    window = builder.alloc_qubits(degree)
    _emit_reduced_product(builder, list(lhs), list(rhs), sink(window))
    for k in range(degree):
        builder.ccx(control, window[k], target[k])
    # The window holds exactly `lhs * rhs`: measure it and restore the phase from the
    # factors, which is what `gen_gf_unmul` is for. No `mr`, no `hmr`, no workspace.
    builder.del_gf2_mul(list(lhs), list(rhs), target=window, field=field)
    # `del_gf2_mul` clears the register but leaves it to the caller to free.
    builder.free(window)


class Gf54MulReducedMixin:
    """`ixor_gf2_mul` replaced by the reduced-window 54-AND circuit."""

    def ixor_gf2_mul(self, lhs, rhs, *, target, field=None, control=True):
        if field is None:
            raise ValueError("the 54-AND multiply needs an explicit GF2Field")
        return emit_gf54_mul_reduced(self, lhs, rhs, target=target, field=field,
                                     control=control)
