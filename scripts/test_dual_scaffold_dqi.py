#!/dev/shm/qvcc-venv313/bin/python
"""Dual-Scaffold Co-Optimization DQI Decoder and Pre-Registered Cost Gate Evaluation.

Status: audit-only

This script implements:
1. AlternativeChainBuilder: Chain 1G 36-qubit inversion scaffold and 48-qubit division scaffold.
2. DualScaffoldDecoder: Custom ProosAndZalkaDecoder subclass wiring Chain 1G into:
   - EEA Division: accumulate_quotient
   - Chien Search: _chien_search_and_forney_mbuc
3. Small-instance validation (m=15, n=6, degree=12) with verify_submission_dqi.verify:
   - Measures exact circuit width (Q) and Toffoli counts
   - Verifies input restoration, zero junk ancillae, and exact MBUC phase equivalence.
4. Full-instance DQI cost-gate evaluation:
   (R_new * T_new) / (R_base * T_base) < (1893 / 1881)^2 ≈ 1.012788 (+1.28%)
"""
from __future__ import annotations

import math
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np

# Set up paths to kickmix, verifiers, and scripts
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/kickmix"))
sys.path.insert(0, str(REPO_ROOT / "apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/verifiers/dqi"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

import kickmix as km
import dqi_reference_circuit as DR
from verify_submission_dqi import verify, ProblemSpecDQI, repetition_overhead
from test_itoh_tsujii_scaffold import (
    AlternativeChainBuilder as BaseChainBuilder,
    GF12Model,
    PRIM_POLY_12,
    DEGREE,
)


class DualScaffoldChainBuilder(BaseChainBuilder):
    """Chain 1G Builder with support for pre-allocated target registers."""

    def init_chain_1g_inverse_with_scaffold(
        self,
        builder: Any,
        inp: Any,
        field: Any,
        target: Any = None,
    ) -> tuple[Any, Any]:
        """Itoh-Tsujii Chain 1G inversion with 36-qubit scaffold (3 intermediate registers).

        Addition chain: 1 -> 2 -> 4 -> 8 -> 10 -> 11
        Intermediates retained at completion:
          r1 holds beta_2
          r2 holds beta_10 (after uncomputing beta_4)
          r3 holds beta_8
        Target holds beta_11^2 = a^-1.
        Scaffold returned: [r1, r2, r3] (36 qubits total).
        """
        deg = field.degree
        if target is None:
            target = builder.alloc_qubits(deg)
        scaffold = builder.alloc_qubits(deg * 3)
        r1 = scaffold[:deg]
        r2 = scaffold[deg : 2 * deg]
        r3 = scaffold[2 * deg : 3 * deg]

        ctrl_1 = builder.alloc_qubits(1)[0]
        builder.x(ctrl_1)

        # Step 1: Compute beta_2 = beta_1^2 * beta_1 into r1
        builder.cx(inp, r2)
        self.square_in_place(builder, r2, 1)
        builder.ixor_gf2_mul(r2, inp, target=r1, field=field, control=ctrl_1)
        self.square_in_place(builder, r2, 1, adjoint=True)
        builder.cx(inp, r2)

        # Step 2: Compute beta_4 = beta_2^4 * beta_2 into r2
        builder.cx(r1, r3)
        self.square_in_place(builder, r3, 2)
        builder.ixor_gf2_mul(r3, r1, target=r2, field=field, control=ctrl_1)
        self.square_in_place(builder, r3, 2, adjoint=True)
        builder.cx(r1, r3)

        # Step 3: Compute beta_8 = beta_4^16 * beta_4 into r3 (borrowing empty target)
        builder.cx(r2, target)
        self.square_in_place(builder, target, 4)
        builder.ixor_gf2_mul(target, r2, target=r3, field=field, control=ctrl_1)
        self.square_in_place(builder, target, 4, adjoint=True)
        builder.cx(r2, target)

        # Step 4: Uncompute beta_4 from r2 using beta_2 in r1 (borrowing empty target)
        builder.cx(r1, target)
        self.square_in_place(builder, target, 2)
        builder.ixor_gf2_mul(target, r1, target=r2, field=field, control=ctrl_1)
        self.square_in_place(builder, target, 2, adjoint=True)
        builder.cx(r1, target)

        # Step 5: Compute beta_10 = beta_8^4 * beta_2 into now-clean r2 (borrowing empty target)
        builder.cx(r3, target)
        self.square_in_place(builder, target, 2)
        builder.ixor_gf2_mul(target, r1, target=r2, field=field, control=ctrl_1)
        self.square_in_place(builder, target, 2, adjoint=True)
        builder.cx(r3, target)

        # Step 6: Compute beta_11 = beta_10^2 * beta_1 into target
        self.square_in_place(builder, r2, 1)
        builder.ixor_gf2_mul(r2, inp, target=target, field=field, control=ctrl_1)
        self.square_in_place(builder, r2, 1, adjoint=True)

        # Step 7: Final in-place square on target to produce a^-1
        self.square_in_place(builder, target, 1)

        builder.x(ctrl_1)
        builder.free(ctrl_1)

        return target, scaffold

    def del_chain_1g_inverse_with_scaffold(
        self,
        builder: Any,
        inp: Any,
        target: Any,
        scaffold: Any,
        field: Any,
    ) -> None:
        """Adjoint uncomputation of Itoh-Tsujii Chain 1G 36-qubit inverse."""
        deg = field.degree
        r1 = scaffold[:deg]
        r2 = scaffold[deg : 2 * deg]
        r3 = scaffold[2 * deg : 3 * deg]

        ctrl_1 = builder.alloc_qubits(1)[0]
        builder.x(ctrl_1)

        # Step 7 reversed: Un-square target from a^-1 to beta_11
        self.square_in_place(builder, target, 1, adjoint=True)

        # Step 6 reversed: Uncompute beta_11 from target
        self.square_in_place(builder, r2, 1)
        builder.ixor_gf2_mul(r2, inp, target=target, field=field, control=ctrl_1)
        self.square_in_place(builder, r2, 1, adjoint=True)

        # Step 5 reversed: Uncompute beta_10 from r2
        builder.cx(r3, target)
        self.square_in_place(builder, target, 2)
        builder.ixor_gf2_mul(target, r1, target=r2, field=field, control=ctrl_1)
        self.square_in_place(builder, target, 2, adjoint=True)
        builder.cx(r3, target)

        # Step 4 reversed: Recompute beta_4 into r2
        builder.cx(r1, target)
        self.square_in_place(builder, target, 2)
        builder.ixor_gf2_mul(target, r1, target=r2, field=field, control=ctrl_1)
        self.square_in_place(builder, target, 2, adjoint=True)
        builder.cx(r1, target)

        # Step 3 reversed: Uncompute beta_8 from r3
        builder.cx(r2, target)
        self.square_in_place(builder, target, 4)
        builder.ixor_gf2_mul(target, r2, target=r3, field=field, control=ctrl_1)
        self.square_in_place(builder, target, 4, adjoint=True)
        builder.cx(r2, target)

        # Step 2 reversed: Uncompute beta_4 from r2
        builder.cx(r1, r3)
        self.square_in_place(builder, r3, 2)
        builder.ixor_gf2_mul(r3, r1, target=r2, field=field, control=ctrl_1)
        self.square_in_place(builder, r3, 2, adjoint=True)
        builder.cx(r1, r3)

        # Step 1 reversed: Uncompute beta_2 from r1
        builder.cx(inp, r2)
        self.square_in_place(builder, r2, 1)
        builder.ixor_gf2_mul(r2, inp, target=r1, field=field, control=ctrl_1)
        self.square_in_place(builder, r2, 1, adjoint=True)
        builder.cx(inp, r2)

        builder.x(ctrl_1)
        builder.free(ctrl_1)

    def init_chain_1g_div_with_scaffold(
        self,
        builder: Any,
        lhs: Any,
        rhs: Any,
        field: Any,
    ) -> tuple[Any, Any]:
        """Chain 1G division: 48-qubit scaffold (36q chain + 12q divisor inverse)."""
        deg = field.degree
        rhs_inv, inv_scaffold = self.init_chain_1g_inverse_with_scaffold(builder, rhs, field)
        ratio = builder.alloc_qubits(deg)

        ctrl_1 = builder.alloc_qubits(1)[0]
        builder.x(ctrl_1)
        builder.ixor_gf2_mul(lhs, rhs_inv, target=ratio, field=field, control=ctrl_1)
        builder.x(ctrl_1)
        builder.free(ctrl_1)

        scaffold = km.array(list(inv_scaffold) + list(rhs_inv))
        return ratio, scaffold

    def del_chain_1g_div_with_scaffold(
        self,
        builder: Any,
        lhs: Any,
        rhs: Any,
        target: Any,
        scaffold: Any,
        field: Any,
    ) -> None:
        """Adjoint uncomputation of Chain 1G division."""
        deg = field.degree
        inv_scaffold = scaffold[: 3 * deg]
        rhs_inv = scaffold[3 * deg : 4 * deg]

        ctrl_1 = builder.alloc_qubits(1)[0]
        builder.x(ctrl_1)
        builder.ixor_gf2_mul(lhs, rhs_inv, target=target, field=field, control=ctrl_1)
        builder.x(ctrl_1)
        builder.free(ctrl_1)

        self.del_chain_1g_inverse_with_scaffold(
            builder, rhs, target=rhs_inv, scaffold=inv_scaffold, field=field
        )


class DualScaffoldDecoder(DR.ProosAndZalkaDecoder):
    """Custom DQI Decoder integrating Chain 1G Dual-Scaffold Co-Optimization.

    Replaces:
    1. EEA accumulate_quotient: Chain 1G 48q scaffold (36q chain + 12q divisor inverse).
    2. Chien Search _chien_search_and_forney_mbuc: Chain 1G 36q inversion scaffold.
    """

    def __init__(self, n_syndromes: int, field: km.GF2Field):
        super().__init__(n_syndromes, field)
        self.chain_builder = DualScaffoldChainBuilder()

    def _multiply_and_add_or_sub(
        self,
        builder: km.CircuitBuilder,
        reg_a: Sequence[km.array],
        reg_b: Sequence[km.array],
        n_A: km.array,
        n_a: km.array,
        n_q: km.array,
        ctrl,
        it: int,
        *,
        is_adjoint: bool,
    ) -> None:
        """Computes the leading-coefficient ratio `q_i` using Chain 1G 48q scaffold."""
        c0_hc, _, _, c3_hc = ctrl
        gf = self.field
        m = len(reg_b)

        def accumulate_quotient(index: int, control) -> None:
            with builder.mark("quotient_div"):
                ratio, scaffold = self.chain_builder.init_chain_1g_div_with_scaffold(
                    builder, reg_b[index], reg_a[index], field=gf
                )
                builder.gf2_iadd(ratio, target=q_i, field=gf, control=control)
                self.chain_builder.del_chain_1g_div_with_scaffold(
                    builder, reg_b[index], reg_a[index], target=ratio, scaffold=scaffold, field=gf
                )
                builder.free(scaffold + ratio)

        def sweep_positions() -> None:
            with builder.mark("sweep_positions"):
                lim_q = min(m - 1, 1 + it)
                lim_q1 = min(m, 1 + it)
                lim_a = min(m - 2, it // 2 + 1)
                lim_A = m - 2
                order = range(m) if is_adjoint else range(m - 1, -1, -1)

                with (
                    builder.unary_iteration(n_q, control=c0_hc) as eq_i,
                    builder.unary_iteration(n_q, control=c3_hc) as eq_ip1,
                    builder.unary_iteration(n_a, control=c3_hc) as cur_a,
                    builder.unary_iteration(n_A, control=c0_hc) as cur_A,
                ):
                    mul = builder.alloc_qubits(1)[0]

                    def endpoint(i: int) -> None:
                        DR._xor_threshold(
                            builder, cur_a, lim_a, mul, c3_hc, i + 1, at_least=True, lowest=1
                        )
                        DR._xor_threshold(
                            builder, cur_A, lim_A, mul, c0_hc, m - i, at_least=True, lowest=1
                        )

                    prev = None
                    for i in order:
                        if prev is None:
                            endpoint(i)
                        else:
                            builder.cx(
                                km.array([
                                    DR._active(cur_a, min(prev, i) + 1, lim_a),
                                    DR._active(cur_A, m - max(prev, i), lim_A),
                                ]),
                                mul,
                            )
                        prev = i

                        swaps = [
                            a
                            for a in (
                                DR._active(eq_i, i, lim_q),
                                DR._active(eq_ip1, i + 1, lim_q1),
                            )
                            if a is not False
                        ]
                        multiplies = i < lim_a or i >= 2

                        if is_adjoint and multiplies:
                            builder.ixor_gf2_mul(
                                q_i, reg_a[i], target=reg_b[i], field=gf, control=mul
                            )
                        if len(swaps) == 1:
                            builder.cswap(swaps[0], q_i, reg_a[i])
                        elif swaps:
                            both = builder.alloc_qubits(1)[0]
                            builder.cx(km.array(swaps), both)
                            builder.cswap(both, q_i, reg_a[i])
                            builder.cx(km.array(swaps), both)
                            builder.free(both)
                        if not is_adjoint and multiplies:
                            builder.ixor_gf2_mul(
                                q_i, reg_a[i], target=reg_b[i], field=gf, control=mul
                            )

                    endpoint(prev)
                    builder.free(mul)

        with builder.mark("multiply_and_add_or_sub"):
            q_i = builder.alloc_qubits(gf.degree)
            leading, trailing = (0, m - 1) if is_adjoint else (m - 1, 0)
            accumulate_quotient(leading, c3_hc if is_adjoint else c0_hc)
            builder.iadd(n_a, target=n_q)
            sweep_positions()
            builder.isub(n_a, target=n_q)
            accumulate_quotient(trailing, c0_hc if is_adjoint else c3_hc)
            builder.free(q_i)

    def _chien_search_and_forney_mbuc(
        self,
        builder: km.CircuitBuilder,
        reg_b: Sequence[km.array],
        error_mbuc: Sequence[km.array],
        n_b: km.array,
        n_B: km.array,
    ) -> None:
        # Signature matches how `build_syndrome_decoding_mbuc` calls it (positional).
        """Applies MBUC phase using Chain 1G 36-qubit inversion scaffold."""
        gf = self.field
        degree = gf.degree
        block_length = (1 << degree) - 1
        primitive = gf.primitive_element
        m = len(reg_b)
        t = (m - 3) // 2

        with builder.mark("pre_align_reg_b"):
            self._pre_align_reg_b(builder, reg_b, n_b, n_B, is_adjoint=False)

        sigma = reg_b[0]
        sigma_prime = reg_b[1]
        omega = reg_b[m // 2 + t]
        sigma_prime_inv = builder.alloc_qubits(degree)
        located = builder.alloc_qubits(degree)
        is_sigma_zero = builder.alloc_qubits(1)[0]
        is_error = builder.alloc_bits(1)

        omega_indices = range(m // 2 + 1, m // 2 + t + 1)
        step_multipliers = [
            (k, gf.pow(primitive, k)) for k in range(1, t + 1)
        ] + [
            (k, gf.pow(primitive, (m - 2 - k) % block_length))
            for k in omega_indices
            if (m - 2 - k) % block_length != 0
        ]

        sum_pairs = (
            [(reg_b[k], omega) for k in range(m // 2 + 1, m // 2 + t)]
            + [(reg_b[k], sigma_prime) for k in range(3, t + 1, 2)]
            + [(reg_b[k], sigma) for k in range(2, t + 1, 2)]
            + [(sigma_prime, sigma)]
        )
        fwd_src = km.array([q for src, _ in sum_pairs for q in src])
        fwd_dst = km.array([q for _, dst in sum_pairs for q in dst])
        rev_src = km.array([q for src, _ in reversed(sum_pairs) for q in src])
        rev_dst = km.array([q for _, dst in reversed(sum_pairs) for q in dst])

        num_positions = min(block_length, len(error_mbuc))
        forney_step = builder.mark("forney_step")
        for j in range(num_positions):
            with forney_step:
                builder.bit_store(is_error[0], True, control=error_mbuc[j])
                builder.push_condition(is_error[0])

                builder.cx(fwd_src, fwd_dst)
                builder.flip_if_equal(sigma, 0, target=is_sigma_zero)

                # Chain 1G 36-qubit inversion scaffold substituting native 48q scaffold
                _, inverse_chain = self.chain_builder.init_chain_1g_inverse_with_scaffold(
                    builder, sigma_prime, field=gf, target=sigma_prime_inv
                )

                builder.init_and(is_sigma_zero, omega, located)
                builder.gf2_phase_by_product(
                    located, sigma_prime_inv, mask=error_mbuc[j], field=gf
                )
                builder.del_and(is_sigma_zero, omega, located)

                self.chain_builder.del_chain_1g_inverse_with_scaffold(
                    builder,
                    sigma_prime,
                    target=sigma_prime_inv,
                    scaffold=inverse_chain,
                    field=gf,
                )
                builder.free(inverse_chain)

                builder.flip_if_equal(sigma, 0, target=is_sigma_zero)
                builder.cx(rev_src, rev_dst)

                builder.pop_condition()
                builder.bit_store(is_error, False)

                if j + 1 < num_positions:
                    for k, step in step_multipliers:
                        builder.gf2_imul(step, target=reg_b[k], field=gf)

        if num_positions > 1:
            for k, step in step_multipliers:
                builder.gf2_idiv(
                    gf.pow(step, num_positions - 1), target=reg_b[k], field=gf
                )

        builder.free(is_error)
        builder.free(located + sigma_prime_inv + km.array([is_sigma_zero]))

        with builder.mark("unalign_reg_b"):
            self._pre_align_reg_b(builder, reg_b, n_b, n_B, is_adjoint=True)


# ============================================================================
# Verification and Benchmarking Battery
# ============================================================================

def run_primitive_isolation_benchmark():
    """Measures primitive width and Toffolis in isolation."""
    print("=" * 80)
    print("1. ISOLATION SYNTHESIS BENCHMARK (GF(2^12))")
    print("=" * 80)
    gf = km.GF2Field(12)
    alt = DualScaffoldChainBuilder()

    # Native Division
    b_nat_div = km.CircuitBuilder()
    lhs = b_nat_div.create_quantum_register(12, name="lhs")
    rhs = b_nat_div.create_quantum_register(12, name="rhs")
    ratio, scaf = b_nat_div.init_gf2_div_with_scaffold(lhs, rhs, field=gf)
    n_scaf_nat_div = len(scaf)
    c_nat_div_fwd = b_nat_div.finish_circuit()

    b_nat_div_full = km.CircuitBuilder()
    lhs = b_nat_div_full.create_quantum_register(12, name="lhs")
    rhs = b_nat_div_full.create_quantum_register(12, name="rhs")
    ratio, scaf = b_nat_div_full.init_gf2_div_with_scaffold(lhs, rhs, field=gf)
    b_nat_div_full.del_gf2_div_with_scaffold(lhs, rhs, target=ratio, scaffold=scaf, field=gf)
    b_nat_div_full.free(scaf + ratio)
    c_nat_div_full = b_nat_div_full.finish_circuit()

    # Chain 1G Division
    b_c1g_div = km.CircuitBuilder()
    lhs = b_c1g_div.create_quantum_register(12, name="lhs")
    rhs = b_c1g_div.create_quantum_register(12, name="rhs")
    ratio, scaf = alt.init_chain_1g_div_with_scaffold(b_c1g_div, lhs, rhs, field=gf)
    n_scaf_c1g_div = len(scaf)
    c_c1g_div_fwd = b_c1g_div.finish_circuit()

    b_c1g_div_full = km.CircuitBuilder()
    lhs = b_c1g_div_full.create_quantum_register(12, name="lhs")
    rhs = b_c1g_div_full.create_quantum_register(12, name="rhs")
    ratio, scaf = alt.init_chain_1g_div_with_scaffold(b_c1g_div_full, lhs, rhs, field=gf)
    alt.del_chain_1g_div_with_scaffold(b_c1g_div_full, lhs, rhs, target=ratio, scaffold=scaf, field=gf)
    b_c1g_div_full.free(scaf + ratio)
    c_c1g_div_full = b_c1g_div_full.finish_circuit()

    # Native Inversion
    b_nat_inv = km.CircuitBuilder()
    inp = b_nat_inv.create_quantum_register(12, name="inp")
    inv, scaf_inv = b_nat_inv.init_gf2_inverse_with_scaffold(inp, field=gf)
    n_scaf_nat_inv = len(scaf_inv)
    c_nat_inv_fwd = b_nat_inv.finish_circuit()

    b_nat_inv_full = km.CircuitBuilder()
    inp = b_nat_inv_full.create_quantum_register(12, name="inp")
    inv, scaf_inv = b_nat_inv_full.init_gf2_inverse_with_scaffold(inp, field=gf)
    b_nat_inv_full.del_gf2_inverse_with_scaffold(inp, target=inv, scaffold=scaf_inv, field=gf)
    b_nat_inv_full.free(scaf_inv + inv)
    c_nat_inv_full = b_nat_inv_full.finish_circuit()

    # Chain 1G Inversion
    b_c1g_inv = km.CircuitBuilder()
    inp = b_c1g_inv.create_quantum_register(12, name="inp")
    inv, scaf_inv = alt.init_chain_1g_inverse_with_scaffold(b_c1g_inv, inp, field=gf)
    n_scaf_c1g_inv = len(scaf_inv)
    c_c1g_inv_fwd = b_c1g_inv.finish_circuit()

    b_c1g_inv_full = km.CircuitBuilder()
    inp = b_c1g_inv_full.create_quantum_register(12, name="inp")
    inv, scaf_inv = alt.init_chain_1g_inverse_with_scaffold(b_c1g_inv_full, inp, field=gf)
    alt.del_chain_1g_inverse_with_scaffold(b_c1g_inv_full, inp, target=inv, scaffold=scaf_inv, field=gf)
    b_c1g_inv_full.free(scaf_inv + inv)
    c_c1g_inv_full = b_c1g_inv_full.finish_circuit()

    metrics = {
        "native_div": {
            "scaffold_q": n_scaf_nat_div,
            "peak_q": c_nat_div_fwd.num_qubits,
            "fwd_toffoli": int(c_nat_div_fwd.operation_statistics().num_toffoli_like),
            "total_toffoli": int(c_nat_div_full.operation_statistics().num_toffoli_like),
        },
        "chain1g_div": {
            "scaffold_q": n_scaf_c1g_div,
            "peak_q": c_c1g_div_fwd.num_qubits,
            "fwd_toffoli": int(c_c1g_div_fwd.operation_statistics().num_toffoli_like),
            "total_toffoli": int(c_c1g_div_full.operation_statistics().num_toffoli_like),
        },
        "native_inv": {
            "scaffold_q": n_scaf_nat_inv,
            "peak_q": c_nat_inv_fwd.num_qubits,
            "fwd_toffoli": int(c_nat_inv_fwd.operation_statistics().num_toffoli_like),
            "total_toffoli": int(c_nat_inv_full.operation_statistics().num_toffoli_like),
        },
        "chain1g_inv": {
            "scaffold_q": n_scaf_c1g_inv,
            "peak_q": c_c1g_inv_fwd.num_qubits,
            "fwd_toffoli": int(c_c1g_inv_fwd.operation_statistics().num_toffoli_like),
            "total_toffoli": int(c_c1g_inv_full.operation_statistics().num_toffoli_like),
        },
    }

    print(f"Native Division:   Scaffold = {metrics['native_div']['scaffold_q']}q, Peak = {metrics['native_div']['peak_q']}q, "
          f"Fwd Toffoli = {metrics['native_div']['fwd_toffoli']}, Total Toffoli = {metrics['native_div']['total_toffoli']}")
    print(f"Chain 1G Division: Scaffold = {metrics['chain1g_div']['scaffold_q']}q, Peak = {metrics['chain1g_div']['peak_q']}q, "
          f"Fwd Toffoli = {metrics['chain1g_div']['fwd_toffoli']}, Total Toffoli = {metrics['chain1g_div']['total_toffoli']}")
    print(f"  --> Delta Division Toffoli per call: +{metrics['chain1g_div']['total_toffoli'] - metrics['native_div']['total_toffoli']} Toffolis")
    print(f"  --> Delta Division scaffold width:  -{metrics['native_div']['scaffold_q'] - metrics['chain1g_div']['scaffold_q']} qubits")

    print(f"\nNative Inversion:   Scaffold = {metrics['native_inv']['scaffold_q']}q, Peak = {metrics['native_inv']['peak_q']}q, "
          f"Fwd Toffoli = {metrics['native_inv']['fwd_toffoli']}, Total Toffoli = {metrics['native_inv']['total_toffoli']}")
    print(f"Chain 1G Inversion: Scaffold = {metrics['chain1g_inv']['scaffold_q']}q, Peak = {metrics['chain1g_inv']['peak_q']}q, "
          f"Fwd Toffoli = {metrics['chain1g_inv']['fwd_toffoli']}, Total Toffoli = {metrics['chain1g_inv']['total_toffoli']}")
    print(f"  --> Delta Inversion Toffoli per call: +{metrics['chain1g_inv']['total_toffoli'] - metrics['native_inv']['total_toffoli']} Toffolis")

    return metrics


def run_small_instance_verification(num_shots: int = 256, log2_delta: float = -10.0):
    """Builds and simulates small-instance decoders (m=15, n=6, degree=12).

    Reference and candidate are verified under identical settings. The verifier's
    acceptance bound tightens with `log2_delta` and with the number of shots, so
    holding one decoder to a stricter setting than the other would make the
    comparison meaningless (and 16 shots cannot certify any decoder at all).
    """
    print("\n" + "=" * 80)
    print("2. SMALL-INSTANCE CIRCUIT SIMULATION & REVERSIBILITY VERIFICATION")
    print(f"   Parameters: m=15, n=6, degree=12, r=2016")
    print(f"   Verification settings shared by both decoders: num_shots={num_shots}, log2_delta={log2_delta}")
    print("=" * 80)

    gf = km.GF2Field(12)
    spec = ProblemSpecDQI(m=15, n=6, degree=12, r=2016, num_shots=num_shots)

    # 1. Reference Circuit
    print("Building reference Proos-Zalka decoder on GF(2^12)...")
    b_ref = km.CircuitBuilder()
    syn_ref = [b_ref.create_quantum_register(12, name=f"S{i}") for i in range(6)]
    err_ref = [b_ref.create_classical_register(12, name=f"E{j}") for j in range(15)]
    dec_ref = DR.ProosAndZalkaDecoder(6, gf)
    dec_ref.build_syndrome_decoding_mbuc(b_ref, syndromes=syn_ref, error_mbuc=err_ref, n_iters=30)
    c_ref = b_ref.finish_circuit()

    print(f"Simulating reference circuit ({num_shots} shots)...")
    t0 = time.time()
    res_ref = verify(c_ref, spec, max_workers=1, batch_size=num_shots, log2_delta=log2_delta)
    t_ref = time.time() - t0
    print(f"  Reference Status:       {res_ref['status']}")
    print(f"  Reference Num Qubits:   {res_ref['num_qubits']}")
    print(f"  Reference Toffoli mean: {res_ref['toffoli_count']:.2f}")
    print(f"  Elapsed:                {t_ref:.2f}s")

    # 2. Dual-Scaffold Candidate Circuit
    print("\nBuilding DualScaffoldDecoder candidate on GF(2^12)...")
    b_cand = km.CircuitBuilder()
    syn_cand = [b_cand.create_quantum_register(12, name=f"S{i}") for i in range(6)]
    err_cand = [b_cand.create_classical_register(12, name=f"E{j}") for j in range(15)]
    dec_cand = DualScaffoldDecoder(6, gf)
    dec_cand.build_syndrome_decoding_mbuc(b_cand, syndromes=syn_cand, error_mbuc=err_cand, n_iters=30)
    c_cand = b_cand.finish_circuit()

    print(f"Simulating DualScaffold circuit ({num_shots} shots)...")
    t0 = time.time()
    res_cand = verify(c_cand, spec, max_workers=1, batch_size=num_shots, log2_delta=log2_delta)
    t_cand = time.time() - t0
    print(f"  Candidate Status:       {res_cand['status']}")
    print(f"  Candidate Num Qubits:   {res_cand['num_qubits']}")
    print(f"  Candidate Toffoli mean: {res_cand['toffoli_count']:.2f}")
    print(f"  Elapsed:                {t_cand:.2f}s")

    assert res_cand["status"] == "PASS", f"Candidate verification failed with {res_cand['status']}"
    assert res_cand["value_errors"] == 0, f"Value errors detected: {res_cand['value_errors']}"
    assert res_cand["phase_errors"] == 0, f"Phase errors detected: {res_cand['phase_errors']}"
    print("\n[SUCCESS] Candidate passed official verification: exact MBUC phase equivalence,")
    print("          syndromes fully restored, 0 junk ancillae left uncomputed.")

    return res_ref, res_cand


def evaluate_full_cost_gate(delta_div: int = 126, delta_inv: int = 126, delta_scaffold: int = 12):
    """Computes the full-scale pre-compilation DQI cost gate.

    `delta_div`, `delta_inv` and `delta_scaffold` are measured in stage 1; the two
    call-count constants and the reference baseline below are pinned assumptions of
    this projection, not measurements at full instance size.
    """
    print("\n" + "=" * 80)
    print("3. FULL-INSTANCE PRE-COMPILATION COST GATE EVALUATION")
    print("=" * 80)

    # Preregistered Reference Constants
    Q_base = 1893
    Q_new = Q_base - delta_scaffold
    T_base = 5725108.382109375
    eps_plus = 0.0008660587301657926
    R_base = repetition_overhead(eps_plus)  # ~1.19126046
    C_base = R_base * (Q_base ** 2) * T_base  # ~24.43946 trillion

    # Subroutine call frequencies in the 222-iteration full decoder
    # EEA division calls: 2 calls per step * 222 steps * 2 (fwd + rev) = 888 calls/shot
    n_div_calls = 888
    # Chien search inversion calls: executed conditionally on error positions.
    # Mean error weight under principal eigenvector distribution for l=35 is ~17.5 positions
    n_inv_calls = 17.5

    delta_T_div = n_div_calls * delta_div
    delta_T_inv = n_inv_calls * delta_inv
    delta_T_total = delta_T_div + delta_T_inv
    T_new = T_base + delta_T_total

    # Cost Gate Formulation
    # Condition: (R_new * T_new) / (R_base * T_base) < (1893 / 1881)^2
    width_ratio_sq = (Q_base / Q_new) ** 2
    feasibility_ceiling_pct = (width_ratio_sq - 1.0) * 100.0
    max_allowable_delta_T = (width_ratio_sq - 1.0) * T_base

    measured_ratio = T_new / T_base
    measured_increase_pct = (measured_ratio - 1.0) * 100.0

    R_new = R_base  # Equivalent decoder retains eps_plus
    C_new = R_new * (Q_new ** 2) * T_new
    cost_change_pct = ((C_new / C_base) - 1.0) * 100.0

    print(f"Reference Baseline Metrics:")
    print(f"  Q_base:               {Q_base} qubits")
    print(f"  T_base:               {T_base:.2f} Toffolis/shot")
    print(f"  Score C_base:         {C_base:.2f} ({C_base / 1e12:.3f} Trillion)")
    print()
    print(f"Candidate Dual-Scaffold Metrics:")
    print(f"  Scaffold saving:      {delta_scaffold} qubits (measured in stage 1: 60q native vs 48q chain 1G)")
    print(f"  Q_new:                {Q_new} qubits ({Q_new - Q_base:+d} qubits, {((Q_new / Q_base) - 1) * 100:+.3f}%)")
    print(f"  EEA Division Calls:   {n_div_calls} calls x (+{delta_div} Toffolis) = +{delta_T_div:,} Toffolis")
    print(f"  Chien Inversion Calls:{n_inv_calls:.1f} calls x (+{delta_inv} Toffolis) = +{delta_T_inv:,.1f} Toffolis")
    print(f"  Total Delta T_mean:   +{delta_T_total:,.1f} Toffolis/shot")
    print(f"  T_new:                {T_new:.2f} Toffolis/shot (+{measured_increase_pct:.4f}%)")
    print(f"  Score C_new:          {C_new:.2f} ({C_new / 1e12:.3f} Trillion, {cost_change_pct:+.3f}%)")
    print()
    print(f"Feasibility Cost Gate Analysis:")
    print(f"  Width Ratio Squared:  (1893 / 1881)^2 = {width_ratio_sq:.6f}")
    print(f"  Feasibility Ceiling:  +{feasibility_ceiling_pct:.4f}% (+{max_allowable_delta_T:,.1f} Toffolis max)")
    print(f"  Actual Gate Increase: +{measured_increase_pct:.4f}% (+{delta_T_total:,.1f} Toffolis)")
    print(f"  Gate Equation Check:  {measured_ratio:.6f} < {width_ratio_sq:.6f} --> {measured_ratio < width_ratio_sq}")

    gate_passed = measured_ratio < width_ratio_sq
    decision = "PROCEED" if gate_passed else "HALT"

    print()
    print(f"GATE DECISION: [{decision}]")
    if not gate_passed:
        print(f"  The measured Toffoli overhead (+{measured_increase_pct:.3f}%) exceeds the width ceiling")
        print(f"  (+{feasibility_ceiling_pct:.3f}%) by {measured_increase_pct / feasibility_ceiling_pct:.2f}x.")
        print(f"  Integrating Dual-Scaffold Co-Optimization worsens complete DQI cost by {cost_change_pct:+.3f}%.")
        print("  Recommendation: HALT candidate integration into the full decoder.")

    return {
        "gate_passed": gate_passed,
        "decision": decision,
        "width_ratio_sq": width_ratio_sq,
        "feasibility_ceiling_pct": feasibility_ceiling_pct,
        "measured_ratio": measured_ratio,
        "measured_increase_pct": measured_increase_pct,
        "delta_T_total": delta_T_total,
        "C_base": C_base,
        "C_new": C_new,
        "cost_change_pct": cost_change_pct,
    }


def main():
    print("=" * 80)
    print("DUAL-SCAFFOLD CO-OPTIMIZATION EVALUATION (OPTION 2)")
    print("=" * 80)
    start_time = time.time()

    # 1. Isolation primitives
    metrics = run_primitive_isolation_benchmark()
    delta_div = metrics["chain1g_div"]["total_toffoli"] - metrics["native_div"]["total_toffoli"]
    delta_inv = metrics["chain1g_inv"]["total_toffoli"] - metrics["native_inv"]["total_toffoli"]
    delta_scaffold = metrics["native_div"]["scaffold_q"] - metrics["chain1g_div"]["scaffold_q"]

    # 2. Small-instance simulation (identical verification settings for both decoders)
    res_ref, res_cand = run_small_instance_verification(num_shots=256, log2_delta=-10.0)

    # 3. Full cost gate
    gate_res = evaluate_full_cost_gate(
        delta_div=delta_div, delta_inv=delta_inv, delta_scaffold=delta_scaffold
    )

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.2f} seconds.")
    print("=" * 80)


if __name__ == "__main__":
    main()
