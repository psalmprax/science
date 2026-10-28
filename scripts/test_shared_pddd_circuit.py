#!/usr/bin/env python3
"""Test shared multi-output PDDD synthesis in 89Q inverse square root circuit."""
from __future__ import annotations

from fractions import Fraction
import io
import json
import math
from pathlib import Path
import sys
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/kickmix"))
sys.path.insert(0, str(REPO_ROOT / "apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/verifiers/invsqrt"))

import kickmix as km
import signed_digit_invsqrt_v3 as core
import scheduled_arithmetic90 as H
from schedule_support90 import optimized_shift, mobius_transform
import portable_exact_domain as domain_compiler
import portable_calibration as cal
import upstream_functions as uf
from verify_submission_invsqrt import verify, ProblemSpecInvSqrt


def xor_many_toffoli(b, a, c, targets):
    targets = list(targets)
    if not targets: return
    first = targets[0]
    for t in targets[1:]: b.cx(first, t)
    b.ccx(a, c, first)
    for t in reversed(targets[1:]): b.cx(first, t)


def init_anf(b, table, address, target, borrowed=()):
    address, target = list(address), list(target); n = len(address)
    coef = mobius_transform(table)
    required = max(0, n - 2); loan = list(borrowed)[:required]
    extra = list(b.alloc_qubits(required - len(loan)))
    scratch = loan + extra
    def emit(mask, control):
        ts = [target[j] for j in range(len(target)) if (coef[mask] >> j) & 1]
        if ts:
            if control is None: b.x(km.array(ts))
            else: b.cx(control, km.array(ts))
    def walk(mask, last, control):
        emit(mask, control)
        for bit in range(last + 1, n):
            child = mask | (1 << bit)
            if not mask: walk(child, bit, address[bit])
            elif bit == n - 1:
                ts = [target[j] for j in range(len(target)) if (coef[child] >> j) & 1]
                xor_many_toffoli(b, control, address[bit], ts)
            else:
                q = scratch[mask.bit_count() - 1]
                b.init_and(control, address[bit], q); walk(child, bit, q)
                b.del_and(control, address[bit], q)
    walk(0, -1, None); b.free(km.array(extra))


def phase_from_records(b, table, address, records):
    address, records = list(address), list(records); n = len(address); k = n // 2
    coef = mobius_transform(table)
    left = {1 << j: address[j] for j in range(k)}
    masks = [m for m in range(1, 1 << k) if m.bit_count() > 1]
    scratch = list(b.alloc_qubits(len(masks) + n - k - 1))
    for mask, q in zip(masks, scratch):
        bit = mask.bit_length() - 1; b.init_and(left[mask ^ (1 << bit)], address[bit], q); left[mask] = q
    carry = scratch[len(masks):]; parity = b.alloc_bits(1)[0]
    def emit(high, hq):
        for low in range(1 << k):
            v = coef[low | (high << k)]
            if not v: continue
            b.bit_store(parity, False)
            while v:
                bit = (v & -v).bit_length() - 1; b.cx(records[bit], parity); v &= v - 1
            if not high and not low: b.neg_if(parity); continue
            b.push_condition(parity)
            if not low: b.z(hq)
            elif not high: b.z(left[low])
            else: b.cz(left[low], hq)
            b.pop_condition()
    def walk(mask, last, q):
        emit(mask, q)
        for j in range(last + 1, n - k):
            child = mask | (1 << j)
            if not mask: walk(child, j, address[k + j])
            else:
                t = carry[mask.bit_count() - 1]; b.init_and(q, address[k + j], t)
                walk(child, j, t); b.del_and(q, address[k + j], t)
    walk(0, -1, None)
    for mask in reversed(masks):
        bit = mask.bit_length() - 1; b.del_and(left[mask ^ (1 << bit)], address[bit], left[mask])
    b.free(km.array(scratch)); b.free(parity)


def boolean_plan(values, order=(7, 4, 2, 3, 0, 1, 6, 5), polarity=104):
    n = len(order); indices = np.arange(1 << n)
    original = sum(((indices >> j) & 1) << order[j] for j in range(n)) ^ polarity
    truth = int.from_bytes(np.packbits(np.asarray(values, dtype=np.uint8)[original], bitorder='little').tobytes(), 'little')
    nodes = []; node_ids = {}
    def xor(a, b): return (a[0] ^ b[0], a[1] ^ b[1])
    cache = {}
    def rec(t, n):
        one = (1 << (1 << n)) - 1
        if t == 0: return (0, frozenset())
        if t == one: return (1, frozenset())
        if (t, n) in cache: return cache[(t, n)]
        half = (1 << (1 << (n - 1))) - 1; lo = t & half; hi = t >> (1 << (n - 1))
        left = rec(lo, n - 1); right = rec(lo ^ hi, n - 1)
        variable = (0, frozenset((n - 1,)))
        if not right[1]: res = xor(left, variable) if right[0] else left
        else:
            key = (n - 1, right)
            if key not in node_ids: node_ids[key] = len(order) + len(nodes); nodes.append(key)
            res = xor(left, (0, frozenset((node_ids[key],))))
        cache[(t, n)] = res
        return res
    result = rec(truth, n)
    return dict(order=tuple(order), polarity=polarity, nodes=nodes, result=result)


def build_shared_plan(vals_dict, order=(4, 7, 2, 1, 0, 3, 6, 5), polarity=66):
    n = len(order)
    indices = np.arange(1 << n)
    original = sum(((indices >> j) & 1) << order[j] for j in range(n)) ^ polarity
    nodes, node_ids = [], {}
    def xor(a, b): return (a[0] ^ b[0], a[1] ^ b[1])
    cache = {}
    def rec(t, level):
        one = (1 << (1 << level)) - 1
        if t == 0: return (0, frozenset())
        if t == one: return (1, frozenset())
        if (t, level) in cache: return cache[(t, level)]
        half = (1 << (1 << (level - 1))) - 1
        lo = t & half; hi = t >> (1 << (level - 1))
        left = rec(lo, level - 1); right = rec(lo ^ hi, level - 1)
        variable = (0, frozenset((level - 1,)))
        if not right[1]:
            res = xor(left, variable) if right[0] else left
        else:
            key = (level - 1, right)
            if key not in node_ids:
                node_ids[key] = len(order) + len(nodes)
                nodes.append(key)
            res = xor(left, (0, frozenset((node_ids[key],))))
        cache[(t, level)] = res
        return res
    results = {}
    for name, vals in vals_dict.items():
        truth = int.from_bytes(np.packbits(vals[original], bitorder="little").tobytes(), "little")
        results[name] = rec(truth, n)
    return {"order": tuple(order), "polarity": polarity, "nodes": nodes, "results": results}


def restore_boolean(b, plan, address, target):
    address = list(address); n = len(address); order = plan['order']; pol = plan['polarity']
    flipped = [address[j] for j in range(n) if (pol >> j) & 1]
    if flipped: b.x(km.array(flipped))
    wires = [address[j] for j in order]
    nodes = list(b.alloc_qubits(len(plan['nodes']))); parity = b.alloc_qubits(1)[0]
    wires += nodes
    def form(f):
        if f[0]: b.x(parity)
        for j in sorted(f[1]): b.cx(wires[j], parity)
    for i, (variable, right) in enumerate(plan['nodes']):
        form(right); b.init_and(wires[variable], parity, nodes[i]); form(right)
    constant, terms = plan['result']
    if constant: b.x(target)
    for term in sorted(terms): b.cx(wires[term], target)
    for i in reversed(range(len(nodes))):
        variable, right = plan['nodes'][i]; form(right)
        b.del_and(wires[variable], parity, nodes[i]); form(right)
    b.free(km.array(nodes)); b.free(parity)
    if flipped: b.x(km.array(flipped))


def restore_shared_boolean(b, plan, address, targets_dict):
    address = list(address); n = len(address); order = plan['order']; pol = plan['polarity']
    flipped = [address[j] for j in range(n) if (pol >> j) & 1]
    if flipped: b.x(km.array(flipped))
    wires = [address[j] for j in order]
    nodes = list(b.alloc_qubits(len(plan['nodes']))); parity = b.alloc_qubits(1)[0]
    wires += nodes
    def form(f):
        if f[0]: b.x(parity)
        for j in sorted(f[1]): b.cx(wires[j], parity)
    for i, (variable, right) in enumerate(plan['nodes']):
        form(right); b.init_and(wires[variable], parity, nodes[i]); form(right)
    for name, target in targets_dict.items():
        constant, terms = plan['results'][name]
        if constant: b.x(target)
        for term in sorted(terms): b.cx(wires[term], target)
    for i in reversed(range(len(nodes))):
        variable, right = plan['nodes'][i]; form(right)
        b.del_and(wires[variable], parity, nodes[i]); form(right)
    b.free(km.array(nodes)); b.free(parity)
    if flipped: b.x(km.array(flipped))


def build_candidate(cfg, tables, *, coarse_choice=None, shared_curve_plan=None):
    extra_reload = 0
    lay = H.layout(cfg, tables)
    split, bounds = H.prepare_derivative_split(H, cfg, tables)
    b = km.CircuitBuilder(); x = b.create_quantum_register(22, name='x'); flags = b.alloc_qubits(5)
    H.normalize(b, x, flags); b.x(x[-1])
    c2, residual, coarse, low, high = [b.alloc_qubits(n) for n in [cfg.w2, 12, 6, 21, 8]]
    full_target = km.array(list(low) + list(residual) + list(c2) + list(high) + list(coarse))
    rest_target = km.array(list(residual) + list(c2) + list(coarse))
    full_table = [0] * 256; rest_table = [0] * 256; high_table = [0] * 256; low_table = [0] * 256
    for (e, a), (k2, _, _) in tables.items():
        key = (e, a); idx = a + 128 * e
        common = lay['lo'][key] | split['residual'][key] << 21 | k2 << 33
        full_table[idx] = common | lay['hi'][key] << (33 + cfg.w2) | split['coarse'][key] << (41 + cfg.w2)
        rest_table[idx] = split['residual'][key] | k2 << 12 | split['coarse'][key] << (12 + cfg.w2)
        high_table[idx] = lay['hi'][key]; low_table[idx] = lay['lo'][key]
    address = km.array(list(x[14:21]) + [flags[0]]); delta = x[:14]
    measured = []
    def erase(reg, table):
        ms = b.alloc_bits(len(reg)); b.hmr(reg, ms)
        measured.append((list(ms), list(table)))
        return list(ms)
    def phase(reg, low_bit):
        ext = b.alloc_qubits(9)
        word = km.array(list(reg) + list(ext) + [x[-1]])
        optimized_shift(b, flags, word, len(reg), False, prune=True)
        for i, q in enumerate(word):
            exponent = low_bit + i - cfg.fy - 10
            if exponent < 0: b.z_pow(q, Fraction(2)**(exponent + 1))
        optimized_shift(b, flags, word, len(reg), True, prune=True); b.free(ext)
    extra_borrow = []
    def adder(a, t, cin=None, sub=False): H.folded_gadd(b, a, t, cin, sub, borrowed=(x[-1], *extra_borrow))
    def inner(inverse):
        rows = cfg.rrow_list()
        for i in reversed(rows) if inverse else rows:
            shift = cfg.sh2[i]
            if shift >= 0: addend = c2[shift:]; target = residual; carry = c2[shift - 1] if shift else None
            else: addend = c2; target = residual[-shift:]; carry = None
            b.cx(delta[i], target); adder(addend, target, carry, sub=inverse); b.cx(delta[i], target)
    def outer(derivative, inverse):
        for i in range(14) if inverse else reversed(range(14)):
            shift = cfg.sh1[i]; carry = derivative[shift - 1] if shift else None
            b.cx(delta[i], low); adder(derivative[shift:], low, carry, sub=inverse); b.cx(delta[i], low)

    init_anf(b, full_table, address, full_target, borrowed=(x[-1],))
    b.x(high[-1]); extra_borrow.append(high[-1])
    inner(False)
    extra_borrow.clear(); b.x(high[-1])

    # Erase curvature (3 bits: c2[4], c2[5], c2[6])
    checkpoint_start = 4
    for j in range(checkpoint_start, 6): b.cx(c2[j + 1], c2[j])
    erased = km.array(list(c2[checkpoint_start:]))
    checkpoint_table = [(tables[(i // 128, i % 128)][0] ^ (tables[(i // 128, i % 128)][0] >> 1)) >> checkpoint_start for i in range(256)]
    checkpoint_ms = erase(erased, checkpoint_table); b.free(erased)

    phase(high, 20); high_ms = erase(high, high_table); b.free(high)

    # Restore coarse[6]
    new = b.alloc_qubits(1)
    coarse = km.array(list(coarse) + list(new))
    cv = np.array([split['coarse'][(i // 128, i % 128)] for i in range(256)], dtype=np.int64)
    c6_vals = ((cv ^ (cv >> 1)) >> 6) & 1
    if coarse_choice is None:
        p_c6 = boolean_plan(c6_vals, order=(7, 6, 5, 3, 0, 1, 2, 4), polarity=144)
    else:
        p_c6 = boolean_plan(c6_vals, order=coarse_choice["order"], polarity=coarse_choice["polarity"])
    restore_boolean(b, p_c6, address, coarse[6])
    extra_reload += len(p_c6['nodes'])

    adder(residual[11:], coarse); derivative = residual[:11] + coarse
    outer(derivative, False); phase(low[cfg.gy:], cfg.gy); outer(derivative, True)
    low_ms = erase(low, low_table); b.free(low)

    # Restore curvature (c2[4], c2[5], c2[6])
    replacement = b.alloc_qubits(3)
    c2 = km.array(list(c2[:checkpoint_start]) + list(replacement))
    targets_dict = {'curve4': c2[4], 'curve5': c2[5], 'curve6': c2[6]}
    restore_shared_boolean(b, shared_curve_plan, address, targets_dict)
    extra_reload += len(shared_curve_plan['nodes'])

    for j in reversed(range(checkpoint_start, 6)):
        b.cx(c2[j + 1], c2[j])

    adder(residual[11:], coarse, sub=True); inner(True)
    rest_target = km.array(list(residual) + list(c2) + list(coarse))
    rest_ms = erase(rest_target, rest_table); b.free(rest_target)

    for k, m in enumerate(checkpoint_ms):
        j = checkpoint_start + k; b.cx(m, rest_ms[12 + j])
        if j < 6: b.cx(m, rest_ms[12 + j + 1])
    records = low_ms + rest_ms[:12 + cfg.w2] + high_ms + rest_ms[12 + cfg.w2:]
    phase_from_records(b, full_table, address, records)

    b.x(x[-1]); H.unnormalize(b, x, flags); b.free(flags)
    circuit = b.finish_circuit()
    return circuit, {"qubits": circuit.num_qubits, "static_toffoli": int(circuit.operation_statistics().num_toffoli_like), "extra_reload": extra_reload}


def main():
    print("1. Loading integer triples and computing curve tables...")
    cfg, tables, _ = core.load_or_build_tables()
    cv_curve = np.array([tables[(i // 128, i % 128)][0] for i in range(256)], dtype=np.int64)
    gray_curve = cv_curve ^ (cv_curve >> 1)
    c4 = (gray_curve >> 4) & 1
    c5 = (gray_curve >> 5) & 1
    c6 = (gray_curve >> 6) & 1

    # The tracked copy in src/ is canonical for reproduction; the artifacts/ copy is the
    # original working output and the /tmp copy predates both. artifacts/ is gitignored,
    # so a clone only has src/shared_davio_results.json.
    shared_pddd_path = next(
        (
            p
            for p in (
                REPO_ROOT / "src/shared_davio_results.json",
                REPO_ROOT / "artifacts/quantum_compilation/invsqrt/shared_davio_results.json",
                Path("/tmp/shared_davio_results.json"),
            )
            if p.exists()
        ),
        None,
    )
    if shared_pddd_path is None:
        raise SystemExit(
            "shared_davio_results.json not found; expected src/shared_davio_results.json"
        )
    shared_cands = json.loads(shared_pddd_path.read_text())["candidates"]
    print(f"Loaded {len(shared_cands)} shared optimal candidates with {shared_cands[0]['nodes']} nodes from {shared_pddd_path}.")

    # We will test candidate 0
    cand0 = shared_cands[0]
    print(f"Candidate 0: order={cand0['order']}, polarity={cand0['polarity']}, nodes={cand0['nodes']}")

    shared_plan = build_shared_plan(
        {"curve4": c4, "curve5": c5, "curve6": c6},
        order=cand0["order"],
        polarity=cand0["polarity"]
    )

    choices_old = json.loads(open(REPO_ROOT / "src/boolean_choices90.json").read())
    c6_choice = choices_old["coarse6"] # keep coarse6 unchanged

    print("\n2. Building raw circuit with shared curve restoration...")
    raw, meta = build_candidate(cfg, tables, coarse_choice=c6_choice, shared_curve_plan=shared_plan)
    print(f"Raw circuit stats: Q={raw.num_qubits}, Static Toffolis={meta['static_toffoli']}, reload={meta['extra_reload']}")

    print("\n3. Running exact domain compiler...")
    opt, comp = domain_compiler.optimize_circuit(raw, linear=True)
    st = opt.operation_statistics()
    print(f"Post-compilation stats: Q={opt.num_qubits}, Static Toffolis={st.num_toffoli_like}, Replacements={len(comp.replacements)}")
    for r in comp.replacements:
        print(f"  Line {r['line']}: {r['reason']}")

    print("\n4. Calibrating rotations...")
    fixed_deltas = json.loads(open(REPO_ROOT / "src/invsqrt_calibration.json").read())
    t = cal._serialize(opt, "kmx").decode()
    lines, rots = cal.parse_kmx(t)
    values = fixed_deltas.get("deltasPhaseUnits", [])
    dropped = set(fixed_deltas.get("droppedOriginalRotationIndices", [18]))
    rep_text = uf.rewrite(lines, rots, {j: int(v) for j, v in enumerate(values) if j not in dropped}, dropped)
    cand = km.Circuit(rep_text)
    
    st_final = cand.operation_statistics()
    print(f"Calibrated stats: Q={cand.num_qubits}, Static Toffolis={st_final.num_toffoli_like}, Rotations={st_final.num_z_pow(min_clifford_hierarchy_level=4)}")

    out_path = Path("/tmp/invsqrt_89q_shared_pddd.kmb")
    cand.to_file(str(out_path), format="kmb")

    print("\n5. Running official verifier (25,600 shots)...")
    spec = ProblemSpecInvSqrt.for_n_p(10, shots_per_x=1)
    spec = ProblemSpecInvSqrt(
        width=spec.width, max_x=25600, beta=spec.beta,
        precision_bits=spec.precision_bits, shots_per_x=1, system_qubits=spec.system_qubits
    )
    receipt = verify(cand, spec)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
