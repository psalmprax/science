"""Portable exact-domain inverse-square-root Clifford compiler.

Requires the credited signed_digit_invsqrt_v3 module on sys.path, NumPy and
kickmix. The default entry point regenerates from readable source and explicit
coefficient data; it has no filesystem layout, fixture or network dependency.
Full truth-vector equality certifies every selected replacement over every
x=1..3,133,452. Classical branches join exact equal functions or unknown.
"""
from __future__ import annotations
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
from collections import Counter
import json
import re
import numpy as np
import kickmix as km

MAX_X = 3_133_452

class Known:
    """Immutable truth vector. Samples only reject matches; equality is exact."""
    __slots__ = ('v', 'signature')
    sample = None

    def __init__(self, values):
        self.v = values
        self.signature = values[Known.sample].tobytes()

class DomainCompiler:
    def __init__(self, nqubits, max_x=MAX_X, linear=False, mbu=False, temp_bit=0):
        self.max_x = max_x
        self.nwords = (max_x + 63) // 64
        Known.sample = np.unique(np.r_[np.arange(min(32, self.nwords)),
            np.linspace(0, self.nwords - 1, min(96, self.nwords), dtype=int)])
        self.zero = Known(np.zeros(self.nwords, dtype=np.uint64))
        ones = np.full(self.nwords, np.uint64(2**64 - 1))
        if max_x % 64:
            ones[-1] = np.uint64((1 << (max_x % 64)) - 1)
        self.one = Known(ones)
        xs = np.arange(1, max_x + 1, dtype=np.uint32)
        self.original = []
        for bit in range(22):
            values = np.zeros(self.nwords * 8, dtype=np.uint8)
            packed = np.packbits(((xs >> bit) & 1).astype(np.uint8), bitorder='little')
            values[:len(packed)] = packed
            self.original.append(Known(values.view(np.uint64)))
        self.q = self.original[:] + [self.zero] * (nqubits - 22)
        self.stack = []
        self.replacements = []
        self.counts = Counter()
        self.join_unknown = 0
        self.output = []
        self.linear = linear
        self.linear_false_candidates = 0
        self.mbu = mbu
        self.temp_bit = temp_bit

    @staticmethod
    def equal(a, b):
        if a is b:
            return True
        if a is None or b is None or a.signature != b.signature:
            return False
        return bool(np.array_equal(a.v, b.v))

    def xor(self, a, b):
        if a is None or b is None:
            return None
        if a is self.zero:
            return b
        if b is self.zero:
            return a
        if a is b:
            return self.zero
        return Known(a.v ^ b.v)

    def band(self, a, b):
        if a is self.zero or b is self.zero:
            return self.zero
        if a is None or b is None:
            return None
        if a is self.one:
            return b
        if b is self.one or a is b:
            return a
        return Known(a.v & b.v)

    def literal(self, function, excluded):
        """Return an untouched live literal equal to the full truth vector."""
        if function is None:
            return None
        for q, value in enumerate(self.q):
            if q not in excluded and self.equal(function, value):
                return (q, False)
        complement = self.xor(function, self.one)
        for q, value in enumerate(self.q):
            if q not in excluded and self.equal(complement, value):
                return (q, True)
        return None

    def linear_terms(self, function, excluded):
        """Find an affine GF(2) expression, then prove it with ALL truth bits.

        Gaussian elimination over sampled bits only proposes a combination.
        It cannot certify a rewrite: the combination is always compared with
        the full domain vector before being accepted. Unsampled dependencies
        can cause missed opportunities, never invalid replacements.
        """
        if not self.linear or function is None:
            return None
        variables = [q for q, value in enumerate(self.q)
                     if q not in excluded and value is not None]
        values = [self.q[q] for q in variables] + [self.one]
        extra_words = []
        # A failed full proof supplies another exact input row to the linear
        # system. Refinement removes sample-induced dependencies rather than
        # stopping at the first invalid proposed combination.
        while True:
            def signature(value):
                raw = value.signature
                if extra_words:
                    raw += value.v[extra_words].tobytes()
                return int.from_bytes(raw, 'little')
            basis = {}
            for index, value in enumerate(values):
                vector = signature(value)
                combination = 1 << index
                while vector:
                    pivot = vector.bit_length()
                    if pivot not in basis:
                        basis[pivot] = vector, combination
                        break
                    existing, coefficients = basis[pivot]
                    vector ^= existing
                    combination ^= coefficients
            vector = signature(function)
            combination = 0
            while vector:
                pivot = vector.bit_length()
                if pivot not in basis:
                    return None
                existing, coefficients = basis[pivot]
                vector ^= existing
                combination ^= coefficients
            selected = [i for i in range(len(values)) if combination >> i & 1]
            exact = self.zero
            for i in selected:
                exact = self.xor(exact, values[i])
            if self.equal(exact, function):
                constant = len(variables) in selected
                return [variables[i] for i in selected if i < len(variables)], constant
            self.linear_false_candidates += 1
            witness = int(np.flatnonzero(exact.v != function.v)[0])
            assert witness not in extra_words
            extra_words.append(witness)

    def record(self, line_number, original, emitted, reason):
        self.replacements.append({'line': line_number, 'original': original,
            'replacement': emitted, 'reason': reason, 'condition_depth': len(self.stack)})
        self.counts[reason] += 1

    def execute(self, line, number):
        parts = line.split()
        if not parts:
            return
        op = parts[0]
        qs = [int(s[1:]) for s in parts[1:] if re.fullmatch(r'q\d+', s)]
        emitted = [line]
        # Conditional inline phases have no effect on computational functions.
        if op in {'Z', 'CZ', 'Z_POW', 'NEG', 'INFO_TAG', 'REGISTER',
                  'APPEND_TO_REGISTER', 'BIT_STORE0', 'BIT_INVERT'}:
            pass
        elif op == 'PUSH_CONDITION':
            self.stack.append(self.q[:])
        elif op == 'POP_CONDITION':
            before = self.stack.pop()
            for q, old in enumerate(before):
                if not self.equal(old, self.q[q]):
                    self.q[q] = None
                    self.join_unknown += 1
        elif op in {'HMR', 'R'}:
            assert 'if' not in parts, 'Conditional measurement/reset needs a joined branch.'
            self.q[qs[0]] = self.zero
        elif op == 'X':
            assert 'if' not in parts, 'Inline basis condition is unsupported.'
            self.q[qs[0]] = self.xor(self.q[qs[0]], self.one)
        elif op == 'CX':
            assert 'if' not in parts, 'Inline basis condition is unsupported.'
            a, b = qs
            self.q[b] = self.xor(self.q[b], self.q[a])
        elif op == 'SWAP':
            assert 'if' not in parts, 'Inline basis condition is unsupported.'
            a, b = qs
            self.q[a], self.q[b] = self.q[b], self.q[a]
        elif op == 'CCX':
            assert 'if' not in parts, 'Inline basis condition is unsupported.'
            a, b, t = qs
            product = self.band(self.q[a], self.q[b])
            reason = None
            if self.equal(product, self.zero):
                emitted, reason = [], 'ccx_zero_product'
            elif self.equal(product, self.one):
                emitted, reason = [f'X q{t}'], 'ccx_constant_one'
            else:
                literal = self.literal(product, {t})
                if literal is not None:
                    q, inverted = literal
                    emitted = ([f'X q{t}'] if inverted else []) + [f'CX q{q} q{t}']
                    reason = 'ccx_live_complement' if inverted else 'ccx_live_literal'
                else:
                    terms = self.linear_terms(product, {t})
                    if terms is not None:
                        variables, constant = terms
                        emitted = ([f'X q{t}'] if constant else [])
                        emitted += [f'CX q{q} q{t}' for q in variables]
                        reason = 'ccx_affine_live_span'
            if reason is None and self.mbu:
                result = self.xor(self.q[t], product)
                terms = None
                if self.equal(result, self.zero):
                    terms = [], False
                elif self.equal(result, self.one):
                    terms = [], True
                else:
                    literal = self.literal(result, {t})
                    if literal is not None:
                        q, inverted = literal
                        terms = [q], inverted
                    elif self.linear:
                        terms = self.linear_terms(result, {t})
                if terms is not None:
                    variables, constant = terms
                    prepare = ([f'X q{t}'] if constant else [])
                    prepare += [f'CX q{q} q{t}' for q in variables]
                    # Before: t = a*b XOR L(other wires).  XOR L makes t=a*b.
                    # HMR contributes (-1)**(m*a*b); conditioned CZ cancels it.
                    # The same Clifford expression then restores t=L.
                    emitted = prepare + [f'HMR q{t} b{self.temp_bit}',
                        f'CZ q{a} q{b} if b{self.temp_bit}'] + prepare
                    reason = 'ccx_output_affine_mbu'
            if reason:
                self.record(number, line, emitted, reason)
            self.q[t] = self.xor(self.q[t], product)
        elif op == 'CSWAP':
            assert 'if' not in parts, 'Inline basis condition is unsupported.'
            c, a, b = qs
            delta = self.xor(self.q[a], self.q[b])
            product = self.band(self.q[c], delta)
            reason = None
            if self.equal(product, self.zero):
                emitted, reason = [], 'cswap_zero_delta'
            elif self.equal(product, delta) and delta is not None:
                emitted, reason = [f'SWAP q{a} q{b}'], 'cswap_unconditional'
            else:
                literal = self.literal(product, {a, b})
                if literal is not None:
                    q, inverted = literal
                    emitted = ([f'X q{a}', f'X q{b}'] if inverted else [])
                    emitted += [f'CX q{q} q{a}', f'CX q{q} q{b}']
                    reason = 'cswap_live_complement' if inverted else 'cswap_live_literal'
                else:
                    terms = self.linear_terms(product, {a, b})
                    if terms is not None:
                        variables, constant = terms
                        emitted = ([f'X q{a}', f'X q{b}'] if constant else [])
                        emitted += [gate for q in variables
                                    for gate in (f'CX q{q} q{a}', f'CX q{q} q{b}')]
                        reason = 'cswap_affine_live_span'
            if reason:
                self.record(number, line, emitted, reason)
            self.q[a] = self.xor(self.q[a], product)
            self.q[b] = self.xor(self.q[b], product)
        else:
            raise AssertionError(f'Unsupported operation {op}: {line}')
        self.output.extend(emitted)

    def compile(self, circuit):
        for number, line in enumerate(str(circuit).splitlines(), 1):
            self.execute(line, number)
            if number % 5000 == 0:
                print(json.dumps({'line': number, 'replacements': len(self.replacements),
                    'unknown_qubits': sum(v is None for v in self.q)}), flush=True)
        assert not self.stack
        assert all(self.equal(self.q[q], self.original[q]) for q in range(22)), 'Input function lost.'
        assert all(self.equal(v, self.zero) for v in self.q[22:]), 'Scratch function is not zero.'
        return km.Circuit('\n'.join(self.output) + '\n')

def optimize_circuit(circuit, linear=False, mbu=False):
    """Open in-memory compiler API; no circuit binary or network is required."""
    compiler = DomainCompiler(circuit.num_qubits, linear=linear, mbu=mbu,
                              temp_bit=circuit.num_bits)
    return compiler.compile(circuit), compiler

def build_selected(cfg=None, tables=None, layout=None):
    """Regenerate the credited frozen arithmetic, then prove five reductions.

    This entry point depends only on the readable frozen source and its
    explicit coefficient data. The CLI's existing KMB is an independent
    canonical regression fixture rather than a generator dependency.
    """
    import signed_digit_invsqrt_v3 as core
    if cfg is None or tables is None:
        assert cfg is None and tables is None
        cfg, tables, _ = core.load_or_build_tables()
    source = core.build_selected(cfg, tables, layout)
    circuit, compiler = optimize_circuit(source)
    assert len(compiler.replacements) == 5
    return circuit
