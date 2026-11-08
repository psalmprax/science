# SPDX-License-Identifier: Apache-2.0
# New derivative-split integration; original upstream notices retained below.
# Sumit Chongder: centered signed-digit generator and integer pipeline,361357149.
# Rudra Bhaskar: derivative split, top-carry reuse, static-high checkpoint,361445283.
# https://www.kaggle.com/code/rudrabhaskar/highest-public-score-direct-polynomial-phasing?scriptVersionId=361445283
# Alex Chilton: shared prefix and empty-target temporary-AND shifts,361472684.
# Kirill Tymchishin: exact Rudra/Alex integration used as the95Q source baseline,361491814.
# https://www.kaggle.com/code/kirilltym/quantum-circuits-qrom-tower42?scriptVersionId=361491814
# Kozykorpesh: explicit8,19,27,1 coefficient model,datasetVersionId21136269.
# Our changes: generic exact coefficient widths/left shifts, normalized-input
# borrowing on this derivative schedule, and the proven original-input rewrite.
# The derivative split and its precision calibrations are credited upstream.

# SPDX-License-Identifier: Apache-2.0
# Arithmetic research candidate. Retained upstream attribution:
# Sumit Chongder: readable centered signed-digit generator and integer pipeline.
# https://www.kaggle.com/code/sumitchongder/notebook2f196e646b?scriptVersionId=361357149
# anazemcev / Artem1410: early high-word phase, folded carry and normalized-input borrowing.
# https://www.kaggle.com/code/anazemcev/quantum-circuits-signed-digit-memory?scriptVersionId=361422248
# Kozykorpesh: exact reachable-domain calibration, Cfg(8,19,27,1), explicit table.
# https://www.kaggle.com/code/kozykappa/quantum-ai-compact-arithmetic-banked-qrom?scriptVersionId=361456406
# Quantum Circuit Toolbox datasetVersionId 21136269, code/domain_tables.py and coefficients.json.
# Alex Chilton: shared prefix flags and temporary-AND moves into zero destinations.
# https://www.kaggle.com/code/alexchilton/where-the-toffolis-go-anatomy-a-cheaper-invsqrt?scriptVersionId=361472684
# This revision combines credited methods. No upstream gain is claimed as original.
# Added own exact rewrite: the original input bound proves x[21]&x[20]==0.
# Linearize the first conjunction in normalization and the final input phase fixup.

# Attribution: Sumit Chongder, Apache-2.0 as published.
# https://www.kaggle.com/code/sumitchongder/notebook2f196e646b
# Read-only source downloaded from latest version 2. Kirill's v9 cites v1.
# This isolated lab regenerates the circuit from readable source; no binaries or
# remote notebook outputs are inputs. Our added exact integer certificate and
# reporting helpers and memory-schedule adaptations follow the original generator.
"""Builds an optimized kickmix circuit for the invsqrt benchmark (Benchmark I).

Status: experimental

|x>|0> -> exp(2 pi i / sqrt(x)) |x>|0>,   x in [1, 3133452], 22-bit register 'x'.

Design (all arithmetic exactly mirrored by the numpy model below):
  1. Normalize x by single-bit binary stages (16, 8, 4, 2, 1) into M = x_norm / 2^21 in [1, 2);
     flags f0..f4 hold the shift count s' in binary.  e = f0 (parity), s = s' >> 1.
     1/sqrt(x) = 2^(s-10) * y,  y = 2^((e-1)/2) / sqrt(M)  in (0.5, 1].
  2. One 256-entry QROM indexed by (7 mantissa bits, e) XORs quadratic coefficients directly into
     the working registers (no separate coefficient copies).
  3. Centered signed-digit Horner: delta_c = sum_i (2 d_i - 1) 2^(i-22) over the 14 low bits.
     R = K1 - C2*delta_c,  y = K0 - R*delta_c.  Each row is an UNcontrolled Gidney addition
     wrapped in CNOT flips (Y - A = ~(~Y + A)), with a rounding carry-in.
  4. The y register is split: rows only touch a 21-bit low register L; the constant high part
     sits in a separate register and the overflow bit is merged in before phasing.
  5. Binary de-normalization (shift by s), one Z_POW per bit, then exact mirror uncompute.
     Lookup uncompute is measurement based (del_lookup); normalization flags are cleared by HMR
     with classically conditioned phase fixups.
Usage: python signed_digit_lab.py OUT.kmb [--baseline] (requires kickmix and NumPy)
"""

import functools
import math
import types
from fractions import Fraction

import numpy as np

try:
    import kickmix as km
except ImportError:
    km = None

W = 22
MB = 21
DB = 14
NSEG = 128
F2, F1, FYR, GY, DROP2 = 8, 19, 27, 1, (5,)  # precision configuration (verified)


# ----------------------------------------------------------------- numpy model / tables
@functools.lru_cache(None)
def fitc(lo, hi, mid, scale, deg=2, iters=80):
    def f(M: np.ndarray) -> np.ndarray:
        _out: np.ndarray = scale / np.sqrt(M)
        return _out

    d = np.linspace(lo - mid, hi - mid, 3001)
    A = np.vander(d, deg + 1, increasing=True)
    w = np.sqrt(mid + d) / scale
    wt = np.ones_like(d)
    for _ in range(iters):  # Lawson iterations -> weighted minimax
        sw = w * np.sqrt(wt)
        coef = np.linalg.lstsq(A * sw[:, None], f(mid + d) * sw, rcond=None)[0]
        err = np.abs((A @ coef - f(mid + d)) * w)
        wt = wt * err
        wt /= wt.sum()
    return coef


class Cfg:
    def __init__(self, f2, f1, fyr, gy, drop2=()):
        self.f2, self.f1, self.fyr, self.gy = f2, f1, fyr, gy
        self.fy = fyr + gy
        self.w2 = f2 - 1
        self.sh2 = [f2 + 22 - i - f1 for i in range(DB)]
        self.sh1 = [f1 + 22 - i - self.fy for i in range(DB)]
        self.drop2 = set(drop2)

    def rrow_list(self):
        return [i for i in reversed(range(DB)) if i not in self.drop2 and self.sh2[i] < self.w2]


def rd(v, s):
    if s < 0:
        return v << (-s)
    r = v >> s
    if s >= 1:
        r = r + ((v >> (s - 1)) & 1)
    return r


def pipeline(cfg, K2, K1, d):
    sig = [2 * ((d >> i) & 1) - 1 for i in range(DB)]
    R = np.full_like(d, K1)
    for i in cfg.rrow_list():
        R = R - sig[i] * rd(K2, cfg.sh2[i])
    Q = np.zeros_like(d)
    for i in reversed(range(DB)):
        Q = Q + sig[i] * rd(R, cfg.sh1[i])
    return R, Q


MAX_X = 3_133_452


def valid_residuals(e, a):
    d: np.ndarray = np.arange(1 << DB, dtype=np.int64)
    xn = (1 << MB) + (a << DB) + d
    mask = (d % 2 == 0) if e else ((xn <= MAX_X) | (d % 4 == 0))
    return d[mask]


def calibrate_segment(cfg, e, a, c_radius=1, b_radius=3):
    d = valid_residuals(e, a)
    lo = 1 + a * 2.0 ** (DB - MB)
    mid = lo + ((1 << DB) - 1) / 2 ** (MB + 1)
    scale = 2 ** ((e - 1) / 2)
    co = fitc(lo, lo + 2.0 ** (DB - MB), mid, scale)
    f = scale / np.sqrt(lo + d * 2.0**-MB)
    target = f * 2**cfg.fy
    w = 1 / f
    K2c = round(co[2] * 2**cfg.f2)
    K1c = round(-co[1] * 2**cfg.f1)
    best = None
    quant = 1 << cfg.gy
    for K2 in range(max(0, K2c - c_radius), min((1 << cfg.w2) - 1, K2c + c_radius) + 1):
        for K1 in range(K1c - b_radius, K1c + b_radius + 1):
            R, Q = pipeline(cfg, K2, K1, d)
            if R.min() < 0:
                continue
            vals = target + Q
            center = 0.5 * (vals.min() + vals.max()) + 0.5 * (quant - 1)
            k0s = np.arange(
                math.floor(center) - quant - 1, math.ceil(center) + quant + 2, dtype=np.int64
            )
            # The stored high anchor and the carry must keep the leading one fixed.
            k0s = k0s[k0s <= (1 << cfg.fy) - 1 + int(Q.min())]
            if len(k0s) == 0:
                continue
            y = ((k0s[:, None] - Q) >> cfg.gy) << cfg.gy
            scores = np.max(abs(y - target) * w, axis=1) / 2**cfg.fy
            i = int(scores.argmin())
            v = float(scores[i])
            K0 = int(k0s[i])
            if best is None or v < best[0]:
                best = (v, K2, K1, K0)
    if best is None:
        raise RuntimeError("No legal coefficient triple")
    return best


def regenerate_tables_exact(cfg, refine_threshold=0.95):
    tabs = {}
    ratios = []
    for e in (0, 1):
        for a in range(128):
            best = calibrate_segment(cfg, e, a)
            if best[0] * 2**24 > refine_threshold:
                alt = calibrate_segment(cfg, e, a, c_radius=4, b_radius=7)
                if alt[0] < best[0]:
                    best = alt
            tabs[(e, a)] = best[1:]
            ratios.append(best[0] * 2**24)
    return tabs, {
        "worst_ratio": max(ratios),
        "worst_key": [int(np.argmax(ratios)) // 128, int(np.argmax(ratios)) % 128],
        "reachable_normalized_inputs": sum(
            len(valid_residuals(e, a)) for e in (0, 1) for a in range(128)
        ),
    }


def build_tables(cfg, search=2):
    """Optional credited exact-domain coefficient regeneration.

    The radius policy is the published Kozykorpesh calibration, rather than
    the old continuous-domain fitter. The default selected build uses the
    explicitly published numeric table below, checked on every original x.
    """
    tables, report = regenerate_tables_exact(cfg)
    return tables, report["worst_ratio"] / 2**24


def layout(cfg, tabs, LB=21, S=20):
    d = np.arange(2**DB, dtype=np.int64)
    assert max(t[0] for t in tabs.values()) < (1 << cfg.w2)
    Rmax = 0
    per = {}
    Ymax = 0
    for key, (K2, K1, K0) in tabs.items():
        R, Q = pipeline(cfg, K2, K1, d)
        assert R.min() >= 0
        Rmax = max(Rmax, int(R.max()), K1)
        per[key] = (int(Q.min()), int(Q.max()))
        Ymax = max(Ymax, K0 - per[key][0])
    w1 = Rmax.bit_length()
    lo_tab, hi_tab = {}, {}
    for key, (_K2, _K1, K0) in tabs.items():
        qmin, qmax = per[key]
        base = K0 % (1 << S)
        j = 0
        while base + j * (1 << S) - qmax < 0:
            j += 1
        k0lo = base + j * (1 << S)
        assert k0lo - qmin < (1 << LB), "layout infeasible"
        lo_tab[key] = k0lo
        hi_tab[key] = (K0 - k0lo) >> S
        assert hi_tab[key] >= 0
    whi = max(max(hi_tab.values()).bit_length(), 1)
    return {"w1": w1, "LB": LB, "S": S, "whi": whi, "lo": lo_tab, "hi": hi_tab}


# ----------------------------------------------------------------- circuit pieces
def and_all_into(b, controls, target):
    k = len(controls)
    if k == 1:
        b.cx(controls[0], target)
        return
    if k == 2:
        b.ccx(controls[0], controls[1], target)
        return
    anc = b.alloc_qubits(k - 2)
    b.init_and(controls[0], controls[1], anc[0])
    for j in range(1, k - 2):
        b.init_and(anc[j - 1], controls[j + 1], anc[j])
    b.ccx(anc[k - 3], controls[k - 1], target)
    for j in reversed(range(1, k - 2)):
        b.del_and(anc[j - 1], controls[j + 1], anc[j])
    b.del_and(controls[0], controls[1], anc[0])
    b.free(anc)


def phase_all(b, controls):
    k = len(controls)
    if k == 1:
        b.z(controls[0])
        return
    if k == 2:
        b.cz(controls[0], controls[1])
        return
    anc = b.alloc_qubits(k - 2)
    b.init_and(controls[0], controls[1], anc[0])
    for j in range(1, k - 2):
        b.init_and(anc[j - 1], controls[j + 1], anc[j])
    b.cz(anc[k - 3], controls[k - 1])
    for j in reversed(range(1, k - 2)):
        b.del_and(anc[j - 1], controls[j + 1], anc[j])
    b.del_and(controls[0], controls[1], anc[0])
    b.free(anc)


NSTAGES = [(4, 16), (3, 8), (2, 4), (1, 2), (0, 1)]


def _baseline_normalize(b, x, f):
    for k, d in NSTAGES:
        top = x[W - d : W]
        if d == 1:
            b.cx(x[W - 1], f[0])
            b.x(f[0])
        else:
            b.x(top)
            and_all_into(b, list(top), f[k])
            b.x(top)
        for i in reversed(range(0, W - d)):
            b.cswap(f[k], x[i], x[i + d])


def _combined_normalize(b, x, f):
    """v3 step 1: f4, f3, f2 (the binary leading-zero count div 4) from one prefix-AND chain on the unshifted x.
    chain[m] = [x[W-m-2:W] == 0] (top m+2 bits zero); p_j = chain[4j-2] = [lz >= 4j].  f4 = p4, f3 = p2 ^ p4,
    f2 = p1 ^ p2 ^ p3 ^ p4 ^ p5 (nested prefixes, so XOR = bit 2 of lz).  19 ANDs instead of 15 + 7 + 3."""
    bits = [x[W - 1 - m] for m in range(20)]  # x21, x20, ..., x2
    b.x(km.array(bits))
    chain = b.alloc_qubits(19)
    b.init_and(bits[0], bits[1], chain[0])
    for m in range(1, 19):
        b.init_and(chain[m - 1], bits[m + 1], chain[m])
    p = {j: chain[4 * j - 2] for j in range(1, 6)}
    b.cx(p[4], f[4])
    b.cx(p[2], f[3])
    b.cx(p[4], f[3])
    for j in range(1, 6):
        b.cx(p[j], f[2])
    for m in reversed(range(1, 19)):
        b.del_and(chain[m - 1], bits[m + 1], chain[m])
    b.del_and(bits[0], bits[1], chain[0])
    b.free(chain)
    b.x(km.array(bits))
    for k, d in NSTAGES:
        top = x[W - d : W]
        if d == 1:
            b.cx(x[W - 1], f[0])
            b.x(f[0])
        elif d == 2:
            b.x(top)
            b.ccx(top[0], top[1], f[1])
            b.x(top)
        for i in reversed(range(0, W - d)):
            b.cswap(f[k], x[i], x[i + d])


# MAX_X < 3*2**20 proves the original two top bits are never both one.
# After inversion, their AND equals XOR with one, on every promised input.
def normalize(b, x, f):
    """v3 step 1: f4, f3, f2 (the binary leading-zero count div 4) from one prefix-AND chain on the unshifted x.
    chain[m] = [x[W-m-2:W] == 0] (top m+2 bits zero); p_j = chain[4j-2] = [lz >= 4j].  f4 = p4, f3 = p2 ^ p4,
    f2 = p1 ^ p2 ^ p3 ^ p4 ^ p5 (nested prefixes, so XOR = bit 2 of lz).  19 ANDs instead of 15 + 7 + 3."""
    bits = [x[W - 1 - m] for m in range(20)]  # x21, x20, ..., x2
    b.x(km.array(bits))
    chain = b.alloc_qubits(19)
    b.x(chain[0])
    b.cx(bits[0], chain[0])
    b.cx(bits[1], chain[0])
    for m in range(1, 19):
        b.init_and(chain[m - 1], bits[m + 1], chain[m])
    p = {j: chain[4 * j - 2] for j in range(1, 6)}
    b.cx(p[4], f[4])
    b.cx(p[2], f[3])
    b.cx(p[4], f[3])
    for j in range(1, 6):
        b.cx(p[j], f[2])
    for m in reversed(range(1, 19)):
        b.del_and(chain[m - 1], bits[m + 1], chain[m])
    b.cx(bits[1], chain[0])
    b.cx(bits[0], chain[0])
    b.x(chain[0])
    b.free(chain)
    b.x(km.array(bits))
    for k, d in NSTAGES:
        top = x[W - d : W]
        if d == 1:
            b.cx(x[W - 1], f[0])
            b.x(f[0])
        elif d == 2:
            b.x(top)
            b.ccx(top[0], top[1], f[1])
            b.x(top)
        for i in reversed(range(0, W - d)):
            b.cswap(f[k], x[i], x[i + d])


def _baseline_unnormalize(b, x, f):
    for k, d in reversed(NSTAGES):
        for i in range(0, W - d):
            b.cswap(f[k], x[i], x[i + d])
        top = x[W - d : W]
        if d == 1:
            b.x(f[0])
            b.cx(x[W - 1], f[0])
        else:
            m = b.alloc_bits(1)[0]
            b.hmr(f[k], m)
            b.push_condition(m)  # fixup runs on half the shots on average
            b.x(top)
            phase_all(b, list(top))
            b.x(top)
            b.pop_condition()
            b.free(m)


def promise_phase_original_top(b, controls):
    """Phase the original top16-zero predicate using its promised bit relation."""
    assert len(controls) == 16
    controls = list(reversed(controls))
    k = len(controls)
    anc = b.alloc_qubits(k - 2)
    b.x(anc[0])
    b.cx(controls[0], anc[0])
    b.cx(controls[1], anc[0])
    for j in range(1, k - 2):
        b.init_and(anc[j - 1], controls[j + 1], anc[j])
    b.cz(anc[k - 3], controls[k - 1])
    for j in reversed(range(1, k - 2)):
        b.del_and(anc[j - 1], controls[j + 1], anc[j])
    b.cx(controls[1], anc[0])
    b.cx(controls[0], anc[0])
    b.x(anc[0])
    b.free(anc)


def unnormalize(b, x, f):
    for k, d in reversed(NSTAGES):
        for i in range(0, W - d):
            b.cswap(f[k], x[i], x[i + d])
        top = x[W - d : W]
        if d == 1:
            b.x(f[0])
            b.cx(x[W - 1], f[0])
        else:
            m = b.alloc_bits(1)[0]
            b.hmr(f[k], m)
            b.push_condition(m)  # fixup runs on half the shots on average
            b.x(top)
            if d == 16:
                promise_phase_original_top(b, list(top))
            else:
                phase_all(b, list(top))
            b.x(top)
            b.pop_condition()
            b.free(m)


def gadd(b, a, t, cin=None, sub=False):
    """t += a + cin (or t -= a + cin), Gidney adder: len(t)-1 Toffolis, uncompute by HMR."""
    a = list(a)
    t = list(t)
    o = len(a)
    n = len(t)
    assert o <= n
    if sub:
        b.x(km.array(t))
    c = b.alloc_qubits(n - 1) if n > 1 else km.array([])

    def C(i: int) -> object:
        return cin if i == 0 else c[i - 1]

    for i in range(0, n - 1):
        if i == 0 and cin is None:
            if o:
                b.init_and(a[0], t[0], C(1))
        elif i < o:
            b.cx(C(i), a[i])
            b.cx(C(i), t[i])
            b.init_and(a[i], t[i], C(i + 1))
            b.cx(C(i), C(i + 1))
        else:
            b.init_and(C(i), t[i], C(i + 1))
    if n - 1 < o:
        b.cx(a[n - 1], t[n - 1])
    if n > 1 or cin is not None:
        b.cx(C(n - 1), t[n - 1])
    for i in reversed(range(0, n - 1)):
        if i == 0 and cin is None:
            if o:
                b.del_and(a[0], t[0], C(1))
                b.cx(a[0], t[0])
        elif i < o:
            b.cx(C(i), C(i + 1))
            b.del_and(a[i], t[i], C(i + 1))
            b.cx(C(i), a[i])
            b.cx(a[i], t[i])
        else:
            b.del_and(C(i), t[i], C(i + 1))
            b.cx(C(i), t[i])
    if n > 1:
        b.free(c)
    if sub:
        b.x(km.array(t))


def _baseline_shift_stages(b, f, reg, L, inverse):
    stages = [(1, 1), (2, 2), (3, 4), (4, 8)]
    occ = []
    o = L
    for _k, d in stages:
        occ.append(o)
        o = min(o + d, len(reg))
    seq = list(enumerate(stages))
    if not inverse:
        for idx, (k, d) in seq:
            for i in reversed(range(0, occ[idx])):
                if i + d < len(reg):
                    b.cswap(f[k], reg[i], reg[i + d])
    else:
        for idx, (k, d) in reversed(seq):
            for i in range(0, occ[idx]):
                if i + d < len(reg):
                    b.cswap(f[k], reg[i], reg[i + d])


def shift_stages(b, f, reg, L, inverse):
    stages = [(1, 1), (2, 2), (3, 4), (4, 8)]
    occ = []
    o = L
    for _k, d in stages:
        occ.append(o)
        o = min(o + d, len(reg))
    seq = list(enumerate(stages))
    if not inverse:
        for idx, (k, d) in seq:
            for i in reversed(range(0, occ[idx])):
                if i + d >= len(reg):
                    continue
                if i + d >= occ[idx]:  # v2: target is |0> for every f
                    b.init_and(f[k], reg[i], reg[i + d])
                    b.cx(reg[i + d], reg[i])
                else:
                    b.cswap(f[k], reg[i], reg[i + d])
    else:
        for idx, (k, d) in reversed(seq):
            for i in range(0, occ[idx]):
                if i + d >= len(reg):
                    continue
                if i + d >= occ[idx]:
                    b.cx(reg[i + d], reg[i])
                    b.del_and(f[k], reg[i], reg[i + d])
                else:
                    b.cswap(f[k], reg[i], reg[i + d])


def build_circuit(cfg, tabs, lay, adder=None):
    # The default preserves the public generator exactly. An injected adder
    # supports isolated, explicitly tested memory experiments.
    if adder is None:
        adder = gadd
    w2, w1, LB, S, whi = cfg.w2, lay["w1"], lay["LB"], lay["S"], lay["whi"]
    gy, fy = cfg.gy, cfg.fy
    b = km.CircuitBuilder()
    x = b.create_quantum_register(W, name="x")
    f = b.alloc_qubits(5)
    c2 = b.alloc_qubits(w2)
    R = b.alloc_qubits(w1)
    Lr = b.alloc_qubits(LB)
    Yh = b.alloc_qubits(whi)
    tgt = km.array(list(Lr) + list(R) + list(c2) + list(Yh))
    table = [0] * 256
    for (e, a), (K2, K1, _K0) in tabs.items():
        table[a + 128 * e] = (
            lay["lo"][(e, a)]
            | (K1 << LB)
            | (K2 << (LB + w1))
            | (lay["hi"][(e, a)] << (LB + w1 + w2))
        )
    addr = km.array([*list(x[14:21]), f[0]])
    delta = x[0:DB]

    def rrows(inverse):
        seq = cfg.rrow_list()
        if inverse:
            seq = list(reversed(seq))
        for i in seq:
            sh = cfg.sh2[i]
            if sh >= 0:
                off = c2[sh:w2]
                t = R[0:w1]
                cin = c2[sh - 1] if sh >= 1 else None
            else:
                off = c2[0:w2]
                t = R[-sh:w1]
                cin = None
            b.cx(delta[i], t)
            adder(b, off, t, cin, sub=inverse)
            b.cx(delta[i], t)

    def yrows(inverse):
        seq = list(reversed(range(DB)))
        if inverse:
            seq.reverse()
        for i in seq:
            sh = cfg.sh1[i]
            assert 0 <= sh < w1
            t = Lr[0:LB]
            cin = R[sh - 1] if sh >= 1 else None
            b.cx(delta[i], t)
            adder(b, R[sh:w1], t, cin, sub=inverse)
            b.cx(delta[i], t)

    normalize(b, x, f)
    b.init_lookup(table=table, address=addr, target=tgt)
    rrows(False)
    yrows(False)
    ext = b.alloc_qubits(10)
    if LB > S:
        adder(b, Lr[S:LB], Yh)
    Cx = km.array(list(Lr[gy:S]) + list(Yh) + list(ext))
    shift_stages(b, f, Cx, S - gy + whi, False)
    for j in range(len(Cx)):
        e = gy + j - fy - 10  # bit weight 2^e turns
        if e < 0:
            b.z_pow(Cx[j], Fraction(2) ** (e + 1))
    shift_stages(b, f, Cx, S - gy + whi, True)
    if LB > S:
        adder(b, Lr[S:LB], Yh, sub=True)
    b.free(ext)
    yrows(True)
    rrows(True)
    b.del_lookup(table=table, address=addr, target=tgt)
    b.free(tgt)
    unnormalize(b, x, f)
    b.free(f)
    return b.finish_circuit()


def main(out):
    cfg = Cfg(F2, F1, FYR, GY, DROP2)
    tabs, worst = build_tables(cfg)
    print(f"model worst relative error / 2^-24 = {worst * 2**24:.3f}")
    lay = layout(cfg, tabs)
    c = build_circuit(cfg, tabs, lay)
    s = c.operation_statistics()
    print(
        "qubits",
        c.num_qubits,
        "toffoli(static)",
        s.num_toffoli_like,
        "rotations",
        s.num_z_pow(min_clifford_hierarchy_level=4),
    )
    c.to_file(out, format="kmb")
    print("wrote", out)


def selected_cfg():
    return Cfg(F2, F1, FYR, GY, DROP2)


def config_record(cfg):
    return {"f2": cfg.f2, "f1": cfg.f1, "fyr": cfg.fyr, "gy": cfg.gy, "drop2": sorted(cfg.drop2)}


_PUBLISHED_COEFFICIENT_ROWS = (
    (67, 184284, 189442910),
    (66, 182154, 188710050),
    (65, 180064, 187985630),
    (63, 178014, 187269499),
    (62, 176002, 186561483),
    (61, 174029, 185861435),
    (60, 172092, 185169210),
    (59, 170191, 184484662),
    (58, 168323, 183807650),
    (57, 166490, 183138038),
    (56, 164690, 182475692),
    (55, 162923, 181820481),
    (54, 161186, 181172277),
    (53, 159480, 180530959),
    (52, 157805, 179896401),
    (51, 156158, 179268491),
    (50, 154540, 178647108),
    (49, 152949, 178032144),
    (48, 151386, 177423486),
    (48, 149849, 176821020),
    (47, 148338, 176224658),
    (46, 146852, 175634293),
    (45, 145391, 175049819),
    (44, 143954, 174471143),
    (44, 142540, 173898157),
    (43, 141150, 173330791),
    (42, 139782, 172768943),
    (42, 138435, 172212512),
    (41, 137110, 171661436),
    (40, 135806, 171115614),
    (40, 134523, 170574956),
    (39, 133260, 170039402),
    (39, 132016, 169508850),
    (38, 130792, 168983243),
    (37, 129587, 168462497),
    (37, 128400, 167946525),
    (36, 127231, 167435275),
    (36, 126080, 166928657),
    (35, 124945, 166426619),
    (35, 123828, 165929073),
    (34, 122728, 165435976),
    (34, 121643, 164947236),
    (33, 120574, 164462814),
    (33, 119521, 163982624),
    (32, 118483, 163506626),
    (32, 117460, 163034740),
    (31, 116452, 162566926),
    (31, 115458, 162103106),
    (30, 114478, 161643244),
    (30, 113512, 161187263),
    (29, 112560, 160735129),
    (29, 111621, 160286771),
    (29, 110694, 159842141),
    (28, 109781, 159401203),
    (28, 108879, 158963885),
    (28, 107991, 158530145),
    (27, 107114, 158099945),
    (27, 106249, 157673220),
    (26, 105396, 157249942),
    (26, 104554, 156830046),
    (26, 103723, 156413494),
    (25, 102902, 156000254),
    (25, 102094, 155590262),
    (25, 101295, 155183486),
    (24, 100506, 154779894),
    (24, 99729, 154379426),
    (24, 98960, 153982048),
    (23, 98202, 153587734),
    (23, 97454, 153196425),
    (23, 96715, 152808089),
    (23, 95985, 152422695),
    (22, 95263, 152040209),
    (22, 94552, 151660581),
    (22, 93849, 151283781),
    (22, 93155, 150909775),
    (21, 92468, 150538541),
    (21, 91791, 150170021),
    (21, 91122, 149804197),
    (20, 90461, 149441041),
    (20, 89807, 149080504),
    (20, 89162, 148722566),
    (20, 88525, 148367192),
    (20, 87894, 148014354),
    (19, 87272, 147664030),
    (19, 86657, 147316174),
    (19, 86048, 146970764),
    (19, 85447, 146627772),
    (18, 84854, 146287182),
    (18, 84266, 145948942),
    (18, 83686, 145613040),
    (18, 83112, 145279444),
    (18, 82544, 144948132),
    (17, 81983, 144619086),
    (17, 81429, 144292261),
    (17, 80881, 143967641),
    (17, 80338, 143645203),
    (17, 79802, 143324921),
    (16, 79272, 143006783),
    (16, 78748, 142690743),
    (16, 78229, 142376789),
    (16, 77716, 142064901),
    (16, 77208, 141755051),
    (16, 76707, 141447221),
    (15, 76210, 141141395),
    (15, 75719, 140837537),
    (15, 75233, 140535633),
    (15, 74753, 140235661),
    (15, 74277, 139937603),
    (15, 73806, 139641437),
    (14, 73340, 139347152),
    (14, 72880, 139054712),
    (14, 72424, 138764106),
    (14, 71973, 138475312),
    (14, 71526, 138188314),
    (14, 71084, 137903096),
    (14, 70647, 137619634),
    (13, 70214, 137337922),
    (13, 69785, 137057924),
    (13, 69361, 136779634),
    (13, 68941, 136503030),
    (13, 68525, 136228098),
    (13, 68113, 135954820),
    (13, 67707, 135683180),
    (13, 67303, 135413164),
    (12, 66903, 135144762),
    (12, 66508, 134877938),
    (12, 66116, 134612690),
    (12, 65728, 134349002),
    (95, 260618, 267912731),
    (93, 257604, 266876315),
    (91, 254649, 265851837),
    (90, 251750, 264839056),
    (88, 248905, 263837774),
    (86, 246114, 262847765),
    (85, 243374, 261868805),
    (83, 240685, 260900715),
    (82, 238045, 259943272),
    (80, 235453, 258996304),
    (79, 232907, 258059600),
    (77, 230407, 257133000),
    (76, 227952, 256216297),
    (75, 225540, 255309329),
    (73, 223170, 254411937),
    (72, 220841, 253523932),
    (71, 218553, 252645160),
    (70, 216303, 251775464),
    (68, 214093, 250914700),
    (67, 211918, 250062693),
    (66, 209782, 249219307),
    (65, 207680, 248384399),
    (64, 205614, 247557825),
    (63, 203581, 246739448),
    (62, 201582, 245929136),
    (61, 199616, 245126756),
    (60, 197681, 244332176),
    (59, 195777, 243545275),
    (58, 193903, 242765929),
    (57, 192059, 241994017),
    (56, 190245, 241229423),
    (55, 188459, 240472030),
    (54, 186700, 239721726),
    (54, 184968, 238978392),
    (53, 183264, 238241942),
    (52, 181585, 237512258),
    (51, 179932, 236789237),
    (51, 178303, 236072769),
    (50, 176700, 235362777),
    (49, 175120, 234659153),
    (48, 173563, 233961801),
    (47, 172029, 233270629),
    (47, 170518, 232585538),
    (46, 169029, 231906458),
    (45, 167561, 231233292),
    (45, 166114, 230565944),
    (44, 164688, 229904352),
    (44, 163283, 229248411),
    (43, 161897, 228598063),
    (42, 160531, 227953219),
    (42, 159184, 227313793),
    (41, 157855, 226679725),
    (41, 156545, 226050925),
    (40, 155253, 225427340),
    (39, 153979, 224808886),
    (39, 152722, 224195486),
    (38, 151482, 223587090),
    (38, 150259, 222983610),
    (37, 149052, 222385000),
    (37, 147861, 221791176),
    (37, 146686, 221202083),
    (36, 145526, 220617669),
    (35, 144382, 220037861),
    (35, 143252, 219462595),
    (35, 142138, 218891815),
    (34, 141037, 218325477),
    (34, 139951, 217763501),
    (33, 138878, 217205852),
    (33, 137820, 216652456),
    (32, 136775, 216103278),
    (32, 135742, 215558246),
    (32, 134723, 215017318),
    (31, 133716, 214480450),
    (31, 132722, 213947574),
    (31, 131741, 213418652),
    (30, 130770, 212893641),
    (30, 129812, 212372477),
    (29, 128866, 211855133),
    (29, 127931, 211341539),
    (29, 127007, 210831665),
    (28, 126095, 210325473),
    (28, 125193, 209822899),
    (28, 124302, 209323909),
    (27, 123422, 208828473),
    (27, 122551, 208336530),
    (27, 121691, 207848046),
    (26, 120841, 207362994),
    (26, 120001, 206881310),
    (26, 119170, 206402968),
    (26, 118350, 205927930),
    (25, 117538, 205456166),
    (25, 116736, 204987618),
    (25, 115943, 204522262),
    (24, 115158, 204060072),
    (24, 114383, 203600991),
    (24, 113616, 203144995),
    (24, 112858, 202692049),
    (23, 112108, 202242127),
    (23, 111366, 201795179),
    (23, 110632, 201351183),
    (23, 109907, 200910105),
    (22, 109189, 200471921),
    (22, 108480, 200036585),
    (22, 107778, 199604069),
    (22, 107083, 199174349),
    (21, 106395, 198747401),
    (21, 105716, 198323176),
    (21, 105044, 197901658),
    (21, 104377, 197482814),
    (20, 103720, 197066632),
    (20, 103068, 196653058),
    (20, 102423, 196242078),
    (20, 101785, 195833664),
    (20, 101154, 195427788),
    (19, 100528, 195024434),
    (19, 99910, 194623560),
    (19, 99297, 194225148),
    (19, 98691, 193829172),
    (19, 98092, 193435607),
    (19, 97497, 193044431),
    (18, 96909, 192655627),
    (18, 96328, 192269155),
    (18, 95751, 191884999),
    (18, 95180, 191503135),
    (18, 94616, 191123543),
    (17, 94056, 190746209),
    (17, 93502, 190371093),
    (17, 92954, 189998179),
)


def load_or_build_tables(path=None, regenerate=False):
    """Read explicit public numeric coefficients; optional open regeneration.

    No circuit binary is reused. The 256-row table is Kozykorpesh's published
    exact-domain calibration, pinned to datasetVersionId 21136269. Keeping
    these ordinary integer triples explicit avoids numerical-environment
    changes silently modifying the scored model. The notebook independently
    certifies all 3,133,452 original inputs before building circuits.
    """
    import json
    from pathlib import Path

    cfg = selected_cfg()
    expected_rows = [list(row) for row in _PUBLISHED_COEFFICIENT_ROWS]
    worst = 0.9628198889674466 / 2**24
    if regenerate:
        tables, worst = build_tables(cfg)
        regenerated = [list(tables[(e, a)]) for e in range(2) for a in range(128)]
        assert regenerated == expected_rows, (
            "Regenerated numeric coefficients differ from the explicit public table."
        )
    else:
        tables = {(i // 128, i % 128): tuple(row) for i, row in enumerate(expected_rows)}
    path = Path(path) if path else None
    if path and path.exists():
        saved = json.loads(path.read_text())
        assert saved["config"] == config_record(cfg)
        assert saved["rows"] == expected_rows, (
            "Cached coefficient table differs from the explicitly selected model."
        )
    elif path:
        path.write_text(
            json.dumps(
                {
                    "source": "Kozykorpesh exact-domain calibration; Atlas datasetVersionId21136269",
                    "config": config_record(cfg),
                    "normalized_worst_relative_error": worst,
                    "regeneration_is_optional": True,
                    "rows": expected_rows,
                },
                indent=2,
            )
        )
    return cfg, tables, worst


def integer_model(xs, cfg, tabs):
    """Exact emitted arithmetic in units 2^-(cfg.fy+10) turns, before mod 1."""
    xs = np.asarray(xs, dtype=np.int64)
    if not ((xs >= 1) & (xs <= 3133452)).all():
        raise ValueError("The supported competition domain is x=1..3133452")
    # All inputs fit exactly in float64; frexp gives the exact binary exponent.
    shifts = 21 - (np.frexp(xs)[1].astype(np.int64) - 1)
    norm = xs << shifts
    assert ((norm >> 21) == 1).all(), "Normalized top-bit borrowing requires x[21]=1"
    parity = shifts & 1
    index = ((norm >> 14) & 127) + 128 * parity
    table = np.asarray([tabs[(e, a)] for e in range(2) for a in range(128)], dtype=np.int64)
    K2, K1, K0 = (table[index, j] for j in range(3))
    d = norm & ((1 << DB) - 1)
    signs = [2 * ((d >> i) & 1) - 1 for i in range(DB)]
    R = K1.copy()
    for i in cfg.rrow_list():
        R -= signs[i] * rd(K2, cfg.sh2[i])
    Q = np.zeros_like(xs)
    for i in reversed(range(DB)):
        Q += signs[i] * rd(R, cfg.sh1[i])
    Y = ((K0 - Q) >> cfg.gy) << cfg.gy
    if (R < 0).any() or (Y < 0).any():
        raise OverflowError("An unsigned intermediate is negative")
    return Y << (shifts >> 1)


def exhaustive_certificate(cfg, tabs, chunk_size=131072):
    """Every promised basis input, exact integer target and tolerance.

    This is an arithmetic certificate. Ancillas, restored inputs and phase
    feed-forward require fresh circuit simulation and the official verifier.
    """
    output_fraction = cfg.fy + 10
    worst_ratio, worst_x, min_margin = 0.0, 0, 1 << 64
    for start in range(1, 3133453, chunk_size):
        xs = np.arange(start, min(start + chunk_size, 3133453), dtype=np.int64)
        values = integer_model(xs, cfg, tabs).astype(np.uint64)
        phase = values << np.uint64(64 - output_fraction)
        roots = [math.isqrt((1 << 130) // int(x)) for x in xs]
        target = np.array([((r + 1) >> 1) & ((1 << 64) - 1) for r in roots], dtype=np.uint64)
        tolerance = np.array([min(r >> 25, 1 << 63) for r in roots], dtype=np.uint64)
        error = np.abs((phase - target).view(np.int64)).view(np.uint64)
        ratio = error / tolerance
        i = int(np.argmax(ratio))
        if float(ratio[i]) > worst_ratio:
            worst_ratio, worst_x = float(ratio[i]), int(xs[i])
        if (error > tolerance).any():
            i = int(np.flatnonzero(error > tolerance)[0])
            return {"pass": False, "first_bad_x": int(xs[i]), "ratio": float(ratio[i])}
        min_margin = min(min_margin, int((tolerance - error).min()))
    return {
        "pass": True,
        "tested_inputs": 3133452,
        "worst_ratio": worst_ratio,
        "worst_x": worst_x,
        "min_margin_units": min_margin,
        "phase_fraction_bits": output_fraction,
    }


def conditional_toffoli_discount(circuit):
    """Mean discount for this emitted family's fair HMR conditional branches."""
    depth = 0
    total = 0
    discount = 0.0
    for line in str(circuit).splitlines():
        opcode = line.split(" ", 1)[0]
        if opcode == "PUSH_CONDITION":
            depth += 1
        elif opcode == "POP_CONDITION":
            depth -= 1
        elif opcode in ["CCX", "CSWAP"]:
            total += 1
            if depth:
                assert depth == 1, "Expected one fair measurement bit in a Toffoli branch."
                discount += 0.5
    assert total == int(circuit.operation_statistics().num_toffoli_like)
    assert depth == 0
    return discount


def estimated_resources(circuit, certificate):
    """Model-based cost; full unchanged official verification remains required."""
    assert certificate["pass"] and certificate["min_margin_units"] > 0
    stats = circuit.operation_statistics()
    rotations = int(stats.num_z_pow(min_clifford_hierarchy_level=4))
    epsilon = math.pi * (certificate["min_margin_units"] / 2**64) / rotations
    per_rotation = 0.56 * math.log2(1 / epsilon) + 5.3
    static_toffoli = int(stats.num_toffoli_like)
    # Fair flag measurements activate the emitted normalization phase-fixup
    # Toffolis on half the shots. Read their actual branch counts so the
    # original schedules discount 11 and the promise rewrite discounts 10.5.
    normalization_discount = conditional_toffoli_discount(circuit)
    expected_toffoli = static_toffoli - normalization_discount
    t_like = int(stats.num_t_like)
    extra = (t_like + rotations * per_rotation) / 4
    return {
        "qubits": circuit.num_qubits,
        "static_toffoli": static_toffoli,
        "expected_mean_toffoli": expected_toffoli,
        "t_like": t_like,
        "rotations": rotations,
        "rotation_epsilon": epsilon,
        "estimated_expected_cost": circuit.num_qubits * (expected_toffoli + extra),
        "static_cost_upper_estimate": circuit.num_qubits * (static_toffoli + extra),
        "full_official_verification_required": True,
    }


# ---- Our isolated phase scheduling, folded carry and borrowed workspace ----
def folded_gadd(b, a, t, cin=None, sub=False, borrowed=()):
    a, t = list(a), list(t)
    o, n = len(a), len(t)
    assert o <= n and n >= 1
    if sub:
        for q in t:
            b.x(q)
    if n == 1:
        if o:
            b.cx(a[0], t[0])
        if cin is not None:
            b.cx(cin, t[0])
        if sub:
            for q in t:
                b.x(q)
        return
    borrowed = list(borrowed)[: max(0, n - 2)]
    count = max(0, n - 2 - len(borrowed))
    allocated = b.alloc_qubits(count) if count else []
    c = borrowed + list(allocated)

    def C(i: int) -> object:
        return cin if i == 0 else c[i - 1]

    for i in range(n - 2):
        if i == 0 and cin is None:
            if o:
                b.init_and(a[0], t[0], C(1))
        elif i < o:
            b.cx(C(i), a[i])
            b.cx(C(i), t[i])
            b.init_and(a[i], t[i], C(i + 1))
            b.cx(C(i), C(i + 1))
        else:
            b.init_and(C(i), t[i], C(i + 1))
    i = n - 2
    carry = C(i)
    if carry is None:
        if i < o:
            b.ccx(a[i], t[i], t[-1])
            b.cx(a[i], t[i])
    elif i < o:
        b.cx(carry, a[i])
        b.cx(carry, t[i])
        b.ccx(a[i], t[i], t[-1])
        b.cx(carry, t[-1])
        b.cx(carry, a[i])
        b.cx(a[i], t[i])
    else:
        b.ccx(carry, t[i], t[-1])
        b.cx(carry, t[i])
    if n - 1 < o:
        b.cx(a[n - 1], t[n - 1])
    for i in reversed(range(n - 2)):
        if i == 0 and cin is None:
            if o:
                b.del_and(a[0], t[0], C(1))
                b.cx(a[0], t[0])
        elif i < o:
            b.cx(C(i), C(i + 1))
            b.del_and(a[i], t[i], C(i + 1))
            b.cx(C(i), a[i])
            b.cx(a[i], t[i])
        else:
            b.del_and(C(i), t[i], C(i + 1))
            b.cx(C(i), t[i])
    if count:
        b.free(allocated)
    if sub:
        for q in t:
            b.x(q)


def truth_table_check(max_width=6, borrowed_count=0):
    class Bits:
        def __init__(self) -> None:
            self.v: list[int] = []

        def alloc_qubits(self, n: int) -> list[int]:
            ids = list(range(len(self.v), len(self.v) + n))
            self.v.extend([0] * n)
            return ids

        def x(self, q: int) -> None:
            self.v[q] ^= 1

        def cx(self, a: int, t: int) -> None:
            self.v[t] ^= self.v[a]

        def ccx(self, a: int, b: int, t: int) -> None:
            self.v[t] ^= self.v[a] & self.v[b]

        def init_and(self, a: int, b: int, t: int) -> None:
            assert self.v[t] == 0
            self.v[t] = self.v[a] & self.v[b]

        def del_and(self, a: int, b: int, t: int) -> None:
            assert self.v[t] == (self.v[a] & self.v[b])
            self.v[t] = 0

        def free(self, reg: list[int]) -> None:
            assert not any(self.v[q] for q in reg)

        def put(self, reg: list[int], value: int) -> None:
            for i, q in enumerate(reg):
                self.v[q] = (value >> i) & 1

        def read(self, reg: list[int]) -> int:
            return sum(self.v[q] << i for i, q in enumerate(reg))

    count = 0
    for n in range(1, max_width + 1):
        for m in range(n + 1):
            for a in range(1 << m):
                for value in range(1 << n):
                    for cin_value in (None, 0, 1):
                        for sub in (False, True):
                            b = Bits()
                            areg = b.alloc_qubits(m)
                            target = b.alloc_qubits(n)
                            cin = b.alloc_qubits(1)[0] if cin_value is not None else None
                            if cin is not None:
                                b.v[cin] = cin_value
                            b.put(areg, a)
                            b.put(target, value)
                            borrowed = b.alloc_qubits(borrowed_count)
                            folded_gadd(b, areg, target, cin, sub, borrowed=borrowed)
                            delta = a + (cin_value or 0)
                            expected = (value - delta if sub else value + delta) % (1 << n)
                            assert b.read(target) == expected, (n, m, a, value, cin_value, sub)
                            assert b.read(areg) == a
                            assert cin is None or b.v[cin] == cin_value
                            assert not any(b.v[q] for q in borrowed)
                            count += 1
    return count


def build_split_circuit(cfg, tabs, lay, adder=None, borrow_top=False):
    if adder is None:
        adder = gadd
    w2, w1, LB, S, whi = cfg.w2, lay["w1"], lay["LB"], lay["S"], lay["whi"]
    b = km.CircuitBuilder()
    x = b.create_quantum_register(W, name="x")
    flags = b.alloc_qubits(5)
    normalize(b, x, flags)
    # Every promised positive input normalizes to x[21]=1. The arithmetic
    # address uses x[14:21] and delta x[:14], so this wire is idle in the core.
    if borrow_top:
        b.x(x[-1])
        raw_adder = adder

        def adder(builder, a, t, cin=None, sub=False):
            return raw_adder(builder, a, t, cin, sub, borrowed=(x[-1],))

    c2 = b.alloc_qubits(w2)
    R = b.alloc_qubits(w1)
    Lr = b.alloc_qubits(LB)
    Yh = b.alloc_qubits(whi)
    remaining = km.array(list(Lr) + list(R) + list(c2))
    target = km.array(list(remaining) + list(Yh))
    table = [0] * 256
    rest_table = [0] * 256
    high_table = [0] * 256
    for (e, a), (K2, K1, _K0) in tabs.items():
        idx = a + 128 * e
        rest_table[idx] = lay["lo"][(e, a)] | (K1 << LB) | (K2 << (LB + w1))
        high_table[idx] = lay["hi"][(e, a)]
        table[idx] = rest_table[idx] | (high_table[idx] << (LB + w1 + w2))
    address = km.array([*list(x[14:21]), flags[0]])
    delta = x[:DB]

    def rrows(inverse):
        seq = cfg.rrow_list()
        if inverse:
            seq = list(reversed(seq))
        for i in seq:
            sh = cfg.sh2[i]
            if sh >= 0:
                off = c2[sh:w2]
                t = R[:w1]
                cin = c2[sh - 1] if sh >= 1 else None
            else:
                off = c2[:w2]
                t = R[-sh:w1]
                cin = None
            b.cx(delta[i], t)
            adder(b, off, t, cin, sub=inverse)
            b.cx(delta[i], t)

    def yrows(inverse):
        seq = list(reversed(range(DB)))
        if inverse:
            seq.reverse()
        for i in seq:
            sh = cfg.sh1[i]
            assert 0 <= sh < w1
            cin = R[sh - 1] if sh >= 1 else None
            b.cx(delta[i], Lr)
            adder(b, R[sh:w1], Lr, cin, sub=inverse)
            b.cx(delta[i], Lr)

    def phase(reg, base_bit):
        ext = b.alloc_qubits(10)
        word = km.array(list(reg) + list(ext))
        shift_stages(b, flags, word, len(reg), False)
        for j in range(len(word)):
            exponent = base_bit + j - cfg.fy - 10
            if exponent < 0:
                b.z_pow(word[j], Fraction(2) ** (exponent + 1))
        shift_stages(b, flags, word, len(reg), True)
        b.free(ext)

    b.init_lookup(table=table, address=address, target=target)
    phase(Yh, S)
    b.del_lookup(table=high_table, address=address, target=Yh)
    b.free(Yh)
    rrows(False)
    yrows(False)
    phase(Lr[cfg.gy : LB], cfg.gy)
    yrows(True)
    rrows(True)
    b.del_lookup(table=rest_table, address=address, target=remaining)
    b.free(remaining)
    if borrow_top:
        b.x(x[-1])
    unnormalize(b, x, flags)
    b.free(flags)
    return b.finish_circuit()


def build_previous_schedule(cfg, tabs, lay=None):
    """Regenerate 91Q circuit from the explicit model and credited schedules."""
    if lay is None:
        lay = layout(cfg, tabs)
    return build_split_circuit(cfg, tabs, lay, adder=folded_gadd, borrow_top=True)


# The derivative-split version of this API is defined below.

# The derivative-split version of this API is defined below.


# The derivative-split version of this API is defined below.


# The derivative-split version of this API is defined below.


def cli():
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Generate the credited signed-digit circuit and memory-schedule adaptation."
    )
    parser.add_argument("output", nargs="?", default="invsqrt.kmb")
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()
    cfg, tabs, _ = load_or_build_tables()
    certificate = exhaustive_certificate(cfg, tabs)
    if not certificate["pass"]:
        raise ArithmeticError(certificate)
    lay = layout(cfg, tabs)
    circuit = (
        build_memory_baseline(cfg, tabs, lay) if args.baseline else build_selected(cfg, tabs, lay)
    )
    circuit.to_file(args.output, format="kmb")
    report = estimated_resources(circuit, certificate)
    report["arithmetic_certificate"] = certificate
    report["variant"] = "sumit_baseline" if args.baseline else selected_descriptor()["variant"]
    Path(args.output + ".resources.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def _rudra_gadd_top_reused(b, source, target, cin=None, sub=False):
    """Gidney addition with n-2 clean carries; the last carry targets the MSB."""
    a, t = list(source), list(target)
    n, m = len(t), len(a)
    assert n >= 1 and m <= n
    if sub:
        b.x(km.array(t))
    carries = b.alloc_qubits(n - 2) if n > 2 else km.array([])
    for i in range(n - 1):
        previous = cin if i == 0 else carries[i - 1]
        last = i == n - 2
        out = t[-1] if last else carries[i]
        if previous is None:
            if i < m:
                (b.ccx if last else b.init_and)(a[i], t[i], out)
        elif i < m:
            b.cx(previous, a[i])
            b.cx(previous, t[i])
            (b.ccx if last else b.init_and)(a[i], t[i], out)
            b.cx(previous, out)
        else:
            (b.ccx if last else b.init_and)(previous, t[i], out)
    if n - 1 < m:
        b.cx(a[n - 1], t[n - 1])
    if n == 1 and cin is not None:
        b.cx(cin, t[0])
    for i in reversed(range(n - 1)):
        previous = cin if i == 0 else carries[i - 1]
        last = i == n - 2
        if previous is None:
            if i < m:
                if not last:
                    b.del_and(a[i], t[i], carries[i])
                b.cx(a[i], t[i])
        elif i < m:
            if not last:
                b.cx(previous, carries[i])
                b.del_and(a[i], t[i], carries[i])
            b.cx(previous, a[i])
            b.cx(a[i], t[i])
        else:
            if not last:
                b.del_and(previous, t[i], carries[i])
            b.cx(previous, t[i])
    if n > 2:
        b.free(carries)
    if sub:
        b.x(km.array(t))


def prepare_derivative_split(reference, cfg, tables):
    d = np.arange(1 << reference.DB, dtype=np.int64)
    coarse, residual = {}, {}
    bounds = {
        "normalized_addresses_checked": 0,
        "min_residual": 1 << 30,
        "max_residual": 0,
        "max_table_residual": 0,
        "max_coarse": 0,
    }
    for key, (k2, k1, _) in tables.items():
        derivative, _ = reference.pipeline(cfg, k2, k1, d)
        base = int(derivative.min()) >> 11
        low = derivative - (base << 11)
        coarse[key], residual[key] = base, k1 - (base << 11)
        assert 0 <= residual[key] < 1 << 12
        assert (low >= 0).all() and (low < 1 << 12).all()
        assert ((base + (low >> 11)) < 1 << 7).all()
        joined = (low & 2047) + ((base + (low >> 11)) << 11)
        assert np.array_equal(joined, derivative)
        bounds["normalized_addresses_checked"] += len(d)
        bounds["min_residual"] = min(bounds["min_residual"], int(low.min()))
        bounds["max_residual"] = max(bounds["max_residual"], int(low.max()))
        bounds["max_table_residual"] = max(bounds["max_table_residual"], residual[key])
        bounds["max_coarse"] = max(bounds["max_coarse"], base)
    assert cfg.w2 >= 6 and len(cfg.rrow_list()) == 8
    return {"coarse": coarse, "residual": residual}, bounds


def _build_derivative_circuit(reference, cfg, tables, layout, split, borrow_top=False):
    assert (layout["w1"], layout["LB"], layout["S"], layout["whi"]) == (18, 21, 20, 8)
    b = km.CircuitBuilder()
    x = b.create_quantum_register(22, name="x")
    flags = b.alloc_qubits(5)
    # These coefficients are still zero during normalization; allocate afterward.
    reference.normalize(b, x, flags)
    if borrow_top:
        b.x(x[-1])

    def adder(builder, a, t, cin=None, sub=False):
        if borrow_top:
            return folded_gadd(builder, a, t, cin, sub, borrowed=(x[-1],))
        return _rudra_gadd_top_reused(builder, a, t, cin, sub)

    c2, residual, coarse, low, high = [b.alloc_qubits(n) for n in [cfg.w2, 12, 7, 21, 8]]
    full_target = km.array(list(low) + list(residual) + list(c2) + list(high) + list(coarse))
    rest_target = km.array(list(low) + list(residual) + list(c2) + list(coarse))
    full_table, rest_table, high_table = [0] * 256, [0] * 256, [0] * 256
    for (parity, address), (k2, _, _) in tables.items():
        key, index = (parity, address), address + 128 * parity
        common = layout["lo"][key] | (split["residual"][key] << 21) | (k2 << 33)
        full_table[index] = (
            common | (layout["hi"][key] << (33 + cfg.w2)) | (split["coarse"][key] << (41 + cfg.w2))
        )
        rest_table[index] = common | (split["coarse"][key] << (33 + cfg.w2))
        high_table[index] = layout["hi"][key]
    address = km.array([*list(x[14:21]), flags[0]])
    delta = x[:14]

    def phase_register(register, low_bit):
        ext = b.alloc_qubits(10)
        word = km.array(list(register) + list(ext))
        reference.shift_stages(b, flags, word, len(register), False)
        for i, qubit in enumerate(word):
            exponent = low_bit + i - cfg.fy - 10
            if exponent < 0:
                b.z_pow(qubit, Fraction(2) ** (exponent + 1))
        reference.shift_stages(b, flags, word, len(register), True)
        b.free(ext)

    def inner(inverse):
        rows = cfg.rrow_list()
        for i in reversed(rows) if inverse else rows:
            shift = cfg.sh2[i]
            # A narrower curvature coefficient can require an exact left shift.
            # Add to the corresponding residual suffix, leaving lower bits intact.
            if shift >= 0:
                addend = c2[shift:]
                target = residual
                carry = c2[shift - 1] if shift else None
            else:
                addend = c2
                target = residual[-shift:]
                carry = None
            b.cx(delta[i], target)
            adder(b, addend, target, carry, sub=inverse)
            b.cx(delta[i], target)

    def outer(derivative, inverse):
        for i in range(14) if inverse else reversed(range(14)):
            shift = cfg.sh1[i]
            carry = derivative[shift - 1] if shift else None
            b.cx(delta[i], low)
            adder(b, derivative[shift:], low, carry, sub=inverse)
            b.cx(delta[i], low)

    b.init_lookup(table=full_table, address=address, target=full_target)
    phase_register(high, 20)
    # High and the shift workspace have been restored. For every X-measurement
    # mask m, unlookup contributes (-1)^(m.high(address)); its phase correction
    # contributes the same sign. Their product is +1, preserving our phase.
    # Address and its parity flag stay live and unchanged until after erasure.
    b.del_lookup(table=high_table, address=address, target=high)
    b.free(high)
    inner(False)
    adder(b, residual[11:], coarse)
    derivative = residual[:11] + coarse
    outer(derivative, False)
    # Include bit20 here. There is no output join: floor((L+2^20H)/4)
    # equals floor(L/4)+2^18H, so the two diagonal phases add exactly.
    phase_register(low[cfg.gy :], cfg.gy)
    outer(derivative, True)
    adder(b, residual[11:], coarse, sub=True)
    inner(True)
    b.del_lookup(table=rest_table, address=address, target=rest_target)
    b.free(rest_target)
    if borrow_top:
        b.x(x[-1])
    reference.unnormalize(b, x, flags)
    b.free(flags)
    return b.finish_circuit()


def _derivative_reference(promise):
    return types.SimpleNamespace(
        **dict(
            globals(),
            normalize=normalize if promise else _combined_normalize,
            unnormalize=unnormalize if promise else _baseline_unnormalize,
        )
    )


def _build_derivative_variant(cfg, tabs, lay, borrow, promise):
    if lay is None:
        lay = layout(cfg, tabs)
    reference = _derivative_reference(promise)
    split, _bounds = prepare_derivative_split(reference, cfg, tabs)
    return _build_derivative_circuit(reference, cfg, tabs, lay, split, borrow_top=borrow)


def build_selected(cfg, tabs, lay=None):
    """Regenerate selected92Q derivative split with borrowed carry and domain rewrite."""
    return _build_derivative_variant(cfg, tabs, lay, borrow=True, promise=True)


def build_memory_baseline(cfg, tabs, lay=None):
    """Same Kozykorpesh coefficient model, Rudra/Kirill derivative schedule:93Q."""
    return _build_derivative_variant(cfg, tabs, lay, borrow=False, promise=False)


def build_borrowed_baseline(cfg, tabs, lay=None):
    """Same derivative schedule with the normalized input bit borrowed:92Q."""
    return _build_derivative_variant(cfg, tabs, lay, borrow=True, promise=False)


def build_combined_baseline(cfg, tabs, lay=None):
    """Compatibility alias for the borrowed-carry comparison."""
    return build_borrowed_baseline(cfg, tabs, lay)


def comparison_report(cfg, tabs, certificate, lay=None):
    """One fixed coefficient table: derivative schedule, borrow, domain rewrite."""
    if lay is None:
        lay = layout(cfg, tabs)
    return {
        "baseline": estimated_resources(build_memory_baseline(cfg, tabs, lay), certificate),
        "borrowed": estimated_resources(build_borrowed_baseline(cfg, tabs, lay), certificate),
        "selected": estimated_resources(build_selected(cfg, tabs, lay), certificate),
        "selected_descriptor": selected_descriptor(),
    }


def selected_descriptor():
    return {
        "variant": "kozy_table_rudra_split_alex_schedule_borrow_and_domain",
        "source_author": "Sumit Chongder; Rudra Bhaskar; Alex Chilton; Kozykorpesh; Kirill Tymchishin; anazemcev / Artem1410",
        "source_url": "https://www.kaggle.com/code/kirilltym/quantum-circuits-qrom-tower42?scriptVersionId=361491814",
        "coefficient_source": "Kozykorpesh Quantum Circuit Toolbox datasetVersionId21136269, code/coefficients.json; same256-row model as our preceding91Q revision",
        "configuration": {"f2": 8, "f1": 19, "fyr": 27, "gy": 1, "drop2": [5]},
        "license_as_published": "Apache-2.0, subject to retained upstream notices",
        "baseline_definition": "Same fixed Kozykorpesh table with the Rudra derivative split and Alex normalization/shifts, before normalized-input carry borrowing:93Q",
        "baseline_canonical_sha256": "9e67c00941ff0623cfaf22f133fe7f70740f8b0a163a4ba64c6fdcd1c9446295",
        "borrowed_canonical_sha256": "f3f8457058aecdcf2650e00b411234899cb01933b79a1f6519c0bf28b49591bc",
        "selected_canonical_sha256": "63f70e623dd995f4f87ca13254f46f991a5ae545d4c20c80e8510f5ff0a49bbd",
        "historical_kirill95_sha256": "497b161bc468654b5b2570cfaa006ce336dec9b8c89f82f048fcca4e6e8f7919",
        "previous_same_table91_sha256": "e3db0cc65b4771b640fbcd73d770d2015c232e9f54bd69479786fb5445ca76b9",
        "measured_counts": {"qubits": 92, "static_toffoli": 1472, "rotations": 45},
        "expected_mean_toffoli": 1461.5,
        "estimated_cost": 164645.89641938862,
        "own_changes_in_this_revision": [
            "Combine the published derivative-split representation with the previously selected explicit Kozykorpesh coefficient model",
            "Parameterize curvature storage and exact left-shift rows without changing the integer pipeline",
            "Borrow normalized x[21] as clean adder carry, restoring it before inverse normalization",
            "Retain our exact original-input top-bit mutual-exclusion rewrite",
        ],
        "upstream_contributions": "Rudra derivative split/top-carry/static-high phase checkpoint; Alex prefix flags/empty-target shift; Kirill exact integration; Sumit model; Kozykorpesh precision calibration",
        "full_official_verification": "Required separately; actual unchanged full-verifier report identifies the selected hash",
    }


if __name__ == "__main__":
    cli()
