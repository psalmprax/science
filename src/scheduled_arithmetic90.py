# SPDX-License-Identifier: Apache-2.0
# Kozykorpesh public V3/run361677445, https://www.kaggle.com/code/kozykappa/quantum-ai-lb-score-6-68108?scriptVersionId=361677445
# Source-only adaptation: explicit coefficients supplied by caller; no Toolbox or binary input.
# Original notices and credited Sumit/Rudra/Alex/Kirill/Artem lineage retained below.
# SPDX-License-Identifier: Apache-2.0
# Retained source credits: Sumit Chongder (signed digits/Gidney arithmetic),
# Rudra Bhaskar (derivative split), Alex Chilton (prefix/AND shifts),
# Kirill Tymchishin (integration), Kozykorpesh (calibrated coefficients),
# anazemcev / Artem1410 (memory schedule and input-domain borrowing).
# This module retains the required arithmetic from the supplied 5.77242 notebook.
import math
from fractions import Fraction
import numpy as np
import kickmix as km
W=22; MB=21; DB=14; NSEG=128; MAX_X=3133452
NSTAGES=[(4,16),(3,8),(2,4),(1,2),(0,1)]


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
    if s < 0: return v << (-s)
    r = v >> s
    if s >= 1: r = r + ((v >> (s - 1)) & 1)
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


def layout(cfg, tabs, LB=21, S=20):
    d = np.arange(2**DB, dtype=np.int64)
    assert max(t[0] for t in tabs.values()) < (1 << cfg.w2)
    Rmax = 0; per = {}; Ymax = 0
    for key, (K2, K1, K0) in tabs.items():
        R, Q = pipeline(cfg, K2, K1, d)
        assert R.min() >= 0
        Rmax = max(Rmax, int(R.max()), K1)
        per[key] = (int(Q.min()), int(Q.max()))
        Ymax = max(Ymax, K0 - per[key][0])
    w1 = Rmax.bit_length()
    lo_tab, hi_tab = {}, {}
    for key, (K2, K1, K0) in tabs.items():
        qmin, qmax = per[key]
        base = K0 % (1 << S); j = 0
        while base + j * (1 << S) - qmax < 0: j += 1
        k0lo = base + j * (1 << S)
        assert k0lo - qmin < (1 << LB), 'layout infeasible'
        lo_tab[key] = k0lo; hi_tab[key] = (K0 - k0lo) >> S
        assert hi_tab[key] >= 0
    whi = max(max(hi_tab.values()).bit_length(), 1)
    return dict(w1=w1, LB=LB, S=S, whi=whi, lo=lo_tab, hi=hi_tab)


def phase_all(b, controls):
    k = len(controls)
    if k == 1: b.z(controls[0]); return
    if k == 2: b.cz(controls[0], controls[1]); return
    anc = b.alloc_qubits(k - 2)
    b.init_and(controls[0], controls[1], anc[0])
    for j in range(1, k - 2): b.init_and(anc[j - 1], controls[j + 1], anc[j])
    b.cz(anc[k - 3], controls[k - 1])
    for j in reversed(range(1, k - 2)): b.del_and(anc[j - 1], controls[j + 1], anc[j])
    b.del_and(controls[0], controls[1], anc[0])
    b.free(anc)


def normalize(b, x, f):
    """v3 step 1: f4, f3, f2 (the binary leading-zero count div 4) from one prefix-AND chain on the unshifted x.
    chain[m] = [x[W-m-2:W] == 0] (top m+2 bits zero); p_j = chain[4j-2] = [lz >= 4j].  f4 = p4, f3 = p2 ^ p4,
    f2 = p1 ^ p2 ^ p3 ^ p4 ^ p5 (nested prefixes, so XOR = bit 2 of lz).  19 ANDs instead of 15 + 7 + 3."""
    bits = [x[W - 1 - m] for m in range(20)]                  # x21, x20, ..., x2
    b.x(km.array(bits))
    chain = b.alloc_qubits(19)
    b.x(chain[0]); b.cx(bits[0],chain[0]); b.cx(bits[1],chain[0])
    for m in range(1, 19): b.init_and(chain[m - 1], bits[m + 1], chain[m])
    p = {j: chain[4 * j - 2] for j in range(1, 6)}
    b.cx(p[4], f[4])
    b.cx(p[2], f[3]); b.cx(p[4], f[3])
    for j in range(1, 6): b.cx(p[j], f[2])
    for m in reversed(range(1, 19)): b.del_and(chain[m - 1], bits[m + 1], chain[m])
    b.cx(bits[1],chain[0]); b.cx(bits[0],chain[0]); b.x(chain[0])
    b.free(chain)
    b.x(km.array(bits))
    for (k, d) in NSTAGES:
        top = x[W - d:W]
        if d == 1:
            b.cx(x[W - 1], f[0]); b.x(f[0])
        elif d == 2:
            b.x(top); b.ccx(top[0], top[1], f[1]); b.x(top)
        for i in reversed(range(0, W - d)): b.cswap(f[k], x[i], x[i + d])


def promise_phase_original_top(b,controls):
    """Phase the original top16-zero predicate using its promised bit relation."""
    assert len(controls)==16
    controls=list(reversed(controls))
    k=len(controls)
    anc=b.alloc_qubits(k-2)
    b.x(anc[0]); b.cx(controls[0],anc[0]); b.cx(controls[1],anc[0])
    for j in range(1,k-2):
        b.init_and(anc[j-1],controls[j+1],anc[j])
    b.cz(anc[k-3],controls[k-1])
    for j in reversed(range(1,k-2)):
        b.del_and(anc[j-1],controls[j+1],anc[j])
    b.cx(controls[1],anc[0]); b.cx(controls[0],anc[0]); b.x(anc[0])
    b.free(anc)


def unnormalize(b, x, f):
    for (k, d) in reversed(NSTAGES):
        for i in range(0, W - d): b.cswap(f[k], x[i], x[i + d])
        top = x[W - d:W]
        if d == 1:
            b.x(f[0]); b.cx(x[W - 1], f[0])
        else:
            m = b.alloc_bits(1)[0]
            b.hmr(f[k], m)
            b.push_condition(m)                     # fixup runs on half the shots on average
            b.x(top)
            if d==16: promise_phase_original_top(b,list(top))
            else: phase_all(b,list(top))
            b.x(top)
            b.pop_condition()
            b.free(m)


def shift_stages(b, f, reg, L, inverse):
    stages = [(1, 1), (2, 2), (3, 4), (4, 8)]
    occ = []; o = L
    for (k, d) in stages: occ.append(o); o = min(o + d, len(reg))
    seq = list(enumerate(stages))
    if not inverse:
        for idx, (k, d) in seq:
            for i in reversed(range(0, occ[idx])):
                if i + d >= len(reg): continue
                if i + d >= occ[idx]:                       # v2: target is |0> for every f
                    b.init_and(f[k], reg[i], reg[i + d]); b.cx(reg[i + d], reg[i])
                else:
                    b.cswap(f[k], reg[i], reg[i + d])
    else:
        for idx, (k, d) in reversed(seq):
            for i in range(0, occ[idx]):
                if i + d >= len(reg): continue
                if i + d >= occ[idx]:
                    b.cx(reg[i + d], reg[i]); b.del_and(f[k], reg[i], reg[i + d])
                else:
                    b.cswap(f[k], reg[i], reg[i + d])


def integer_model(xs,cfg,tabs):
    """Exact emitted arithmetic in units 2^-(cfg.fy+10) turns, before mod 1."""
    xs=np.asarray(xs,dtype=np.int64)
    if not ((xs>=1)&(xs<=3133452)).all():
        raise ValueError('The supported competition domain is x=1..3133452')
    # All inputs fit exactly in float64; frexp gives the exact binary exponent.
    shifts=21-(np.frexp(xs)[1].astype(np.int64)-1)
    norm=xs << shifts
    assert ((norm >> 21)==1).all(), 'Normalized top-bit borrowing requires x[21]=1'
    parity=shifts&1
    index=((norm >> 14)&127)+128*parity
    table=np.asarray([tabs[(e,a)] for e in range(2) for a in range(128)],dtype=np.int64)
    K2,K1,K0=(table[index,j] for j in range(3))
    d=norm&((1 << DB)-1)
    signs=[2*((d >> i)&1)-1 for i in range(DB)]
    R=K1.copy()
    for i in cfg.rrow_list():
        R-=signs[i]*rd(K2,cfg.sh2[i])
    Q=np.zeros_like(xs)
    for i in reversed(range(DB)):
        Q+=signs[i]*rd(R,cfg.sh1[i])
    Y=((K0-Q) >> cfg.gy) << cfg.gy
    if (R<0).any() or (Y<0).any():
        raise OverflowError('An unsigned intermediate is negative')
    return Y << (shifts >> 1)


def conditional_toffoli_discount(circuit):
    """Mean discount for this emitted family's fair HMR conditional branches."""
    depth=0
    total=0
    discount=0.
    for line in str(circuit).splitlines():
        opcode=line.split(' ',1)[0]
        if opcode=='PUSH_CONDITION':depth+=1
        elif opcode=='POP_CONDITION':depth-=1
        elif opcode in ['CCX','CSWAP']:
            total+=1
            if depth:
                assert depth==1,'Expected one fair measurement bit in a Toffoli branch.'
                discount+=.5
    assert total==int(circuit.operation_statistics().num_toffoli_like)
    assert depth==0
    return discount


def estimated_resources(circuit,certificate):
    """Model-based cost; full unchanged official verification remains required."""
    assert certificate['pass'] and certificate['min_margin_units']>0
    stats=circuit.operation_statistics()
    rotations=int(stats.num_z_pow(min_clifford_hierarchy_level=4))
    epsilon=math.pi*(certificate['min_margin_units']/2**64)/rotations
    per_rotation=0.56*math.log2(1/epsilon)+5.3
    static_toffoli=int(stats.num_toffoli_like)
    # Fair flag measurements activate the emitted normalization phase-fixup
    # Toffolis on half the shots. Read their actual branch counts so the
    # original schedules discount 11 and the promise rewrite discounts 10.5.
    normalization_discount=conditional_toffoli_discount(circuit)
    expected_toffoli=static_toffoli-normalization_discount
    t_like=int(stats.num_t_like)
    extra=(t_like+rotations*per_rotation)/4
    return {'qubits':circuit.num_qubits,'static_toffoli':static_toffoli,
            'expected_mean_toffoli':expected_toffoli,'t_like':t_like,
            'rotations':rotations,'rotation_epsilon':epsilon,
            'estimated_expected_cost':circuit.num_qubits*(expected_toffoli+extra),
            'static_cost_upper_estimate':circuit.num_qubits*(static_toffoli+extra),
            'full_official_verification_required':True}


def folded_gadd(b,a,t,cin=None,sub=False,borrowed=()):
    a,t=list(a),list(t)
    o,n=len(a),len(t)
    assert o<=n and n>=1
    if sub:
        for q in t:b.x(q)
    if n==1:
        if o:b.cx(a[0],t[0])
        if cin is not None:b.cx(cin,t[0])
        if sub:
            for q in t:b.x(q)
        return
    borrowed=list(borrowed)[:max(0,n-2)]
    count=max(0,n-2-len(borrowed))
    allocated=b.alloc_qubits(count) if count else []
    c=borrowed+list(allocated)
    def C(i):return cin if i==0 else c[i-1]
    for i in range(n-2):
        if i==0 and cin is None:
            if o:b.init_and(a[0],t[0],C(1))
        elif i<o:
            b.cx(C(i),a[i]);b.cx(C(i),t[i])
            b.init_and(a[i],t[i],C(i+1));b.cx(C(i),C(i+1))
        else:
            b.init_and(C(i),t[i],C(i+1))
    i=n-2
    carry=C(i)
    if carry is None:
        if i<o:
            b.ccx(a[i],t[i],t[-1]);b.cx(a[i],t[i])
    elif i<o:
        b.cx(carry,a[i]);b.cx(carry,t[i])
        b.ccx(a[i],t[i],t[-1]);b.cx(carry,t[-1])
        b.cx(carry,a[i]);b.cx(a[i],t[i])
    else:
        b.ccx(carry,t[i],t[-1]);b.cx(carry,t[i])
    if n-1<o:b.cx(a[n-1],t[n-1])
    for i in reversed(range(n-2)):
        if i==0 and cin is None:
            if o:b.del_and(a[0],t[0],C(1));b.cx(a[0],t[0])
        elif i<o:
            b.cx(C(i),C(i+1));b.del_and(a[i],t[i],C(i+1))
            b.cx(C(i),a[i]);b.cx(a[i],t[i])
        else:
            b.del_and(C(i),t[i],C(i+1));b.cx(C(i),t[i])
    if count:b.free(allocated)
    if sub:
        for q in t:b.x(q)


def prepare_derivative_split(reference, cfg, tables):
    d = np.arange(1 << reference.DB, dtype=np.int64)
    coarse, residual = {}, {}
    bounds = {'normalized_addresses_checked': 0, 'min_residual': 1 << 30,
              'max_residual': 0, 'max_table_residual': 0, 'max_coarse': 0}
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
        bounds['normalized_addresses_checked'] += len(d)
        bounds['min_residual'] = min(bounds['min_residual'], int(low.min()))
        bounds['max_residual'] = max(bounds['max_residual'], int(low.max()))
        bounds['max_table_residual'] = max(bounds['max_table_residual'], residual[key])
        bounds['max_coarse'] = max(bounds['max_coarse'], base)
    assert cfg.w2 >= 6 and len(cfg.rrow_list()) == 8
    return {'coarse': coarse, 'residual': residual}, bounds
