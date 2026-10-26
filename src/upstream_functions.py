# SPDX-License-Identifier: Apache-2.0
# Original algorithm and functions: mizeroluckygall, Kaggle Rotation Calibration V3.
# Pinned primary source: https://www.kaggle.com/code/mizeroluckygall/quantum-circuits-rotation-calibration-39-char?scriptVersionId=361537580
# Upstream credits lightningv08, Quantum Circuits | Memory & Algebra, for exhaustive phase calibration.
# Exact function bodies extracted verbatim; notebook setup, Input scan and packaging omitted.
# Apache-2.0 confirmed from anonymous pinned Kaggle license footer (see upstream_primary_license.json).
from decimal import Decimal, localcontext
from fractions import Fraction
import math
import numpy as np
from scipy.optimize import linprog
UNIT = 2 ** 64

def parse_kmx(text):
    """Return (lines, rotations); rotation = dict(line, qubit, t, cond, counted)."""
    lines = text.splitlines()
    rots = []
    for i, line in enumerate(lines):
        tok = line.split('#')[0].split()
        if not tok or tok[0] != 'Z_POW':
            continue
        cond = tok[tok.index('if') + 1] if 'if' in tok else None
        t = Fraction(Decimal(tok[2]))
        counted = (t * 4).denominator != 1          # finer than T: a synthesized rotation
        rots.append(dict(line=i, qubit=tok[1], t=t, cond=cond, counted=counted))
    return lines, rots


def instrument(lines, rots, first_record_qubit):
    """Copy of the circuit where rotation j becomes CX qubit -> record_j."""
    out = list(lines)
    for j, r in enumerate(rots):
        rec = f'q{first_record_qubit + j}'
        out[r['line']] = f"CX {r['qubit']} {rec}" + (f" if {r['cond']}" if r['cond'] else '')
    return '\n'.join(out) + '\n'


def theta_units(t):
    """Z_POW exponent (half turns) -> phase units (2^-64 turn), exact if representable."""
    v = t * (UNIT // 2)
    return int(round(v)), v.denominator == 1


def exponent_text(t):
    """Exact decimal text for a dyadic exponent, in kickmix's no-exponent format."""
    with localcontext() as ctx:
        ctx.prec = 400
        d = Decimal(t.numerator) / Decimal(t.denominator)
    s = format(d, 'f')
    if '.' not in s:
        s += '.0'
    return s


def rewrite(lines, rots, deltas_units, dropped):
    out = list(lines)
    remove = set()
    for j, r in enumerate(rots):
        if j in dropped:
            remove.add(r['line'])
            continue
        if deltas_units.get(j, 0):
            t = r['t'] + Fraction(int(deltas_units[j]), UNIT // 2)
            out[r['line']] = f"Z_POW {r['qubit']} {exponent_text(t)}" + (f" if {r['cond']}" if r['cond'] else '')
    return '\n'.join(l for i, l in enumerate(out) if i not in remove) + '\n'


def fit(err_units, tol_units, bits, free, fixed_shift_units, *, bound_units, scale=2.0 ** 28,
        start=4096, add=400, max_iter=60, verbose=True):
    """Maximise m s.t. |err + shift + B[:,free] d| <= tol - m for every input.

    err_units: int64 signed error of the current circuit; tol_units: uint64;
    bits: (n_inputs, n_rot) uint8; fixed_shift_units: float array added to err
    (the effect of dropping rotations). Returns (deltas, margin_units).
    """
    n = len(err_units)
    e = (err_units.astype(np.float64) + fixed_shift_units) / scale
    tol = tol_units.astype(np.float64) / scale
    Bf = bits[:, free].astype(np.float64)
    k = len(free)
    active = set(np.linspace(0, n - 1, start, dtype=np.int64).tolist())
    active.update(np.argsort(tol - np.abs(e))[:start].tolist())
    for it in range(max_iter):
        ii = np.fromiter(sorted(active), dtype=np.int64)
        A = np.vstack([np.c_[Bf[ii], np.ones(len(ii))], np.c_[-Bf[ii], np.ones(len(ii))]])
        b = np.r_[tol[ii] - e[ii], tol[ii] + e[ii]]
        res = linprog(np.r_[np.zeros(k), -1.0], A_ub=A, b_ub=b,
                      bounds=[(-bound_units / scale, bound_units / scale)] * k + [(None, None)], method='highs')
        if not res.success:
            raise RuntimeError(res.message)
        d, m = res.x[:k], res.x[k]
        corrected = e + Bf @ d
        slack = tol - np.abs(corrected)
        worst = float(slack.min())
        if verbose:
            print(f'    LP iter {it}: rows {len(ii):6d}  lp margin {m*scale:.4g}  true margin {worst*scale:.4g} units', flush=True)
        if worst >= m - 1e-9 * max(1.0, abs(m)):
            break
        active.update(np.argpartition(slack, add)[:add].tolist())
    return np.rint(d * scale).astype(np.int64), worst * scale


def exact_margin(err_units, tol_units, bits, deltas_full_units):
    """Integer check of the final, rounded corrections over every input."""
    shift = bits.astype(np.int64) @ deltas_full_units.astype(np.int64)
    corrected = err_units.astype(np.int64) + shift
    slack = tol_units.astype(np.int64) - np.abs(corrected)
    return int(slack.min()), int(np.argmin(slack))


def collect(km, IV, spec, text, rots, *, batch=16384, seed=20261023, log=print):
    """Simulate every input: true phase error (units) and every rotation's bit."""
    c0 = km.Circuit(text)
    Q = c0.num_qubits
    ci = km.Circuit(instrument(text.splitlines(), rots, Q))
    xr, _ = IV.validate_registers(c0, spec)
    rec = km.array([km.q(Q + j) for j in range(len(rots))])
    xs = np.arange(1, spec.max_x + 1, dtype=np.uint64)
    n = len(xs)
    phase = np.empty(n, dtype=np.uint64)
    bits = np.empty((n, len(rots)), dtype=np.uint8)
    s0 = km.Simulator(batch_size=batch, seed=seed, debug_print=False)
    s1 = km.Simulator(batch_size=batch, seed=seed + 1, debug_print=False)
    for start in range(0, n, batch):
        x = np.resize(xs[start:start + batch], batch)
        take = min(batch, n - start)
        for s, c in ((s0, c0), (s1, ci)):
            s.clear_for_shot(); s.write_across_shots(xr, IV.to_bits(x, spec.width)); s.do(c)
        phase[start:start + take] = [IV.half_turns_to_units(s0.read_shot_phase(j)) for j in range(take)]
        bits[start:start + take] = s1.read_across_shots(rec)[:, :take].T
        if start // batch % 40 == 0:
            log(f'  simulated {start + take:,}/{n:,}')
    target, tol = IV.reference_values(spec, xs)
    err = (phase - np.asarray(target, dtype=np.uint64)).view(np.int64)
    return xs, err, np.asarray(tol, dtype=np.uint64), bits, Q


def search(err, tol, bits, rots, report, rotation_t_count, *, max_drop=8, log=print):
    """Try dropping 0..max_drop smallest rotations; return candidates with cost estimates."""
    counted = [j for j, r in enumerate(rots) if r['counted']]
    weight = {j: abs(float(min(rots[j]['t'] % 2, 2 - rots[j]['t'] % 2))) for j in counted}
    order = sorted(counted, key=lambda j: weight[j])
    Qn, tof, tcount = report['num_qubits'], report['toffoli_count'], report['t_count']
    theta = np.array([theta_units(r['t'])[0] for r in rots], dtype=np.float64)

    def estimate(n_rot, margin_units):
        if margin_units <= 0:
            return float('inf')
        eps = math.pi * (margin_units / UNIT) / n_rot
        return Qn * (tof + (tcount + n_rot * rotation_t_count(eps)) / 4)

    m0, _ = exact_margin(err, tol, bits, np.zeros(len(rots), dtype=np.int64))
    base = estimate(len(counted), m0)
    log(f'baseline: {len(counted)} rotations, margin {m0:.6g} units, estimated cost {base:,.3f} '
        f'(verifier reported {report["score"]:,.3f})')
    rows = [dict(drop=0, dropped=[], deltas=np.zeros(len(rots), dtype=np.int64), margin=m0, estimate=base, fitted=False)]
    for k in range(0, max_drop + 1):
        dropped = order[:k]
        free = [j for j in counted if j not in dropped]
        shift = -(bits[:, dropped].astype(np.float64) @ theta[dropped]) if dropped else np.zeros(len(err))
        bound = max([4 * abs(theta[j]) for j in dropped] + [2.0 ** 36])
        log(f'drop {k}: ' + (', '.join(f'{float(rots[j]["t"]):.3e}' for j in dropped) or 'none'))
        try:
            d_free, _ = fit(err, tol, bits, free, shift, bound_units=bound, verbose=False)
        except RuntimeError as exc:
            log(f'    LP failed: {exc}'); continue
        deltas = np.zeros(len(rots), dtype=np.int64)
        deltas[free] = d_free
        for j in dropped:
            deltas[j] = -int(theta_units(rots[j]['t'])[0])
        m, worst_i = exact_margin(err, tol, bits, deltas)
        est = estimate(len(free), m)
        log(f'    exact margin {m:.6g} units, estimated cost {est:,.3f}')
        rows.append(dict(drop=k, dropped=dropped, deltas=deltas, margin=m, estimate=est, fitted=True))
        if m <= 0:
            break
    return rows
