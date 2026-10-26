# SPDX-License-Identifier: Apache-2.0
# Adapted from mizeroluckygall, Quantum Circuits | Rotation Calibration, V3:
# https://www.kaggle.com/code/mizeroluckygall/quantum-circuits-rotation-calibration-39-char?scriptVersionId=361537580
# Upstream credits lightningv08, Quantum Circuits | Memory & Algebra, for
# exhaustive phase calibration. See upstream_functions.py for unchanged bodies.
# Local changes: explicit Circuit/spec API, strict supported-circuit audit,
# chunked LP scans and integer products, canonical hash checks, independent
# measurement replay, and report-only output without verifier/submission calls.
"""Exhaustive rotation calibration for a supported basis-state phase oracle.

``calibrate(circuit, IV, spec, output_path, baseline_report=None)`` accepts the
unchanged official inverse-square-root verifier module IV. It does not search
Inputs, read ZIPs, contact Kaggle, or call IV.verify. Its output requires a full
official verification before use. No verifier salt/hash is optimized.

Only unconditioned, exactly representable Z_POW rotations are calibrated.
Before the final rotation, classical conditions may guard diagonal phases;
arithmetic is deterministic on basis states and HMR resets its target.
After the final rotation the original cleanup may contain conditional arithmetic.
This excludes arbitrary quantum
circuits whose rotation controls can depend on measurement outcomes.
"""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.optimize import linprog

_path = Path(__file__).with_name("upstream_functions.py")
_spec = importlib.util.spec_from_file_location("_vqcc_upstream_rotcal", _path)
_upstream = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_upstream)
UNIT = _upstream.UNIT
parse_kmx = _upstream.parse_kmx
instrument = _upstream.instrument
theta_units = _upstream.theta_units
exponent_text = _upstream.exponent_text
rewrite = _upstream.rewrite
UPSTREAM_URL = "https://www.kaggle.com/code/mizeroluckygall/quantum-circuits-rotation-calibration-39-char?scriptVersionId=361537580"


def _serialize(circuit, fmt):
    stream = io.BytesIO()
    circuit.to_file(stream, format=fmt)
    return stream.getvalue()


def audit_supported(text, rots):
    """Conservative structural guard for deterministic rotation activations.

    Before the final rotation, HMR results may only control diagonal gates.
    Therefore the Z_POW activation bits do not depend on their outcomes.
    Original cleanup after the final rotation cannot change earlier activation
    bits, so its conditional arithmetic is permitted. This alone does not prove
    MBU phase corrections; replay and the official verifier remain necessary.
    """
    metadata = {"REGISTER", "APPEND_TO_REGISTER", "INFO_TAG"}
    basis = {"X", "Y", "CX", "CY", "CCX", "CCY", "SWAP", "CSWAP", "R", "HMR"}
    diagonal = {"Z", "CZ", "CCZ", "Z_POW"}
    classical = {"BIT_STORE0", "BIT_STORE1", "BIT_INVERT", "BIT_XOR", "NEG"}
    depth = 0
    last_rotation_line = max((r["line"] for r in rots), default=-1)
    unknown = []
    conditional_nondiagonal = []
    conditional_cleanup = 0
    for i, line in enumerate(text.splitlines()):
        tok = line.split("#", 1)[0].split()
        if not tok:
            continue
        op = tok[0]
        if op == "PUSH_CONDITION":
            depth += 1
            continue
        if op == "POP_CONDITION":
            depth -= 1
            if depth < 0:
                raise ValueError("Unbalanced POP_CONDITION")
            continue
        if op not in metadata | basis | diagonal | classical:
            unknown.append((i + 1, op))
        conditioned = depth > 0 or "if" in tok
        if conditioned and op in basis:
            if i < last_rotation_line:
                conditional_nondiagonal.append((i + 1, op))
            else:
                conditional_cleanup += 1
        if conditioned and op == "Z_POW":
            raise ValueError("Conditioned Z_POW is outside this calibration scope")
    if depth:
        raise ValueError("Unbalanced PUSH_CONDITION")
    if unknown or conditional_nondiagonal:
        raise ValueError(f"Unsupported operations: {unknown[:8]}; conditioned arithmetic: {conditional_nondiagonal[:8]}")
    for rotation in rots:
        if rotation["cond"] is not None:
            raise ValueError("Conditioned rotations are unsupported")
        if not theta_units(rotation["t"])[1]:
            raise ValueError("Every original exponent must be exact on the 2^-64-turn lattice")
    return {"supported": True, "rotationsUnconditioned": True,
            "measurementIndependentActivationBits": True, "exactPhaseLattice": True,
            "conditionalCleanupOperationsAfterLastRotation": conditional_cleanup}


def _matvec(bits, vector, columns=None, *, chunk=16384, dtype=np.float64):
    """A small temporary replaces the upstream full float/int64 matrix."""
    result = np.empty(len(bits), dtype=dtype)
    vector = np.asarray(vector, dtype=dtype)
    for begin in range(0, len(bits), chunk):
        end = min(begin + chunk, len(bits))
        block = bits[begin:end]
        if columns is not None:
            block = block[:, columns]
        result[begin:end] = block.astype(dtype) @ vector
    return result


def exact_margin(err_units, tol_units, bits, deltas_full_units, *, chunk=16384):
    """Exact rounded-correction check on every collected input, without overflow."""
    deltas = np.asarray(deltas_full_units, dtype=np.int64)
    largest_error = max(abs(int(err_units.min())), abs(int(err_units.max())))
    if largest_error + sum(abs(int(d)) for d in deltas) >= 1 << 63:
        raise ValueError("Correction can overflow signed 64-bit phase arithmetic")
    if int(np.max(tol_units)) >= 1 << 63:
        raise ValueError("Half-turn tolerance needs a wider integer checker")
    minimum, worst = None, None
    for begin in range(0, len(bits), chunk):
        end = min(begin + chunk, len(bits))
        shift = bits[begin:end].astype(np.int64) @ deltas
        corrected = np.asarray(err_units[begin:end], dtype=np.int64) + shift
        slack = np.asarray(tol_units[begin:end], dtype=np.int64) - np.abs(corrected)
        local = int(np.argmin(slack))
        value = int(slack[local])
        if minimum is None or value < minimum:
            minimum, worst = value, begin + local
    return minimum, worst


def fit(err_units, tol_units, bits, free, fixed_shift_units, *, bound_units,
        scale=2.0 ** 28, start=4096, add=400, max_iter=60, chunk=16384, log=None):
    """Maximize worst slack using cutting planes; every iteration scans all x."""
    n, k = len(err_units), len(free)
    e = (err_units.astype(np.float64) + fixed_shift_units) / scale
    tol = tol_units.astype(np.float64) / scale
    active = set(np.linspace(0, n - 1, min(start, n), dtype=np.int64).tolist())
    initial = tol - np.abs(e)
    count = min(start, n)
    active.update(np.argpartition(initial, count - 1)[:count].tolist())
    converged = False
    for iteration in range(max_iter):
        ii = np.fromiter(sorted(active), dtype=np.int64)
        selected = bits[ii][:, free].astype(np.float64)
        A = np.vstack([np.c_[selected, np.ones(len(ii))],
                       np.c_[-selected, np.ones(len(ii))]])
        rhs = np.r_[tol[ii] - e[ii], tol[ii] + e[ii]]
        result = linprog(np.r_[np.zeros(k), -1.0], A_ub=A, b_ub=rhs,
                         bounds=[(-bound_units / scale, bound_units / scale)] * k + [(None, None)],
                         method="highs")
        if not result.success:
            raise RuntimeError(result.message)
        delta, margin = result.x[:k], result.x[k]
        corrected = e + _matvec(bits, delta, free, chunk=chunk)
        slack = tol - np.abs(corrected)
        actual = float(slack.min())
        if log:
            log(f"LP {iteration}: active={len(ii)}, lp_margin={margin*scale:.3f}, exhaustive_margin={actual*scale:.3f}")
        if actual >= margin - 1e-9 * max(1.0, abs(margin)):
            converged = True
            break
        count = min(add, n)
        new_rows = set(np.argpartition(slack, count - 1)[:count].tolist())
        if new_rows.issubset(active):
            break
        active.update(new_rows)
    return np.rint(delta * scale).astype(np.int64), actual * scale, {
        "iterations": iteration + 1, "activeInputs": len(active), "lpConverged": converged,
        "exhaustiveScans": iteration + 1,
    }


def collect(km, IV, spec, text, rots, *, batch=16384, seed=20261023,
            replay_measurements=True, log=print):
    """Visit every x; check input/scratch and optionally replay a second RNG trace.

    The fixed seeds provide reproducibility, never candidate selection. The
    structural audit proves activation-bit independence. Equal phases under two
    traces are a diagnostic; they are not a replacement for all official shots.
    """
    circuit = km.Circuit(text)
    Q = circuit.num_qubits
    recorded = km.Circuit(instrument(text.splitlines(), rots, Q))
    xr, scratch = IV.validate_registers(circuit, spec)
    rec = km.array([km.q(Q + j) for j in range(len(rots))])
    xs = np.arange(1, spec.max_x + 1, dtype=np.uint64)
    phase = np.empty(len(xs), dtype=np.uint64)
    bits = np.empty((len(xs), len(rots)), dtype=np.uint8)
    simulations = [km.Simulator(batch_size=batch, seed=seed + j, debug_print=False)
                   for j in range(3 if replay_measurements else 2)]
    for begin in range(0, len(xs), batch):
        take = min(batch, len(xs) - begin)
        values = np.resize(xs[begin:begin + batch], batch)
        for index, sim in enumerate(simulations):
            sim.clear_for_shot()
            sim.write_across_shots(xr, IV.to_bits(values, spec.width))
            sim.do(recorded if index == 1 else circuit)
            if not np.array_equal(IV.from_bits(sim.read_across_shots(xr))[:take], values[:take]):
                raise ValueError("Input register was not restored during exhaustive collection")
            if len(scratch) and sim.read_across_shots(scratch)[:, :take].any():
                raise ValueError("Original scratch was not cleared during exhaustive collection")
        phases = np.asarray([IV.half_turns_to_units(simulations[0].read_shot_phase(j))
                             for j in range(take)], dtype=np.uint64)
        if replay_measurements:
            replay = np.asarray([IV.half_turns_to_units(simulations[2].read_shot_phase(j))
                                 for j in range(take)], dtype=np.uint64)
            if not np.array_equal(phases, replay):
                raise ValueError("Phase depends on measurement trace; do not calibrate this model")
        phase[begin:begin + take] = phases
        bits[begin:begin + take] = simulations[1].read_across_shots(rec)[:, :take].T
        if log and begin // batch % 40 == 0:
            log(f"Collected {begin + take:,}/{len(xs):,} inputs")
    target, tolerance = IV.reference_values(spec, xs)
    error = (phase - np.asarray(target, dtype=np.uint64)).view(np.int64)
    return xs, error, np.asarray(tolerance, dtype=np.uint64), bits, Q


def search(err, tol, bits, rots, report, rotation_t_count, *, max_drop=8,
           bound_floor_units=2.0 ** 36, chunk=16384, log=print, fit_options=None,
           stop_on_invalid_margin=True):
    """Upstream smallest-angle drop prefixes, followed by exact integer checks."""
    counted = [j for j, r in enumerate(rots) if r["counted"]]
    order = sorted(counted, key=lambda j: abs(float(min(rots[j]["t"] % 2, 2 - rots[j]["t"] % 2))))
    # A Clifford Z_POW(1) is exactly 2^63 phase units. It is never varied or
    # dropped, so avoid putting that unsigned value into a signed int64 array.
    counted_set = set(counted)
    theta = np.array([theta_units(r["t"])[0] if j in counted_set else 0
                      for j, r in enumerate(rots)], dtype=np.int64)

    def estimate(n_rotations, margin):
        if margin <= 0:
            return math.inf
        eps = math.pi * (margin / UNIT) / n_rotations if n_rotations else None
        synthesis = n_rotations * rotation_t_count(eps) if n_rotations else 0
        return report["num_qubits"] * (report["toffoli_count"] + (report["t_count"] + synthesis) / 4)

    baseline, worst = exact_margin(err, tol, bits, np.zeros(len(rots), dtype=np.int64), chunk=chunk)
    rows = [{"drop": 0, "dropped": [], "deltas": np.zeros(len(rots), dtype=np.int64),
             "margin": baseline, "worstIndex": worst, "estimate": estimate(len(counted), baseline), "fitted": False}]
    for count in range(min(max_drop, len(counted)) + 1):
        dropped = order[:count]
        free = [j for j in counted if j not in dropped]
        shift = -_matvec(bits, theta[dropped], dropped, chunk=chunk, dtype=np.int64) if dropped else np.zeros(len(err), dtype=np.int64)
        bound = max([4 * abs(int(theta[j])) for j in dropped] + [bound_floor_units])
        options = dict(fit_options or {})
        try:
            delta, _, diagnostics = fit(err, tol, bits, free, shift,
                                         bound_units=bound, chunk=chunk, **options)
        except RuntimeError as exc:
            rows.append({"drop": count, "dropped": dropped, "fitted": False,
                         "estimate": math.inf, "margin": -1, "lpError": str(exc)})
            continue
        deltas = np.zeros(len(rots), dtype=np.int64)
        deltas[free] = delta
        deltas[dropped] = -theta[dropped]
        margin, worst = exact_margin(err, tol, bits, deltas, chunk=chunk)
        row = {"drop": count, "dropped": dropped, "deltas": deltas, "margin": margin,
               "worstIndex": worst, "estimate": estimate(len(free), margin), "fitted": True,
               **diagnostics}
        rows.append(row)
        if log:
            log(f"Drop {count}: exact margin={margin:,} units, estimated cost={row['estimate']:,.3f}")
        if margin <= 0 and stop_on_invalid_margin:
            break
    return rows


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def apply_fixed_calibration(circuit, fixed_deltas, output_path=None, *, km=None):
    """Reproduce a prior calibration without simulation or LP; bind both hashes.

    fixed_deltas is a dict or JSON path written by run_two_pinned.py. Its
    canonical source hash and candidate hash are mandatory. The output still
    inherits whatever verification status was independently established for
    that exact candidate; this helper does not run or replace verification.
    """
    if km is None:
        import kickmix as km
    if not isinstance(fixed_deltas, dict):
        fixed_deltas = json.loads(Path(fixed_deltas).read_text())
    source_hash = hashlib.sha256(_serialize(circuit, "kmb")).hexdigest()
    if source_hash != fixed_deltas["sourceCircuitSha256"]:
        raise ValueError("Fixed calibration source circuit hash mismatch")
    if fixed_deltas.get("phaseUnitsPerTurn") != UNIT:
        raise ValueError("Unexpected phase units")
    text = _serialize(circuit, "kmx").decode()
    lines, rots = parse_kmx(text)
    audit_supported(text, rots)
    values = fixed_deltas["deltasPhaseUnits"]
    if len(values) != len(rots):
        raise ValueError("Fixed calibration rotation count mismatch")
    dropped = set(fixed_deltas["droppedOriginalRotationIndices"])
    if not dropped.issubset(range(len(rots))):
        raise ValueError("Invalid dropped rotation index")
    replacement = rewrite(lines, rots, {j: int(v) for j, v in enumerate(values) if j not in dropped}, dropped)
    candidate = km.Circuit(replacement)
    if hashlib.sha256(_serialize(candidate, "kmb")).hexdigest() != fixed_deltas["candidateCircuitSha256"]:
        raise ValueError("Fixed calibration candidate hash mismatch")
    if candidate.num_qubits != circuit.num_qubits:
        raise ValueError("Fixed calibration changed qubit count")
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        candidate.to_file(str(output_path), format="kmb")
    return candidate


def calibrate(circuit, IV, spec, output_path, baseline_report=None, *, km=None,
              max_drop=8, batch=16384, chunk=16384, seed=20261023,
              replay_measurements=True, fit_options=None, log=print):
    """Write one estimated-improving candidate and JSON; never run IV.verify.

    baseline_report may be the unchanged official full report for this canonical
    circuit. If omitted, static operation counts give an upper-bound resource
    estimate, sufficient to compare angle changes with fixed arithmetic. A report
    with a mismatched canonical hash is rejected. The selected candidate is
    recollected over the entire domain and its integer margin must reproduce.
    """
    if km is None:
        import kickmix as km
    started = time.perf_counter()
    output_path = Path(output_path)
    source_bytes = _serialize(circuit, "kmb")
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    text = _serialize(circuit, "kmx").decode()
    lines, rots = parse_kmx(text)
    if not rots:
        raise ValueError("Circuit has no rotations to calibrate")
    audit = audit_supported(text, rots)
    IV.validate_registers(circuit, spec)
    resource_source = "static upper-bound operation counts"
    if baseline_report is not None:
        if baseline_report.get("circuit_sha256") != source_hash:
            raise ValueError("Baseline report does not bind this canonical circuit hash")
        if baseline_report.get("num_qubits") != circuit.num_qubits:
            raise ValueError("Baseline report qubit count mismatch")
        resources = baseline_report
        resource_source = "canonical-hash-bound caller report"
    else:
        stats = circuit.operation_statistics()
        resources = {"num_qubits": circuit.num_qubits,
                     "toffoli_count": stats.num_toffoli_like, "t_count": stats.num_t_like}
    xs, err, tol, bits, _ = collect(km, IV, spec, text, rots, batch=batch, seed=seed,
                                  replay_measurements=replay_measurements, log=log)
    rows = search(err, tol, bits, rots, resources, IV.rotation_t_count,
                  max_drop=max_drop, chunk=chunk, log=log, fit_options=fit_options)
    baseline = rows[0]
    best = min((r for r in rows if r.get("fitted") and r["margin"] > 0),
               key=lambda r: r["estimate"], default=None)
    selected = best is not None and best["estimate"] < baseline["estimate"]
    result = {"upstream": UPSTREAM_URL, "sourceCircuitSha256": source_hash,
              "resourceSource": resource_source, "structuralAudit": audit,
              "allInputsVisited": len(xs), "maxX": int(spec.max_x),
              "seedPurpose": "Fixed reproducibility only; no secret salt, hash search or seed selection",
              "measurementReplay": replay_measurements,
              "rows": rows, "selectedImprovement": selected,
              "officialVerificationRun": False, "officialStatus": "NOT_RUN"}
    if selected:
        dropped = set(best["dropped"])
        delta = {j: int(v) for j, v in enumerate(best["deltas"]) if j not in dropped}
        new_text = rewrite(lines, rots, delta, dropped)
        new_lines, new_rots = parse_kmx(new_text)
        audit_supported(new_text, new_rots)
        del xs, err, tol, bits
        xs2, err2, tol2, bits2, _ = collect(km, IV, spec, new_text, new_rots, batch=batch,
                                           seed=seed, replay_measurements=replay_measurements, log=log)
        actual_margin, worst = exact_margin(err2, tol2, bits2, np.zeros(len(new_rots), dtype=np.int64), chunk=chunk)
        if actual_margin <= 0 or actual_margin != best["margin"]:
            raise ValueError(f"Rewritten exact margin {actual_margin} differs from model {best['margin']}")
        candidate = km.Circuit(new_text)
        if candidate.num_qubits != circuit.num_qubits:
            raise ValueError("Rotation rewrite changed qubit count")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        candidate.to_file(str(output_path), format="kmb")
        result.update({"candidatePath": str(output_path),
                       "candidateCircuitSha256": hashlib.sha256(_serialize(candidate, "kmb")).hexdigest(),
                       "candidateRotationCount": sum(r["counted"] for r in new_rots),
                       "candidateMarginUnits": actual_margin, "candidateWorstX": int(xs2[worst]),
                       "estimatedCost": best["estimate"], "baselineEstimatedCost": baseline["estimate"],
                       "status": "NEEDS_FULL_OFFICIAL_VERIFICATION"})
    else:
        result["status"] = "NO_ESTIMATED_IMPROVEMENT"
    result["elapsedSeconds"] = time.perf_counter() - started
    result = _jsonable(result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path = output_path.with_suffix(".calibration.json")
    report_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    result["reportPath"] = str(report_path)
    return result
