"""Experiment 03: Direct quadratic polynomial phase compiler for inverse square root.

Status: experimental

Directly fits piecewise degree-2 quadratic polynomials to ``1 / sqrt(M)`` over
the normalized mantissa ``M`` in ``[1, 4)``, evaluates by two-stage
column-truncated Horner's rule, and de-normalizes in-place with a
unary-controlled shift -- avoiding the Newton-Raphson ``square / multiply_m /
correct / multiply_y0`` stages of the reference circuit.

Interval & arithmetic scheme (exact model/circuit parity):

* After normalization ``x_norm = x << 2s`` and ``M = x_norm / 2^(width-2)``.
* The power-of-two interval index is ``idx = x_norm >> (width - k)`` in
  ``[2^(k-2), 2^k)`` (or ``i = idx - 2^(k-2)`` in ``[0, 3 * 2^(k-2))``).
* ``delta = M - m0`` bits are read directly from ``x_norm[width - k - j]``
  with weight ``2^-(k - 2 + j)`` for ``j = 1 .. delta_bits``.
* Both ``direct_poly_model`` and ``build_direct_poly_circuit`` evaluate the
  exact two-stage column-truncated Horner recurrence:
  ``r1 = c1 - sum_j bit_j(delta) * (c2 >> m_j)``,
  ``y0 = c0 - sum_j bit_j(delta) * (r1 >> m_j)``,
  followed by ``y1 = y0 << s``.

Estimated cost and verified measured cost are kept as separate fields.
"""
from __future__ import annotations

import dataclasses
import functools
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from apedp_dyn.quantum_compilation.candidates import candidate_id_for, sha256_file
from apedp_dyn.quantum_compilation.contracts import VerifierContract
from apedp_dyn.quantum_compilation.experiment import (
    Candidate,
    CandidateArtifact,
    CandidateScreen,
    Proposer,
)
from apedp_dyn.quantum_compilation.invsqrt_qrom_line import reference_phase_values

__all__ = [
    "DEFAULT_CONFIG",
    "DIRECT_POLY_CONFIGS",
    "DirectPolyProposer",
    "PolyPrecision",
    "build_direct_poly_circuit",
    "direct_poly_model",
    "evaluate_direct_poly_model",
    "fit_polynomials",
]

TOLERANCE_EXPONENT = 24  # relative phase tolerance 2^-24, matching the reference.

DEFAULT_CONFIG: tuple[int, int, int] = (8, 28, 14)  # (index_bits, coef_bits, delta_bits)
DIRECT_POLY_CONFIGS: list[tuple[int, int, int]] = [
    (8, 28, 14),
    (8, 29, 14),
    (9, 28, 13),
    (9, 29, 13),
    (10, 28, 12),
    (8, 27, 14),
    (9, 27, 13),
]


@dataclasses.dataclass(frozen=True)
class PolyPrecision:
    """Precision parameters for the direct degree-2 polynomial phase compiler."""

    index_bits: int = 8   # top k bits of x_norm; N = 3 * 2^(index_bits - 2) active intervals
    coef_bits: int = 28   # fixed-point width of (c0, c1, c2) and Horner accumulator y0
    delta_bits: int = 14  # number of fractional delta bits after the interval prefix

    def output_bits(self, n_p: int) -> int:
        return self.coef_bits + n_p

    def num_intervals(self) -> int:
        return 3 * (1 << (self.index_bits - 2))


def _contract_n_p(contract: VerifierContract) -> int:
    return int(contract.constants["N_P"])


def _max_x(n_p: int) -> int:
    return int(3 * (2**n_p - 2) ** 2)


def _rotation_t_count(eps: float) -> float:
    return 0.56 * math.log2(1.0 / eps) + 5.3


def _delta_rows(n_p: int, prec: PolyPrecision) -> list[tuple[int, int]]:
    """Return ``(x_norm_bit_index, shift_weight_m)`` pairs for ``delta``."""
    width = 2 * n_p + 2
    k = prec.index_bits
    wc = prec.coef_bits
    rows: list[tuple[int, int]] = []
    for j in range(1, prec.delta_bits + 1):
        bit_idx = width - k - j
        m_weight = k - 2 + j
        if bit_idx < 0 or m_weight >= wc:
            break
        rows.append((bit_idx, m_weight))
    return rows


def fit_polynomials(n_p: int, prec: PolyPrecision) -> list[list[int]]:
    """Fit per-interval degree-2 quadratic polynomials to ``1/sqrt(M)`` on ``[1, 4)``.

    Returns ``table[interval][j]`` for ``interval in 0 .. num_intervals - 1``
    and ``j in (0, 1, 2)`` corresponding to ``[c0, -c1, c2]`` scaled by
    ``2^coef_bits``.  ``c1`` is negated so the subtracting Horner steps evaluate
    ``c0 + c1 * delta + c2 * delta^2``, and ``c0`` is centered against the exact
    column-truncation bias on each interval.
    """
    width = 2 * n_p + 2
    k = prec.index_bits
    wc = prec.coef_bits
    if k < 2 or k > width:
        raise ValueError(f"index_bits must lie in [2, {width}], got {k}")
    first = 1 << (k - 2)
    num_intervals = prec.num_intervals()
    limit = (1 << wc) - 1
    d_rows = _delta_rows(n_p, prec)
    low_width = width - k
    sample_low = np.unique(
        np.linspace(0, max(0, (1 << low_width) - 1), 129, dtype=np.int64)
    )
    cheb_j: np.ndarray = np.arange(3, dtype=np.float64)
    cheb_unit = 0.5 * (1.0 - np.cos((2.0 * cheb_j + 1.0) * np.pi / 6.0))

    table: list[list[int]] = []
    for i in range(num_intervals):
        idx = first + i
        m0 = idx / first
        m1 = (idx + 1) / first
        h = m1 - m0
        nodes = h * cheb_unit
        vals = 1.0 / np.sqrt(m0 + nodes)
        coeffs = np.polynomial.polynomial.polyfit(nodes, vals, 2)

        c0_base = max(0, min(limit, round(float(coeffs[0]) * (1 << wc))))
        c1_base = max(0, min(limit, round(float(-coeffs[1]) * (1 << wc))))
        c2_val = max(0, min(limit, round(float(coeffs[2]) * (1 << wc))))

        m_vals = m0 + sample_low.astype(np.float64) / (1 << (width - 2))
        exact_y0 = (1.0 / np.sqrt(m_vals)) * (1 << wc)
        t_c2 = np.zeros_like(sample_low)
        for bit_idx, m_weight in d_rows:
            t_c2 = t_c2 + ((sample_low >> bit_idx) & 1) * (c2_val >> m_weight)

        best_err = float("inf")
        best_c0, best_c1 = c0_base, c1_base
        for dc1 in (-2, -1, 0, 1, 2):
            c1_try = max(0, min(limit, c1_base + dc1))
            r1 = c1_try - t_c2
            t_r1 = np.zeros_like(sample_low)
            for bit_idx, m_weight in d_rows:
                t_r1 = t_r1 + ((sample_low >> bit_idx) & 1) * (r1 >> m_weight)
            diff = (c0_base - t_r1) - exact_y0
            dc0 = -round(0.5 * float(diff.max() + diff.min()))
            c0_try = max(0, min(limit, c0_base + dc0))
            err = float(np.abs(((c0_try - t_r1) - exact_y0) / exact_y0).max())
            if err < best_err:
                best_err = err
                best_c0, best_c1 = c0_try, c1_try

        table.append([best_c0, best_c1, c2_val])
    return table


@functools.cache
def _poly_table(n_p: int, prec: PolyPrecision) -> tuple[tuple[int, int, int], ...]:
    """Return the 2^index_bits lookup table (padded with zeros below 2^(k-2))."""
    k = prec.index_bits
    first = 1 << (k - 2)
    active = fit_polynomials(n_p, prec)
    padded: list[tuple[int, int, int]] = [(0, 0, 0)] * first
    for row in active:
        padded.append((int(row[0]), int(row[1]), int(row[2])))
    return tuple(padded)


def direct_poly_model(n_p: int, prec: PolyPrecision, xs: np.ndarray) -> np.ndarray:
    """Exact fixed-point output for every input ``x`` matching the circuit bit-for-bit.

    ``result / 2^output_bits(n_p)`` approximates ``1 / sqrt(x)``.
    """
    width = 2 * n_p + 2
    wc = prec.coef_bits
    k = prec.index_bits
    if k < 2 or k > width:
        raise ValueError(f"index_bits must lie in [2, {width}], got {k}")
    xs = np.asarray(xs, dtype=np.int64)
    if not ((1 <= xs) & (xs < 2**width)).all():
        raise ValueError("x must lie in [1, 2^(width - 1)).")

    # 1. Normalize: s leading zero pairs, x_norm = x << (2 * s).
    s = (width - (np.floor(np.log2(xs)).astype(np.int64) + 1)) // 2
    x_norm = xs << (2 * s)

    # 2. Power-of-two interval bit-slicing: idx = x_norm >> (width - k).
    idx = x_norm >> (width - k)
    table: np.ndarray = np.asarray(_poly_table(n_p, prec), dtype=np.int64)
    c0 = table[idx, 0]
    c1 = table[idx, 1]
    c2 = table[idx, 2]

    # 3. Two-stage column-truncated Horner evaluation matching ref._add_row.
    d_rows = _delta_rows(n_p, prec)
    t_c2 = np.zeros_like(xs)
    for bit_idx, m_weight in d_rows:
        ctl = (x_norm >> bit_idx) & 1
        t_c2 = t_c2 + ctl * (c2 >> m_weight)
    r1 = c1 - t_c2
    if ((r1 < 0) | (r1 >= 2**wc)).any():
        raise OverflowError("Horner intermediate r1 does not fit its register.")

    t_r1 = np.zeros_like(xs)
    for bit_idx, m_weight in d_rows:
        ctl = (x_norm >> bit_idx) & 1
        t_r1 = t_r1 + ctl * (r1 >> m_weight)
    y0 = c0 - t_r1
    if ((y0 <= 0) | (y0 >= 2**wc)).any():
        raise OverflowError("Horner output y0 does not fit its register.")

    # 4. De-normalize: y1 = y0 << s (with output_bits = wc + n_p).
    return np.asarray(y0 << s, dtype=np.int64)  # type: ignore[no-any-return]


@functools.cache
def _evaluate_direct_poly_full_domain(
    n_p: int,
    prec: PolyPrecision,
    precision_bits: int = TOLERANCE_EXPONENT,
) -> tuple[bool, str | None, dict[str, Any]]:
    return _evaluate_direct_poly_array(n_p, prec, precision_bits, xs=None)


def _evaluate_direct_poly_array(
    n_p: int,
    prec: PolyPrecision,
    precision_bits: int = TOLERANCE_EXPONENT,
    *,
    xs: np.ndarray | None = None,
) -> tuple[bool, str | None, dict[str, Any]]:
    max_x = _max_x(n_p)
    if xs is None:
        xs_arr: np.ndarray = np.arange(1, max_x + 1, dtype=np.int64)
        domain_kind = "full-domain"
    else:
        xs_arr = np.asarray(xs, dtype=np.int64)
        domain_kind = "full-domain" if xs_arr.size >= max_x else "subset"
    output_bits = prec.output_bits(n_p)
    if output_bits > 64:
        return (
            False,
            "model screen rejected: output_bits exceeds 64-bit phase accumulator",
            {"error": "output_bits exceeds 64-bit phase accumulator"},
        )
    try:
        y1 = direct_poly_model(n_p, prec, xs_arr)
        phase = y1.astype(np.uint64) << np.uint64(64 - output_bits)
        target, tol = reference_phase_values(
            n_p, xs_arr if xs is not None else None, precision_bits=precision_bits
        )
        error = (phase - target).view(np.int64)
        if (error == np.iinfo(np.int64).min).any():
            raise OverflowError("phase difference saturates signed 64-bit accumulator")
        used = np.abs(error).view(np.uint64)
        ratio = np.divide(
            used, tol, out=np.zeros(xs_arr.size, dtype=np.float64), where=tol > 0
        )
        worst = float(ratio.max())
        worst_x = int(xs_arr[int(np.argmax(ratio))])
        min_margin_units = int((tol.astype(np.int64) - used.astype(np.int64)).min())
    except (OverflowError, ValueError, IndexError, FloatingPointError) as exc:
        return (
            False,
            f"model screen rejected: {exc}",
            {"error": str(exc)},
        )

    metrics = {
        "worst_error_over_tolerance": worst,
        "error_tolerance_ratio": worst,
        "worst_x": worst_x,
        "min_phase_margin_units": min_margin_units,
        "margin_units": min_margin_units,
        "domain_size": int(xs_arr.size),
        "domain_kind": domain_kind,
        "precision": dataclasses.asdict(prec),
    }
    if not math.isfinite(worst) or worst >= 1.0 or min_margin_units <= 0:
        return (
            False,
            f"worst error/tolerance {worst:.6f} >= 1.0",
            metrics,
        )
    return True, None, metrics


def evaluate_direct_poly_model(
    n_p: int,
    prec: PolyPrecision,
    precision_bits: int = TOLERANCE_EXPONENT,
    *,
    xs: np.ndarray | None = None,
) -> tuple[bool, str | None, dict[str, Any]]:
    """Evaluate ``prec`` over the full domain (cached) or an explicit ``xs`` subset."""
    if xs is None:
        return _evaluate_direct_poly_full_domain(n_p, prec, precision_bits)
    return _evaluate_direct_poly_array(n_p, prec, precision_bits, xs=xs)


def build_direct_poly_circuit(n_p: int, prec: PolyPrecision) -> Any:
    """Build the exact Kickmix circuit for direct degree-2 polynomial compilation."""
    import invsqrt.nr_reference_circuit as ref

    import kickmix as km

    width = 2 * n_p + 2
    wc = prec.coef_bits
    k = prec.index_bits
    if k < 2 or k > width:
        raise ValueError(f"index_bits must lie in [2, {width}], got {k}")

    table = _poly_table(n_p, prec)
    d_rows = _delta_rows(n_p, prec)
    first = 1 << (k - 2)

    c2_lo = k - 1
    c2_hi = max((r[2].bit_length() for r in table), default=c2_lo)
    c2_len = max(1, c2_hi - c2_lo)

    builder = km.CircuitBuilder()
    x = builder.create_quantum_register(width, name="x")
    y0_reg = builder.alloc_qubits(wc)
    flags = builder.alloc_qubits(n_p)
    c1_reg = builder.alloc_qubits(wc)
    c2_reg = builder.alloc_qubits(c2_len)
    scratch = builder.alloc_qubits(max(wc, k - 1, n_p))
    y1 = km.array(list(y0_reg) + list(scratch[:n_p]))
    ancillas = scratch[: k - 1]

    def _lookup() -> None:
        bits = [x[width - 1 - level] for level in range(k)]

        def load(row: int, control: Any) -> None:
            c0_v, c1_v, c2_v = table[row]
            for t in range(wc):
                if (c0_v >> t) & 1:
                    builder.cx(control, y0_reg[t])
                if (c1_v >> t) & 1:
                    builder.cx(control, c1_reg[t])
            for t in range(c2_lo, c2_hi):
                if (c2_v >> t) & 1:
                    builder.cx(control, c2_reg[t - c2_lo])

        def node(control: Any, level: int, lo: int, hi: int) -> None:
            if hi <= first:
                return
            valid = max(lo, first)
            if hi - valid == 1 or all(table[r] == table[valid] for r in range(valid, hi)):
                load(valid, control)
                return
            mid = (lo + hi) // 2
            bit_q = bits[level]
            ancilla = ancillas[level - 1]
            if mid <= first:
                builder.init_and(control, bit_q, ancilla)
                node(ancilla, level + 1, mid, hi)
                builder.del_and(control, bit_q, ancilla)
                return
            builder.x(bit_q)
            builder.init_and(control, bit_q, ancilla)
            builder.x(bit_q)
            node(ancilla, level + 1, lo, mid)
            builder.cx(control, ancilla)
            node(ancilla, level + 1, mid, hi)
            builder.del_and(control, bit_q, ancilla)

        half = 1 << (k - 1)
        builder.x(bits[0])
        node(bits[0], 1, 0, half)
        builder.x(bits[0])
        node(bits[0], 1, half, 2 * half)

    def _horner(inv: bool) -> None:
        def step1() -> None:
            for bit_idx, m_weight in d_rows:
                terms = [
                    (c2_reg[i - c2_lo], i - m_weight)
                    for i in range(max(m_weight, c2_lo), c2_hi)
                ]
                if terms:
                    ref._add_row(
                        builder, inv, x[bit_idx], terms, c1_reg, scratch, subtract=True
                    )

        def step2() -> None:
            for bit_idx, m_weight in d_rows:
                terms = [(c1_reg[i], i - m_weight) for i in range(m_weight, wc)]
                if terms:
                    ref._add_row(
                        builder, inv, x[bit_idx], terms, y0_reg, scratch, subtract=True
                    )

        if inv:
            step2()
            step1()
        else:
            step1()
            step2()

    def _denormalize(inv: bool) -> None:
        for f_idx in (reversed(range(n_p)) if inv else range(n_p)):
            lane = y1[f_idx : f_idx + wc + 1]
            if inv:
                builder.right_rotate(lane, 1, control=flags[f_idx])
            else:
                builder.left_rotate(lane, 1, control=flags[f_idx])

    def compute(inverse: bool) -> None:
        stages = [
            ("normalize", lambda inv: ref._normalize(builder, inv, x, flags)),
            ("lookup", lambda inv: _lookup()),
            ("horner", lambda inv: _horner(inv)),
            ("denormalize", lambda inv: _denormalize(inv)),
        ]
        for name, fn in (reversed(stages) if inverse else stages):
            with builder.mark(name):
                fn(inverse)  # type: ignore[no-untyped-call]

    compute(False)
    ref._phase_by_bits(builder, y1)
    compute(True)
    for reg in (y0_reg, flags, c1_reg, c2_reg, scratch):
        builder.free(reg)
    return builder.finish_circuit()


class DirectPolyProposer(Proposer):
    """Direct degree-2 polynomial phase compiler proposer for ``invsqrt``."""

    generator_id = "invsqrt-direct-poly-v2"

    def __init__(self, configs: list[tuple[int, ...]] | None = None) -> None:
        self.configs = list(configs or DIRECT_POLY_CONFIGS)

    def generate(
        self,
        contract: VerifierContract,
        incumbent: Candidate | None,
        budget: int,
    ) -> list[Candidate]:
        candidates: list[Candidate] = []
        seen_ids: set[str] = set()
        for config in self.configs:
            if len(candidates) >= budget:
                break
            prec = PolyPrecision(
                index_bits=config[0],
                coef_bits=config[1],
                delta_bits=config[2],
            )
            parameters = dataclasses.asdict(prec)
            candidate_id = candidate_id_for(
                "invsqrt",
                self.generator_id,
                parameters,
                contract_sha256=contract.contract_sha256,
            )
            if candidate_id in seen_ids:
                continue
            seen_ids.add(candidate_id)
            candidates.append(
                Candidate(
                    candidate_id=candidate_id,
                    generator=self.generator_id,
                    parameters=parameters,
                    parent_candidate_id=incumbent.candidate_id if incumbent else None,
                )
            )
        return candidates

    def screen(self, candidate: Candidate, contract: VerifierContract) -> CandidateScreen:
        try:
            prec = PolyPrecision(**candidate.parameters)
        except TypeError as exc:
            return CandidateScreen(
                valid=False,
                reason=f"invalid precision fields: {exc}",
                metrics={"error": str(exc)},
            )
        n_p = _contract_n_p(contract)
        valid, reason, metrics = evaluate_direct_poly_model(n_p, prec)
        return CandidateScreen(
            valid=valid,
            reason=reason,
            metrics=dict(metrics),
        )

    def compile(self, candidate: Candidate, contract: VerifierContract) -> CandidateArtifact:
        prec = PolyPrecision(**candidate.parameters)
        n_p = _contract_n_p(contract)
        _valid, _reason, screen_metrics = evaluate_direct_poly_model(n_p, prec)
        margin_units = int(screen_metrics.get("min_phase_margin_units", 0))
        ensure_runtime(contract)
        circuit = build_direct_poly_circuit(n_p, prec)
        artifact_dir = Path("artifacts/quantum_compilation/invsqrt_direct_poly")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / f"{candidate.candidate_id}.kmb"
        circuit.to_file(artifact_path, format="kmb")
        digest = sha256_file(artifact_path)
        statistics = circuit.operation_statistics()
        rotations = int(statistics.num_z_pow(min_clifford_hierarchy_level=4))
        t_like = int(statistics.num_t_like)
        return CandidateArtifact(
            path=artifact_path.resolve(),
            sha256=digest,
            metadata={
                "num_qubits": int(circuit.num_qubits),
                "toffoli_like": int(statistics.num_toffoli_like),
                "t_like": t_like,
                "rotations": rotations,
                "min_phase_margin_units": margin_units,
                "margin_units": margin_units,
                "precision": dataclasses.asdict(prec),
            },
        )

    def measure_resources(
        self, artifact: CandidateArtifact, contract: VerifierContract
    ) -> float:
        meta = dict(artifact.metadata)
        n_p = _contract_n_p(contract)
        if not meta and artifact.path.is_file():
            ensure_runtime(contract)
            import kickmix as km

            circuit = km.Circuit.from_file(artifact.path)
            stats = circuit.operation_statistics()
            meta = {
                "num_qubits": int(circuit.num_qubits),
                "toffoli_like": int(stats.num_toffoli_like),
                "t_like": int(stats.num_t_like),
                "rotations": int(stats.num_z_pow(min_clifford_hierarchy_level=4)),
            }

        num_qubits = int(meta["num_qubits"])
        toffoli_like = float(meta["toffoli_like"])
        rotations = int(meta["rotations"])
        t_base = float(meta.get("t_like", 1.0))
        margin_units = int(meta.get("min_phase_margin_units", meta.get("margin_units", 0)))
        if margin_units <= 0 and "precision" in meta:
            prec = PolyPrecision(**meta["precision"])
            _, _, screen_metrics = evaluate_direct_poly_model(n_p, prec)
            margin_units = int(screen_metrics.get("min_phase_margin_units", 0))

        t_per_toffoli = float(contract.constants.get("T_PER_TOFFOLI", 4.0))
        if rotations > 0 and margin_units > 0:
            eps = math.pi * float(margin_units) / (2.0**64) / rotations
            t_count = t_base + rotations * _rotation_t_count(eps)
        else:
            t_count = t_base
        return float(num_qubits * (toffoli_like + t_count / t_per_toffoli))


def ensure_runtime(contract: VerifierContract) -> tuple[Any, Any]:
    """Import kickmix and the shipped invsqrt verifier/reference modules."""
    for path in (
        str(contract.kickmix_root),
        str(contract.verifier_root.parent),
        str(contract.verifier_root),
    ):
        if path not in sys.path:
            sys.path.insert(0, path)
    import invsqrt.nr_reference_circuit as nr_reference_circuit
    import invsqrt.verify_submission_invsqrt as verify_submission_invsqrt

    import kickmix  # noqa: F401

    expected = contract.verifier_root.resolve()
    actual = Path(verify_submission_invsqrt.__file__).resolve().parent
    if actual != expected:
        raise RuntimeError(
            f"imported invsqrt from {actual}, expected {expected}"
        )
    return nr_reference_circuit, verify_submission_invsqrt
