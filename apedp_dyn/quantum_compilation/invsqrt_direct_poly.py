"""Experiment 03: Direct polynomial phase compiler for inverse square root.

Status: experimental

Directly fits piecewise cubic polynomials to ``1 / sqrt(M)`` over the
normalized mantissa ``M`` in ``[1, 4)``, evaluates by Horner's rule, and
de-normalizes with a unary-controlled shift -- avoiding the Newton-Raphson
``square / multiply_m / correct / multiply_y0`` stages of the reference
circuit.

Interval scheme (exact power-of-two addressing):

* After normalization ``x_norm = x << 2s`` and ``M = x_norm / 2^(width-2)``.
* The interval index is ``i = (x_norm >> (width - k)) - 2^(k-2)`` with
  ``N = 3 * 2^(k-2)`` intervals; the selector register holds the raw top
  ``k`` bits of ``x_norm`` (clean bit extraction, no subtraction).
* ``delta = M - interval_base`` is computed with selector-controlled constant
  subtractions: ``base = sum_j sel_j * 2^(wc-k+2+j)`` (no borrows since the
  low bits of ``base`` are zero).

``nr_reference_circuit``'s alternating-sign Horner (with negated odd
coefficients) evaluates the standard polynomial
``c0 + c1*delta + c2*delta^2 + c3*delta^3``.

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

__all__ = [
    "DirectPolyProposer",
    "PolyPrecision",
    "build_direct_poly_circuit",
    "direct_poly_model",
    "fit_polynomials",
]

TOLERANCE_EXPONENT = 24  # relative phase tolerance 2^-24, matching the reference.

DEFAULT_CONFIG = (8, 26, 12, 28, 28, 29)  # index_bits, coef_bits, delta_bits, square_bits, product_bits, result_bits
DIRECT_POLY_CONFIGS = [
    (8, 26, 12, 28, 28, 29),
    (9, 26, 12, 29, 29, 30),
    (8, 27, 12, 29, 28, 29),
    (9, 25, 12, 28, 28, 29),
    (7, 26, 12, 28, 28, 29),
    (8, 28, 12, 30, 29, 30),
    (9, 27, 12, 30, 29, 30),
]

# candidate_id -> phase margin (fraction of a turn), set by screen(), used by
# compile()/measure_resources() for rotation-synthesis cost.
_SCREEN_MARGIN_CACHE: dict[str, float] = {}


@dataclasses.dataclass(frozen=True)
class PolyPrecision:
    index_bits: int = 8      # selector bits; N = 3 * 2^(index_bits - 2) intervals
    coef_bits: int = 26      # coefficient and Horner intermediate width
    delta_bits: int = 12     # (informational; Horner uses all significant delta bits)
    square_bits: int = 28    # intermediate register width
    product_bits: int = 28   # product register width (+2 integer bits)
    result_bits: int = 29    # output register width (+1 integer bit)

    def output_bits(self, n_p: int) -> int:
        return self.result_bits + n_p + 1

    def num_intervals(self) -> int:
        return 3 * (1 << (self.index_bits - 2))


def _contract_n_p(contract: VerifierContract) -> int:
    return int(contract.constants["N_P"])


def _max_x(n_p: int) -> int:
    return int(3 * (2**n_p - 2) ** 2)


def fit_polynomials(n_p: int, prec: PolyPrecision) -> list[list[int]]:
    """Fit per-interval cubic polynomials to ``1/sqrt(M)`` on ``[1, 4)``.

    Returns ``table[interval][j]`` = coefficient of ``delta^j`` times
    ``2^coef_bits``, where ``delta = M - interval_base``.  Coefficients c1 and
    c3 are negated so that NR's alternating-sign Horner
    (``c0 - c1*delta + c2*delta^2 - c3*delta^3``) evaluates the standard
    polynomial ``c0 + c1*delta + c2*delta^2 + c3*delta^3``.
    """
    num_intervals = prec.num_intervals()
    table: list[list[int]] = []
    for i in range(num_intervals):
        m0 = 1.0 + 3.0 * i / num_intervals
        m1 = 1.0 + 3.0 * (i + 1) / num_intervals
        delta_nodes = np.linspace(0, m1 - m0, 9)
        delta_values = 1.0 / np.sqrt(m0 + delta_nodes)
        delta_coeffs = np.polynomial.polynomial.polyfit(delta_nodes, delta_values, 3)
        row = [round(c * 2**prec.coef_bits) for c in delta_coeffs]
        # Negate odd coefficients so NR's alternating-sign Horner yields
        # c0 + c1*delta + c2*delta^2 + c3*delta^3.
        row[1] = -row[1]
        row[3] = -row[3]
        table.append(row)
    # Clamp coefficients to fit in wc-bit signed fixed point: [-2^wc, 2^wc - 1].
    limit = 1 << prec.coef_bits
    for row in table:
        for j in range(4):
            row[j] = max(-limit, min(limit - 1, row[j]))
    return table


@functools.cache
def _poly_table(n_p: int, prec: PolyPrecision) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(row) for row in fit_polynomials(n_p, prec))


def direct_poly_model(n_p: int, prec: PolyPrecision, xs: np.ndarray) -> np.ndarray:
    """Exact fixed-point output for every input ``x`` (no circuit built).

    ``result / 2^output_bits(n_p)`` approximates ``1 / sqrt(x)``.
    """
    width = 2 * n_p + 2
    wc = prec.coef_bits
    xs = np.asarray(xs, dtype=np.int64)
    if not ((1 <= xs) & (xs < 2**width)).all():
        raise ValueError("x must lie in [1, 2^(width - 1)).")

    # 1. Normalize: s leading zero pairs, M = x_norm / 2^(width - 2) in [1, 4).
    s = (width - (np.floor(np.log2(xs)).astype(np.int64) + 1)) // 2
    x_norm = xs << (2 * s)
    # M_int = M * 2^wc (wide enough for the raw mantissa).
    shift_m = width - 2 - wc
    if shift_m >= 0:
        M_int = x_norm >> shift_m
    else:
        M_int = x_norm << (-shift_m)

    # 2. Interval lookup: i = (x_norm >> (width - k)) - 2^(k-2).
    num_intervals = prec.num_intervals()
    one_fixed = 1 << wc
    three_fixed = 3 << wc
    interval: np.ndarray = ((M_int - one_fixed) * num_intervals) // three_fixed
    interval = np.clip(interval, 0, num_intervals - 1)

    table = np.array(_poly_table(n_p, prec), dtype=np.int64)
    c0 = table[interval, 0]
    c1 = table[interval, 1]
    c2 = table[interval, 2]
    c3 = table[interval, 3]

    # 3. delta = M - interval_base, in fixed point.
    interval_base: np.ndarray = one_fixed + (interval * three_fixed) // num_intervals
    delta = M_int - interval_base

    # 4. NR's alternating-sign Horner with negated odd coefficients.
    #    y0 = c0 - T(delta*(c1 - T(delta*(c2 - T(delta*c3)))))
    def trunc_product(value: np.ndarray) -> np.ndarray:
        return value >> prec.coef_bits  # type: ignore[no-any-return]

    t3 = trunc_product(delta * c3)
    r2 = c2 - t3
    t2r = trunc_product(delta * r2)
    r1 = c1 - t2r
    t1r = trunc_product(delta * r1)
    y0 = c0 - t1r

    if ((y0 <= 0) | (y0 >= 2**wc)).any():
        raise OverflowError("Horner intermediate does not fit its register.")

    # 5. De-normalize: result / 2^output_bits = y0 / 2^wc * 2^(s - n_p).
    shift = prec.output_bits(n_p) - wc + s - n_p
    result = np.where(shift >= 0, y0 << shift, y0 >> -shift)
    return result  # type: ignore[no-any-return]


def build_direct_poly_circuit(n_p: int, prec: PolyPrecision) -> Any:
    """Build the exact Kickmix circuit for direct polynomial compilation."""
    import invsqrt.nr_reference_circuit as ref
    import kickmix as km

    width = 2 * n_p + 2
    out_bits = prec.output_bits(n_p)
    wc = prec.coef_bits
    k = prec.index_bits
    if k < 2 or k > width:
        raise ValueError(f"index_bits must lie in [2, {width}], got {k}")
    m_shift = max(0, wc - (width - 2))
    builder = km.CircuitBuilder()

    x = builder.create_quantum_register(width, name="x")
    y1 = builder.alloc_qubits(out_bits)
    flags = builder.alloc_qubits(n_p)
    delta_reg = builder.alloc_qubits(max(wc, width + m_shift))
    scratch = builder.alloc_qubits(
        max(wc, prec.square_bits, prec.product_bits, k - 1)
    )

    coefs = [builder.alloc_qubits(wc) for _ in range(4)]
    selectors = builder.alloc_qubits(k)
    y0 = coefs[0]  # NR convention: Horner result lives in coefs[0].

    def compute(inverse: bool) -> None:
        stages = [
            ("normalize", lambda inv: ref._normalize(builder, inv, x, flags)),
            ("select_interval", lambda inv: _select_interval_direct(builder, x, selectors, k)),
            (
                "load_coefficients",
                lambda inv: _load_coefficients_direct(builder, prec, x, selectors, coefs, scratch),
            ),
            (
                "compute_delta",
                lambda inv: _compute_delta_direct(builder, inv, x, delta_reg, selectors, prec),
            ),
            (
                "horner",
                lambda inv: _horner_direct(builder, inv, prec, delta_reg, coefs, scratch),
            ),
            (
                "denormalize",
                lambda inv: _denormalize_direct(builder, flags, y0, y1, scratch, n_p),
            ),
        ]
        for name, fn in (reversed(stages) if inverse else stages):
            with builder.mark(name):
                fn(inverse)  # type: ignore[no-untyped-call]

    compute(False)
    ref._phase_by_bits(builder, y1)
    compute(True)
    for reg in (y1, flags, delta_reg, scratch, *coefs, selectors):
        builder.free(reg)
    return builder.finish_circuit()


def _select_interval_direct(builder: Any, x: Any, selectors: Any, k: int) -> None:
    """Copy the raw top ``k`` bits of ``x_norm`` into ``selectors``.

    ``sel_j = x_norm[width - k + j]`` so the selector value is
    ``s = x_norm >> (width - k)``; the interval index is ``s - 2^(k-2)``.
    Plain CNOTs (self-inverse; safe to re-apply during uncompute).
    """
    width = len(x)
    shift = width - k
    for j in range(k):
        builder.cx(x[shift + j], selectors[j])


def _load_coefficients_direct(
    builder: Any,
    prec: PolyPrecision,
    x: Any,
    selectors: Any,
    coefs: list[Any],
    scratch: Any,
) -> None:
    """Load polynomial coefficients via a per-interval AND tree.

    For each interval the (possibly complemented) selector bits are ANDed
    into a chain of ancillas; a single-control CX per coefficient bit then
    XORs the interval's coefficient value.  The chain is uncomputed
    immediately, so a second call restores the initial state.
    """
    table = _poly_table(10, prec)
    k = prec.index_bits
    wc = prec.coef_bits
    num_intervals = prec.num_intervals()
    base_offset = 1 << (k - 2)
    ancillas = scratch[: k - 1]

    for i in range(num_intervals):
        s = i + base_offset
        for j in range(k):
            if ((s >> j) & 1) == 0:
                builder.x(selectors[j])
        # AND chain: anc[0] = sel0 & sel1; anc[j-1] = anc[j-2] & sel_j.
        builder.init_and(selectors[0], selectors[1], ancillas[0])
        for j in range(2, k):
            builder.init_and(ancillas[j - 2], selectors[j], ancillas[j - 1])
        control = ancillas[k - 2]
        for cj, reg in enumerate(coefs):
            value = table[i][cj]
            for t in range(wc):
                if (value >> t) & 1:
                    builder.cx(control, reg[t])
        # Uncompute the chain (reverse order).
        for j in range(k - 1, 1, -1):
            builder.del_and(ancillas[j - 2], selectors[j], ancillas[j - 1])
        builder.del_and(selectors[0], selectors[1], ancillas[0])
        for j in range(k):
            if ((s >> j) & 1) == 0:
                builder.x(selectors[j])


def _compute_delta_direct(
    builder: Any,
    inv: bool,
    x: Any,
    delta_reg: Any,
    selectors: Any,
    prec: PolyPrecision,
) -> None:
    """Compute ``delta = M * 2^wc - interval_base`` as a pure bit copy.

    With ``s = x_norm >> (width-k)`` and ``base = s * 2^(wc-k+2)``:
    ``delta = x_norm * 2^m_shift - base = 2^m_shift * (x_norm mod 2^(width-k))``
    where ``m_shift = wc - (width - 2)``.  So ``delta`` is just the low
    ``width-k`` bits of ``x_norm`` shifted up by ``m_shift`` -- plain CNOTs,
    exactly self-inverse, no borrow chain.
    """
    width = len(x)
    k = prec.index_bits
    m_shift = prec.coef_bits - (width - 2)
    low_bits = width - k
    reg_bits = len(delta_reg)

    if m_shift >= 0:
        for t in range(min(low_bits, reg_bits - m_shift)):
            builder.cx(x[t], delta_reg[t + m_shift])
    else:
        for t in range(min(low_bits + m_shift, reg_bits)):
            builder.cx(x[t - m_shift], delta_reg[t])


def _horner_direct(
    builder: Any,
    inv: bool,
    prec: PolyPrecision,
    delta_reg: Any,
    coefs: list[Any],
    scratch: Any,
) -> None:
    """Horner: y0 = c0 + c1*delta + c2*delta^2 + c3*delta^3 (result in coefs[0]).

    Uses NR's ``_add_row`` with ``delta_reg`` bits as controls.  ``delta`` is
    bounded by the interval width, so bits of weight ``2^-m`` live at position
    ``wc - m`` for ``m in [index_bits - 1, wc)``; lower ``m`` are provably zero.
    """
    import invsqrt.nr_reference_circuit as ref

    wc = prec.coef_bits
    k = prec.index_bits
    c0, c1, c2, c3 = coefs

    steps = ((c2, c3), (c1, c2), (c0, c1))
    for acc, mult in (reversed(steps) if inv else steps):
        for m in range(k - 1, wc):
            control = delta_reg[wc - m]
            terms = [(mult[i], i - m) for i in range(m, wc)]
            if terms:
                ref._add_row(builder, inv, control, terms, acc, scratch, subtract=True)


def _denormalize_direct(
    builder: Any,
    flags: Any,
    y0: Any,
    y1: Any,
    scratch: Any,
    n_p: int,
) -> None:
    """De-normalize: ``y1[t + shift_base + s] ^= y0[t]`` for each shift ``s``.

    ``flags`` is unary (``flags[k] = 1`` iff ``s >= k + 1``), so the exact-``s``
    control is simple: ``NOT flags[0]`` for ``s = 0``, ``flags[s-1] AND NOT
    flags[s]`` for ``1 <= s < n_p``, and ``flags[n_p-1]`` for ``s = n_p``.
    Writing into the full ``y1`` (not just ``p``) keeps the high phase bits
    for large shifts.  The block is self-inverse.
    """
    wc = len(y0)
    output_bits = len(y1)
    shift_base = output_bits - wc - n_p
    anc = scratch[0]

    for t in range(wc):
        # s = 0
        target = t + shift_base
        if 0 <= target < output_bits:
            builder.x(flags[0])
            builder.cx(flags[0], y1[target])
            builder.x(flags[0])
        # 1 <= s < n_p
        for s_val in range(1, n_p):
            target = t + shift_base + s_val
            if 0 <= target < output_bits:
                builder.x(flags[s_val])
                builder.init_and(flags[s_val - 1], flags[s_val], anc)
                builder.cx(anc, y1[target])
                builder.del_and(flags[s_val - 1], flags[s_val], anc)
                builder.x(flags[s_val])
        # s = n_p
        target = t + shift_base + n_p
        if 0 <= target < output_bits:
            builder.cx(flags[n_p - 1], y1[target])


class DirectPolyProposer(Proposer):  # type: ignore[misc]
    """Direct polynomial phase compiler proposer for ``invsqrt``."""

    generator_id = "invsqrt-direct-poly-v1"

    def __init__(self, configs: list[tuple[int, ...]] | None = None) -> None:
        self.configs = list(configs or DIRECT_POLY_CONFIGS)

    def generate(
        self,
        contract: VerifierContract,
        incumbent: Candidate | None,
        budget: int,
    ) -> list[Candidate]:
        candidates: list[Candidate] = []
        for config in self.configs:
            if len(candidates) >= budget:
                break
            prec = PolyPrecision(
                index_bits=config[0],
                coef_bits=config[1],
                delta_bits=config[2],
                square_bits=config[3],
                product_bits=config[4],
                result_bits=config[5],
            )
            parameters = dataclasses.asdict(prec)
            candidate_id = candidate_id_for(
                "invsqrt",
                self.generator_id,
                parameters,
                contract_sha256=contract.contract_sha256,
            )
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
        prec = PolyPrecision(**candidate.parameters)
        n_p = _contract_n_p(contract)
        max_x = _max_x(n_p)
        xs: np.ndarray = np.arange(1, max_x + 1, dtype=np.int64)
        exact = 1.0 / np.sqrt(xs.astype(np.float64))
        tolerance = (2.0 ** -TOLERANCE_EXPONENT) * exact
        try:
            approx = direct_poly_model(n_p, prec, xs) / 2.0 ** prec.output_bits(n_p)
            worst = float((np.abs(approx - exact) / tolerance).max())
            margin = float((tolerance - np.abs(approx - exact)).min())
        except (OverflowError, ValueError, IndexError, FloatingPointError) as exc:
            return CandidateScreen(
                valid=False,
                reason=f"model screen rejected: {exc}",
                metrics={"error": str(exc)},
            )
        if not math.isfinite(worst) or worst >= 1.0:
            return CandidateScreen(
                valid=False,
                reason=f"worst error/tolerance {worst:.6f} >= 1.0",
                metrics={"worst_error_over_tolerance": worst, "margin_units": margin},
            )
        _SCREEN_MARGIN_CACHE[candidate.candidate_id] = margin
        return CandidateScreen(
            valid=True,
            reason=None,
            metrics={
                "worst_error_over_tolerance": worst,
                "margin_units": margin,
                "precision": dataclasses.asdict(prec),
            },
        )

    def compile(self, candidate: Candidate, contract: VerifierContract) -> CandidateArtifact:
        prec = PolyPrecision(**candidate.parameters)
        n_p = _contract_n_p(contract)
        ensure_runtime(contract)
        circuit = build_direct_poly_circuit(n_p, prec)
        artifact_dir = Path("artifacts/quantum_compilation/invsqrt_direct_poly")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / f"{candidate.candidate_id}.kmb"
        circuit.to_file(artifact_path, format="kmb")
        digest = sha256_file(artifact_path)
        statistics = circuit.operation_statistics()
        rotations = int(statistics.num_z_pow(min_clifford_hierarchy_level=4))
        margin = _SCREEN_MARGIN_CACHE.pop(candidate.candidate_id, 0.0)
        return CandidateArtifact(
            path=artifact_path.resolve(),
            sha256=digest,
            metadata={
                "num_qubits": int(circuit.num_qubits),
                "toffoli_like": int(statistics.num_toffoli_like),
                "rotations": rotations,
                "margin_units": margin,
                "precision": dataclasses.asdict(prec),
            },
        )

    def measure_resources(
        self, artifact: CandidateArtifact, contract: VerifierContract
    ) -> float:
        meta = artifact.metadata
        num_qubits = int(meta["num_qubits"])
        toffoli_like = float(meta["toffoli_like"])
        rotations = int(meta["rotations"])
        margin = float(meta["margin_units"])
        _, verifier = ensure_runtime(contract)
        if rotations > 0 and margin > 0:
            eps = math.pi * margin / 2.0**64 / rotations
            t_count = 1.0 + rotations * verifier.rotation_t_count(eps)
        else:
            t_count = 1.0
        return float(num_qubits * (toffoli_like + t_count / 4.0))

    def verify(
        self,
        artifact: CandidateArtifact,
        contract: VerifierContract,
        *,
        python_executable: str | None = None,
        timeout_s: float = 1800.0,
        salt: str = "",
    ) -> Any:
        from apedp_dyn.quantum_compilation.referee import run_invsqrt_verifier

        return run_invsqrt_verifier(
            contract,
            artifact.path,
            python_executable=python_executable or sys.executable,
            salt=salt,
            timeout_s=timeout_s,
        )


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
