"""Experiment 02: QROM piecewise-linear seed proposer for inverse square root.

Status: experimental

Adapts ``scripts/nr_qrom.py`` from the ``apedp_q_alternatives`` scaffold into an
APEDP-Q :class:`~apedp_dyn.quantum_compilation.experiment.Proposer`.

The proposer varies the piecewise-linear QROM seed precision
``(index_bits, coef_bits, slope_bits, delta_bits, square_bits, product_bits,
result_bits)``, screens each point with the exact fixed-point model over the
full inverse-square-root domain in 64-bit phase units (2^-64 turns, matching the
official verifier), compiles the exact ``.kmb`` circuit, and lets the
:class:`~apedp_dyn.quantum_compilation.experiment.ExperimentRunner` verify it
with the official competition verifier.

Estimated cost (from circuit statistics + rotation synthesis) and verified
measured cost (from the official verifier) are kept as separate fields and
never conflated.
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
    "QROM_LINE_CONFIGS",
    "QPrecision",
    "QromLineProposer",
    "build_qrom_line_circuit",
    "evaluate_qrom_line_model",
    "fixed_point_model",
    "line_table",
    "reference_phase_values",
]

TOLERANCE_EXPONENT = 24  # relative phase tolerance 2^-24, matching the reference.
_MASK64 = (1 << 64) - 1

# Precision configurations from scripts/run_invsqrt_qrom_sweep.py and expanded search
# (index_bits, coef_bits, slope_bits, delta_bits, square_bits, product_bits, result_bits).
QROM_LINE_CONFIGS: list[tuple[int, int, int, int, int, int, int]] = [
    (6, 18, 13, 13, 28, 27, 28),
    (6, 18, 12, 12, 28, 27, 28),
    (7, 15, 12, 11, 28, 27, 28),
    (7, 15, 13, 12, 28, 27, 28),
    (7, 16, 12, 10, 28, 27, 28),
    (7, 16, 11, 11, 28, 27, 27),
    (7, 16, 12, 11, 28, 27, 27),
    (7, 16, 12, 11, 28, 27, 28),
    (7, 16, 12, 12, 28, 27, 27),
    (7, 16, 12, 12, 28, 27, 28),
    (7, 17, 12, 12, 28, 27, 28),
    (7, 16, 12, 11, 27, 26, 28),
    (7, 17, 12, 11, 28, 27, 28),
    (8, 18, 13, 12, 29, 28, 29),
    (8, 18, 13, 12, 30, 29, 30),
    # Expanded configurations (deduplicated)
    (7, 17, 13, 12, 28, 27, 28),
    (7, 16, 13, 12, 29, 28, 28),
    (8, 17, 12, 11, 28, 27, 29),
    (6, 19, 13, 14, 28, 27, 28),
    (7, 18, 13, 12, 29, 28, 29),
    (8, 16, 12, 10, 29, 28, 29),
    (7, 15, 11, 10, 27, 26, 27),
]


@dataclasses.dataclass(frozen=True)
class QPrecision:
    index_bits: int = 7
    coef_bits: int = 16
    slope_bits: int = 12
    delta_bits: int = 10
    square_bits: int = 28
    product_bits: int = 27
    result_bits: int = 28

    def output_bits(self, n_p: int) -> int:
        return self.result_bits + n_p + 1


def _contract_n_p(contract: VerifierContract) -> int:
    return int(contract.constants["N_P"])


def _max_x(n_p: int) -> int:
    return int(3 * (2**n_p - 2) ** 2)


def _rotation_t_count(eps: float) -> float:
    return 0.56 * math.log2(1.0 / eps) + 5.3


@functools.cache
def _full_domain_reference_values(
    n_p: int, precision_bits: int = TOLERANCE_EXPONENT
) -> tuple[np.ndarray, np.ndarray]:
    """Exact 64-bit phase target and tolerance in units of 2^-64 turns for 1..max_x.

    Matches ``verify_submission_invsqrt.reference_values`` for ``beta = 1`` using
    exact 130-bit integer square roots without requiring the ``kickmix`` extension.
    """
    max_x = _max_x(n_p)
    num = 1 << 130
    shift = precision_bits + 1
    cap = 1 << 63
    rs = [math.isqrt(num // x) for x in range(1, max_x + 1)]
    target = np.array([((r + 1) >> 1) & _MASK64 for r in rs], dtype=np.uint64)
    tol = np.array(
        [min(r >> shift, cap) if shift >= 0 else min(r << -shift, cap) for r in rs],
        dtype=np.uint64,
    )
    return target, tol


def reference_phase_values(
    n_p: int,
    xs: np.ndarray | None = None,
    *,
    precision_bits: int = TOLERANCE_EXPONENT,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(target, tol)`` in 2^-64-turn uint64 units for ``xs``."""
    max_x = _max_x(n_p)
    if xs is None:
        return _full_domain_reference_values(n_p, precision_bits)
    xs_arr = np.asarray(xs, dtype=np.int64)
    if xs_arr.size == max_x and int(xs_arr[0]) == 1 and int(xs_arr[-1]) == max_x:
        return _full_domain_reference_values(n_p, precision_bits)
    num = 1 << 130
    shift = precision_bits + 1
    cap = 1 << 63
    rs = [math.isqrt(num // int(x)) for x in xs_arr.tolist()]
    target = np.array([((r + 1) >> 1) & _MASK64 for r in rs], dtype=np.uint64)
    tol = np.array(
        [min(r >> shift, cap) if shift >= 0 else min(r << -shift, cap) for r in rs],
        dtype=np.uint64,
    )
    return target, tol


def ensure_runtime(contract: VerifierContract) -> tuple[Any, Any]:
    """Import kickmix and the shipped invsqrt verifier/reference modules.

    The bundled ``kickmix`` extension targets CPython 3.13, so this must run
    under that interpreter.
    """
    for path in (
        str(contract.kickmix_root),
        str(contract.verifier_root.parent),
        str(contract.verifier_root),
    ):
        if path not in sys.path:
            sys.path.insert(0, path)
    import invsqrt.nr_reference_circuit as nr_reference_circuit
    import invsqrt.verify_submission_invsqrt as verify_submission_invsqrt

    import kickmix  # noqa: F401 - import is the capability check

    expected = contract.verifier_root.resolve()
    actual = Path(verify_submission_invsqrt.__file__).resolve().parent
    if actual != expected:
        raise RuntimeError(
            f"imported invsqrt from {actual}, expected {expected}; refuse to "
            "verify against a different copy of the benchmark."
        )
    return nr_reference_circuit, verify_submission_invsqrt


@functools.cache
def line_table(
    n_p: int, prec: QPrecision
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    k, wc, wb = prec.index_bits, prec.coef_bits, prec.slope_bits
    a_tab: list[int] = [0] * (1 << k)
    b_tab: list[int] = [0] * (1 << k)
    for i in range(1 << (k - 2), 1 << k):
        m0 = i / 2 ** (k - 2)
        m1 = (i + 1) / 2 ** (k - 2)
        slope = (m0**-0.5 - m1**-0.5) / (m1 - m0)
        ms = np.linspace(m0, m1, 4097)
        gap = float((m0**-0.5 - slope * (ms - m0) - ms**-0.5).max())
        a_tab[i] = round((m0**-0.5 - gap / 2) * 2**wc)
        b_tab[i] = round(slope * 2**wb)
    return tuple(a_tab), tuple(b_tab)


def _delta_rows(n_p: int, prec: QPrecision) -> list[tuple[int, int]]:
    width = 2 * n_p + 2
    k = prec.index_bits
    rows: list[tuple[int, int]] = []
    for j in range(1, prec.delta_bits + 1):
        bit = width - k - j
        if bit < 0:
            break
        rows.append((bit, prec.coef_bits - prec.slope_bits - (k - 2 + j)))
    return rows


def fixed_point_model(n_p: int, prec: QPrecision, xs: np.ndarray) -> np.ndarray:
    """Exact fixed-point output for every input ``x`` (no circuit built)."""
    width = 2 * n_p + 2
    wc = prec.coef_bits
    wsq = prec.square_bits
    wz = prec.product_bits
    wp = prec.result_bits
    xs = np.asarray(xs, dtype=np.int64)
    if (xs <= 0).any():
        raise ValueError("Inputs must be positive")

    def bit(v: Any, i: int) -> Any:
        return (v >> i) & 1

    def product(
        rows: list[tuple[Any, int]],
        operand: np.ndarray,
        out_bits: int,
        operand_bits: int,
        limit: int,
    ) -> np.ndarray:
        acc: np.ndarray = np.zeros_like(xs)
        for ctl, weight in rows:
            e = out_bits - operand_bits + weight
            shifted = operand << e if e >= 0 else operand >> -e
            acc = acc + (ctl * shifted)
        if (acc >= 2**limit).any():
            raise OverflowError("Intermediate register overflow")
        return acc

    s = (width - (np.floor(np.log2(xs)).astype(np.int64) + 1)) // 2
    x_norm = xs << (2 * s)
    a_tab_raw, b_tab_raw = line_table(n_p, prec)
    a_tab: np.ndarray = np.asarray(a_tab_raw, dtype=np.int64)
    b_tab: np.ndarray = np.asarray(b_tab_raw, dtype=np.int64)
    index = x_norm >> (width - prec.index_bits)
    b = b_tab[index]
    y0 = a_tab[index].copy()
    for row_bit, shift in _delta_rows(n_p, prec):
        y0 = y0 - bit(x_norm, row_bit) * (
            b << shift if shift >= 0 else b >> -shift
        )
    if ((y0 <= 0) | (y0 >= 2**wc)).any():
        raise OverflowError("y0 does not fit its register")
    y = {k: bit(y0, wc - k) for k in range(1, wc + 1)}
    square = np.zeros_like(xs)
    for k in range(1, wc + 1):
        if 2 * k <= wsq:
            square = square + (y[k] << (wsq - 2 * k))
        for m2 in range(k + 1, wc + 1):
            if k + m2 - 1 <= wsq:
                square = square + ((y[k] & y[m2]) << (wsq - (k + m2 - 1)))
    m_rows = [(bit(x_norm, i), i - width + 2) for i in range(width)]
    z = product(m_rows, square, wz, wsq, wz + 2)
    correction: np.ndarray = (3 << wz) - z
    p: np.ndarray = product([(y[k], -k) for k in range(1, wc + 1)], correction, wp, wz, wp + 1)
    return np.asarray(p << s, dtype=np.int64)  # type: ignore[no-any-return]


@functools.cache
def _evaluate_qrom_line_full_domain(
    n_p: int,
    prec: QPrecision,
    precision_bits: int = TOLERANCE_EXPONENT,
) -> tuple[bool, str | None, dict[str, Any]]:
    return _evaluate_qrom_line_array(n_p, prec, precision_bits, xs=None)


def _evaluate_qrom_line_array(
    n_p: int,
    prec: QPrecision,
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
        y1 = fixed_point_model(n_p, prec, xs_arr)
        phase = y1.astype(np.uint64) << np.uint64(64 - output_bits)
        if xs is None:
            target, tol = _full_domain_reference_values(n_p, precision_bits)
        else:
            target, tol = reference_phase_values(n_p, xs_arr, precision_bits=precision_bits)
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


def evaluate_qrom_line_model(
    n_p: int,
    prec: QPrecision,
    precision_bits: int = TOLERANCE_EXPONENT,
    *,
    xs: np.ndarray | None = None,
) -> tuple[bool, str | None, dict[str, Any]]:
    """Evaluate ``prec`` over the full domain (cached) or an explicit ``xs`` subset."""
    if xs is None:
        return _evaluate_qrom_line_full_domain(n_p, prec, precision_bits)
    return _evaluate_qrom_line_array(n_p, prec, precision_bits, xs=xs)


def build_qrom_line_circuit(n_p: int, prec: QPrecision) -> Any:
    """Build the exact Kickmix circuit for one QPrecision point."""
    import invsqrt.nr_reference_circuit as ref

    import kickmix as km

    width = 2 * n_p + 2
    out_bits = prec.output_bits(n_p)
    builder = km.CircuitBuilder()
    x = builder.create_quantum_register(width, name="x")
    y1 = builder.alloc_qubits(out_bits)
    flags = builder.alloc_qubits(n_p)
    y0 = builder.alloc_qubits(prec.coef_bits)
    breg = builder.alloc_qubits(prec.slope_bits)
    square = builder.alloc_qubits(prec.square_bits)
    z = builder.alloc_qubits(prec.product_bits + 2)
    scratch = builder.alloc_qubits(
        max(
            prec.square_bits,
            prec.product_bits + 2,
            prec.result_bits + 1,
            prec.coef_bits,
            prec.index_bits - 1,
        )
    )
    p = y1[: len(y1) - len(flags)]
    ancillas = scratch[: prec.index_bits - 1]

    def _lookup(
        builder: Any, x: Any, ancillas: Any, y0: Any, breg: Any
    ) -> Any:
        k = prec.index_bits
        a_tab, b_tab = line_table(n_p, prec)
        bits = [x[width - 1 - level] for level in range(k)]
        first = 1 << (k - 2)

        def load(row: int, control: Any) -> None:
            for reg, value in ((y0, a_tab[row]), (breg, b_tab[row])):
                for t in range(len(reg)):
                    if (value >> t) & 1:
                        builder.cx(control, reg[t])

        def node(
            control: Any, level: int, lo: int, hi: int
        ) -> None:
            if hi <= first:
                return
            valid = max(lo, first)
            if hi - valid == 1 or all(
                a_tab[r] == a_tab[valid] and b_tab[r] == b_tab[valid]
                for r in range(valid, hi)
            ):
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

    def _linear(
        builder: Any, inverse: Any, x: Any, breg: Any, y0: Any, scratch: Any
    ) -> None:
        for row_bit, shift in _delta_rows(n_p, prec):
            terms = [
                (breg[t], t + shift)
                for t in range(len(breg))
                if 0 <= t + shift < len(y0)
            ]
            if terms:
                ref._add_row(builder, inverse, x[row_bit], terms, y0, scratch, subtract=True)

    def compute(inverse: Any) -> None:
        stages: list[tuple[str, Any]] = [
            ("normalize", lambda inv: ref._normalize(builder, inv, x, flags)),
            ("lookup", lambda inv: _lookup(builder, x, ancillas, y0, breg)),
            ("linear", lambda inv: _linear(builder, inv, x, breg, y0, scratch)),
            ("square", lambda inv: ref._square(builder, inv, y0, square, scratch)),
            ("multiply_m", lambda inv: ref._multiply_m(builder, inv, x, square, z, scratch)),
            ("correct", lambda inv: ref._correct(builder, inv, z)),
            ("multiply_y0", lambda inv: ref._multiply_y0(builder, inv, y0, z, p, scratch)),
            ("denormalize", lambda inv: ref._denormalize(builder, inv, flags, y1)),
        ]
        for _name, fn in (reversed(stages) if inverse else stages):
            with builder.mark(_name):
                fn(inverse)

    compute(False)
    ref._phase_by_bits(builder, y1)
    compute(True)
    for reg in (y1, flags, y0, breg, square, z, scratch):
        builder.free(reg)
    return builder.finish_circuit()


class QromLineProposer(Proposer):
    """QROM piecewise-linear seed proposer for ``invsqrt``."""

    generator_id = "invsqrt-qrom-line-v1"

    def __init__(self, configs: list[tuple[int, ...]] | None = None) -> None:
        self.configs = list(configs or QROM_LINE_CONFIGS)

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
            prec = QPrecision(
                index_bits=config[0],
                coef_bits=config[1],
                slope_bits=config[2],
                delta_bits=config[3],
                square_bits=config[4],
                product_bits=config[5],
                result_bits=config[6],
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
            prec = QPrecision(**candidate.parameters)
        except TypeError as exc:
            return CandidateScreen(
                valid=False,
                reason=f"invalid precision fields: {exc}",
                metrics={"error": str(exc)},
            )
        n_p = _contract_n_p(contract)
        valid, reason, metrics = evaluate_qrom_line_model(n_p, prec)
        return CandidateScreen(
            valid=valid,
            reason=reason,
            metrics=dict(metrics),
        )

    def compile(self, candidate: Candidate, contract: VerifierContract) -> CandidateArtifact:
        prec = QPrecision(**candidate.parameters)
        n_p = _contract_n_p(contract)
        _valid, _reason, screen_metrics = evaluate_qrom_line_model(n_p, prec)
        margin_units = int(screen_metrics.get("min_phase_margin_units", 0))
        ensure_runtime(contract)
        circuit = build_qrom_line_circuit(n_p, prec)
        artifact_dir = Path("artifacts/quantum_compilation/invsqrt_qrom_line")
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
            prec = QPrecision(**meta["precision"])
            _, _, screen_metrics = evaluate_qrom_line_model(n_p, prec)
            margin_units = int(screen_metrics.get("min_phase_margin_units", 0))

        t_per_toffoli = float(contract.constants.get("T_PER_TOFFOLI", 4.0))
        if rotations > 0 and margin_units > 0:
            eps = math.pi * float(margin_units) / (2.0**64) / rotations
            t_count = t_base + rotations * _rotation_t_count(eps)
        else:
            t_count = t_base
        return float(num_qubits * (toffoli_like + t_count / t_per_toffoli))
