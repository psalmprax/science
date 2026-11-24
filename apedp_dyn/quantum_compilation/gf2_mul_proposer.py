"""The GF(2^12) multiplication tensor as a proposer, not a claim.

Status: experimental

:mod:`apedp_dyn.quantum.field_tensor_search` can build the exact 12x12x12
multiplication tensor, verify a bilinear decomposition against all 144
coefficients over GF(2), and price a compiled multiplier. What it had no way to
do was hand a candidate to the quantum track: the seeded decompositions and the
published 51-Toffoli incumbent lived in a benchmark script, and "fewer than 51
Toffolis" was a sentence in a document. This module is the bridge. Each
decomposition becomes a :class:`CircuitCandidate` whose proposer-side screen is
recorded in :class:`~apedp_dyn.quantum_compilation.ledger.QuantumCandidateLedger`.

Two limits are structural, not incidental:

* **The screen is exact, and it is still a screen.** Every one of the 144 tensor
  coefficients is reconstructed from ``U``, ``V`` and ``W`` over GF(2), so a
  refused candidate really does fail to multiply. That says nothing about a
  submitted circuit, though, so no candidate here can reach a
  correctness-claiming status: the official verifier is the only judge, and a
  bare multiplier has no verifier contract to submit to. Every candidate stays
  ``PROPOSED`` or ``SCREEN_REJECTED``, and its receipt slot stays empty until
  someone builds the full decoder submission it would belong to.
* **A rank is not a Toffoli count.** ``compile_reversible_multiplier`` prices
  ``rank + 12`` on the field engine's own model for a controlled call, which is
  not the DQI decoder's measured per-multiply cost. A candidate carries the
  model's number and says so; the target comparison is made on that number and
  nowhere else.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from apedp_dyn.quantum.field_tensor_search import (
    FieldSpec,
    TensorDecomposition,
    build_54_reference,
    build_field_spec,
    build_karatsuba_reference,
    build_multiplication_tensor,
    build_schoolbook_reference,
    compile_reversible_multiplier,
    evaluate_multiplier,
    load_published_51_reference,
    search_flip_graph,
    verify_tensor_decomposition,
)
from apedp_dyn.quantum_compilation.candidates import (
    CircuitCandidate,
    candidate_id_for,
    sha256_file,
)
from apedp_dyn.quantum_compilation.ledger import QuantumCandidateLedger

__all__ = [
    "BENCHMARK",
    "PUBLISHED_INCUMBENT_TOFFOLI",
    "TARGET_TOFFOLI",
    "Gf2MultiplicationProposer",
    "Gf2Screen",
    "screen_report",
]

#: The pinned field's multiply is the object, not a competition submission, so
#: this is deliberately not one of ``ledger.BENCHMARK_ORDER``.
BENCHMARK = "gf2mul"

#: The published October 2026 hand-tuned incumbent (specification §4.1). It is an
#: implementation result, not a proof of minimal tensor rank.
PUBLISHED_INCUMBENT_TOFFOLI = 51

#: The target is *strictly fewer* Toffolis than the incumbent.
TARGET_TOFFOLI = PUBLISHED_INCUMBENT_TOFFOLI

_GENERATOR_ID = "gf2-mul-tensor-v1"
_COEFFICIENTS = 12 * 12 * 12


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in name).strip("-").lower()


def _digest(*arrays: np.ndarray) -> str:
    payload = b"".join(np.ascontiguousarray(a, dtype=np.uint8).tobytes() for a in arrays)
    return hashlib.sha256(payload).hexdigest()


@dataclasses.dataclass(frozen=True, slots=True)
class Gf2Screen:
    """The proposer-side verdict on one decomposition. Never a promotion."""

    name: str
    valid: bool
    reason: str | None
    rank: int
    toffoli: int
    cnot: int
    ancilla: int
    depth: int
    meets_target: bool
    tensor_mismatches: int
    resources: dict[str, Any]

    def record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "valid": self.valid,
            "reason": self.reason,
            "rank": self.rank,
            "toffoli": self.toffoli,
            "cnot": self.cnot,
            "ancilla": self.ancilla,
            "depth": self.depth,
            "meets_target": self.meets_target,
            "target_toffoli_exclusive": TARGET_TOFFOLI,
            "tensor_mismatches": self.tensor_mismatches,
            "coefficients_checked": _COEFFICIENTS,
            "resources": dict(self.resources),
            "screen_only": True,
        }


class Gf2MultiplicationProposer:
    """Emits GF(2^12) multiplier candidates from the exact tensor engine.

    ``spec_sha256`` plays the role ``contract_sha256`` plays for a competition
    candidate: it pins the field, the tensor and the target, so a candidate id
    changes when any of those does, and two decompositions cannot inherit each
    other's status across different pinning.
    """

    def __init__(
        self,
        *,
        tensor: np.ndarray | None = None,
        field_spec: FieldSpec | None = None,
        flip_graph_steps: int = 40,
        flip_graph_seeds: tuple[int, ...] = (0, 1, 2),
    ) -> None:
        if flip_graph_steps < 1:
            raise ValueError(f"flip_graph_steps must be positive, got {flip_graph_steps}")
        self.field_spec = field_spec if field_spec is not None else build_field_spec()
        self.tensor = build_multiplication_tensor() if tensor is None else tensor
        if self.tensor.shape != (12, 12, 12):
            raise ValueError(f"the multiplication tensor must be 12x12x12, got {self.tensor.shape}")
        self.flip_graph_steps = int(flip_graph_steps)
        self.flip_graph_seeds = tuple(int(s) for s in flip_graph_seeds)
        payload = json.dumps(
            {
                "benchmark": BENCHMARK,
                "field": dataclasses.asdict(self.field_spec),
                "tensor_sha256": hashlib.sha256(self.tensor.tobytes()).hexdigest(),
                "incumbent_toffoli": PUBLISHED_INCUMBENT_TOFFOLI,
                "target_toffoli_exclusive": TARGET_TOFFOLI,
                "toffoli_model": "rank + 12 (controlled) on the field engine's own model",
            },
            sort_keys=True,
        ).encode("utf-8")
        self.spec_sha256 = hashlib.sha256(payload).hexdigest()

    def seeds(self) -> list[TensorDecomposition]:
        """The pinned decomposition hierarchy, plus flip-graph variants of the best one.

        The published incumbent is included on purpose: it arrives with all-zero
        ``U``, ``V`` and ``W``, and screening it is how the gap between "a
        published count" and "a decomposition this checkout has" stays visible.
        """
        decompositions = [
            build_schoolbook_reference(),
            build_karatsuba_reference(),
            build_54_reference(),
            load_published_51_reference(),
        ]
        composed = decompositions[2]
        for seed in self.flip_graph_seeds:
            variant = search_flip_graph(composed, max_steps=self.flip_graph_steps, seed=seed)
            if variant is not None:
                decompositions.append(variant)
        return decompositions

    def screen(self, decomp: TensorDecomposition) -> Gf2Screen:
        """Reconstruct all 144 coefficients and price the compiled multiplier."""
        exact, mismatches = verify_tensor_decomposition(self.tensor, decomp)
        circuit = compile_reversible_multiplier(decomp)
        resources = evaluate_multiplier(circuit)
        reason = None
        if not exact:
            reason = (
                f"decomposition_not_exact: {len(mismatches)} of {_COEFFICIENTS} tensor "
                "coefficients disagree with the reconstructed U*V*W over GF(2)"
            )
        return Gf2Screen(
            name=decomp.name,
            valid=exact,
            reason=reason,
            rank=int(decomp.rank),
            toffoli=int(circuit.toffoli_count),
            cnot=int(circuit.cnot_count),
            ancilla=int(circuit.ancilla_count),
            depth=int(circuit.circuit_depth),
            meets_target=bool(exact and circuit.toffoli_count < TARGET_TOFFOLI),
            tensor_mismatches=len(mismatches),
            resources=dict(resources),
        )

    def parameters(self, decomp: TensorDecomposition) -> dict[str, Any]:
        """The identity of a decomposition, small enough to live in a candidate id."""
        return {
            "name": decomp.name,
            "rank": int(decomp.rank),
            "category": str(decomp.metadata.get("category", "")),
            "decomposition_sha256": _digest(decomp.U, decomp.V, decomp.W),
        }

    def write_artifact(self, decomp: TensorDecomposition, directory: Path | str) -> Path:
        """Write the decomposition itself, so the screen can be re-run from disk."""
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        path = target / f"{_slug(decomp.name)}.decomposition.json"
        payload = {
            "benchmark": BENCHMARK,
            "field": dataclasses.asdict(self.field_spec),
            "tensor_sha256": hashlib.sha256(self.tensor.tobytes()).hexdigest(),
            "spec_sha256": self.spec_sha256,
            "name": decomp.name,
            "rank": int(decomp.rank),
            "metadata": dict(decomp.metadata),
            "U": np.asarray(decomp.U, dtype=int).tolist(),
            "V": np.asarray(decomp.V, dtype=int).tolist(),
            "W": np.asarray(decomp.W, dtype=int).tolist(),
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def candidate(
        self,
        decomp: TensorDecomposition,
        directory: Path | str,
        *,
        screen: Gf2Screen | None = None,
    ) -> CircuitCandidate:
        """One ``CircuitCandidate`` in ``PROPOSED``; nothing is verified here."""
        verdict = self.screen(decomp) if screen is None else screen
        path = self.write_artifact(decomp, directory)
        parameters = self.parameters(decomp)
        return CircuitCandidate(
            candidate_id=candidate_id_for(
                BENCHMARK, _GENERATOR_ID, parameters, contract_sha256=self.spec_sha256
            ),
            benchmark=BENCHMARK,
            generator_id=_GENERATOR_ID,
            parameters=parameters,
            artifact_path=str(path),
            artifact_sha256=sha256_file(path),
            contract_sha256=self.spec_sha256,
            screen_metrics=verdict.record(),
            screen_score_estimate=float(verdict.toffoli),
        )

    def propose(
        self,
        directory: Path | str,
        *,
        ledger: QuantumCandidateLedger | None = None,
        decompositions: list[TensorDecomposition] | None = None,
    ) -> list[CircuitCandidate]:
        """Screen the candidate set and, if given a ledger, record the screen.

        The ledger call is ``record_screen`` and nothing else: a screen can only
        propose or reject. An exact decomposition recorded as ``PROPOSED`` still
        has no correctness claim, because only a verifier receipt can make one.
        """
        candidates: list[CircuitCandidate] = []
        for decomp in self.seeds() if decompositions is None else decompositions:
            screen = self.screen(decomp)
            candidate = self.candidate(decomp, directory, screen=screen)
            if ledger is not None:
                ledger.register(candidate)
                candidate = ledger.record_screen(
                    candidate,
                    passed=screen.valid,
                    metrics=screen.record(),
                    score_estimate=float(screen.toffoli),
                    rejection_reason=screen.reason,
                )
            candidates.append(candidate)
        return candidates


def screen_report(candidates: list[CircuitCandidate]) -> dict[str, Any]:
    """A read-out that cannot be mistaken for a leaderboard result."""
    rows = [
        {
            "name": candidate.parameters.get("name"),
            "rank": candidate.screen_metrics.get("rank"),
            "valid": candidate.screen_metrics.get("valid"),
            "toffoli": candidate.screen_metrics.get("toffoli"),
            "meets_target": candidate.screen_metrics.get("meets_target"),
            "reason": candidate.screen_metrics.get("reason"),
        }
        for candidate in candidates
    ]
    exact = [c for c in candidates if c.screen_metrics.get("valid")]
    best = min(exact, key=lambda c: c.screen_metrics["toffoli"]) if exact else None
    return {
        "benchmark": BENCHMARK,
        "published_incumbent_toffoli": PUBLISHED_INCUMBENT_TOFFOLI,
        "target_toffoli_exclusive": TARGET_TOFFOLI,
        "candidates": rows,
        "exact_candidates": len(exact),
        "best_exact_name": None if best is None else str(best.parameters.get("name")),
        "best_exact_toffoli": None if best is None else int(best.screen_metrics["toffoli"]),
        "target_met": None if best is None else bool(best.screen_metrics["meets_target"]),
        "note": (
            "Proposer-side screen only: every coefficient of the multiplication tensor is "
            "reconstructed over GF(2) and the multiplier is priced on the field engine's own "
            "controlled model. No official verifier has priced a submitted circuit."
        ),
    }
