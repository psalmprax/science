r"""Magic State Cultivation & Distillation Module for Fault-Tolerant Quantum Discovery.

Status: experimental

Scientific Scope & Algorithmic Foundation:
- Models magic state cultivation and distillation protocols (Bravyi-Kitaev 15-to-1,
  5-to-1, and parameterized continuous non-Clifford purification filters).
- Magic states $|T\rangle = \cos(\pi/8)|0\rangle + \sin(\pi/8)|1\rangle$ are the critical
  non-Clifford resource required to achieve universal fault-tolerant quantum computation
  via magic state injection into Clifford+T circuits.
- The module computes output infidelity, yield/acceptance probability, physical
  qubit overhead, and distillation circuit depth across multi-round schedules.
- Note on physical attribution: Google's Willow architecture demonstrated surface-code
  quantum memory and logical error reduction below threshold ($d=3, 5, 7$), not multi-round
  magic-state distillation. The continuous cultivation filter in this module is a stylized
  phenomenological benchmark model for continuous non-Clifford state purification.
- Features `MagicStateCultivationProposer` which extends `ProposerBase` to propose
  candidate distillation circuit schedules conforming to `AGENTS.md §1.1`.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from apedp_dyn.grammar.proposer_interface import (
    CandidateProposal,
    ProposerBase,
    StrategyArchetype,
)

# Bravyi-Kitaev 15-to-1 pseudo-threshold: 35 * eps^3 = eps => eps = 1/sqrt(35) ~= 0.169;
# accounting for higher-order terms yields empirical pseudo-threshold ~0.141.
BK_15_TO_1_PSEUDO_THRESHOLD: float = 0.141
DEFAULT_CULTIVATION_STRENGTH: float = 40.0
DEFAULT_BACKACTION_COEFF: float = 0.1


class DistillationProtocol(str, Enum):
    """Preregistered quantum distillation protocols."""
    BRAVYI_KITAEV_15_TO_1 = "bravyi_kitaev_15_to_1"
    FIVE_TO_ONE = "five_to_one"
    CONTINUOUS_CULTIVATION_FILTER = "continuous_cultivation_filter"
    # Backward compatibility alias
    WILLOW_CULTIVATION = "continuous_cultivation_filter"


@dataclass(frozen=True)
class DistillationRoundResult:
    """Quantitative performance of a single magic state distillation stage."""
    round_index: int
    input_infidelity: float
    output_infidelity: float
    yield_probability: float
    physical_qubits_required: int
    suppression_factor: float


@dataclass
class DistillationScheduleResult:
    """Full multi-round distillation cascade results."""
    protocol: DistillationProtocol
    rounds: list[DistillationRoundResult]
    initial_infidelity: float
    final_infidelity: float
    cumulative_yield: float
    total_physical_qubits: int
    meets_threshold: bool
    summary_formula: str


def compute_15_to_1_round(epsilon_in: float, code_distance: int = 3) -> DistillationRoundResult:
    r"""Compute output infidelity and acceptance probability for 15-to-1 Bravyi-Kitaev distillation.

    Output error scaling:
        \epsilon_{out} = 35 \epsilon_{in}^3 + O(\epsilon_{in}^4)
    Yield probability:
        P_{acc} \approx 1 - 35 \epsilon_{in}
    Physical qubits required:
        N_{qubits} \approx 15 \times d^2
    """
    if not (0.0 <= epsilon_in < 0.5):
        raise ValueError(f"Input infidelity must be in [0, 0.5), got {epsilon_in}")

    epsilon_out = float(35.0 * (epsilon_in ** 3.0))
    # Cap at epsilon_in for physical validity if epsilon_in is above pseudo-threshold
    if epsilon_in > BK_15_TO_1_PSEUDO_THRESHOLD:
        epsilon_out = epsilon_in

    yield_prob = max(0.0, float(1.0 - 35.0 * epsilon_in))
    suppression = epsilon_in / max(1e-18, epsilon_out)
    qubits = 15 * (code_distance ** 2)

    return DistillationRoundResult(
        round_index=1,
        input_infidelity=epsilon_in,
        output_infidelity=epsilon_out,
        yield_probability=yield_prob,
        physical_qubits_required=qubits,
        suppression_factor=suppression,
    )


def compute_cultivation_filter_round(
    epsilon_in: float,
    cultivation_filter_strength: float = DEFAULT_CULTIVATION_STRENGTH,
    code_distance: int = 5,
) -> DistillationRoundResult:
    r"""Model continuous magic state cultivation filter.

    Stylized phenomenological model of non-Clifford purification filters:
        \epsilon_{out} = \frac{\epsilon_{in}}{\kappa} + c \times \epsilon_{in}^2
    where \kappa represents the initial attenuation factor and c represents residual non-linear backaction.
    """
    if not (0.0 <= epsilon_in < 0.5):
        raise ValueError(f"Input infidelity must be in [0, 0.5), got {epsilon_in}")

    epsilon_out = float(epsilon_in / cultivation_filter_strength + DEFAULT_BACKACTION_COEFF * (epsilon_in ** 2.0))
    epsilon_out = min(epsilon_out, epsilon_in)
    yield_prob = max(0.01, float(1.0 - 5.0 * epsilon_in))
    suppression = epsilon_in / max(1e-18, epsilon_out)
    qubits = 17 * (code_distance ** 2)

    return DistillationRoundResult(
        round_index=1,
        input_infidelity=epsilon_in,
        output_infidelity=epsilon_out,
        yield_probability=yield_prob,
        physical_qubits_required=qubits,
        suppression_factor=suppression,
    )


# Backward-compatible alias
compute_willow_cultivation_round = compute_cultivation_filter_round


def simulate_distillation_cascade(
    initial_infidelity: float,
    target_infidelity: float = 1e-6,
    protocol: DistillationProtocol = DistillationProtocol.WILLOW_CULTIVATION,
    max_rounds: int = 5,
    base_code_distance: int = 5,
) -> DistillationScheduleResult:
    """Simulate multi-round cascade until target infidelity is achieved or max rounds reached."""
    rounds: list[DistillationRoundResult] = []
    current_eps = initial_infidelity
    cum_yield = 1.0
    total_qubits = 0

    for r_idx in range(1, max_rounds + 1):
        d = base_code_distance + (r_idx - 1) * 2  # Growing code distance across rounds
        if protocol == DistillationProtocol.BRAVYI_KITAEV_15_TO_1:
            res = compute_15_to_1_round(current_eps, code_distance=d)
        elif protocol == DistillationProtocol.WILLOW_CULTIVATION:
            res = compute_willow_cultivation_round(current_eps, cultivation_filter_strength=40.0, code_distance=d)
        else:  # Five-to-one
            eps_out = float(min(current_eps, 5.0 * (current_eps ** 2.0)))
            y_p = max(0.0, float(1.0 - 5.0 * current_eps))
            res = DistillationRoundResult(
                round_index=r_idx,
                input_infidelity=current_eps,
                output_infidelity=eps_out,
                yield_probability=y_p,
                physical_qubits_required=5 * (d ** 2),
                suppression_factor=current_eps / max(1e-18, eps_out),
            )

        # Update round index in result
        round_res = DistillationRoundResult(
            round_index=r_idx,
            input_infidelity=res.input_infidelity,
            output_infidelity=res.output_infidelity,
            yield_probability=res.yield_probability,
            physical_qubits_required=res.physical_qubits_required,
            suppression_factor=res.suppression_factor,
        )
        rounds.append(round_res)
        cum_yield *= round_res.yield_probability
        total_qubits = max(total_qubits, round_res.physical_qubits_required)
        current_eps = round_res.output_infidelity

        if current_eps <= target_infidelity:
            break

    meets_target = current_eps <= target_infidelity
    formula = (
        f"MagicStateSchedule({protocol.value}, rounds={len(rounds)}, "
        f"eps_in={initial_infidelity:.2e} -> eps_out={current_eps:.2e}, "
        f"yield={cum_yield:.3f}, footprint={total_qubits} qubits)"
    )

    return DistillationScheduleResult(
        protocol=protocol,
        rounds=rounds,
        initial_infidelity=initial_infidelity,
        final_infidelity=current_eps,
        cumulative_yield=cum_yield,
        total_physical_qubits=total_qubits,
        meets_threshold=meets_target,
        summary_formula=formula,
    )


class MagicStateCultivationProposer(ProposerBase):
    """Generates candidate quantum magic state cultivation & distillation schedules.

    Conforms to ProposerBase and emits CandidateProposal objects evaluated against
    fault-tolerant target fidelity (F >= 0.9999 or eps <= 1e-4) and resource footprints.
    """

    def __init__(
        self,
        target_infidelity: float = 1e-4,
        max_physical_qubits: int = 2000,
        random_seed: int = 42,
    ):
        self.target_infidelity = target_infidelity
        self.max_physical_qubits = max_physical_qubits
        self.random_seed = random_seed

    @property
    def proposer_id(self) -> str:
        return "magic_state_cultivation_proposer_v1"

    def propose_schedules(
        self,
        input_infidelities: Sequence[float] = (0.01, 0.005, 0.001),
        candidate_prefix: str = "H-QMAGIC",
    ) -> list[CandidateProposal]:
        """Propose candidate cultivation schedules across input noise regimes."""
        proposals: list[CandidateProposal] = []
        protocols = [
            DistillationProtocol.WILLOW_CULTIVATION,
            DistillationProtocol.BRAVYI_KITAEV_15_TO_1,
            DistillationProtocol.FIVE_TO_ONE,
        ]

        idx = 1
        for eps_in in input_infidelities:
            for proto in protocols:
                sched = simulate_distillation_cascade(
                    initial_infidelity=eps_in,
                    target_infidelity=self.target_infidelity,
                    protocol=proto,
                    base_code_distance=5,
                )

                # Score hypothesis: high fidelity, high yield, low qubit footprint
                fidelity_score = 1.0 - min(1.0, sched.final_infidelity / self.target_infidelity)
                qubit_penalty = min(1.0, sched.total_physical_qubits / self.max_physical_qubits)
                heuristic_score = float(0.5 * fidelity_score + 0.3 * sched.cumulative_yield + 0.2 * (1.0 - qubit_penalty))

                cand_id = f"{candidate_prefix}-{proto.name[:4]}-EPS{int(eps_in*1000):03d}-{idx:02d}"
                proposal = CandidateProposal(
                    proposer_id=self.proposer_id,
                    candidate_id=cand_id,
                    formula=sched.summary_formula,
                    strategy_archetype=StrategyArchetype.CONTINUOUS_FACTOR,
                    declared_scale_invariant=True,
                    declared_nominal_dependencies=("epsilon_in", "rounds", "qubits"),
                    parameters={"heuristic_score": heuristic_score, "final_infidelity": sched.final_infidelity},
                    metadata={
                        "protocol": proto.value,
                        "initial_infidelity": eps_in,
                        "final_infidelity": sched.final_infidelity,
                        "cumulative_yield": sched.cumulative_yield,
                        "total_physical_qubits": sched.total_physical_qubits,
                        "rounds_count": len(sched.rounds),
                        "meets_threshold": sched.meets_threshold,
                    },
                )
                proposals.append(proposal)
                idx += 1

        return proposals

    def fit_and_propose(self, *args: Any, **kwargs: Any) -> list[CandidateProposal]:
        """Generate candidate proposals conforming to ProposerBase."""
        infidelities = kwargs.get("input_infidelities", (0.01, 0.005, 0.001))
        prefix = kwargs.get("candidate_prefix", "H-QMAGIC")
        return self.propose_schedules(input_infidelities=infidelities, candidate_prefix=prefix)

