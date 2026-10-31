"""APEDP-QED: quantum-assisted equation discovery, as a proposer family.

Status: experimental

The question this package exists to answer is narrow and falsifiable: given
noisy observations of a system and a library of candidate terms, can a
quantum-assisted search recover the governing equation as well as the classical
proposers, at the same budget? It is *not* "can a quantum computer do physics",
and nothing here is allowed to slide from the first question to the second.

What exists, and what each piece is for:

:mod:`~apedp_dyn.qed.library`
    The term library: fixed, named scalar functions of the observed state. It is
    the whole hypothesis space, so a term absent here can never be proposed.
:mod:`~apedp_dyn.qed.objective`
    ``J(z) = mean residual**2 + lambda * |z|`` with the coefficients re-fitted by
    least squares for every selection, and a hard budget on distinct fits. The
    budget is the resource that makes six proposers comparable.
:mod:`~apedp_dyn.qed.tasks`
    A task: development observations, the holdout trajectory, the library, and
    the generating equation. Proposers see an objective over development rows
    only; the holdout and the answer stay with the referee.
:mod:`~apedp_dyn.qed.arms`
    Six search mechanisms over the selection vector -- hill climbing, simulated
    annealing, a genetic algorithm, genetic programming over a set-building
    language, RIME operators on a relaxed selection vector, and QAOA over an
    approximate quadratic surrogate.
:mod:`~apedp_dyn.qed.qaoa`
    The circuit: QUBO construction from exact singleton and pair fits, Ising-free
    diagonal phase separation, a mixer, angle optimisation by pattern search, and
    measurement. Classical simulation only.
:mod:`~apedp_dyn.qed.referee`
    Adjudication against the holdout: structural recovery, forward rollout,
    bootstrap intervals, parameter stability, and ablation separability. A
    different equation that predicts the holdout equally well is ``ALTERNATIVE``
    -- a non-identifiability result, not a win and not a failure.
:mod:`~apedp_dyn.qed.records`
    The experiment record: task, exhaustive fairness sweep, per-arm rows, an
    advantage block computed from those rows, and the limits in writing.
:mod:`~apedp_dyn.qed.proposer`
    The standard ``CandidateProposal`` adapter, so QED competes inside APEDP-X
    and never adjudicates itself.

Boundaries, stated here because this package is easy to over-read. The QAOA arm
optimises a quadratic surrogate of a non-quadratic objective, on a dense
statevector simulator, and is scored by the real objective afterwards -- so it
is a search heuristic, not an implementation of a published quantum advantage.
The experiment is a synthetic system with a ten-term library and a known answer;
the fact that the penalised optimum equals the truth is a property of the
benchmark, verified by an exhaustive sweep, and says nothing about how easy real
discovery is. Stochastic derivative estimation, multiple systems, multiple
libraries, hardware execution and any claim of computational advantage are all
absent and named as absent in ``docs/QED_QUANTUM_EQUATION_DISCOVERY.md``.
"""
from __future__ import annotations

from apedp_dyn.qed.arms import (
    ABSTAINED,
    ARMS,
    SEARCHED,
    ArmAbstained,
    ArmResult,
    boolean_gp,
    genetic_algorithm,
    qaoa_surrogate,
    random_restart,
    rime_relaxed,
    run_arm,
    simulated_annealing,
)
from apedp_dyn.qed.library import (
    OSCILLATOR_TERMS,
    LibraryRefusal,
    Term,
    TermLibrary,
    oscillator_library,
)
from apedp_dyn.qed.objective import BudgetExhausted, Fit, SupportObjective
from apedp_dyn.qed.proposer import QuantumEquationProposer
from apedp_dyn.qed.qaoa import (
    QaoaRun,
    Qubo,
    QuboRefusal,
    build_quadratic_surrogate,
    mask_to_support,
    qaoa_state,
    run_qaoa,
    support_to_mask,
)
from apedp_dyn.qed.records import (
    LIMITATIONS,
    RECORD,
    SCHEMA,
    build_record,
    oracle_sweep,
    write_record,
)
from apedp_dyn.qed.referee import (
    ABSTAIN,
    ALTERNATIVE,
    HOLDOUT_R2_FLOOR,
    INVALID,
    STABILITY_TOLERANCE,
    VALID,
    Adjudication,
    Check,
    ablation,
    adjudicate,
    bootstrap_interval,
    parameter_stability,
    reference_scores,
    rollout_scores,
)
from apedp_dyn.qed.tasks import (
    DiscoveryTask,
    RolloutRefusal,
    TaskRefusal,
    damped_oscillator_task,
    simulate_oscillator,
)

__all__ = [
    "ABSTAIN",
    "ABSTAINED",
    "ALTERNATIVE",
    "ARMS",
    "HOLDOUT_R2_FLOOR",
    "INVALID",
    "LIMITATIONS",
    "OSCILLATOR_TERMS",
    "RECORD",
    "SCHEMA",
    "SEARCHED",
    "STABILITY_TOLERANCE",
    "VALID",
    "Adjudication",
    "ArmAbstained",
    "ArmResult",
    "BudgetExhausted",
    "Check",
    "DiscoveryTask",
    "Fit",
    "LibraryRefusal",
    "QaoaRun",
    "QuantumEquationProposer",
    "Qubo",
    "QuboRefusal",
    "RolloutRefusal",
    "SupportObjective",
    "TaskRefusal",
    "Term",
    "TermLibrary",
    "ablation",
    "adjudicate",
    "boolean_gp",
    "bootstrap_interval",
    "build_quadratic_surrogate",
    "build_record",
    "damped_oscillator_task",
    "genetic_algorithm",
    "mask_to_support",
    "oracle_sweep",
    "oscillator_library",
    "parameter_stability",
    "qaoa_state",
    "qaoa_surrogate",
    "random_restart",
    "reference_scores",
    "rime_relaxed",
    "rollout_scores",
    "run_arm",
    "run_qaoa",
    "simulate_oscillator",
    "simulated_annealing",
    "support_to_mask",
    "write_record",
]
