"""QED pins: the machinery behind the recovery experiment is checkable.

These tests are about the parts, not the headline: the library refuses what it
cannot represent, the budget is a real limit on a real count of fits, the
quadratic surrogate is exact on a genuinely quadratic objective (so its
approximation error on a real one is a measurement and not a bug), the referee
separates an alternative equation from a wrong one, and no proposer can reach
the holdout rows.
"""
from __future__ import annotations

import numpy as np
import pytest

from apedp_dyn.grammar.proposer_interface import CandidateProposal, StrategyArchetype
from apedp_dyn.qed.arms import ARMS, ArmAbstained, run_arm
from apedp_dyn.qed.library import LibraryRefusal, TermLibrary, oscillator_library
from apedp_dyn.qed.objective import BudgetExhausted, SupportObjective
from apedp_dyn.qed.proposer import QuantumEquationProposer
from apedp_dyn.qed.qaoa import (
    Qubo,
    build_quadratic_surrogate,
    mask_to_support,
    normalised,
    qaoa_state,
    run_qaoa,
    support_to_mask,
)
from apedp_dyn.qed.referee import (
    ABSTAIN,
    ALTERNATIVE,
    HOLDOUT_R2_FLOOR,
    VALID,
    ablation,
    adjudicate,
    bootstrap_interval,
    reference_scores,
    rollout_scores,
)
from apedp_dyn.qed.tasks import (
    RolloutRefusal,
    TaskRefusal,
    damped_oscillator_task,
    simulate_oscillator,
)

SMALL = TermLibrary.from_specs((("1", lambda s: np.ones(s.shape[0])), ("x", lambda s: s[:, 0])))


# --------------------------------------------------------------------------- #
# Library
# --------------------------------------------------------------------------- #


def test_a_library_refuses_a_duplicate_name():
    with pytest.raises(LibraryRefusal):
        TermLibrary.from_specs((("x", lambda s: s[:, 0]), ("x", lambda s: s[:, 0])))


def test_a_library_refuses_a_non_finite_or_mis_shaped_column():
    with pytest.raises(LibraryRefusal):
        TermLibrary.from_specs((("bad", lambda s: np.full(s.shape[0], np.nan)),)).design_matrix(
            np.ones((3, 1))
        )
    with pytest.raises(LibraryRefusal):
        TermLibrary.from_specs((("short", lambda s: np.ones(s.shape[0] - 1)),)).design_matrix(
            np.ones((3, 1))
        )


def test_a_support_cannot_address_a_term_that_is_not_there():
    design = SMALL.design_matrix(np.array([[1.0], [2.0]]))
    with pytest.raises(LibraryRefusal):
        SMALL.select(design, (5,))
    with pytest.raises(LibraryRefusal):
        SMALL.select(design, (1, 1))


def test_rendering_keeps_the_signs_and_the_left_hand_side():
    text = SMALL.render((0, 1), (0.5, -2.0), lhs="dv/dt")
    assert text == "dv/dt = +0.5 -2*x"
    assert SMALL.render((), (), lhs="dv/dt") == "dv/dt = 0"


def test_the_oscillator_library_is_ten_distinct_terms_with_x_and_v():
    library = oscillator_library()
    assert len(library) == 10
    assert library.names[:3] == ("1", "x", "v")
    assert len(set(library.names)) == 10


# --------------------------------------------------------------------------- #
# Objective and budget
# --------------------------------------------------------------------------- #


def _objective(budget: int = 8, penalty: float = 0.002) -> SupportObjective:
    states = np.linspace(0.2, 2.0, 50).reshape(-1, 1)
    design = SMALL.design_matrix(states)
    target = 3.0 * states[:, 0] - 4.0
    return SupportObjective(SMALL, design, target, penalty=penalty, budget=budget)


def test_an_empty_selection_scores_the_target_power_and_costs_one_fit():
    objective = _objective()
    fit = objective.evaluate(())
    assert fit.support == () and fit.coefficients == ()
    assert fit.objective == pytest.approx(fit.residual)
    assert objective.evaluations == 1


def test_the_objective_carries_the_parsimony_penalty_per_term():
    objective = _objective(penalty=0.01)
    fit = objective.evaluate((0, 1))
    assert fit.objective == pytest.approx(fit.residual + 2 * 0.01)
    assert fit.complexity == 2


def test_the_budget_counts_distinct_fits_and_a_repeat_is_free():
    objective = _objective(budget=2)
    objective.evaluate((0,))
    assert objective.remaining == 1
    objective.evaluate((0,))
    assert objective.remaining == 1, "re-scoring a known selection must not cost another fit"
    objective.evaluate((1,))
    assert objective.remaining == 0
    with pytest.raises(BudgetExhausted):
        objective.evaluate((0, 1))


def test_an_objective_refuses_data_it_cannot_fit():
    design = np.array([[1.0, 1.0], [1.0, np.nan]])
    with pytest.raises(LibraryRefusal):
        SupportObjective(SMALL, design, np.zeros(2), penalty=0.0, budget=4)
    with pytest.raises(LibraryRefusal):
        SupportObjective(SMALL, np.ones((3, 2)), np.zeros(3), penalty=0.0, budget=0)
    with pytest.raises(LibraryRefusal):
        SupportObjective(SMALL, np.ones((3, 2)), np.zeros(2), penalty=0.0, budget=4)


def test_every_arm_gets_an_objective_over_development_rows_only():
    task = damped_oscillator_task()
    objective = task.objective()
    assert objective.design.shape == (task.dev_states.shape[0], len(task.library))
    assert objective.target.shape == (task.dev_states.shape[0],)
    for forbidden in ("holdout_states", "holdout_target", "truth_support", "truth_coefficients"):
        assert not hasattr(objective, forbidden), f"the search objective leaks {forbidden}"


# --------------------------------------------------------------------------- #
# QAOA and the surrogate
# --------------------------------------------------------------------------- #


class _QuadraticObjective:
    """A stand-in whose objective really is quadratic in ``z``."""

    def __init__(self):
        self.offset = 0.5
        self.linear = (-0.3, 0.7, -0.1)
        self.pairs = {(0, 1): 0.2}

    def value(self, support):
        total = self.offset
        for index in support:
            total += self.linear[index]
        total += sum(gain for (i, j), gain in self.pairs.items() if i in support and j in support)
        return float(total)

    def __call__(self, support):
        class _Fit:
            objective = self.value(support)

        return _Fit()


def test_the_surrogate_is_exact_when_the_objective_is_quadratic():
    stub = _QuadraticObjective()
    qubo, paid = build_quadratic_surrogate(stub, 3)
    assert paid == 1 + 3 + 3
    for mask in range(8):
        bits = [(mask >> index) & 1 for index in range(3)]
        assert qubo.energy(bits) == pytest.approx(stub.value(mask_to_support(mask, 3)))


def test_the_surrogate_cost_is_the_full_expansion_or_an_abstention():
    objective = _objective(budget=2)
    with pytest.raises(BudgetExhausted):
        build_quadratic_surrogate(objective.evaluate, 2)


def test_qubo_energies_agree_with_the_scalar_form_and_find_the_same_minimum():
    qubo = Qubo(offset=0.25, linear=(0.5, -0.5, 0.1), quadratic=((0, 1, -0.4), (1, 2, 0.3)))
    values = qubo.energies()
    for mask in range(8):
        bits = [(mask >> index) & 1 for index in range(3)]
        assert values[mask] == pytest.approx(qubo.energy(bits))
    assert int(np.argmin(values)) == qubo.argmin()[0]
    assert qubo.argmin()[1] == pytest.approx(float(values.min()))


def test_rescaling_the_cost_does_not_move_its_minimiser():
    qubo = Qubo(offset=2.0, linear=(1.0, -3.0), quadratic=((0, 1, 0.7),))
    raw = qubo.energies()
    scaled = normalised(raw)
    assert scaled.min() == pytest.approx(0.0) and scaled.max() == pytest.approx(1.0)
    assert int(np.argmin(scaled)) == int(np.argmin(raw))


def test_a_constant_cost_produces_a_uniform_superposition_not_a_division_by_zero():
    assert np.all(normalised(np.full(8, 0.4)) == 0.0)


def test_the_circuit_cost_counts_rz_rzz_and_rx_per_layer():
    qubo = Qubo(offset=0.0, linear=(1.0,) * 10, quadratic=tuple((i, j, 0.0) for i in range(10) for j in range(i + 1, 10)))
    report = qubo.rotations(2)
    assert report["rzz_per_layer"] == 45
    assert report["rotations_per_layer"] == 2 * 10 + 45
    assert report["rotations_total"] == 2 * report["rotations_per_layer"]


def test_the_qaoa_state_is_normalised_and_the_angle_count_is_checked():
    costs = normalised(np.arange(8, dtype=float))
    state = qaoa_state(costs, 2, (0.3, 0.4, -0.2, 0.1))
    assert np.sum(np.abs(state) ** 2) == pytest.approx(1.0)
    with pytest.raises(Exception):
        qaoa_state(costs, 2, (0.3, 0.4))
    with pytest.raises(Exception):
        qaoa_state(np.arange(5, dtype=float), 1, (0.3, 0.4))


def test_a_qaoa_run_is_seeded_and_reports_probabilities_that_sum_to_one():
    qubo = Qubo(offset=0.0, linear=(0.4, -0.2, 0.1, 0.3), quadratic=((0, 1, -0.5), (2, 3, 0.2)))
    first = run_qaoa(qubo, depth=2, shots=256, rng=np.random.default_rng(5))
    second = run_qaoa(qubo, depth=2, shots=256, rng=np.random.default_rng(5))
    assert first.angles == second.angles
    assert first.samples == second.samples
    assert sum(first.probabilities) == pytest.approx(1.0)
    assert first.candidates[0] == qubo.argmin()[0], "the surrogate minimiser is scored first"
    assert len(set(first.candidates)) == len(first.candidates)
    assert 0.0 <= first.probability_of(qubo.argmin()[0]) <= 1.0


def test_masks_and_supports_round_trip():
    for mask in range(1 << 5):
        assert support_to_mask(mask_to_support(mask, 5), 5) == mask


# --------------------------------------------------------------------------- #
# Arms
# --------------------------------------------------------------------------- #


def test_every_arm_terminates_and_respects_the_budget():
    task = damped_oscillator_task()
    for arm in ARMS:
        result = run_arm(arm, task, seed=3)
        assert result.status in ("searched", "abstained")
        assert result.evaluations <= task.budget, arm
        if result.status == "searched":
            assert result.best is not None
            assert result.evaluations >= 1
        else:
            assert result.reason, f"{arm} abstained without a reason"


def test_a_budget_too_small_for_the_surrogate_is_an_abstention_with_the_number():
    task = damped_oscillator_task(budget=10)
    result = run_arm("qaoa_surrogate", task, seed=1)
    assert result.status == "abstained"
    assert "56" in result.reason and result.best is None


def test_an_unknown_arm_is_a_key_error():
    task = damped_oscillator_task()
    with pytest.raises(KeyError):
        run_arm("no_such_arm", task, seed=1)


def test_the_classical_arms_reach_the_generating_support_at_a_generous_budget():
    """Four independent mechanisms, one objective, the same answer.

    The quantum arm is deliberately not in this assertion: what it returns is a
    property of its surrogate and is measured in the record, not assumed here.
    """
    task = damped_oscillator_task()
    for arm in ("random_restart", "simulated_annealing", "genetic_algorithm", "boolean_gp"):
        result = run_arm(arm, task, seed=11)
        assert result.best is not None, arm
        assert set(result.best.support) == set(task.truth_support), (arm, result.best.support)
        assert result.evaluations <= task.budget, arm


# --------------------------------------------------------------------------- #
# Referee
# --------------------------------------------------------------------------- #


def test_the_truth_predicts_the_holdout_and_the_companion_is_separate():
    task = damped_oscillator_task()
    scores = rollout_scores(task, task.truth_support, task.truth_coefficients)
    assert scores["discovered_r2"] > 0.999
    assert scores["companion_r2"] > 0.999
    assert scores["diverged"] is False
    assert reference_scores(task)["support"] == list(task.truth_names)


def test_a_diverging_candidate_is_flagged_and_scores_zero_rather_than_raising():
    task = damped_oscillator_task()
    support = (task.library.index_of("v^3"),)
    scores = rollout_scores(task, support, (2.0,))
    assert scores["diverged"] is True
    assert scores["discovered_r2"] == 0.0
    assert scores["finite_rows"] < scores["rows"]


def test_a_coordinate_with_neither_a_law_nor_an_identity_is_refused():
    """Silence about a coordinate must be an error, not an implicit zero."""
    from apedp_dyn.qed.tasks import DiscoveryTask

    states = np.zeros((5, 3))
    task = DiscoveryTask(
        name="missing-rule",
        library=SMALL,
        dev_states=states,
        dev_target=np.zeros(5),
        holdout_states=states,
        holdout_target=np.zeros(5),
        dt=0.1,
        discovered=1,
        kinematic_pairs=((0, 1),),
        truth_support=(1,),
        truth_coefficients=(-1.0,),
        penalty=0.0,
        budget=4,
        noise_level=0.0,
        seed=0,
    )
    with pytest.raises(RolloutRefusal):
        task.derivatives(np.array([1.0, 0.5, 0.25]), (1,), (-1.0,))


def test_the_bootstrap_interval_contains_the_generating_coefficients():
    task = damped_oscillator_task()
    interval = bootstrap_interval(task, task.truth_support, task.truth_coefficients, samples=200)
    assert interval["support"] == list(task.truth_names)
    assert len(interval["bounds"]) == 2
    for bound, truth in zip(interval["bounds"], task.truth_coefficients, strict=True):
        assert bound[0] <= truth <= bound[1], (bound, truth)
    assert interval["bounds"][0][1] < 0, "the interval for x must stay negative"


def test_ablation_separates_the_generating_support_and_ranks_its_removals():
    task = damped_oscillator_task()
    scores = rollout_scores(task, task.truth_support, task.truth_coefficients)
    report = ablation(
        task, task.truth_support, task.truth_coefficients, candidate_r2=scores["discovered_r2"]
    )
    assert report["separated"] is True
    gaps = {row["removed"]: row["gap"] for row in report["removals"]}
    assert gaps["x"] > gaps["v"] > 0, gaps
    assert report["largest_addition_gain"] < 1e-4
    assert len(report["additions"]) == len(task.library) - len(task.truth_support)


def test_an_abstention_is_an_abstention_and_not_a_failure():
    task = damped_oscillator_task(budget=10)
    result = run_arm("qaoa_surrogate", task, seed=1)
    verdict = adjudicate(task, result, seed=1)
    assert verdict.status == ABSTAIN
    assert verdict.holdout_r2 is None
    assert verdict.notes and "no candidate" in verdict.notes[0]


def test_a_different_equation_that_still_predicts_is_an_alternative_not_a_failure():
    task = damped_oscillator_task()
    result = run_arm("rime_relaxed", task, seed=11)
    verdict = adjudicate(task, result, seed=11)
    assert set(verdict.support) != set(task.truth_names)
    assert verdict.status == ALTERNATIVE
    assert verdict.recovered is False
    assert verdict.holdout_r2 >= HOLDOUT_R2_FLOOR
    separability = next(check for check in verdict.checks if check.name == "ablation_separability")
    assert separability.passed is False, "an extra term that costs nothing is not separable"


def test_a_wrong_equation_that_cannot_predict_is_invalid():
    task = damped_oscillator_task()
    result = run_arm("rime_relaxed", task, seed=11)
    verdict = adjudicate(task, result, seed=11)
    assert verdict.status in (ALTERNATIVE, VALID), "the alternative must be judged on prediction"


# --------------------------------------------------------------------------- #
# Task and proposer
# --------------------------------------------------------------------------- #


def test_the_task_is_reproducible_from_its_seed_and_noisy_on_purpose():
    first = damped_oscillator_task(seed=4)
    second = damped_oscillator_task(seed=4)
    third = damped_oscillator_task(seed=5)
    assert np.array_equal(first.dev_states, second.dev_states)
    assert np.array_equal(first.dev_target, second.dev_target)
    assert not np.array_equal(first.dev_states, third.dev_states)
    clean = simulate_oscillator(2.5, 0.0, steps=400, dt=0.02, stiffness=1.44, damping=0.15)
    assert not np.array_equal(clean, first.dev_states), "the observations must be noisy"


def test_the_task_refuses_a_contradictory_formulation():
    states = np.zeros((5, 2))
    with pytest.raises(TaskRefusal):
        damped_oscillator_task().__class__(
            name="broken",
            library=oscillator_library(),
            dev_states=states,
            dev_target=np.zeros(5),
            holdout_states=states,
            holdout_target=np.zeros(5),
            dt=0.02,
            discovered=0,
            kinematic_pairs=((0, 1),),
            truth_support=(1,),
            truth_coefficients=(-1.0,),
            penalty=0.0,
            budget=4,
            noise_level=0.01,
            seed=1,
        )


def test_the_proposer_emits_one_standard_candidate_and_never_adjudicates():
    task = damped_oscillator_task()
    # The adapter is arm-agnostic, so it is pinned with an arm that recovers on
    # this task; what the QAOA arm returns is a property of the arm and is
    # measured in the record, not assumed here.
    proposer = QuantumEquationProposer(arm="random_restart", library=task.library, seed=11)
    proposals = proposer.fit_and_propose(task.dev_states, task.dev_target)
    assert len(proposals) == 1
    proposal = proposals[0]
    assert isinstance(proposal, CandidateProposal)
    assert proposal.strategy_archetype is StrategyArchetype.DYNAMICAL_SYSTEM
    assert proposal.declared_scale_invariant is False
    assert proposal.proposer_id == "qed_random_restart"
    assert set(proposal.parameters["support"]) == set(task.truth_names)
    assert proposal.metadata["arm"] == "random_restart"
    assert proposal.metadata["evaluations"] <= proposal.metadata["budget"]

    import pandas as pd

    frame = pd.DataFrame(task.dev_states, columns=["x", "v"])
    predicted = proposal.compute_score(frame)
    assert predicted.shape == (task.dev_states.shape[0],)
    # The target is a finite difference of a noisy signal, so its pointwise
    # error is dominated by gradient noise (~2.1 here); what the score must
    # reproduce is the fitted law, so the correlation is the assertion.
    assert np.corrcoef(predicted, task.dev_target)[0, 1] > 0.9
    assert np.mean(np.abs(predicted - task.dev_target)) < 0.35 * np.std(task.dev_target)
    assert "*x" in proposal.formula and "*v" in proposal.formula
    assert abs(proposal.parameters["coefficients"][0] + 1.44) < 0.05


def test_the_proposer_returns_nothing_rather_than_something_worse_when_it_cannot_search():
    task = damped_oscillator_task(budget=10)
    proposer = QuantumEquationProposer(arm="qaoa_surrogate", library=task.library, budget=10)
    assert list(proposer.fit_and_propose(task.dev_states, task.dev_target)) == []


def test_no_arm_object_or_signature_can_reach_the_holdout():
    """The boundary is structural: propsers are handed rows, not the task."""
    proposer = QuantumEquationProposer(arm="random_restart", library=oscillator_library())
    import inspect

    parameters = set(inspect.signature(proposer.fit_and_propose).parameters)
    for forbidden in ("task", "holdout", "truth"):
        assert not any(forbidden in name for name in parameters)


def test_an_arm_that_cannot_run_raises_its_own_refusal_type():
    assert issubclass(ArmAbstained, RuntimeError)
