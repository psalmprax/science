import ApedpChecked

/-!
Prints the axioms every theorem depends on. `scripts/check_lean_checked.py`
runs this file and fails if any line names an axiom other than Lean's own
`propext`, `Quot.sound` and `Classical.choice`, or names `sorryAx`.
-/

open ApedpChecked

#print axioms Dim.mul_def
#print axioms Dim.div_def
#print axioms Dim.pow_def
#print axioms Dim.mul_comm
#print axioms Dim.mul_assoc
#print axioms Dim.mul_one
#print axioms Dim.one_mul
#print axioms Dim.div_self
#print axioms Dim.mul_div_cancel
#print axioms Dim.pow_zero
#print axioms Dim.pow_one
#print axioms Dim.mul_pow
#print axioms Dim.pow_add
#print axioms Dim.kinetic_energy_consistent
#print axioms Dim.newtonian_gravity_consistent
#print axioms Dim.pendulum_period_consistent
#print axioms Dim.momentum_form_of_kinetic_energy
#print axioms Dim.momentum_is_not_energy
#print axioms Dim.force_is_not_energy
#print axioms monomial_scale
#print axioms bivariate_monomial_scale
#print axioms Tracker.halted_is_terminal
#print axioms Tracker.only_the_next_stage
#print axioms Tracker.failure_halts
#print axioms Tracker.pass_advances_by_one
#print axioms Tracker.step_bounds
#print axioms Tracker.no_shortcut
#print axioms Tracker.stage_needs_as_many_evaluations
#print axioms Tracker.nothing_follows_a_failure
#print axioms Tracker.six_passes_reach_stage_six
#print axioms Tracker.skipping_is_refused
#print axioms Slice.burned_refuses_everyone
#print axioms Slice.nothing_after_a_burn
#print axioms Slice.claim_preserves_burned
#print axioms Slice.budget_is_respected
#print axioms Slice.own_slice_again
#print axioms Slice.newcomer_refused_when_spent
#print axioms Slice.granted_claim_is_recorded
#print axioms Slice.single_use
