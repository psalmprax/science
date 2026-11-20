/-!
# The promotion state machine, as a model

A candidate is promoted one stage at a time. A stage can be evaluated only when
it is the next one, each evaluation either passes or fails, and a failure halts
the candidate for good. This file states that machine as a pure function and
proves the properties the referee relies on.

This is a model. It mirrors the rules `PromotionTracker` implements in Python
(`apedp_dyn/epistemic/promotion_lifecycle.py`); it is not that code, and
nothing here proves the Python agrees with it. What it does establish is that
the *rules* have the properties claimed for them.
-/

namespace ApedpChecked

/-- Where a candidate stands: the highest stage it has passed, and whether a
failure has halted it. -/
structure Tracker where
  highest : Nat
  halted : Bool
  deriving DecidableEq, Repr

namespace Tracker

/-- A candidate nothing has been decided about. -/
def init : Tracker := ⟨0, false⟩

/-- Evaluate `stage` with outcome `ok`. `none` means the evaluation is refused:
the candidate is halted, or `stage` is not the next one. -/
def step (t : Tracker) (stage : Nat) (ok : Bool) : Option Tracker :=
  if t.halted then none
  else if stage ≠ t.highest + 1 then none
  else if ok then some ⟨stage, false⟩
  else some ⟨t.highest, true⟩

/-- Evaluate a sequence of `(stage, outcome)` pairs; refused as a whole if any
one of them is refused. -/
def run (t : Tracker) : List (Nat × Bool) → Option Tracker
  | [] => some t
  | (stage, ok) :: rest =>
    match step t stage ok with
    | none => none
    | some t' => run t' rest

/-- A halted candidate cannot be evaluated again, at any stage, with any outcome. -/
theorem halted_is_terminal (t : Tracker) (h : t.halted = true) (stage : Nat) (ok : Bool) :
    step t stage ok = none := by
  simp [step, h]

/-- Only the next stage can be evaluated: no stage is skipped and none is re-run. -/
theorem only_the_next_stage (t t' : Tracker) (stage : Nat) (ok : Bool)
    (h : step t stage ok = some t') : stage = t.highest + 1 := by
  unfold step at h
  split at h
  · contradiction
  · split at h
    · contradiction
    · rename_i hne
      exact Decidable.of_not_not hne

/-- A failed evaluation halts the candidate and advances nothing. -/
theorem failure_halts (t t' : Tracker) (stage : Nat)
    (h : step t stage false = some t') : t'.halted = true ∧ t'.highest = t.highest := by
  unfold step at h
  split at h
  · contradiction
  · split at h
    · contradiction
    · simp at h
      subst h
      exact ⟨rfl, rfl⟩

/-- A passed evaluation advances the candidate by exactly one stage. -/
theorem pass_advances_by_one (t t' : Tracker) (stage : Nat)
    (h : step t stage true = some t') : t'.highest = t.highest + 1 ∧ t'.halted = false := by
  have hs := only_the_next_stage t t' stage true h
  unfold step at h
  split at h
  · contradiction
  · split at h
    · contradiction
    · simp at h
      subst h
      exact ⟨hs, rfl⟩

/-- One evaluation never moves the highest stage down, and never up by more than one. -/
theorem step_bounds (t t' : Tracker) (stage : Nat) (ok : Bool)
    (h : step t stage ok = some t') :
    t.highest ≤ t'.highest ∧ t'.highest ≤ t.highest + 1 := by
  cases ok with
  | false =>
    have := (failure_halts t t' stage h).2
    omega
  | true =>
    have := (pass_advances_by_one t t' stage h).1
    omega

/-- No sequence of evaluations raises the highest stage by more than its length:
there is no shortcut to stage six in fewer than six evaluations. -/
theorem no_shortcut (events : List (Nat × Bool)) :
    ∀ (t t' : Tracker), run t events = some t' → t'.highest ≤ t.highest + events.length := by
  induction events with
  | nil =>
    intro t t' h
    simp [run] at h
    subst h
    simp
  | cons event rest ih =>
    intro t t' h
    obtain ⟨stage, ok⟩ := event
    unfold run at h
    cases hstep : step t stage ok with
    | none => simp [hstep] at h
    | some mid =>
      simp [hstep] at h
      have h1 := (step_bounds t mid stage ok hstep).2
      have h2 := ih mid t' h
      simp only [List.length_cons]
      omega

/-- Reaching stage `n` from nothing takes at least `n` evaluations. -/
theorem stage_needs_as_many_evaluations (events : List (Nat × Bool)) (t' : Tracker)
    (h : run init events = some t') : t'.highest ≤ events.length := by
  have := no_shortcut events init t' h
  simpa [init] using this

/-- Once a candidate has failed, every continuation is refused. -/
theorem nothing_follows_a_failure (t t' : Tracker) (stage : Nat)
    (h : step t stage false = some t') (next : Nat) (ok : Bool) (rest : List (Nat × Bool)) :
    run t' ((next, ok) :: rest) = none := by
  have hh := (failure_halts t t' stage h).1
  simp [run, halted_is_terminal t' hh]

/-- The intended path exists: six passes in order reach stage six. -/
theorem six_passes_reach_stage_six :
    run init [(1, true), (2, true), (3, true), (4, true), (5, true), (6, true)]
      = some ⟨6, false⟩ := by decide

/-- Skipping a stage is refused. -/
theorem skipping_is_refused : run init [(1, true), (3, true)] = none := by decide

end Tracker

end ApedpChecked
