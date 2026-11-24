/-!
# The confirmation-holdout guard, as a model

A confirmation holdout may be inspected by a bounded number of distinct
candidates, and once a candidate fails on it nobody may inspect it again. This
file states that rule as a pure function and proves that it has the properties
the anti-snooping discipline depends on.

This is a model of the rule `FullLineageExperimentLedger` implements in Python
(`apedp_dyn/epistemic/experiment_ledger.py`), with candidates as natural
numbers. It is not that code, and it says nothing about slice identity, file
persistence or concurrency.
-/

namespace ApedpChecked

/-- A holdout slice: who has inspected it, and whether a failure burned it. -/
structure Slice where
  inspectors : List Nat
  burned : Bool
  deriving DecidableEq, Repr

namespace Slice

/-- A slice nobody has looked at. -/
def fresh : Slice := ⟨[], false⟩

/-- Candidate `c` asks to inspect the slice under an inspection budget.
`none` means refused. -/
def claim (budget : Nat) (s : Slice) (c : Nat) : Option Slice :=
  if s.burned then none
  else if c ∈ s.inspectors then some s
  else if s.inspectors.length < budget then some { s with inspectors := c :: s.inspectors }
  else none

/-- A candidate fails on the slice. -/
def burn (s : Slice) : Slice := { s with burned := true }

/-- A burned slice refuses everyone, including the candidate that burned it. -/
theorem burned_refuses_everyone (budget : Nat) (s : Slice) (c : Nat) (h : s.burned = true) :
    claim budget s c = none := by
  simp [claim, h]

/-- Burning is permanent: no claim is granted after it, whatever happened before. -/
theorem nothing_after_a_burn (budget : Nat) (s : Slice) (c : Nat) :
    claim budget (burn s) c = none :=
  burned_refuses_everyone budget (burn s) c rfl

/-- A granted claim never un-burns a slice. -/
theorem claim_preserves_burned (budget : Nat) (s s' : Slice) (c : Nat)
    (h : claim budget s c = some s') : s'.burned = s.burned := by
  unfold claim at h
  split at h
  · contradiction
  · split at h
    · simp at h
      subst h
      rfl
    · split at h
      · simp at h
        subst h
        rfl
      · contradiction

/-- The number of distinct inspectors never exceeds the budget. -/
theorem budget_is_respected (budget : Nat) (s s' : Slice) (c : Nat)
    (within : s.inspectors.length ≤ budget) (h : claim budget s c = some s') :
    s'.inspectors.length ≤ budget := by
  unfold claim at h
  split at h
  · contradiction
  · split at h
    · simp at h
      subst h
      exact within
    · split at h
      · rename_i hlt
        simp at h
        subst h
        simp only [List.length_cons]
        omega
      · contradiction

/-- A candidate that already inspected the slice may look again, and that costs nothing. -/
theorem own_slice_again (budget : Nat) (s : Slice) (c : Nat)
    (alive : s.burned = false) (mine : c ∈ s.inspectors) : claim budget s c = some s := by
  simp [claim, alive, mine]

/-- With a spent budget, a candidate that has not inspected the slice is refused. -/
theorem newcomer_refused_when_spent (budget : Nat) (s : Slice) (c : Nat)
    (spent : budget ≤ s.inspectors.length) (new : c ∉ s.inspectors) :
    claim budget s c = none := by
  have full : ¬ s.inspectors.length < budget := by omega
  simp [claim, new, full]

/-- A granted claim records the candidate as an inspector. -/
theorem granted_claim_is_recorded (budget : Nat) (s s' : Slice) (c : Nat)
    (h : claim budget s c = some s') : c ∈ s'.inspectors := by
  unfold claim at h
  split at h
  · contradiction
  · split at h
    · rename_i hmem
      simp at h
      subst h
      exact hmem
    · split at h
      · simp at h
        subst h
        simp
      · contradiction

/-- The single-use default: after one candidate, a second distinct one is refused. -/
theorem single_use (a b : Nat) (distinct : b ≠ a) (s : Slice)
    (h : claim 1 fresh a = some s) : claim 1 s b = none := by
  have hs : s = ⟨[a], false⟩ := by
    simp [claim, fresh] at h
    exact h.symm
  subst hs
  simp [claim, distinct]

end Slice

end ApedpChecked
