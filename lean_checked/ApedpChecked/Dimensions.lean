/-!
# Dimensional algebra

A physical dimension is a vector of integer exponents over mass, length and
time. Multiplying quantities adds exponents, dividing subtracts them, and a
law is dimensionally consistent when both sides carry the same vector. That
makes dimensional consistency of a monomial law a decidable statement, and the
theorems below are checked by the Lean kernel with no axioms declared here.

What this file does not do: it says nothing about whether a law is *true*, and
it does not connect to the Python referee. It establishes the algebra and
checks a few laws by hand.
-/

namespace ApedpChecked

/-- Exponents of mass, length and time. -/
structure Dim where
  massExp : Int
  lengthExp : Int
  timeExp : Int
  deriving DecidableEq, Repr

namespace Dim

/-- The dimension of a pure number. -/
def one : Dim := ⟨0, 0, 0⟩

def mul (a b : Dim) : Dim := ⟨a.massExp + b.massExp, a.lengthExp + b.lengthExp, a.timeExp + b.timeExp⟩

def inv (a : Dim) : Dim := ⟨-a.massExp, -a.lengthExp, -a.timeExp⟩

def pow (a : Dim) (n : Int) : Dim := ⟨a.massExp * n, a.lengthExp * n, a.timeExp * n⟩

instance : Mul Dim := ⟨mul⟩
instance : Div Dim := ⟨fun a b => mul a (inv b)⟩
instance : HPow Dim Int Dim := ⟨pow⟩

theorem mul_def (a b : Dim) :
    a * b = ⟨a.massExp + b.massExp, a.lengthExp + b.lengthExp, a.timeExp + b.timeExp⟩ := rfl

theorem div_def (a b : Dim) :
    a / b = ⟨a.massExp + -b.massExp, a.lengthExp + -b.lengthExp, a.timeExp + -b.timeExp⟩ := rfl

theorem pow_def (a : Dim) (n : Int) :
    a ^ n = ⟨a.massExp * n, a.lengthExp * n, a.timeExp * n⟩ := rfl

/-! ## The algebra: dimensions form a commutative group under multiplication -/

theorem mul_comm (a b : Dim) : a * b = b * a := by
  simp only [mul_def, Dim.mk.injEq]
  omega

theorem mul_assoc (a b c : Dim) : a * b * c = a * (b * c) := by
  simp only [mul_def, Dim.mk.injEq]
  omega

theorem mul_one (a : Dim) : a * one = a := by
  cases a
  simp [mul_def, one]

theorem one_mul (a : Dim) : one * a = a := by
  cases a
  simp [mul_def, one]

theorem div_self (a : Dim) : a / a = one := by
  simp only [div_def, one, Dim.mk.injEq]
  omega

/-- Dividing by a quantity undoes multiplying by it. -/
theorem mul_div_cancel (a b : Dim) : a * b / b = a := by
  cases a
  cases b
  simp only [mul_def, div_def, Dim.mk.injEq]
  omega

/-! ## Powers -/

theorem pow_zero (a : Dim) : a ^ (0 : Int) = one := by
  simp [pow_def, one]

theorem pow_one (a : Dim) : a ^ (1 : Int) = a := by
  cases a
  simp [pow_def]

/-- A product raised to a power is the product of the powers. -/
theorem mul_pow (a b : Dim) (n : Int) : (a * b) ^ n = a ^ n * b ^ n := by
  simp only [mul_def, pow_def, Int.add_mul]

/-- Exponents add. -/
theorem pow_add (a : Dim) (m n : Int) : a ^ (m + n) = a ^ m * a ^ n := by
  simp only [mul_def, pow_def, Int.mul_add]

/-! ## Base and derived dimensions -/

def mass : Dim := ⟨1, 0, 0⟩
def length : Dim := ⟨0, 1, 0⟩
def time : Dim := ⟨0, 0, 1⟩

def velocity : Dim := length / time
def acceleration : Dim := velocity / time
def force : Dim := mass * acceleration
def energy : Dim := force * length
def momentum : Dim := mass * velocity

/-- The dimension Newton's gravitational constant must carry. -/
def gravitationalConstant : Dim := length ^ (3 : Int) / (mass * time ^ (2 : Int))

/-! ## Laws checked for dimensional consistency -/

/-- `E = m v²`: kinetic energy has the dimension of energy. -/
theorem kinetic_energy_consistent : mass * velocity ^ (2 : Int) = energy := by decide

/-- `F = G m₁ m₂ / r²`: Newtonian gravity has the dimension of force. -/
theorem newtonian_gravity_consistent :
    gravitationalConstant * mass * mass / length ^ (2 : Int) = force := by decide

/-- `T² = L / g`: the square of a pendulum's period. -/
theorem pendulum_period_consistent : time ^ (2 : Int) = length / acceleration := by decide

/-- `p² / m`: momentum squared over mass is an energy. -/
theorem momentum_form_of_kinetic_energy : momentum ^ (2 : Int) / mass = energy := by decide

/-! The check can fail: these are refuted, not merely unproven. -/

/-- Momentum is not an energy, so `E = m v` is dimensionally inconsistent. -/
theorem momentum_is_not_energy : mass * velocity ≠ energy := by decide

/-- A force is not an energy: `E = F` is missing a length. -/
theorem force_is_not_energy : force ≠ energy := by decide

end Dim

/-! ## Scale homogeneity of monomials

Rescaling every input of a monomial by `c` rescales its value by a fixed power
of `c`. This is the property a "scale homogeneity" invariant asserts; here it
is proved for one- and two-variable monomials over the natural numbers.
-/

theorem monomial_scale (c x k : Nat) : (c * x) ^ k = c ^ k * x ^ k :=
  Nat.mul_pow c x k

/-- `x^a * y^b` is homogeneous of degree `a + b`. -/
theorem bivariate_monomial_scale (c x y a b : Nat) :
    (c * x) ^ a * (c * y) ^ b = c ^ (a + b) * (x ^ a * y ^ b) := by
  rw [Nat.mul_pow, Nat.mul_pow, Nat.pow_add]
  ac_rfl

end ApedpChecked
