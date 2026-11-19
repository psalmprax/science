//! Signed Pauli strings, and the exact conjugation tables of the Clifford set.
//!
//! The tables are *derived* from the 2x2 and 4x4 unitaries at first use, the way
//! `apedp_dyn/vqc/stabilizer.py` derives them, rather than transcribed as sign
//! patterns. A hand-written table is a second, undocumented statement of the
//! same physics and the first place a relative phase goes wrong; deriving it
//! means the only claim to check is the gate matrix itself, and that claim is
//! checked by the Python-Rust parity corpus in `tests/test_vqc_native_parity.py`.
//!
//! Phase convention: the sign of a Pauli string is `i**phase` with
//! `phase` in `0..4`, so `-1` is phase 2 and `+i` is phase 1. Every conjugate
//! below is exact in that representation (the entries of `U P U^dagger` are
//! always `0` or `+-1` or `+-i` for a Clifford `U`), so "exact agreement with
//! Python" is a statement about integers, not about a floating-point tolerance.

use std::sync::OnceLock;

use crate::bits::Bitset;

/// `I`, `X`, `Y`, `Z` as small integers: the row/column order of every table.
pub const I: u8 = 0;
pub const X: u8 = 1;
pub const Y: u8 = 2;
pub const Z: u8 = 3;

/// The label names, for reports and parity failures.
pub const LABEL_CHARS: [char; 4] = ['I', 'X', 'Y', 'Z'];

/// One complex number with real arithmetic. The gate and Pauli entries are all
/// in `{0, +-1, +-i}` (apart from `H`, whose `1/sqrt 2` cancels in `U P U^d`),
/// so `f64` is exact here and `total_cmp` is never needed.
#[derive(Clone, Copy, Debug, PartialEq)]
struct C {
    re: f64,
    im: f64,
}

impl C {
    const fn new(re: f64, im: f64) -> Self {
        C { re, im }
    }
    const ZERO: C = C::new(0.0, 0.0);
    const ONE: C = C::new(1.0, 0.0);

    fn add(self, other: C) -> C {
        C::new(self.re + other.re, self.im + other.im)
    }
    fn mul(self, other: C) -> C {
        C::new(
            self.re * other.re - self.im * other.im,
            self.re * other.im + self.im * other.re,
        )
    }
    fn conj(self) -> C {
        C::new(self.re, -self.im)
    }
    fn close_to(self, other: C, tolerance: f64) -> bool {
        (self.re - other.re).abs() <= tolerance && (self.im - other.im).abs() <= tolerance
    }
}

/// `i**power`, for `power` reduced mod 4.
fn i_pow(power: i32) -> C {
    match power.rem_euclid(4) {
        0 => C::new(1.0, 0.0),
        1 => C::new(0.0, 1.0),
        2 => C::new(-1.0, 0.0),
        _ => C::new(0.0, -1.0),
    }
}

/// A row-major `dim x dim` matrix.
#[derive(Clone, Debug)]
struct Mat {
    dim: usize,
    entries: Vec<C>,
}

impl Mat {
    fn new(dim: usize, entries: Vec<C>) -> Self {
        assert_eq!(entries.len(), dim * dim, "matrix shape");
        Mat { dim, entries }
    }

    fn at(&self, row: usize, column: usize) -> C {
        self.entries[row * self.dim + column]
    }

    fn mul(&self, other: &Mat) -> Mat {
        assert_eq!(self.dim, other.dim, "matrix dimensions");
        let mut out = vec![C::ZERO; self.dim * self.dim];
        for row in 0..self.dim {
            for column in 0..self.dim {
                let mut total = C::ZERO;
                for inner in 0..self.dim {
                    total = total.add(self.at(row, inner).mul(other.at(inner, column)));
                }
                out[row * self.dim + column] = total;
            }
        }
        Mat::new(self.dim, out)
    }

    fn adjoint(&self) -> Mat {
        let mut out = vec![C::ZERO; self.dim * self.dim];
        for row in 0..self.dim {
            for column in 0..self.dim {
                out[row * self.dim + column] = self.at(column, row).conj();
            }
        }
        Mat::new(self.dim, out)
    }

    /// `A (x) B`, first factor varying slowest -- NumPy's `kron` order, which is
    /// also "the first target is the first tensor factor".
    fn kron(&self, other: &Mat) -> Mat {
        let dim = self.dim * other.dim;
        let mut out = vec![C::ZERO; dim * dim];
        for row in 0..self.dim {
            for column in 0..self.dim {
                let scalar = self.at(row, column);
                for sub_row in 0..other.dim {
                    for sub_column in 0..other.dim {
                        let target_row = row * other.dim + sub_row;
                        let target_column = column * other.dim + sub_column;
                        out[target_row * dim + target_column] =
                            scalar.mul(other.at(sub_row, sub_column));
                    }
                }
            }
        }
        Mat::new(dim, out)
    }
}

fn pauli_matrix(label: u8) -> Mat {
    match label {
        I => Mat::new(2, vec![C::ONE, C::ZERO, C::ZERO, C::ONE]),
        X => Mat::new(2, vec![C::ZERO, C::ONE, C::ONE, C::ZERO]),
        Y => Mat::new(
            2,
            vec![C::ZERO, C::new(0.0, -1.0), C::new(0.0, 1.0), C::ZERO],
        ),
        Z => Mat::new(2, vec![C::ONE, C::ZERO, C::ZERO, C::new(-1.0, 0.0)]),
        _ => panic!("label {label} is not a Pauli"),
    }
}

/// The one-qubit Cliffords this tier executes, in the order of the tables.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gate1 {
    H,
    X,
    Y,
    Z,
    S,
    Sdg,
}

/// The two-qubit Cliffords this tier executes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gate2 {
    Cx,
    Cz,
    Swap,
}

/// Whatever `apedp_dyn.vqc.gates.GATE_SPECS` calls a Clifford gate, named.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Clifford {
    One(Gate1),
    Two(Gate2),
}

impl Clifford {
    /// The gate of this name, or `None` for a gate this tier does not execute.
    pub fn from_name(name: &str) -> Option<Clifford> {
        match name {
            "h" => Some(Clifford::One(Gate1::H)),
            "x" => Some(Clifford::One(Gate1::X)),
            "y" => Some(Clifford::One(Gate1::Y)),
            "z" => Some(Clifford::One(Gate1::Z)),
            "s" => Some(Clifford::One(Gate1::S)),
            "sdg" => Some(Clifford::One(Gate1::Sdg)),
            "cx" => Some(Clifford::Two(Gate2::Cx)),
            "cz" => Some(Clifford::Two(Gate2::Cz)),
            "swap" => Some(Clifford::Two(Gate2::Swap)),
            _ => None,
        }
    }

    pub fn arity(self) -> usize {
        match self {
            Clifford::One(_) => 1,
            Clifford::Two(_) => 2,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Clifford::One(Gate1::H) => "h",
            Clifford::One(Gate1::X) => "x",
            Clifford::One(Gate1::Y) => "y",
            Clifford::One(Gate1::Z) => "z",
            Clifford::One(Gate1::S) => "s",
            Clifford::One(Gate1::Sdg) => "sdg",
            Clifford::Two(Gate2::Cx) => "cx",
            Clifford::Two(Gate2::Cz) => "cz",
            Clifford::Two(Gate2::Swap) => "swap",
        }
    }
}

fn single_matrix(gate: Gate1) -> Mat {
    let half = C::new(std::f64::consts::FRAC_1_SQRT_2, 0.0);
    let entries = match gate {
        Gate1::H => vec![half, half, half, C::new(-half.re, 0.0)],
        Gate1::X => vec![C::ZERO, C::ONE, C::ONE, C::ZERO],
        Gate1::Y => vec![C::ZERO, C::new(0.0, -1.0), C::new(0.0, 1.0), C::ZERO],
        Gate1::Z => vec![C::ONE, C::ZERO, C::ZERO, C::new(-1.0, 0.0)],
        Gate1::S => vec![C::ONE, C::ZERO, C::ZERO, C::new(0.0, 1.0)],
        Gate1::Sdg => vec![C::ONE, C::ZERO, C::ZERO, C::new(0.0, -1.0)],
    };
    Mat::new(2, entries)
}

fn pair_matrix(gate: Gate2) -> Mat {
    let mut entries = vec![C::ZERO; 16];
    for first in 0..2 {
        for second in 0..2 {
            let column = 2 * first + second;
            let (row, value) = match gate {
                Gate2::Cx => (2 * first + (first ^ second), C::ONE),
                Gate2::Swap => (2 * second + first, C::ONE),
                Gate2::Cz => (
                    column,
                    if first == 1 && second == 1 {
                        C::new(-1.0, 0.0)
                    } else {
                        C::ONE
                    },
                ),
            };
            entries[row * 4 + column] = value;
        }
    }
    Mat::new(4, entries)
}

/// How one gate maps `P_label`: `U P U^dagger == i**delta * P_label`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SingleRule {
    pub delta: i32,
    pub label: u8,
}

/// How one gate maps `P_first (x) P_second`, with its exact phase.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PairRule {
    pub delta: i32,
    pub first: u8,
    pub second: u8,
}

const GATES_ONE: [Gate1; 6] = [Gate1::H, Gate1::X, Gate1::Y, Gate1::Z, Gate1::S, Gate1::Sdg];
const GATES_TWO: [Gate2; 3] = [Gate2::Cx, Gate2::Cz, Gate2::Swap];

fn gate1_index(gate: Gate1) -> usize {
    GATES_ONE.iter().position(|candidate| *candidate == gate).expect("known gate")
}

fn gate2_index(gate: Gate2) -> usize {
    GATES_TWO.iter().position(|candidate| *candidate == gate).expect("known gate")
}

/// The tightest tolerance that still separates two distinct Pauli matrices:
/// their entries differ by at least 1 in the real or imaginary part.
const TOLERANCE: f64 = 1e-9;

fn snap(matrix: &Mat) -> (i32, u8) {
    for phase in 0..4 {
        let scalar = i_pow(phase);
        for label in 0..4u8 {
            let candidate = pauli_matrix(label);
            let matches = (0..matrix.dim * matrix.dim).all(|index| {
                let expected =
                    scalar.mul(candidate.entries[index]);
                matrix.entries[index].close_to(expected, TOLERANCE)
            });
            if matches {
                return (phase, label);
            }
        }
    }
    panic!("the conjugated operator is not a Pauli matrix: not a Clifford gate")
}

fn snap_pair(matrix: &Mat) -> (i32, u8, u8) {
    for phase in 0..4 {
        let scalar = i_pow(phase);
        for first in 0..4u8 {
            for second in 0..4u8 {
                let candidate = pauli_matrix(first).kron(&pauli_matrix(second));
                let matches = (0..16).all(|index| {
                    let expected = scalar.mul(candidate.entries[index]);
                    matrix.entries[index].close_to(expected, TOLERANCE)
                });
                if matches {
                    return (phase, first, second);
                }
            }
        }
    }
    panic!("the conjugated operator is not a two-qubit Pauli: not a Clifford gate")
}

fn build_single() -> [[SingleRule; 4]; 6] {
    let mut table = [[SingleRule { delta: 0, label: I }; 4]; 6];
    for (index, gate) in GATES_ONE.iter().enumerate() {
        let unitary = single_matrix(*gate);
        let dagger = unitary.adjoint();
        for label in 0..4u8 {
            let conjugated = unitary.mul(&pauli_matrix(label)).mul(&dagger);
            let (delta, new_label) = snap(&conjugated);
            // A gate that does not fix the identity is not a gate.
            if label == I && (delta, new_label) != (0, I) {
                panic!("{gate:?} does not fix the identity");
            }
            table[index][label as usize] = SingleRule { delta, label: new_label };
        }
    }
    table
}

fn build_pair() -> [[[PairRule; 4]; 4]; 3] {
    let mut table = [[[PairRule { delta: 0, first: I, second: I }; 4]; 4]; 3];
    for (index, gate) in GATES_TWO.iter().enumerate() {
        let unitary = pair_matrix(*gate);
        let dagger = unitary.adjoint();
        for left in 0..4u8 {
            for right in 0..4u8 {
                let operator = pauli_matrix(left).kron(&pauli_matrix(right));
                let conjugated = unitary.mul(&operator).mul(&dagger);
                let (delta, new_left, new_right) = snap_pair(&conjugated);
                table[index][left as usize][right as usize] =
                    PairRule { delta, first: new_left, second: new_right };
            }
        }
    }
    table
}

fn build_product() -> [[(i32, u8); 4]; 4] {
    let mut table = [[(0, I); 4]; 4];
    for left in 0..4u8 {
        for right in 0..4u8 {
            table[left as usize][right as usize] =
                snap(&pauli_matrix(left).mul(&pauli_matrix(right)));
        }
    }
    table
}

fn single_table() -> &'static [[SingleRule; 4]; 6] {
    static TABLE: OnceLock<[[SingleRule; 4]; 6]> = OnceLock::new();
    TABLE.get_or_init(build_single)
}

fn pair_table() -> &'static [[[PairRule; 4]; 4]; 3] {
    static TABLE: OnceLock<[[[PairRule; 4]; 4]; 3]> = OnceLock::new();
    TABLE.get_or_init(build_pair)
}

fn product_table() -> &'static [[(i32, u8); 4]; 4] {
    static TABLE: OnceLock<[[(i32, u8); 4]; 4]> = OnceLock::new();
    TABLE.get_or_init(build_product)
}

/// `U P_label U^dagger` for a one-qubit Clifford.
pub fn conjugate_one(gate: Gate1, label: u8) -> SingleRule {
    single_table()[gate1_index(gate)][label as usize]
}

/// `U (P_first (x) P_second) U^dagger` for a two-qubit Clifford.
pub fn conjugate_pair(gate: Gate2, first: u8, second: u8) -> PairRule {
    pair_table()[gate2_index(gate)][first as usize][second as usize]
}

/// `P_left P_right == i**delta * P_label`.
pub fn product_pair(left: u8, right: u8) -> (i32, u8) {
    product_table()[left as usize][right as usize]
}

/// The exact product of two signed Pauli strings.
pub fn multiply(left: &[u8], left_phase: i32, right: &[u8], right_phase: i32) -> (Vec<u8>, i32) {
    assert_eq!(left.len(), right.len(), "Pauli strings of different length");
    let mut labels = Vec::with_capacity(left.len());
    let mut phase = (left_phase + right_phase).rem_euclid(4);
    for (a, b) in left.iter().zip(right.iter()) {
        // Equal labels multiply to the identity with phase 0, so `P P = I`
        // (Hermitian and unitary) is a special case rather than a table entry.
        if a == b {
            labels.push(I);
        } else if *a == I {
            labels.push(*b);
        } else if *b == I {
            labels.push(*a);
        } else {
            let (delta, label) = product_pair(*a, *b);
            phase = (phase + delta).rem_euclid(4);
            labels.push(label);
        }
    }
    (labels, phase)
}

/// The characteristic vector of a Pauli string: `x` bits low, `z` bits high.
pub fn to_mask(labels: &[u8]) -> Bitset {
    let width = labels.len();
    let mut mask = Bitset::zeros(2 * width);
    for (position, label) in labels.iter().enumerate() {
        if *label == X || *label == Y {
            mask.set(position);
        }
        if *label == Z || *label == Y {
            mask.set(width + position);
        }
    }
    mask
}

/// `'+XXI'`-style text for one signed Pauli string.
pub fn format_string(labels: &[u8], phase: i32) -> String {
    let sign = match phase.rem_euclid(4) {
        0 => "+",
        1 => "+i",
        2 => "-",
        _ => "-i",
    };
    let body: String = labels.iter().map(|label| LABEL_CHARS[*label as usize]).collect();
    format!("{sign}{body}")
}

/// A single-qubit Pauli string on `num_qubits`, `Z_q` by default.
pub fn unit_string(num_qubits: usize, qubit: usize, label: u8) -> Vec<u8> {
    let mut labels = vec![I; num_qubits];
    labels[qubit] = label;
    labels
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_pauli_product_table_is_the_quaternion_rule() {
        // X Y = iZ, Y X = -iZ, and every generator squares to the identity.
        assert_eq!(product_pair(X, Y), (1, Z));
        assert_eq!(product_pair(Y, X), (3, Z));
        assert_eq!(product_pair(Z, X), (1, Y));
        assert_eq!(product_pair(X, Z), (3, Y));
        assert_eq!(product_pair(Y, Z), (1, X));
        assert_eq!(product_pair(Z, Y), (3, X));
        for label in [I, X, Y, Z] {
            assert_eq!(product_pair(label, label), (0, I), "P^2 = I for {label}");
        }
    }

    #[test]
    fn ascii_random_gate_tables_are_the_textbook_ones() {
        // H swaps X and Z and flips Y: H Y H = -Y.
        assert_eq!(conjugate_one(Gate1::H, X), SingleRule { delta: 0, label: Z });
        assert_eq!(conjugate_one(Gate1::H, Z), SingleRule { delta: 0, label: X });
        assert_eq!(conjugate_one(Gate1::H, Y), SingleRule { delta: 2, label: Y });
        // S sends X to Y and Y to -X.
        assert_eq!(conjugate_one(Gate1::S, X), SingleRule { delta: 0, label: Y });
        assert_eq!(conjugate_one(Gate1::S, Y), SingleRule { delta: 2, label: X });
        // CX: X_c -> X_c X_t, Z_t -> Z_c Z_t.
        assert_eq!(
            conjugate_pair(Gate2::Cx, X, I),
            PairRule { delta: 0, first: X, second: X }
        );
        assert_eq!(
            conjugate_pair(Gate2::Cx, I, Z),
            PairRule { delta: 0, first: Z, second: Z }
        );
        // CZ: X_c -> X_c Z_t.
        assert_eq!(
            conjugate_pair(Gate2::Cz, X, I),
            PairRule { delta: 0, first: X, second: Z }
        );
        // SWAP exchanges the labels and nothing else.
        assert_eq!(
            conjugate_pair(Gate2::Swap, X, Z),
            PairRule { delta: 0, first: Z, second: X }
        );
    }

    #[test]
    fn every_single_gate_conjugation_is_a_permutation_of_the_pauli_group() {
        for gate in GATES_ONE {
            let mut seen = Vec::new();
            for label in 0..4u8 {
                seen.push(conjugate_one(gate, label).label);
            }
            seen.sort_unstable();
            assert_eq!(seen, vec![I, X, Y, Z], "{gate:?} is not invertible");
        }
        for gate in GATES_TWO {
            let mut count = 0;
            for left in 0..4u8 {
                for right in 0..4u8 {
                    let rule = conjugate_pair(gate, left, right);
                    assert!(rule.first < 4 && rule.second < 4);
                    count += 1;
                }
            }
            assert_eq!(count, 16, "{gate:?} does not act on the 16 two-qubit Paulis");
        }
    }

    #[test]
    fn multiplying_strings_carries_the_phase_through_every_label() {
        // (X (x) Z) * (Y (x) Z) = (iZ) (x) I
        let (labels, phase) = multiply(&[X, Z], 0, &[Y, Z], 0);
        assert_eq!(labels, vec![Z, I]);
        assert_eq!(phase, 1);
        let (labels, phase) = multiply(&[X, Z], 1, &[X, Z], 1);
        assert_eq!(labels, vec![I, I]);
        assert_eq!(phase, 2);
    }

    #[test]
    fn mask_placement_matches_the_python_encoding() {
        let mask = to_mask(&[X, Y, Z, I]);
        // 4 columns per half: x half in bits 0..4, z half in 4..8.
        // X sets only its x bit, Y sets both halves, Z only its z bit.
        assert!(mask.get(0) && mask.get(1) && !mask.get(2) && !mask.get(3));
        assert!(!mask.get(4) && mask.get(5) && mask.get(6) && !mask.get(7));
        assert_eq!(mask.len(), 8);
    }

    #[test]
    fn strings_are_formatted_like_the_python_reports() {
        assert_eq!(format_string(&[X, Z, I], 0), "+XZI");
        assert_eq!(format_string(&[Z, Z], 2), "-ZZ");
        assert_eq!(format_string(&[I], 1), "+iI");
    }
}
