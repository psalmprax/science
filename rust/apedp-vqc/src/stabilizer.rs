//! The stabilizer tableau: `n` commuting signed Pauli generators, updated by
//! Clifford conjugation, measured one qubit at a time.
//!
//! This is a port of `apedp_dyn/vqc/stabilizer.py::StabilizerState`, and the
//! port is meant to be indistinguishable from it: the generator order, the
//! "first generator that anticommutes" choice, the GF(2) elimination order and
//! the phase arithmetic are all the same, so for a *deterministic* quantity
//! (an image, a phase, a probability, a distribution) the two implementations
//! must return the same answer exactly, not approximately. The only place they
//! may differ is where a random number is drawn, and there the distributions
//! agree rather than the draws.
//!
//! What this buys, per the review this module answers: 100-qubit sampling stops
//! costing Python interpreter time per gate per shot, which is the dominant
//! term at that width.

use std::collections::BTreeMap;

use crate::bits::Bitset;
use crate::pauli::{
    conjugate_one, conjugate_pair, format_string, multiply, to_mask, unit_string, Clifford, X, Y, Z,
};

/// A refusal, carrying the reason in the same words the Python tier uses.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VqcError {
    /// The circuit or state is outside this tier: named, never approximated.
    Unsupported(String),
    /// A caller mistake: a shot count, an index, an impossible branch.
    Value(String),
}

impl VqcError {
    pub fn message(&self) -> &str {
        match self {
            VqcError::Unsupported(message) | VqcError::Value(message) => message,
        }
    }
}

/// `splitmix64`, used to turn a seed into a stream of uniforms.
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        // Any seed is legal, including zero, because splitmix64 is a bijection
        // with no fixed point at zero.
        Rng { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A uniform in `[0, 1)` with 53 bits of precision.
    pub fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }
}

/// A pure stabilizer state on `n` qubits.
#[derive(Clone, Debug)]
pub struct Stabilizer {
    num_qubits: usize,
    generators: Vec<Vec<u8>>,
    phases: Vec<i32>,
}

impl Stabilizer {
    /// `|0...0>`, the `+1` eigenspace of every `Z_q`.
    pub fn new(num_qubits: usize) -> Result<Self, VqcError> {
        if num_qubits < 1 {
            return Err(VqcError::Value(format!(
                "num_qubits must be positive, got {num_qubits}"
            )));
        }
        let mut generators = Vec::with_capacity(num_qubits);
        for qubit in 0..num_qubits {
            generators.push(unit_string(num_qubits, qubit, Z));
        }
        Ok(Stabilizer {
            num_qubits,
            generators,
            phases: vec![0; num_qubits],
        })
    }

    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// The generators as `'+ZZI'`-style strings, for reports and tests.
    pub fn generator_strings(&self) -> Vec<String> {
        self.generators
            .iter()
            .zip(self.phases.iter())
            .map(|(labels, phase)| format_string(labels, *phase))
            .collect()
    }

    /// A pivot-indexed GF(2) basis of the generator group, with its phases.
    fn echelon(&self) -> Result<BTreeMap<usize, (Bitset, i32)>, VqcError> {
        let mut basis: BTreeMap<usize, (Bitset, i32)> = BTreeMap::new();
        for (labels, phase) in self.generators.iter().zip(self.phases.iter()) {
            let mut mask = to_mask(labels);
            let mut current = phase.rem_euclid(4);
            loop {
                let Some(pivot) = mask.lowest() else {
                    // A dependency: the phases must cancel, or the state would be
                    // its own negative and this is not a stabilizer at all.
                    if current % 2 != 0 {
                        return Err(VqcError::Unsupported(
                            "the generators are not a consistent stabilizer state".to_string(),
                        ));
                    }
                    break;
                };
                match basis.get(&pivot) {
                    None => {
                        basis.insert(pivot, (mask, current));
                        break;
                    }
                    Some((existing, existing_phase)) => {
                        mask.xor_assign(existing);
                        current = (current + existing_phase).rem_euclid(4);
                    }
                }
            }
        }
        Ok(basis)
    }

    /// The phase `i**p` with which `labels` appears in the group, or `None`.
    pub fn phase_of(&self, labels: &[u8]) -> Result<Option<i32>, VqcError> {
        let basis = self.echelon()?;
        let mut mask = to_mask(labels);
        let mut phase = 0i32;
        while let Some(pivot) = mask.lowest() {
            let Some((existing, existing_phase)) = basis.get(&pivot) else {
                return Ok(None);
            };
            mask.xor_assign(existing);
            phase = (phase + existing_phase).rem_euclid(4);
        }
        Ok(Some(phase))
    }

    /// The phase of a group element, refusing an imaginary one.
    pub fn expectation(&self, labels: &[u8]) -> Result<f64, VqcError> {
        if labels.len() != self.num_qubits {
            return Err(VqcError::Value(format!(
                "expected {} labels, got {}",
                self.num_qubits,
                labels.len()
            )));
        }
        match self.phase_of(labels)? {
            None => Ok(0.0),
            Some(phase) if phase % 2 != 0 => Err(VqcError::Unsupported(
                "a Hermitian operator cannot have an imaginary eigenvalue here".to_string(),
            )),
            Some(phase) => Ok(if phase == 0 { 1.0 } else { -1.0 }),
        }
    }

    /// Conjugate every generator by one Clifford gate.
    pub fn apply(&mut self, gate: Clifford, targets: &[usize]) -> Result<(), VqcError> {
        if targets.len() != gate.arity() {
            return Err(VqcError::Value(format!(
                "{} takes {} qubits, got {}",
                gate.name(),
                gate.arity(),
                targets.len()
            )));
        }
        for target in targets {
            if *target >= self.num_qubits {
                return Err(VqcError::Value(format!("qubit {target} out of range")));
            }
        }
        match gate {
            Clifford::One(one) => {
                let qubit = targets[0];
                for index in 0..self.generators.len() {
                    let rule = conjugate_one(one, self.generators[index][qubit]);
                    self.generators[index][qubit] = rule.label;
                    self.phases[index] = (self.phases[index] + rule.delta).rem_euclid(4);
                }
            }
            Clifford::Two(two) => {
                let (first, second) = (targets[0], targets[1]);
                for index in 0..self.generators.len() {
                    let rule = conjugate_pair(
                        two,
                        self.generators[index][first],
                        self.generators[index][second],
                    );
                    self.generators[index][first] = rule.first;
                    self.generators[index][second] = rule.second;
                    self.phases[index] = (self.phases[index] + rule.delta).rem_euclid(4);
                }
            }
        }
        Ok(())
    }

    /// Index of the first generator that anticommutes with `Z_q`, if any.
    fn anticommuting(&self, qubit: usize) -> Option<usize> {
        self.generators
            .iter()
            .position(|labels| labels[qubit] == X || labels[qubit] == Y)
    }

    /// `P(Z_q = 1)`: `0.5`, or exactly `0.0` / `1.0`.
    pub fn outcome_of(&self, qubit: usize) -> Result<f64, VqcError> {
        if qubit >= self.num_qubits {
            return Err(VqcError::Value(format!("qubit {qubit} out of range")));
        }
        if self.anticommuting(qubit).is_some() {
            return Ok(0.5);
        }
        let labels = unit_string(self.num_qubits, qubit, Z);
        match self.phase_of(&labels)? {
            None => Err(VqcError::Unsupported(format!(
                "Z_{qubit} commutes with every generator but is not in the group"
            ))),
            Some(phase) => Ok(if phase == 2 { 1.0 } else { 0.0 }),
        }
    }

    /// Project onto `Z_q = (-1)**outcome` and rewrite the generators.
    fn collapse(&mut self, qubit: usize, outcome: bool) -> Result<(), VqcError> {
        let Some(index) = self.anticommuting(qubit) else {
            return Ok(());
        };
        let anticommuting = self.generators[index].clone();
        let anticommuting_phase = self.phases[index];
        for other in 0..self.generators.len() {
            if other == index {
                continue;
            }
            if self.generators[other][qubit] == X || self.generators[other][qubit] == Y {
                let (labels, phase) = multiply(
                    &self.generators[other],
                    self.phases[other],
                    &anticommuting,
                    anticommuting_phase,
                );
                self.generators[other] = labels;
                self.phases[other] = phase;
            }
        }
        self.generators[index] = unit_string(self.num_qubits, qubit, Z);
        self.phases[index] = if outcome { 2 } else { 0 };
        Ok(())
    }

    /// Measure `Z_q`, collapsing onto the branch; `outcome` forces the branch.
    pub fn measure(
        &mut self,
        qubit: usize,
        outcome: Option<bool>,
        rng: &mut Rng,
    ) -> Result<bool, VqcError> {
        if qubit >= self.num_qubits {
            return Err(VqcError::Value(format!("qubit {qubit} out of range")));
        }
        let probability_one = self.outcome_of(qubit)?;
        let outcome = match outcome {
            None => rng.uniform() < probability_one,
            Some(forced) => {
                if (probability_one == 0.0 && forced) || (probability_one == 1.0 && !forced) {
                    return Err(VqcError::Value(
                        "branch has zero probability; it cannot be executed".to_string(),
                    ));
                }
                forced
            }
        };
        self.collapse(qubit, outcome)?;
        Ok(outcome)
    }

    /// `Measure` followed by a conditional `X`: the qubit ends in `|0>`.
    pub fn reset(&mut self, qubit: usize, rng: &mut Rng) -> Result<(), VqcError> {
        if self.measure(qubit, None, rng)? {
            self.apply(Clifford::from_name("x").expect("x is Clifford"), &[qubit])?;
        }
        Ok(())
    }

    /// One Z-basis outcome for every qubit, in order; destructively measures.
    pub fn sample(&mut self, rng: &mut Rng) -> Result<Vec<bool>, VqcError> {
        let mut bits = Vec::with_capacity(self.num_qubits);
        for qubit in 0..self.num_qubits {
            bits.push(self.measure(qubit, None, rng)?);
        }
        Ok(bits)
    }

    /// The exact probability of one Z-basis outcome, by chained measurement.
    pub fn probability_of_bitstring(&self, bits: &[bool]) -> Result<f64, VqcError> {
        if bits.len() != self.num_qubits {
            return Err(VqcError::Value(format!(
                "expected {} bits, got {}",
                self.num_qubits,
                bits.len()
            )));
        }
        let mut state = self.clone();
        let mut probability = 1.0f64;
        for (qubit, bit) in bits.iter().enumerate() {
            let one = state.outcome_of(qubit)?;
            if one == 0.5 {
                probability *= 0.5;
            } else if (one == 1.0) != *bit {
                return Ok(0.0);
            }
            state.collapse(qubit, *bit)?;
        }
        Ok(probability)
    }

    /// Every outcome with nonzero probability, for `n <= 12`.
    pub fn distribution(&self) -> Result<Vec<(Vec<bool>, f64)>, VqcError> {
        if self.num_qubits > 12 {
            return Err(VqcError::Unsupported(format!(
                "enumerating 2**{} outcomes is not a report anyone reads; use \
                 probability_of_bitstring or sample",
                self.num_qubits
            )));
        }
        let mut outcomes = Vec::new();
        for index in 0..(1usize << self.num_qubits) {
            let bits: Vec<bool> = (0..self.num_qubits)
                .map(|qubit| index >> (self.num_qubits - 1 - qubit) & 1 == 1)
                .collect();
            let probability = self.probability_of_bitstring(&bits)?;
            if probability != 0.0 {
                outcomes.push((bits, probability));
            }
        }
        Ok(outcomes)
    }

    /// Refuse a tableau that is not `n` independent, commuting generators.
    pub fn validate(&self) -> Result<(), VqcError> {
        let n = self.num_qubits;
        if self.generators.len() != n || self.phases.len() != n {
            return Err(VqcError::Unsupported(format!(
                "{n} qubits need {n} generators, got {} generators and {} phases",
                self.generators.len(),
                self.phases.len()
            )));
        }
        let masks: Vec<Bitset> = self
            .generators
            .iter()
            .map(|labels| to_mask(labels))
            .collect();
        let low = Bitset::from_bits(&vec![true; n]);
        for first in 0..n {
            for second in (first + 1)..n {
                let x_first = masks[first].clone();
                let z_first = shift_right(&masks[first], n);
                let x_second = masks[second].clone();
                let z_second = shift_right(&masks[second], n);
                // Symplectic product: one iff the two strings anticommute.
                if x_first.parity_of(&z_second, &z_first, &x_second) {
                    return Err(VqcError::Unsupported(format!(
                        "generators {first} and {second} anticommute ({}, {}), so there is no \
                         simultaneous eigenspace and no state",
                        self.generator_strings()[first],
                        self.generator_strings()[second]
                    )));
                }
            }
        }
        let _ = low;
        if self.echelon()?.len() != n {
            return Err(VqcError::Unsupported(format!(
                "the {n} generators span only rank {}, so the group has fewer than 2**{n} \
                 elements and the state is not pure: {:?}",
                self.echelon()?.len(),
                self.generator_strings()
            )));
        }
        Ok(())
    }
}

/// The `z` half of a mask, aligned so that bit `q` is qubit `q`.
fn shift_right(mask: &Bitset, width: usize) -> Bitset {
    let mut out = Bitset::zeros(width);
    for index in 0..width {
        if mask.get(width + index) {
            out.set(index);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pauli::{I, LABEL_CHARS, X, Z};

    fn ghz(num_qubits: usize) -> Stabilizer {
        let mut state = Stabilizer::new(num_qubits).unwrap();
        state
            .apply(Clifford::from_name("h").unwrap(), &[0])
            .unwrap();
        for qubit in 1..num_qubits {
            state
                .apply(Clifford::from_name("cx").unwrap(), &[qubit - 1, qubit])
                .unwrap();
        }
        state
    }

    #[test]
    fn the_initial_state_stabilizes_every_z() {
        let mut state = Stabilizer::new(4).unwrap();
        for qubit in 0..4 {
            assert_eq!(state.outcome_of(qubit).unwrap(), 0.0);
            let expected: String = (0..4)
                .map(|position| if position == qubit { 'Z' } else { 'I' })
                .collect();
            assert_eq!(state.generator_strings()[qubit], format!("+{expected}"));
        }
        assert_eq!(state.sample(&mut Rng::new(7)).unwrap(), vec![false; 4]);
    }

    #[test]
    fn the_ghz_state_has_the_closed_form_stabilizers() {
        let state = ghz(5);
        let x_all = vec![X; 5];
        assert_eq!(state.phase_of(&x_all).unwrap(), Some(0));
        for qubit in 0..4 {
            let mut labels = vec![I; 5];
            labels[qubit] = Z;
            labels[qubit + 1] = Z;
            assert_eq!(state.phase_of(&labels).unwrap(), Some(0));
        }
        let mut z0 = vec![I; 5];
        z0[0] = Z;
        assert_eq!(state.phase_of(&z0).unwrap(), None);
    }

    #[test]
    fn ghz_sampling_sees_only_the_two_parity_branches() {
        let state = ghz(6);
        let distribution = state.distribution().unwrap();
        assert_eq!(distribution.len(), 2);
        for (bits, probability) in distribution {
            assert_eq!(bits.iter().filter(|bit| **bit).count() % 2, 0);
            assert!((probability - 0.5).abs() < 1e-15);
        }
    }

    #[test]
    fn measurement_collapses_onto_the_sampled_branch() {
        let mut state = ghz(3);
        let mut rng = Rng::new(11);
        let outcome = state.measure(0, None, &mut rng).unwrap();
        // After measuring one qubit the chain is a definite bitstring.
        let bits = state.sample(&mut rng).unwrap();
        assert!(bits.iter().all(|bit| *bit == bits[0]));
        assert_eq!(bits[0], outcome);
    }

    #[test]
    fn a_zero_probability_branch_is_refused_by_name() {
        let mut state = Stabilizer::new(2).unwrap();
        let mut rng = Rng::new(1);
        let error = state.measure(0, Some(true), &mut rng).unwrap_err();
        assert_eq!(
            error,
            VqcError::Value("branch has zero probability; it cannot be executed".to_string())
        );
    }

    #[test]
    fn reset_returns_a_qubit_to_zero() {
        for seed in 0..8 {
            let mut state = ghz(4);
            let mut rng = Rng::new(seed);
            state.reset(2, &mut rng).unwrap();
            assert_eq!(state.outcome_of(2).unwrap(), 0.0);
        }
    }

    #[test]
    fn the_validate_path_refuses_anticommuting_generators() {
        let mut state = Stabilizer::new(2).unwrap();
        state.generators[0] = vec![X, I];
        state.generators[1] = vec![Z, I];
        let error = state.validate().unwrap_err();
        assert!(error.message().contains("anticommute"), "{error:?}");
    }

    #[test]
    fn label_chars_cover_every_label_used_above() {
        assert_eq!(LABEL_CHARS[Z as usize], 'Z');
    }
}
